import copy
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from brain_state_common import CACHE_ROOT, CHANNEL_NAMES, MODEL_ROOT, electrode_adjacency, load_preprocessed_recording, recording_normalizations, torch_device
from brain_state_model import effective_rank, exponential_moving_average
from visualization_model import FactorizedBrainStateModel, FactorizedEEGDataset, augment_eeg_v2, build_visualization_encoder_from_config, factorized_losses


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = PROJECT_ROOT / "universal_brain_recorder" / "labeled_dataset"
AUTOENCODER_MANIFEST = LABELED_ROOT / "autoencoder_manifest.csv"
CACHE_DIR = CACHE_ROOT
MODEL_DIR = MODEL_ROOT
TARGET_SAMPLE_RATE = 125.0
LOW_HZ = 0.5
HIGH_HZ = 45.0
NOTCH_HZ = 50.0
WINDOW_SECONDS_MULTI = [2.0, 4.0, 8.0]
FUTURE_HORIZON_SECONDS = [0.5, 2.0, 8.0]
STEP_SECONDS = 1.0
CLIP_VALUE = 12.0
SPATIAL_FEATURES = 10
TEMPORAL_WIDTH = 112
EMBEDDING_DIM = 160
SESSION_DIM = 48
ARTIFACT_DIM = 32
DROPOUT = 0.12
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1.8e-4
WEIGHT_DECAY = 2e-4
EMA_DECAY = 0.996
EARLY_STOPPING_PATIENCE = 5
MODEL_SELECTION_START_EPOCH = 6
NUM_WORKERS = 0
DEVICE = "auto"
RANDOM_SEED = 317
VALIDATION_FRACTION = 0.20
CROSS_SESSION_SAMPLES_PER_SESSION = 2
ARTIFACT_SAMPLES_PER_CLASS = 2
INITIAL_MODEL_PATH = MODEL_DIR / "visualization_model.pt"
LOSS_WEIGHTS = {
    "invariance": 8.0,
    "variance": 12.0,
    "covariance": 0.5,
    "private_consistency": 0.3,
    "future_prediction": 2.0,
    "acceleration": 1.2,
    "scale_consistency": 4.0,
    "reconstruction": 1.5,
    "task_classification": 1.5,
    "state_classification": 0.8,
    "session_private": 0.6,
    "artifact_private": 0.6,
    "session_adversarial": 0.0,
    "artifact_adversarial": 0.75,
    "conditional_session_adversarial": 1.0,
    "task_cross_session": 0.75,
    "state_cross_session": 0.35,
    "task_session_alignment": 1.5,
    "state_session_alignment": 0.75,
    "artifact_alignment": 0.5,
    "orthogonality": 1.0,
    "reliability": 0.10,
    "scale_entropy": 0.02,
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coarse_state_label(value, task_family):
    label = str(value)
    if label in {"unlabeled_transition", "invalid_stream_sample"}:
        return label
    if task_family == "image_emotion_cognition":
        if label.startswith("fixation"):
            return "fixation"
        if label.startswith("washout"):
            return "washout"
        if label.startswith("break"):
            return "break"
        if label.startswith("image_"):
            return re.sub(r"_\d+$", "", label)
    if task_family == "basic_motor_cognition":
        return re.sub(r"_\d+$", "", label)
    if task_family == "ssvep_flicker":
        if label.startswith("flicker"):
            return "flicker"
        if label.startswith("rest") or label.startswith("baseline"):
            return "non_flicker"
    return re.sub(r"_\d+$", "", label)


def confidence_weight(value):
    return {"high": 1.0, "medium": 0.70, "low": 0.35}.get(str(value).lower(), 0.50)


def load_training_sources():
    manifest = pd.read_csv(AUTOENCODER_MANIFEST)
    raw = manifest[manifest["is_raw_authoritative_recording"].astype(bool)].copy()
    raw = raw[raw["recommended_for_autoencoder"].astype(bool)].copy()
    raw = raw.sort_values(["primary_task_family", "session_id"]).reset_index(drop=True)
    recordings = []
    labels = []
    for index, row in raw.iterrows():
        source_path = PROJECT_ROOT / row["source_eeg_path"]
        label_path = PROJECT_ROOT / row["sample_labels_path"]
        print(f"Loading source {index + 1}/{len(raw)}: {source_path}")
        recording = load_preprocessed_recording(source_path, CACHE_DIR, TARGET_SAMPLE_RATE, LOW_HZ, HIGH_HZ, NOTCH_HZ)
        label_frame = pd.read_csv(
            label_path,
            usecols=["canonical_task_family", "canonical_task_label", "segment_index", "is_valid_eeg_sample"],
        )
        source_rate = float(row["sample_frequency_hz"])
        ratio = source_rate / TARGET_SAMPLE_RATE
        factor = int(round(ratio))
        if abs(ratio - factor) > 1e-8:
            raise ValueError(f"Non-integer label downsampling ratio for {source_path}")
        label_frame = label_frame.iloc[::factor].reset_index(drop=True)
        usable_length = min(recording.data.shape[1], len(label_frame))
        recording.data = recording.data[:, :usable_length]
        label_frame = label_frame.iloc[:usable_length]
        task_family = str(row["primary_task_family"])
        state_labels = np.asarray([
            coarse_state_label(value, task_family)
            for value in label_frame["canonical_task_label"].astype(str).tolist()
        ], dtype=object)
        recordings.append(recording)
        labels.append({
            "task_family": task_family,
            "state_label": state_labels,
            "segment_index": label_frame["segment_index"].to_numpy(dtype=np.int32),
            "is_valid": label_frame["is_valid_eeg_sample"].to_numpy(dtype=bool),
            "session_id": str(row["session_id"]),
            "label_weight": confidence_weight(row["label_confidence"]),
        })
    return raw, recordings, labels


def repeated_task_families(labels):
    counts = {}
    for label in labels:
        family = label["task_family"]
        counts[family] = counts.get(family, 0) + 1
    return {family for family, count in counts.items() if count >= 2}


def temporal_validation_ranges(recordings, labels=None):
    training_ranges = {}
    validation_ranges = {}
    longest_window = int(round(max(WINDOW_SECONDS_MULTI) * TARGET_SAMPLE_RATE))
    longest_future = int(round(max(FUTURE_HORIZON_SECONDS) * TARGET_SAMPLE_RATE))
    nearest_future = int(round(min(FUTURE_HORIZON_SECONDS) * TARGET_SAMPLE_RATE))
    for index, recording in enumerate(recordings):
        sample_count = recording.data.shape[1]
        if labels is not None and labels[index]["task_family"] == "eyes_open_closed_and_blinks":
            training_ranges[index] = (0, sample_count)
            validation_ranges[index] = (sample_count + 1, sample_count)
            continue
        boundary = int(round(sample_count * (1.0 - VALIDATION_FRACTION)))
        training_ranges[index] = (0, boundary - longest_future)
        validation_ranges[index] = (boundary + longest_window + nearest_future, sample_count)
    return training_ranges, validation_ranges


def vocabularies(labels, training_indices, repeated_families):
    task_to_index = {value: index for index, value in enumerate(sorted(repeated_families))}
    states = set()
    for label in labels:
        family = label["task_family"]
        if family not in repeated_families:
            continue
        for value in np.unique(label["state_label"]):
            if value not in {"unlabeled_transition", "invalid_stream_sample"}:
                states.add(f"{family}::{value}")
    session_to_index = {
        labels[index]["session_id"]: position
        for position, index in enumerate(training_indices)
    }
    artifact_to_index = {"clean_reference": 0, "blink": 1, "jaw_emg": 2}
    return task_to_index, {value: index for index, value in enumerate(sorted(states))}, session_to_index, artifact_to_index


def conditional_session_vocabulary(labels, repeated_families):
    mapping = {}
    counts = []
    for family in sorted(repeated_families):
        sessions = sorted(label["session_id"] for label in labels if label["task_family"] == family)
        for index, session_id in enumerate(sessions):
            mapping[(family, session_id)] = index
        counts.append(len(sessions))
    return mapping, counts


def balanced_sampler(dataset):
    recordings = np.asarray(dataset.recording_window_indices, dtype=int)
    unique, counts = np.unique(recordings, return_counts=True)
    count_by_recording = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_recording[value] for value in recordings], dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True)


class SessionBalancedCrossTaskBatchSampler:
    def __init__(self, dataset, batch_size, seed):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.iteration = 0
        self.batch_count = max(1, len(dataset) // self.batch_size)
        self.by_recording = {}
        self.by_task_state_session = {}
        self.by_recording_artifact = {}
        for dataset_index, (recording, task, state, session, artifact) in enumerate(zip(
            dataset.recording_window_indices,
            dataset.task_window_indices,
            dataset.state_window_indices,
            dataset.session_window_indices,
            dataset.artifact_window_indices,
        )):
            self.by_recording.setdefault(recording, []).append(dataset_index)
            if task >= 0 and state >= 0 and session >= 0:
                self.by_task_state_session.setdefault(task, {}).setdefault(state, {}).setdefault(session, []).append(dataset_index)
            if artifact >= 0:
                self.by_recording_artifact.setdefault(recording, {}).setdefault(artifact, []).append(dataset_index)
        self.recordings = sorted(self.by_recording)
        self.shared_states = {
            task: sorted(state for state, sessions in states.items() if len(sessions) >= 2)
            for task, states in self.by_task_state_session.items()
        }
        self.shared_states = {task: states for task, states in self.shared_states.items() if states}
        self.artifact_recordings = sorted(
            recording for recording, artifacts in self.by_recording_artifact.items()
            if 0 in artifacts and any(artifact > 0 for artifact in artifacts)
        )

    def __len__(self):
        return self.batch_count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        for _ in range(self.batch_count):
            batch = []
            for task, shared_states in sorted(self.shared_states.items()):
                state = int(rng.choice(shared_states))
                sessions_by_state = self.by_task_state_session[task][state]
                sessions = sorted(sessions_by_state)
                selected_sessions = rng.choice(sessions, size=2, replace=False)
                for session in selected_sessions:
                    choices = sessions_by_state[int(session)]
                    selected = rng.choice(choices, size=CROSS_SESSION_SAMPLES_PER_SESSION, replace=len(choices) < CROSS_SESSION_SAMPLES_PER_SESSION)
                    batch.extend(int(value) for value in selected)
            for recording in self.artifact_recordings:
                artifacts = self.by_recording_artifact[recording]
                artifact = int(rng.choice(sorted(value for value in artifacts if value > 0)))
                for label in [0, artifact]:
                    choices = artifacts[label]
                    selected = rng.choice(choices, size=ARTIFACT_SAMPLES_PER_CLASS, replace=len(choices) < ARTIFACT_SAMPLES_PER_CLASS)
                    batch.extend(int(value) for value in selected)
            while len(batch) < self.batch_size:
                recording = int(rng.choice(self.recordings))
                batch.append(int(rng.choice(self.by_recording[recording])))
            rng.shuffle(batch)
            yield batch[:self.batch_size]


def adversarial_schedule(epoch, epochs):
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, epochs - 1)))
    return float(2.0 / (1.0 + math.exp(-8.0 * progress)) - 1.0)


def validation_selection_score(metrics):
    score = metrics["total"]
    for name in ["session_adversarial", "artifact_adversarial", "conditional_session_adversarial"]:
        score -= LOSS_WEIGHTS[name] * metrics[name]
    return float(score)


def run_epoch(model, loader, optimizer, scaler, device, train_mode, adversarial_strength, ema_model=None):
    model.train(train_mode)
    totals = {}
    sample_count = 0
    for batch in loader:
        batch = [value.to(device, non_blocking=True) for value in batch]
        previous, current, futures, task_labels, state_labels, session_labels, artifact_labels, recording_indices, stable_previous, stable_futures, domain_valid, label_weights, conditional_session_labels = batch
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, metrics = factorized_losses(
                    model,
                    previous,
                    current,
                    futures,
                    task_labels,
                    state_labels,
                    session_labels,
                    artifact_labels,
                    recording_indices,
                    stable_previous,
                    stable_futures,
                    domain_valid,
                    label_weights,
                    conditional_session_labels,
                    LOSS_WEIGHTS,
                    adversarial_strength,
                )
            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    exponential_moving_average(ema_model, model, EMA_DECAY)
        batch_count = current.shape[0]
        sample_count += batch_count
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_count
    return {name: value / max(1, sample_count) for name, value in totals.items()}


def collect_embeddings(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    state_rows = []
    session_rows = []
    artifact_rows = []
    task_rows = []
    state_label_rows = []
    session_label_rows = []
    artifact_label_rows = []
    task_prediction_rows = []
    state_prediction_rows = []
    session_prediction_rows = []
    dropout_similarities = []
    scale_disagreements = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            current = batch[1].to(device)
            task_labels = batch[3]
            state_labels = batch[4]
            session_labels = batch[5]
            artifact_labels = batch[6]
            factors = model.encoder.encode_factors(current, return_scales=True)
            state = factors["state"]
            state_rows.append(state.cpu().numpy())
            session_rows.append(factors["session"].cpu().numpy())
            artifact_rows.append(factors["artifact"].cpu().numpy())
            task_rows.append(task_labels.numpy())
            state_label_rows.append(state_labels.numpy())
            session_label_rows.append(session_labels.numpy())
            artifact_label_rows.append(artifact_labels.numpy())
            task_prediction_rows.append(torch.argmax(model.task_head(state), dim=1).cpu().numpy())
            state_prediction_rows.append(torch.argmax(model.state_head(state), dim=1).cpu().numpy())
            session_prediction_rows.append(torch.argmax(model.session_private_head(factors["session"]), dim=1).cpu().numpy())
            normalized_scales = F.normalize(factors["state_scales"], dim=2)
            normalized_state = F.normalize(state, dim=1)[:, None, :]
            scale_disagreements.extend((1.0 - torch.sum(normalized_scales * normalized_state, dim=2)).mean(dim=1).cpu().numpy().tolist())
            corrupted = current.clone()
            channel_indices = torch.arange(len(current), device=device) % current.shape[1]
            corrupted[torch.arange(len(current), device=device), channel_indices] = 0.0
            corrupted_state = model.encoder(corrupted)
            dropout_similarities.extend(F.cosine_similarity(state, corrupted_state, dim=1).cpu().numpy().tolist())
    return {
        "state": np.concatenate(state_rows),
        "session": np.concatenate(session_rows),
        "artifact": np.concatenate(artifact_rows),
        "task_labels": np.concatenate(task_rows),
        "state_labels": np.concatenate(state_label_rows),
        "session_labels": np.concatenate(session_label_rows),
        "artifact_labels": np.concatenate(artifact_label_rows),
        "task_predictions": np.concatenate(task_prediction_rows),
        "state_predictions": np.concatenate(state_prediction_rows),
        "session_predictions": np.concatenate(session_prediction_rows),
        "scale_disagreement_mean": float(np.mean(scale_disagreements)),
        "single_channel_dropout_cosine_mean": float(np.mean(dropout_similarities)),
    }


def balanced_accuracy_or_none(labels, predictions):
    valid = labels >= 0
    if not np.any(valid) or len(np.unique(labels[valid])) < 2:
        return None
    return float(balanced_accuracy_score(labels[valid], predictions[valid]))


def posthoc_probe_accuracy(embedding, labels):
    valid = labels >= 0
    embedding = embedding[valid]
    labels = labels[valid]
    values, counts = np.unique(labels, return_counts=True)
    usable = values[counts >= 4]
    keep = np.isin(labels, usable)
    embedding = embedding[keep]
    labels = labels[keep]
    if len(usable) < 2:
        return None
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    model.fit(embedding[train_indices], labels[train_indices])
    return float(balanced_accuracy_score(labels[test_indices], model.predict(embedding[test_indices])))


def temporal_metrics(embeddings, dataset):
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    velocities = []
    accelerations = []
    for index in range(1, len(dataset.indices)):
        first_recording, first_end = dataset.indices[index - 1]
        second_recording, second_end = dataset.indices[index]
        if first_recording == second_recording and second_end - first_end == dataset.step_samples:
            velocities.append(float(np.linalg.norm(normalized[index] - normalized[index - 1])))
    for index in range(2, len(dataset.indices)):
        first_recording, first_end = dataset.indices[index - 2]
        second_recording, second_end = dataset.indices[index - 1]
        third_recording, third_end = dataset.indices[index]
        if first_recording == second_recording == third_recording and second_end - first_end == third_end - second_end == dataset.step_samples:
            accelerations.append(float(np.linalg.norm(normalized[index] - 2.0 * normalized[index - 1] + normalized[index - 2])))
    return {
        "normalized_adjacent_distance_median": float(np.median(velocities)),
        "normalized_adjacent_distance_p95": float(np.percentile(velocities, 95)),
        "normalized_acceleration_median": float(np.median(accelerations)),
        "normalized_acceleration_p95": float(np.percentile(accelerations, 95)),
    }


def conditional_session_probe_metrics(embedding, task_labels, session_labels):
    by_task = {}
    for task in sorted(np.unique(task_labels[task_labels >= 0]).tolist()):
        mask = task_labels == task
        by_task[str(task)] = posthoc_probe_accuracy(embedding[mask], session_labels[mask])
    values = [value for value in by_task.values() if value is not None]
    return by_task, float(np.mean(values)) if values else None


def representation_metrics(model, training_dataset, validation_dataset, device):
    training = collect_embeddings(model, training_dataset, device)
    validation = collect_embeddings(model, validation_dataset, device)
    conditional_by_task, conditional_mean = conditional_session_probe_metrics(
        training["state"],
        training["task_labels"],
        training["session_labels"],
    )
    quality = {
        "embedding_effective_rank_validation": effective_rank(validation["state"]),
        "heldout_time_task_balanced_accuracy": balanced_accuracy_or_none(validation["task_labels"], validation["task_predictions"]),
        "heldout_time_state_balanced_accuracy": balanced_accuracy_or_none(validation["state_labels"], validation["state_predictions"]),
        "training_session_private_balanced_accuracy": balanced_accuracy_or_none(training["session_labels"], training["session_predictions"]),
        "posthoc_session_probe_from_state_balanced_accuracy": posthoc_probe_accuracy(training["state"], training["session_labels"]),
        "posthoc_session_probe_from_private_balanced_accuracy": posthoc_probe_accuracy(training["session"], training["session_labels"]),
        "conditional_session_probe_from_state_by_task": conditional_by_task,
        "conditional_session_probe_from_state_mean": conditional_mean,
        "posthoc_artifact_probe_from_state_balanced_accuracy": posthoc_probe_accuracy(training["state"], training["artifact_labels"]),
        "posthoc_artifact_probe_from_private_balanced_accuracy": posthoc_probe_accuracy(training["artifact"], training["artifact_labels"]),
        "single_channel_dropout_cosine_mean": validation["single_channel_dropout_cosine_mean"],
        "scale_disagreement_mean": validation["scale_disagreement_mean"],
        "training_windows_evaluated": int(len(training["state"])),
        "validation_windows_evaluated": int(len(validation["state"])),
    }
    quality.update(temporal_metrics(validation["state"], validation_dataset))
    artifact_probe = quality["posthoc_artifact_probe_from_state_balanced_accuracy"]
    quality["provisional_invariance_gate_passed"] = bool(
        conditional_mean is not None
        and conditional_mean <= 0.75
        and artifact_probe is not None
        and artifact_probe <= 0.80
        and quality["heldout_time_task_balanced_accuracy"] is not None
        and quality["heldout_time_task_balanced_accuracy"] >= 0.85
        and quality["single_channel_dropout_cosine_mean"] >= 0.97
        and quality["normalized_adjacent_distance_median"] <= 0.35
    )
    return quality


def initialize_backbone(encoder):
    model_path = INITIAL_MODEL_PATH
    if not model_path.exists():
        return None
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    architecture = str(bundle.get("config", {}).get("architecture"))
    if architecture == "channel_aware_multiscale_time_frequency_v2":
        encoder.backbone.load_state_dict(bundle["encoder_state"])
    elif architecture == "visualization_model_v1":
        backbone_state = {
            name.removeprefix("backbone."): value
            for name, value in bundle["encoder_state"].items()
            if name.startswith("backbone.")
        }
        encoder.backbone.load_state_dict(backbone_state)
    else:
        return None
    return str(model_path)


def main():
    set_random_seed(RANDOM_SEED)
    device = torch_device(DEVICE)
    source_frame, recordings, labels = load_training_sources()
    repeated_families = repeated_task_families(labels)
    if not repeated_families:
        raise RuntimeError("Visualization Model training requires at least one task family recorded in multiple sessions")
    training_indices = list(range(len(labels)))
    validation_indices = list(range(len(labels)))
    training_ranges, validation_ranges = temporal_validation_ranges(recordings, labels)
    task_to_index, state_to_index, session_to_index, artifact_to_index = vocabularies(labels, training_indices, repeated_families)
    conditional_session_to_index, conditional_session_counts = conditional_session_vocabulary(labels, repeated_families)
    centers, scales = recording_normalizations(recordings)
    global_center = np.median(centers[training_indices], axis=0).astype(np.float32)
    global_scale = np.median(scales[training_indices], axis=0).astype(np.float32)
    max_window_samples = int(round(max(WINDOW_SECONDS_MULTI) * TARGET_SAMPLE_RATE))
    shortest_window_samples = int(round(min(WINDOW_SECONDS_MULTI) * TARGET_SAMPLE_RATE))
    step_samples = int(round(STEP_SECONDS * TARGET_SAMPLE_RATE))
    future_horizon_samples = [int(round(value * TARGET_SAMPLE_RATE)) for value in FUTURE_HORIZON_SECONDS]
    dataset_arguments = (
        recordings,
        labels,
        max_window_samples,
        shortest_window_samples,
        step_samples,
        future_horizon_samples,
        centers,
        scales,
        task_to_index,
        state_to_index,
        session_to_index,
        artifact_to_index,
        repeated_families,
    )
    training_dataset = FactorizedEEGDataset(dataset_arguments[0], dataset_arguments[1], training_indices, *dataset_arguments[2:], training_ranges, CLIP_VALUE, conditional_session_to_index)
    validation_dataset = FactorizedEEGDataset(dataset_arguments[0], dataset_arguments[1], validation_indices, *dataset_arguments[2:], validation_ranges, CLIP_VALUE, conditional_session_to_index)
    training_loader = DataLoader(
        training_dataset,
        batch_sampler=SessionBalancedCrossTaskBatchSampler(training_dataset, BATCH_SIZE, RANDOM_SEED),
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    adjacency = electrode_adjacency(CHANNEL_NAMES)
    config = {
        "architecture": "visualization_model_v1",
        "channel_count": len(CHANNEL_NAMES),
        "spatial_features": SPATIAL_FEATURES,
        "temporal_width": TEMPORAL_WIDTH,
        "embedding_dim": EMBEDDING_DIM,
        "session_dim": SESSION_DIM,
        "artifact_dim": ARTIFACT_DIM,
        "nuisance_residualization": True,
        "dropout": DROPOUT,
        "sample_rate": TARGET_SAMPLE_RATE,
        "window_seconds": max(WINDOW_SECONDS_MULTI),
        "window_seconds_multi": WINDOW_SECONDS_MULTI,
        "future_horizon_seconds": FUTURE_HORIZON_SECONDS,
        "step_seconds": STEP_SECONDS,
        "clip_value": CLIP_VALUE,
        "normalization_strategy": "per_session_channelwise_robust_median_mad",
        "frequency_bands_hz": [[0.5, 4.0], [4.0, 8.0], [8.0, 12.0], [12.0, 16.0], [16.0, 24.0], [24.0, 32.0], [32.0, 45.0]],
        "low_hz": LOW_HZ,
        "high_hz": HIGH_HZ,
        "notch_hz": NOTCH_HZ,
        "canonical_label_version": "brainz_canonical_tasks_v1",
        "validation_strategy": "per_session_final_time_block_with_window_and_future_embargo_single_blink_segment_training_only",
        "validation_fraction": VALIDATION_FRACTION,
    }
    encoder = build_visualization_encoder_from_config(config, adjacency)
    initialized_from = initialize_backbone(encoder)
    model = FactorizedBrainStateModel(
        encoder,
        len(task_to_index),
        len(state_to_index),
        len(session_to_index),
        len(artifact_to_index),
        len(FUTURE_HORIZON_SECONDS),
        len(CHANNEL_NAMES),
        conditional_session_counts=conditional_session_counts,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS), eta_min=LEARNING_RATE * 0.05)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    history = []
    best_validation = math.inf
    best_state = None
    epochs_without_improvement = 0
    print("Device:", device)
    print("Initialized from:", initialized_from)
    print("Repeated task families:", sorted(repeated_families))
    print("Training sessions:", [labels[index]["session_id"] for index in training_indices])
    print("Validation mode: leakage-embargoed final time blocks from every session")
    print("Training windows:", len(training_dataset))
    print("Validation windows:", len(validation_dataset))
    for epoch in range(1, EPOCHS + 1):
        adversarial_strength = adversarial_schedule(epoch, EPOCHS)
        training_metrics = run_epoch(model, training_loader, optimizer, scaler, device, True, adversarial_strength, ema_model)
        validation_metrics = run_epoch(ema_model, validation_loader, optimizer, scaler, device, False, 0.0)
        scheduler.step()
        validation_score = validation_selection_score(validation_metrics)
        history.append({
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "adversarial_strength": adversarial_strength,
            "selection_score": validation_score,
            "train": training_metrics,
            "validation": validation_metrics,
        })
        print(
            f"Epoch {epoch:03d} train={training_metrics['total']:.4f} "
            f"validation={validation_score:.4f} "
            f"task={validation_metrics['task_classification']:.4f} "
            f"future={validation_metrics['future_prediction']:.4f} "
            f"domain={adversarial_strength:.3f}"
        )
        if epoch >= MODEL_SELECTION_START_EPOCH and validation_score < best_validation:
            best_validation = validation_score
            best_state = {name: value.detach().cpu().clone() for name, value in ema_model.state_dict().items()}
            epochs_without_improvement = 0
        elif epoch >= MODEL_SELECTION_START_EPOCH:
            epochs_without_improvement += 1
        if epoch >= MODEL_SELECTION_START_EPOCH and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping")
            break
    if best_state is None:
        best_state = {name: value.detach().cpu().clone() for name, value in ema_model.state_dict().items()}
    ema_model.load_state_dict(best_state)
    quality = representation_metrics(ema_model, training_dataset, validation_dataset, device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"visualization_model_{stamp}.pt"
    metrics_path = MODEL_DIR / f"visualization_model_metrics_{stamp}.json"
    bundle = {
        "format_version": 3,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder_state": {name.replace("encoder.", "", 1): value for name, value in best_state.items() if name.startswith("encoder.")},
        "task_head_state": {name.replace("task_head.", "", 1): value for name, value in best_state.items() if name.startswith("task_head.")},
        "state_head_state": {name.replace("state_head.", "", 1): value for name, value in best_state.items() if name.startswith("state_head.")},
        "config": config,
        "channel_names": CHANNEL_NAMES,
        "adjacency": adjacency,
        "normalization_center": global_center,
        "normalization_scale": global_scale,
        "task_to_index": task_to_index,
        "state_to_index": state_to_index,
        "session_to_index": session_to_index,
        "artifact_to_index": artifact_to_index,
        "conditional_session_to_index": conditional_session_to_index,
        "conditional_session_counts": conditional_session_counts,
        "repeated_task_families": sorted(repeated_families),
        "training_recordings": [str(recordings[index].path) for index in training_indices],
        "validation_recordings": [str(recordings[index].path) for index in validation_indices],
        "training_session_ids": [labels[index]["session_id"] for index in training_indices],
        "validation_session_ids": [labels[index]["session_id"] for index in validation_indices],
        "initialized_from": initialized_from,
        "initialization_scope": "backbone_only",
        "quality": quality,
    }
    torch.save(bundle, model_path)
    metrics = {
        "model_path": str(model_path),
        "device": str(device),
        "architecture": config["architecture"],
        "training_window_count": len(training_dataset),
        "validation_window_count": len(validation_dataset),
        "best_validation_loss": best_validation,
        "quality": quality,
        "history": history,
        "config": config,
        "loss_weights": LOSS_WEIGHTS,
        "task_to_index": task_to_index,
        "state_to_index": state_to_index,
        "session_to_index": session_to_index,
        "artifact_to_index": artifact_to_index,
        "conditional_session_to_index": {
            f"{family}::{session_id}": index
            for (family, session_id), index in conditional_session_to_index.items()
        },
        "conditional_session_counts": conditional_session_counts,
        "initialized_from": initialized_from,
        "initialization_scope": "backbone_only",
        "training_sources": source_frame.to_dict(orient="records"),
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)
    (MODEL_DIR / "latest_model_path.txt").write_text(str(model_path), encoding="utf-8")
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Representation quality:", quality)


if __name__ == "__main__":
    main()
