import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from brain_state_common import CHANNEL_NAMES, MODEL_ROOT, electrode_adjacency, recording_normalizations, torch_device
from brain_state_model import exponential_moving_average
from train_visualization_model import (
    ARTIFACT_DIM,
    BATCH_SIZE,
    CLIP_VALUE,
    DROPOUT,
    EMA_DECAY,
    EMBEDDING_DIM,
    FUTURE_HORIZON_SECONDS,
    HIGH_HZ,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    LOW_HZ,
    NOTCH_HZ,
    NUM_WORKERS,
    RANDOM_SEED,
    SESSION_DIM,
    SPATIAL_FEATURES,
    STEP_SECONDS,
    TARGET_SAMPLE_RATE,
    TEMPORAL_WIDTH,
    VALIDATION_FRACTION,
    WEIGHT_DECAY,
    WINDOW_SECONDS_MULTI,
    SessionBalancedCrossTaskBatchSampler,
    adversarial_schedule,
    conditional_session_vocabulary,
    initialize_backbone,
    load_training_sources,
    representation_metrics,
    set_random_seed,
    temporal_validation_ranges,
    vocabularies,
)
from universal_model import TagAwareEEGDataset, UniversalBrainStateModel, universal_losses


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_ROOT = PROJECT_ROOT / "sessions"
MODEL_DIR = MODEL_ROOT
RUN_MODE = "full"
DEVICE = "auto"
EXPERIMENT_EPOCHS = 6
FULL_EPOCHS = 28
EARLY_STOPPING_PATIENCE = 6
MODEL_SELECTION_START_EPOCH = 5
MIN_TAG_SESSION_COUNT = 2
AUDIT_SESSION_IDS = {
    "session_20260717_205918_scrolling_reels",
    "session_20260717_221735_puzzle_games",
    "session_20260717_222249_sexual_arousing_visualization",
}
EXPERIMENTS = [
    {
        "name": "factorized_baseline",
        "tag_head_type": "mlp",
        "tag_classification": 0.0,
        "tag_relation": 0.0,
        "prototype_alignment": 0.0,
        "tag_consistency": 0.0,
        "session_style_invariance": 0.0,
    },
    {
        "name": "multilabel_mlp",
        "tag_head_type": "mlp",
        "tag_classification": 1.5,
        "tag_relation": 0.0,
        "prototype_alignment": 0.0,
        "tag_consistency": 0.0,
        "session_style_invariance": 0.0,
    },
    {
        "name": "prototype_relational",
        "tag_head_type": "prototype",
        "tag_classification": 1.5,
        "tag_relation": 1.0,
        "prototype_alignment": 0.35,
        "tag_consistency": 0.0,
        "session_style_invariance": 0.0,
    },
    {
        "name": "prototype_relational_consistent",
        "tag_head_type": "prototype",
        "tag_classification": 1.5,
        "tag_relation": 1.0,
        "prototype_alignment": 0.35,
        "tag_consistency": 0.20,
        "session_style_invariance": 0.0,
    },
]
INVARIANCE_EXPERIMENTS = [
    {
        "name": "prototype_relational_strong_domain",
        "tag_head_type": "prototype",
        "tag_classification": 1.5,
        "tag_relation": 1.0,
        "prototype_alignment": 0.35,
        "tag_consistency": 0.0,
        "session_style_invariance": 0.0,
        "nuisance_initial_weight": 0.35,
        "base_loss_overrides": {
            "conditional_session_adversarial": 3.0,
            "task_cross_session": 1.5,
            "state_cross_session": 0.75,
            "task_session_alignment": 3.0,
            "state_session_alignment": 1.5,
            "orthogonality": 1.5,
        },
    },
    {
        "name": "prototype_relational_style_invariant",
        "tag_head_type": "prototype",
        "tag_classification": 1.5,
        "tag_relation": 1.0,
        "prototype_alignment": 0.35,
        "tag_consistency": 0.15,
        "session_style_invariance": 2.0,
        "nuisance_initial_weight": 0.35,
        "base_loss_overrides": {
            "conditional_session_adversarial": 3.0,
            "task_cross_session": 1.5,
            "state_cross_session": 0.75,
            "task_session_alignment": 3.0,
            "state_session_alignment": 1.5,
            "orthogonality": 1.5,
        },
    },
]
COHESION_EXPERIMENTS = [
    {
        "name": "prototype_relational_spatiotemporal_cohesive",
        "tag_head_type": "prototype",
        "tag_classification": 1.5,
        "tag_relation": 1.0,
        "prototype_alignment": 0.35,
        "tag_consistency": 0.0,
        "session_style_invariance": 0.0,
        "base_loss_overrides": {
            "future_prediction": 2.75,
            "acceleration": 2.25,
            "scale_consistency": 5.5,
            "reliability": 0.15,
        },
    },
]
WINNING_CONFIGURATION = {
    "name": "prototype_relational_spatiotemporal_cohesive",
    "tag_head_type": "prototype",
    "tag_classification": 1.5,
    "tag_relation": 1.0,
    "prototype_alignment": 0.35,
    "tag_consistency": 0.0,
    "session_style_invariance": 0.0,
    "base_loss_overrides": {
        "future_prediction": 2.75,
        "acceleration": 2.25,
        "scale_consistency": 5.5,
        "reliability": 0.15,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def session_tags(session_id):
    path = SESSIONS_ROOT / session_id / "session.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("tags", [])


def tag_vocabulary(labels, training_indices):
    counts = {}
    tags_by_recording = []
    for label in labels:
        tags = session_tags(label["session_id"])
        tags_by_recording.append(tags)
    for index in training_indices:
        for tag in set(tags_by_recording[index]):
            if tag != "unknown_context":
                counts[tag] = counts.get(tag, 0) + 1
    vocabulary = sorted(tag for tag, count in counts.items() if count >= MIN_TAG_SESSION_COUNT)
    tag_to_index = {tag: index for index, tag in enumerate(vocabulary)}
    vectors = np.zeros((len(labels), len(vocabulary)), dtype=np.float32)
    for recording_index, tags in enumerate(tags_by_recording):
        for tag in tags:
            if tag in tag_to_index:
                vectors[recording_index, tag_to_index[tag]] = 1.0
    return vocabulary, tag_to_index, vectors, counts


def repeated_families_for_indices(labels, indices):
    counts = {}
    for index in indices:
        family = labels[index]["task_family"]
        counts[family] = counts.get(family, 0) + 1
    return {family for family, count in counts.items() if count >= 2}


def conditional_vocabulary_for_indices(labels, repeated_families, indices):
    subset = [labels[index] for index in indices]
    mapping = {}
    counts = []
    for family in sorted(repeated_families):
        sessions = sorted(label["session_id"] for label in subset if label["task_family"] == family)
        for index, session_id in enumerate(sessions):
            mapping[(family, session_id)] = index
        counts.append(len(sessions))
    return mapping, counts


def build_datasets(recordings, labels, training_indices, audit_indices, tag_vectors):
    repeated_families = repeated_families_for_indices(labels, training_indices)
    task_to_index, state_to_index, session_to_index, artifact_to_index = vocabularies(labels, training_indices, repeated_families)
    conditional_mapping, conditional_counts = conditional_vocabulary_for_indices(labels, repeated_families, training_indices)
    centers, scales = recording_normalizations(recordings)
    training_ranges, validation_ranges = temporal_validation_ranges(recordings, labels)
    max_window_samples = int(round(max(WINDOW_SECONDS_MULTI) * TARGET_SAMPLE_RATE))
    shortest_window_samples = int(round(min(WINDOW_SECONDS_MULTI) * TARGET_SAMPLE_RATE))
    step_samples = int(round(STEP_SECONDS * TARGET_SAMPLE_RATE))
    future_samples = [int(round(value * TARGET_SAMPLE_RATE)) for value in FUTURE_HORIZON_SECONDS]
    common = (
        recordings,
        labels,
        max_window_samples,
        shortest_window_samples,
        step_samples,
        future_samples,
        centers,
        scales,
        task_to_index,
        state_to_index,
        session_to_index,
        artifact_to_index,
        repeated_families,
    )
    training = TagAwareEEGDataset(*common[:2], training_indices, *common[2:], end_ranges=training_ranges, clip_value=CLIP_VALUE, conditional_session_to_index=conditional_mapping, tag_vectors=tag_vectors)
    validation = TagAwareEEGDataset(*common[:2], training_indices, *common[2:], end_ranges=validation_ranges, clip_value=CLIP_VALUE, conditional_session_to_index=conditional_mapping, tag_vectors=tag_vectors)
    audit = None
    if audit_indices:
        audit = TagAwareEEGDataset(*common[:2], audit_indices, *common[2:], end_ranges=None, clip_value=CLIP_VALUE, conditional_session_to_index=conditional_mapping, tag_vectors=tag_vectors)
    metadata = {
        "repeated_families": repeated_families,
        "task_to_index": task_to_index,
        "state_to_index": state_to_index,
        "session_to_index": session_to_index,
        "artifact_to_index": artifact_to_index,
        "conditional_mapping": conditional_mapping,
        "conditional_counts": conditional_counts,
        "centers": centers,
        "scales": scales,
    }
    return training, validation, audit, metadata


def positive_tag_weights(tag_vectors, training_indices, device):
    session_targets = tag_vectors[training_indices]
    positives = session_targets.sum(axis=0)
    negatives = len(training_indices) - positives
    weights = np.clip(negatives / np.maximum(positives, 1.0), 0.5, 8.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def make_model(configuration, tag_count, metadata, device):
    adjacency = electrode_adjacency(CHANNEL_NAMES)
    config = {
        "architecture": "universal_tag_semantic_model_v1",
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
        "canonical_label_version": "brainz_canonical_tasks_v2",
        "tag_head_type": configuration["tag_head_type"],
        "minimum_tag_session_count": MIN_TAG_SESSION_COUNT,
    }
    from visualization_model import build_visualization_encoder_from_config
    encoder = build_visualization_encoder_from_config(config, adjacency)
    initialized_from = initialize_backbone(encoder)
    nuisance_weight = float(configuration.get("nuisance_initial_weight", 0.12))
    if encoder.nuisance_residualization:
        nuisance_logit = math.log(nuisance_weight / max(1e-6, 1.0 - nuisance_weight))
        encoder.nuisance_logits.data.fill_(nuisance_logit)
    model = UniversalBrainStateModel(
        encoder,
        len(metadata["task_to_index"]),
        len(metadata["state_to_index"]),
        len(metadata["session_to_index"]),
        len(metadata["artifact_to_index"]),
        len(FUTURE_HORIZON_SECONDS),
        len(CHANNEL_NAMES),
        conditional_session_counts=metadata["conditional_counts"],
        tag_count=tag_count,
        tag_head_type=configuration["tag_head_type"],
    ).to(device)
    return model, config, adjacency, initialized_from


def run_epoch(model, loader, optimizer, scaler, device, train_mode, adversarial_strength, positive_weights, universal_weights, base_weights, ema_model=None):
    model.train(train_mode)
    totals = {}
    sample_count = 0
    for batch in loader:
        batch = [value.to(device, non_blocking=True) for value in batch]
        previous, current, futures, task_labels, state_labels, session_labels, artifact_labels, recording_indices, stable_previous, stable_futures, domain_valid, label_weights, conditional_labels, tag_targets = batch
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, metrics = universal_losses(
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
                    conditional_labels,
                    tag_targets,
                    base_weights,
                    universal_weights,
                    positive_weights,
                    adversarial_strength,
                )
            if train_mode:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                exponential_moving_average(ema_model, model, EMA_DECAY)
        count = current.shape[0]
        sample_count += count
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * count
    return {name: value / max(1, sample_count) for name, value in totals.items()}


def collect_tag_embeddings(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    embeddings = []
    probabilities = []
    targets = []
    recordings = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            current = batch[1].to(device)
            state = model.encoder(current)
            embeddings.append(state.cpu().numpy())
            probabilities.append(torch.sigmoid(model.tag_logits(state)).cpu().numpy())
            targets.append(batch[13].numpy())
            recordings.append(batch[7].numpy())
    return {
        "embedding": np.concatenate(embeddings),
        "probability": np.concatenate(probabilities),
        "target": np.concatenate(targets),
        "recording": np.concatenate(recordings),
    }


def aggregate_sessions(collected):
    rows = []
    for recording in sorted(np.unique(collected["recording"]).tolist()):
        mask = collected["recording"] == recording
        rows.append({
            "recording": int(recording),
            "embedding": collected["embedding"][mask].mean(axis=0),
            "probability": collected["probability"][mask].mean(axis=0),
            "target": collected["target"][mask][0],
        })
    return {
        "recording": np.asarray([row["recording"] for row in rows]),
        "embedding": np.stack([row["embedding"] for row in rows]),
        "probability": np.stack([row["probability"] for row in rows]),
        "target": np.stack([row["target"] for row in rows]),
    }


def multilabel_metrics(target, probability):
    prediction = probability >= 0.5
    return {
        "micro_f1": float(f1_score(target, prediction, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "sample_jaccard": float(jaccard_score(target, prediction, average="samples", zero_division=0)),
    }


def downstream_session_probe(training, audit):
    probabilities = np.zeros_like(audit["target"], dtype=np.float32)
    for tag_index in range(training["target"].shape[1]):
        labels = training["target"][:, tag_index]
        if len(np.unique(labels)) < 2:
            probabilities[:, tag_index] = float(labels[0])
            continue
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=RANDOM_SEED),
        )
        classifier.fit(training["embedding"], labels)
        probabilities[:, tag_index] = classifier.predict_proba(audit["embedding"])[:, 1]
    return multilabel_metrics(audit["target"], probabilities)


def semantic_geometry_metrics(aggregated):
    embedding = aggregated["embedding"]
    tags = aggregated["target"]
    embedding = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-9)
    embedding_similarity = embedding @ embedding.T
    intersection = tags @ tags.T
    union = tags.sum(axis=1, keepdims=True) + tags.sum(axis=1, keepdims=True).T - intersection
    tag_similarity = intersection / np.maximum(union, 1.0)
    upper = np.triu_indices(len(embedding), k=1)
    correlation = spearmanr(embedding_similarity[upper], tag_similarity[upper]).statistic
    return {"session_embedding_tag_jaccard_spearman": float(correlation) if np.isfinite(correlation) else 0.0}


def evaluation_metrics(model, training_dataset, validation_dataset, audit_dataset, device):
    quality = representation_metrics(model, training_dataset, validation_dataset, device)
    training = aggregate_sessions(collect_tag_embeddings(model, training_dataset, device))
    validation = aggregate_sessions(collect_tag_embeddings(model, validation_dataset, device))
    quality["heldout_time_session_tag_metrics"] = multilabel_metrics(validation["target"], validation["probability"])
    quality.update(semantic_geometry_metrics(training))
    if audit_dataset is not None:
        audit = aggregate_sessions(collect_tag_embeddings(model, audit_dataset, device))
        quality["unseen_session_direct_tag_metrics"] = multilabel_metrics(audit["target"], audit["probability"])
        quality["unseen_session_downstream_probe_tag_metrics"] = downstream_session_probe(training, audit)
        combined = {
            "embedding": np.concatenate([training["embedding"], audit["embedding"]]),
            "target": np.concatenate([training["target"], audit["target"]]),
        }
        quality["all_session_semantic_geometry"] = semantic_geometry_metrics(combined)
    return quality


def selection_score(quality):
    direct = quality.get("unseen_session_direct_tag_metrics", {}).get("macro_f1", 0.0)
    probe = quality.get("unseen_session_downstream_probe_tag_metrics", {}).get("macro_f1", 0.0)
    geometry = quality.get("all_session_semantic_geometry", {}).get("session_embedding_tag_jaccard_spearman", 0.0)
    robustness = quality.get("single_channel_dropout_cosine_mean", 0.0)
    smoothness = quality.get("normalized_adjacent_distance_median", 1.0)
    session_probe = quality.get("conditional_session_probe_from_state_mean")
    session_penalty = session_probe if session_probe is not None else 0.5
    return float(2.0 * direct + 2.5 * probe + geometry + robustness - smoothness - 0.5 * session_penalty)


def train_configuration(configuration, source_frame, recordings, labels, training_indices, audit_indices, epochs, save_final):
    set_seed(RANDOM_SEED)
    device = torch_device(DEVICE)
    vocabulary, tag_to_index, tag_vectors, tag_counts = tag_vocabulary(labels, training_indices)
    training_dataset, validation_dataset, audit_dataset, metadata = build_datasets(recordings, labels, training_indices, audit_indices, tag_vectors)
    training_loader = DataLoader(
        training_dataset,
        batch_sampler=SessionBalancedCrossTaskBatchSampler(training_dataset, BATCH_SIZE, RANDOM_SEED),
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    model, config, adjacency, initialized_from = make_model(configuration, len(vocabulary), metadata, device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=LEARNING_RATE * 0.05)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    positive_weights = positive_tag_weights(tag_vectors, training_indices, device)
    universal_weights = {key: float(configuration[key]) for key in ["tag_classification", "tag_relation", "prototype_alignment", "tag_consistency", "session_style_invariance"]}
    base_weights = {**LOSS_WEIGHTS, **configuration.get("base_loss_overrides", {})}
    history = []
    best_score = math.inf
    best_state = None
    stale = 0
    print(f"\nConfiguration: {configuration['name']} | train={len(training_dataset)} validation={len(validation_dataset)} audit={len(audit_dataset) if audit_dataset else 0} tags={len(vocabulary)}")
    for epoch in range(1, epochs + 1):
        strength = adversarial_schedule(epoch, epochs)
        train_metrics = run_epoch(model, training_loader, optimizer, scaler, device, True, strength, positive_weights, universal_weights, base_weights, ema_model)
        validation_metrics = run_epoch(ema_model, validation_loader, optimizer, scaler, device, False, 0.0, positive_weights, universal_weights, base_weights)
        scheduler.step()
        score = validation_metrics["total"] - base_weights["conditional_session_adversarial"] * validation_metrics["conditional_session_adversarial"]
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "selection_loss": score})
        print(f"{configuration['name']} epoch={epoch:02d} train={train_metrics['total']:.4f} val={score:.4f} tag={validation_metrics['tag_classification']:.4f} relation={validation_metrics['tag_relation']:.4f}")
        if epoch >= min(MODEL_SELECTION_START_EPOCH, epochs) and score < best_score:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in ema_model.state_dict().items()}
            stale = 0
        elif epoch >= min(MODEL_SELECTION_START_EPOCH, epochs):
            stale += 1
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        best_state = {name: value.detach().cpu().clone() for name, value in ema_model.state_dict().items()}
    ema_model.load_state_dict(best_state)
    quality = evaluation_metrics(ema_model, training_dataset, validation_dataset, audit_dataset, device)
    result = {
        "configuration": configuration,
        "selection_score": selection_score(quality),
        "best_validation_loss": float(best_score),
        "quality": quality,
        "history": history,
        "tag_vocabulary": vocabulary,
        "tag_to_index": tag_to_index,
        "tag_session_counts": tag_counts,
        "training_session_ids": [labels[index]["session_id"] for index in training_indices],
        "audit_session_ids": [labels[index]["session_id"] for index in audit_indices],
        "initialized_from": initialized_from,
    }
    if save_final:
        centers = metadata["centers"]
        scales = metadata["scales"]
        global_center = np.median(centers[training_indices], axis=0).astype(np.float32)
        global_scale = np.median(scales[training_indices], axis=0).astype(np.float32)
        bundle = {
            "format_version": 4,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "encoder_state": {name.replace("encoder.", "", 1): value for name, value in best_state.items() if name.startswith("encoder.")},
            "tag_model_state": {
                name: value for name, value in best_state.items()
                if name.startswith("tag_")
            },
            "task_head_state": {name.replace("task_head.", "", 1): value for name, value in best_state.items() if name.startswith("task_head.")},
            "state_head_state": {name.replace("state_head.", "", 1): value for name, value in best_state.items() if name.startswith("state_head.")},
            "config": config,
            "channel_names": CHANNEL_NAMES,
            "adjacency": adjacency,
            "normalization_center": global_center,
            "normalization_scale": global_scale,
            "tag_vocabulary": vocabulary,
            "tag_to_index": tag_to_index,
            "task_to_index": metadata["task_to_index"],
            "state_to_index": metadata["state_to_index"],
            "session_to_index": metadata["session_to_index"],
            "artifact_to_index": metadata["artifact_to_index"],
            "conditional_session_to_index": metadata["conditional_mapping"],
            "conditional_session_counts": metadata["conditional_counts"],
            "repeated_task_families": sorted(metadata["repeated_families"]),
            "training_recordings": [str(recordings[index].path) for index in training_indices],
            "training_session_ids": [labels[index]["session_id"] for index in training_indices],
            "quality": quality,
            "winning_configuration": configuration,
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "universal_brain_model.pt"
        metrics_path = MODEL_DIR / "universal_brain_model_metrics.json"
        torch.save(bundle, model_path)
        (MODEL_DIR / "latest_universal_model_path.txt").write_text(str(model_path), encoding="utf-8")
        result["model_path"] = str(model_path)
        result["metrics_path"] = str(metrics_path)
        metrics_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main():
    set_random_seed(RANDOM_SEED)
    source_frame, recordings, labels = load_training_sources()
    usable_indices = list(range(len(labels)))
    if RUN_MODE in {"experiments", "invariance_experiments", "cohesion_experiment"}:
        audit_indices = [index for index in usable_indices if labels[index]["session_id"] in AUDIT_SESSION_IDS]
        training_indices = [index for index in usable_indices if index not in audit_indices]
        results = []
        if RUN_MODE == "experiments":
            candidates = EXPERIMENTS
        elif RUN_MODE == "invariance_experiments":
            candidates = INVARIANCE_EXPERIMENTS
        else:
            candidates = COHESION_EXPERIMENTS
        for configuration in candidates:
            results.append(train_configuration(configuration, source_frame, recordings, labels, training_indices, audit_indices, EXPERIMENT_EPOCHS, False))
        results.sort(key=lambda row: row["selection_score"], reverse=True)
        output = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audit_strategy": "complete unseen sessions excluded from neural training",
            "results": results,
            "winner": results[0]["configuration"],
        }
        filenames = {
            "experiments": "universal_model_experiments.json",
            "invariance_experiments": "universal_model_invariance_experiments.json",
            "cohesion_experiment": "universal_model_cohesion_experiment.json",
        }
        filename = filenames[RUN_MODE]
        path = MODEL_DIR / filename
        path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print("Experiment winner:", results[0]["configuration"])
        print("Experiment summary:", path)
    elif RUN_MODE == "full":
        result = train_configuration(WINNING_CONFIGURATION, source_frame, recordings, labels, usable_indices, [], FULL_EPOCHS, True)
        print("Saved model:", result["model_path"])
        print("Saved metrics:", result["metrics_path"])
        print("Representation quality:", result["quality"])
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
