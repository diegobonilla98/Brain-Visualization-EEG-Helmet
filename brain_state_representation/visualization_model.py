import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset

from brain_state_common import CHANNEL_NAMES, electrode_adjacency, normalize_window
from brain_state_model_v2 import MultiScaleBrainStateEncoder, augment_eeg_v2, vicreg_loss_v2


class FactorizedEEGDataset(Dataset):
    def __init__(
        self,
        recordings,
        labels,
        recording_indices,
        max_window_samples,
        shortest_window_samples,
        step_samples,
        future_horizon_samples,
        centers,
        scales,
        family_to_index,
        state_to_index,
        session_to_index,
        artifact_to_index,
        repeated_families,
        end_ranges=None,
        clip_value=12.0,
        conditional_session_to_index=None,
    ):
        self.recordings = recordings
        self.labels = labels
        self.max_window_samples = int(max_window_samples)
        self.shortest_window_samples = int(shortest_window_samples)
        self.step_samples = int(step_samples)
        self.future_horizon_samples = tuple(int(value) for value in future_horizon_samples)
        self.centers = np.asarray(centers, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.family_to_index = family_to_index
        self.state_to_index = state_to_index
        self.session_to_index = session_to_index
        self.artifact_to_index = artifact_to_index
        self.repeated_families = set(repeated_families)
        self.end_ranges = end_ranges
        self.conditional_session_to_index = conditional_session_to_index or {}
        self.clip_value = float(clip_value)
        self.indices = []
        self.recording_window_indices = []
        self.task_window_indices = []
        self.session_window_indices = []
        self.state_window_indices = []
        self.artifact_window_indices = []
        max_horizon = max(self.future_horizon_samples)
        for recording_index in recording_indices:
            sample_count = recordings[recording_index].data.shape[1]
            first_end = self.max_window_samples + self.future_horizon_samples[0]
            last_end = sample_count - max_horizon
            if self.end_ranges is not None:
                range_start, range_end = self.end_ranges[recording_index]
                first_end = max(first_end, int(range_start))
                last_end = min(last_end, int(range_end))
            for end in range(first_end, max(first_end, last_end + 1), self.step_samples):
                self.indices.append((recording_index, end))
                self.recording_window_indices.append(recording_index)
                self.task_window_indices.append(self.family_to_index.get(labels[recording_index]["task_family"], -1))
                self.session_window_indices.append(self.session_to_index.get(labels[recording_index]["session_id"], -1))
                state_index, state_label, _ = self.label_at_end(recording_index, end)
                self.state_window_indices.append(state_index)
                self.artifact_window_indices.append(self.artifact_index(labels[recording_index]["task_family"], state_label))

    def __len__(self):
        return len(self.indices)

    def label_at_end(self, recording_index, end):
        label_data = self.labels[recording_index]
        start = max(0, end - self.shortest_window_samples)
        valid = label_data["is_valid"][start:end]
        state_values = label_data["state_label"][start:end]
        usable = valid & (state_values != "unlabeled_transition") & (state_values != "invalid_stream_sample")
        if len(usable) == 0 or np.mean(usable) < 0.80:
            return -1, "unlabeled_transition", -1
        values, counts = np.unique(state_values[usable], return_counts=True)
        state_label = str(values[int(np.argmax(counts))])
        state_key = f"{label_data['task_family']}::{state_label}"
        segment_values = label_data["segment_index"][start:end][usable]
        segment_index = int(np.median(segment_values)) if len(segment_values) > 0 else -1
        return self.state_to_index.get(state_key, -1), state_label, segment_index

    def artifact_index(self, task_family, state_label):
        if task_family == "eyes_open_closed_and_blinks":
            name = "blink" if state_label == "strong_blinks" else "clean_reference"
            return self.artifact_to_index[name]
        if task_family == "jaw_clench":
            name = "jaw_emg" if state_label == "jaw_clench" else "clean_reference"
            return self.artifact_to_index[name]
        return -1

    def normalized_window(self, recording_index, end):
        data = self.recordings[recording_index].data
        window = data[:, end - self.max_window_samples:end]
        return normalize_window(
            window,
            self.centers[recording_index],
            self.scales[recording_index],
            self.clip_value,
        )

    def __getitem__(self, index):
        recording_index, end = self.indices[index]
        label_data = self.labels[recording_index]
        nearest_horizon = self.future_horizon_samples[0]
        current = self.normalized_window(recording_index, end)
        previous = self.normalized_window(recording_index, end - nearest_horizon)
        futures = np.stack([
            self.normalized_window(recording_index, end + horizon)
            for horizon in self.future_horizon_samples
        ])
        current_state, current_label, current_segment = self.label_at_end(recording_index, end)
        previous_state, previous_label, previous_segment = self.label_at_end(recording_index, end - nearest_horizon)
        future_metadata = [self.label_at_end(recording_index, end + horizon) for horizon in self.future_horizon_samples]
        stable_previous = (
            current_label not in {"unlabeled_transition", "invalid_stream_sample"}
            and previous_label == current_label
            and previous_segment == current_segment
        )
        stable_futures = [
            label == current_label and segment == current_segment and state >= 0
            for state, label, segment in future_metadata
        ]
        task_family = label_data["task_family"]
        session_id = label_data["session_id"]
        task_index = self.family_to_index.get(task_family, -1)
        session_index = self.session_to_index.get(session_id, -1)
        artifact_index = self.artifact_index(task_family, current_label)
        domain_valid = task_family in self.repeated_families and session_index >= 0
        conditional_session_index = self.conditional_session_to_index.get((task_family, session_id), -1)
        return (
            torch.from_numpy(previous),
            torch.from_numpy(current),
            torch.from_numpy(futures),
            torch.tensor(task_index, dtype=torch.long),
            torch.tensor(current_state, dtype=torch.long),
            torch.tensor(session_index, dtype=torch.long),
            torch.tensor(artifact_index, dtype=torch.long),
            torch.tensor(recording_index, dtype=torch.long),
            torch.tensor(stable_previous, dtype=torch.bool),
            torch.tensor(stable_futures, dtype=torch.bool),
            torch.tensor(domain_valid, dtype=torch.bool),
            torch.tensor(float(label_data["label_weight"]), dtype=torch.float32),
            torch.tensor(conditional_session_index, dtype=torch.long),
        )


class ResidualStateProjection(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
        )
        self.norm = nn.LayerNorm(dimension)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x):
        return self.norm(x + self.delta(x))


class FactorizedMultiScaleBrainStateEncoder(nn.Module):
    def __init__(
        self,
        channel_count=32,
        spatial_features=10,
        temporal_width=112,
        embedding_dim=160,
        session_dim=48,
        artifact_dim=32,
        dropout=0.12,
        adjacency=None,
        sample_rate=125.0,
        window_seconds=(2.0, 4.0, 8.0),
        nuisance_residualization=False,
    ):
        super().__init__()
        if adjacency is None:
            adjacency = electrode_adjacency(CHANNEL_NAMES[:channel_count])
        self.channel_count = int(channel_count)
        self.embedding_dim = int(embedding_dim)
        self.session_dim = int(session_dim)
        self.artifact_dim = int(artifact_dim)
        self.sample_rate = float(sample_rate)
        self.window_seconds = tuple(float(value) for value in window_seconds)
        self.window_samples = tuple(int(round(value * sample_rate)) for value in self.window_seconds)
        self.nuisance_residualization = bool(nuisance_residualization)
        self.backbone = MultiScaleBrainStateEncoder(
            channel_count=channel_count,
            spatial_features=spatial_features,
            temporal_width=temporal_width,
            embedding_dim=embedding_dim,
            dropout=dropout,
            adjacency=adjacency,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
        )
        self.state_projection = ResidualStateProjection(embedding_dim, dropout)
        self.session_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, session_dim),
            nn.LayerNorm(session_dim),
        )
        self.artifact_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, artifact_dim),
            nn.LayerNorm(artifact_dim),
        )
        if self.nuisance_residualization:
            self.session_removal = nn.Sequential(
                nn.Linear(session_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.artifact_removal = nn.Sequential(
                nn.Linear(artifact_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.nuisance_logits = nn.Parameter(torch.full((2,), -2.0))
            nn.init.zeros_(self.session_removal[-1].weight)
            nn.init.zeros_(self.session_removal[-1].bias)
            nn.init.zeros_(self.artifact_removal[-1].weight)
            nn.init.zeros_(self.artifact_removal[-1].bias)
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))

    def encode_factors(self, x, return_scales=False):
        shared, shared_scales, scale_weights, bands, reliability = self.backbone(x, return_scales=True)
        session = self.session_projection(shared)
        artifact = self.artifact_projection(shared)
        if self.nuisance_residualization:
            nuisance_weights = torch.sigmoid(self.nuisance_logits)
            residual = shared - nuisance_weights[0] * self.session_removal(session) - nuisance_weights[1] * self.artifact_removal(artifact)
            state = self.state_projection(residual)
        else:
            state = self.state_projection(shared)
        output = {"state": state, "session": session, "artifact": artifact}
        if return_scales:
            batch_size, scale_count, dimension = shared_scales.shape
            flat_scales = shared_scales.reshape(batch_size * scale_count, dimension)
            if self.nuisance_residualization:
                repeated_session = session[:, None, :].expand(-1, scale_count, -1).reshape(batch_size * scale_count, -1)
                repeated_artifact = artifact[:, None, :].expand(-1, scale_count, -1).reshape(batch_size * scale_count, -1)
                nuisance_weights = torch.sigmoid(self.nuisance_logits)
                flat_scales = flat_scales - nuisance_weights[0] * self.session_removal(repeated_session) - nuisance_weights[1] * self.artifact_removal(repeated_artifact)
            state_scales = self.state_projection(flat_scales)
            output["state_scales"] = state_scales.reshape(batch_size, scale_count, dimension)
            output["scale_weights"] = scale_weights
            output["bands"] = bands
            output["reliability"] = reliability
        return output

    def forward(self, x, return_scales=False):
        factors = self.encode_factors(x, return_scales=return_scales)
        if return_scales:
            return (
                factors["state"],
                factors["state_scales"],
                factors["scale_weights"],
                factors["bands"],
                factors["reliability"],
            )
        return factors["state"]

    def forward_longest(self, x):
        shared = self.backbone.forward_longest(x)
        if self.nuisance_residualization:
            session = self.session_projection(shared)
            artifact = self.artifact_projection(shared)
            nuisance_weights = torch.sigmoid(self.nuisance_logits)
            shared = shared - nuisance_weights[0] * self.session_removal(session) - nuisance_weights[1] * self.artifact_removal(artifact)
        return self.state_projection(shared)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def gradient_reverse(x, strength):
    return GradientReversalFunction.apply(x, strength)


class FactorizedBrainStateModel(nn.Module):
    def __init__(self, encoder, task_count, state_count, session_count, artifact_count, horizon_count, channel_count=32, band_count=7, conditional_session_counts=None):
        super().__init__()
        self.encoder = encoder
        dimension = encoder.embedding_dim
        self.future_predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension))
            for _ in range(horizon_count)
        ])
        factor_dimension = dimension + encoder.session_dim + encoder.artifact_dim
        self.spectral_decoder = nn.Sequential(
            nn.Linear(factor_dimension, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, channel_count * band_count),
        )
        self.task_head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.10), nn.Linear(dimension, task_count))
        self.state_head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.10), nn.Linear(dimension, state_count))
        self.session_private_head = nn.Linear(encoder.session_dim, session_count)
        self.session_adversary = nn.Sequential(nn.Linear(dimension, dimension // 2), nn.GELU(), nn.Linear(dimension // 2, session_count))
        self.artifact_private_head = nn.Linear(encoder.artifact_dim, artifact_count)
        self.artifact_adversary = nn.Sequential(nn.Linear(dimension, dimension // 2), nn.GELU(), nn.Linear(dimension // 2, artifact_count))
        conditional_session_counts = conditional_session_counts or [0] * task_count
        self.conditional_session_adversaries = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, dimension),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(dimension, max(1, int(count))),
            )
            for count in conditional_session_counts
        ])
        self.channel_count = int(channel_count)
        self.band_count = int(band_count)


def weighted_cross_entropy(logits, labels, sample_weights):
    valid = labels >= 0
    if not torch.any(valid):
        return logits.new_tensor(0.0)
    losses = F.cross_entropy(logits[valid], labels[valid], reduction="none")
    weights = sample_weights[valid]
    return torch.sum(losses * weights) / weights.sum().clamp_min(1e-6)


def cross_session_contrastive_loss(embedding, labels, sessions, temperature=0.12):
    valid = (labels >= 0) & (sessions >= 0)
    if torch.sum(valid) < 3:
        return embedding.new_tensor(0.0)
    embedding = F.normalize(embedding[valid], dim=1)
    labels = labels[valid]
    sessions = sessions[valid]
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (labels[:, None] == labels[None, :]) & (sessions[:, None] != sessions[None, :]) & ~identity
    logits = similarity - torch.max(similarity, dim=1, keepdim=True).values.detach()
    denominator = torch.exp(logits) * ~identity
    log_probability = logits - torch.log(denominator.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positive.sum(dim=1)
    usable = positive_count > 0
    if not torch.any(usable):
        return embedding.new_tensor(0.0)
    return -((positive * log_probability).sum(dim=1) / positive_count.clamp_min(1))[usable].mean()


def factor_orthogonality(first, second):
    first = (first - first.mean(dim=0)) / (first.std(dim=0) + 1e-4)
    second = (second - second.mean(dim=0)) / (second.std(dim=0) + 1e-4)
    covariance = first.T @ second / max(1, first.shape[0] - 1)
    return covariance.square().mean()


def distribution_alignment(first, second):
    mean_loss = F.smooth_l1_loss(first.mean(dim=0), second.mean(dim=0))
    scale_loss = F.smooth_l1_loss(first.std(dim=0, unbiased=False), second.std(dim=0, unbiased=False))
    return mean_loss + 0.25 * scale_loss


def session_distribution_alignment(embedding, task_labels, state_labels, session_labels, match_state):
    losses = []
    for task in torch.unique(task_labels[task_labels >= 0]).tolist():
        task_mask = task_labels == int(task)
        states = torch.unique(state_labels[task_mask & (state_labels >= 0)]).tolist() if match_state else [None]
        for state in states:
            group_mask = task_mask if state is None else task_mask & (state_labels == int(state))
            sessions = torch.unique(session_labels[group_mask & (session_labels >= 0)]).tolist()
            for first_index in range(len(sessions)):
                for second_index in range(first_index + 1, len(sessions)):
                    first = embedding[group_mask & (session_labels == int(sessions[first_index]))]
                    second = embedding[group_mask & (session_labels == int(sessions[second_index]))]
                    if len(first) >= 2 and len(second) >= 2:
                        losses.append(distribution_alignment(first, second))
    return torch.stack(losses).mean() if losses else embedding.new_tensor(0.0)


def within_recording_artifact_alignment(embedding, artifact_labels, recording_indices):
    losses = []
    for recording in torch.unique(recording_indices).tolist():
        recording_mask = recording_indices == int(recording)
        clean = embedding[recording_mask & (artifact_labels == 0)]
        if len(clean) < 2:
            continue
        artifacts = torch.unique(artifact_labels[recording_mask & (artifact_labels > 0)]).tolist()
        for artifact in artifacts:
            affected = embedding[recording_mask & (artifact_labels == int(artifact))]
            if len(affected) >= 2:
                losses.append(distribution_alignment(clean, affected))
    return torch.stack(losses).mean() if losses else embedding.new_tensor(0.0)


def conditional_session_adversarial_loss(model, embedding, task_labels, conditional_session_labels, sample_weights, strength):
    losses = []
    for task, head in enumerate(model.conditional_session_adversaries):
        valid = (task_labels == task) & (conditional_session_labels >= 0)
        if torch.sum(valid) < 2 or len(torch.unique(conditional_session_labels[valid])) < 2:
            continue
        logits = head(gradient_reverse(embedding[valid], strength))
        losses.append(weighted_cross_entropy(logits, conditional_session_labels[valid], sample_weights[valid]))
    return torch.stack(losses).mean() if losses else embedding.new_tensor(0.0)


def cosine_prediction_loss(predictor, source, target, valid):
    if not torch.any(valid):
        return source.new_tensor(0.0)
    prediction = F.normalize(predictor(source[valid]), dim=1)
    target = F.normalize(target[valid].detach(), dim=1)
    return torch.mean(1.0 - torch.sum(prediction * target, dim=1))


def factorized_losses(
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
    weights,
    adversarial_strength,
    return_factors=False,
):
    augmented = augment_eeg_v2(current, model.encoder.adjacency)
    factors = model.encoder.encode_factors(current, return_scales=True)
    augmented_factors = model.encoder.encode_factors(augmented)
    state = factors["state"]
    state_augmented = augmented_factors["state"]
    with torch.no_grad():
        state_previous = model.encoder.forward_longest(previous)
        future_states = [model.encoder.forward_longest(futures[:, index]) for index in range(futures.shape[1])]
    invariance, variance, covariance = vicreg_loss_v2(state, state_augmented)
    private_consistency = 0.5 * (
        F.smooth_l1_loss(factors["session"], augmented_factors["session"])
        + F.smooth_l1_loss(factors["artifact"], augmented_factors["artifact"])
    )
    future_prediction = state.new_tensor(0.0)
    for index, future_state in enumerate(future_states):
        future_prediction = future_prediction + cosine_prediction_loss(
            model.future_predictors[index],
            state,
            future_state,
            stable_futures[:, index],
        )
    future_prediction = future_prediction / max(1, len(future_states))
    stable_acceleration = stable_previous & stable_futures[:, 0]
    if torch.any(stable_acceleration):
        velocity_before = state[stable_acceleration] - state_previous[stable_acceleration]
        velocity_after = future_states[0][stable_acceleration] - state[stable_acceleration]
        acceleration = F.smooth_l1_loss(velocity_after, velocity_before)
    else:
        acceleration = state.new_tensor(0.0)
    normalized_scales = F.normalize(factors["state_scales"], dim=2)
    normalized_state = F.normalize(state, dim=1)[:, None, :]
    scale_consistency = torch.mean(1.0 - torch.sum(normalized_scales * normalized_state, dim=2))
    target_bands = factors["bands"][-1].detach()
    combined = torch.cat([state, factors["session"], factors["artifact"]], dim=1)
    decoded = model.spectral_decoder(combined).reshape(current.shape[0], model.channel_count, model.band_count)
    reconstruction = F.smooth_l1_loss(decoded, target_bands)
    task_classification = weighted_cross_entropy(model.task_head(state), task_labels, label_weights)
    state_classification = weighted_cross_entropy(model.state_head(state), state_labels, label_weights)
    session_private = weighted_cross_entropy(model.session_private_head(factors["session"]), session_labels, torch.ones_like(label_weights))
    artifact_private = weighted_cross_entropy(model.artifact_private_head(factors["artifact"]), artifact_labels, label_weights)
    domain_labels = torch.where(domain_valid, session_labels, torch.full_like(session_labels, -1))
    session_adversarial = weighted_cross_entropy(
        model.session_adversary(gradient_reverse(state, adversarial_strength)),
        domain_labels,
        torch.ones_like(label_weights),
    )
    artifact_adversarial = weighted_cross_entropy(
        model.artifact_adversary(gradient_reverse(state, adversarial_strength)),
        artifact_labels,
        label_weights,
    )
    conditional_session_adversarial = conditional_session_adversarial_loss(
        model,
        state,
        task_labels,
        conditional_session_labels,
        label_weights,
        adversarial_strength,
    )
    task_cross_session = cross_session_contrastive_loss(state, task_labels, session_labels)
    state_cross_session = cross_session_contrastive_loss(state, state_labels, session_labels)
    task_session_alignment = session_distribution_alignment(state, task_labels, state_labels, session_labels, False)
    state_session_alignment = session_distribution_alignment(state, task_labels, state_labels, session_labels, True)
    artifact_alignment = within_recording_artifact_alignment(state, artifact_labels, recording_indices)
    orthogonality = factor_orthogonality(state, factors["session"]) + factor_orthogonality(state, factors["artifact"])
    reliability = torch.stack(factors["reliability"], dim=1)
    reliability_regularization = torch.mean((torch.mean(reliability, dim=2) - 0.75).square())
    scale_weights = factors["scale_weights"]
    scale_entropy = -torch.mean(torch.sum(scale_weights * torch.log(scale_weights.clamp_min(1e-8)), dim=1))
    total = (
        weights["invariance"] * invariance
        + weights["variance"] * variance
        + weights["covariance"] * covariance
        + weights["private_consistency"] * private_consistency
        + weights["future_prediction"] * future_prediction
        + weights["acceleration"] * acceleration
        + weights["scale_consistency"] * scale_consistency
        + weights["reconstruction"] * reconstruction
        + weights["task_classification"] * task_classification
        + weights["state_classification"] * state_classification
        + weights["session_private"] * session_private
        + weights["artifact_private"] * artifact_private
        + weights["session_adversarial"] * session_adversarial
        + weights["artifact_adversarial"] * artifact_adversarial
        + weights["conditional_session_adversarial"] * conditional_session_adversarial
        + weights["task_cross_session"] * task_cross_session
        + weights["state_cross_session"] * state_cross_session
        + weights["task_session_alignment"] * task_session_alignment
        + weights["state_session_alignment"] * state_session_alignment
        + weights["artifact_alignment"] * artifact_alignment
        + weights["orthogonality"] * orthogonality
        + weights["reliability"] * reliability_regularization
        - weights["scale_entropy"] * scale_entropy
    )
    metrics = {
        "total": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "private_consistency": private_consistency,
        "future_prediction": future_prediction,
        "acceleration": acceleration,
        "scale_consistency": scale_consistency,
        "reconstruction": reconstruction,
        "task_classification": task_classification,
        "state_classification": state_classification,
        "session_private": session_private,
        "artifact_private": artifact_private,
        "session_adversarial": session_adversarial,
        "artifact_adversarial": artifact_adversarial,
        "conditional_session_adversarial": conditional_session_adversarial,
        "task_cross_session": task_cross_session,
        "state_cross_session": state_cross_session,
        "task_session_alignment": task_session_alignment,
        "state_session_alignment": state_session_alignment,
        "artifact_alignment": artifact_alignment,
        "orthogonality": orthogonality,
        "reliability": reliability_regularization,
        "scale_entropy": scale_entropy,
    }
    if return_factors:
        return total, metrics, factors
    return total, metrics


def build_visualization_encoder_from_config(config, adjacency=None):
    return FactorizedMultiScaleBrainStateEncoder(
        channel_count=int(config["channel_count"]),
        spatial_features=int(config["spatial_features"]),
        temporal_width=int(config["temporal_width"]),
        embedding_dim=int(config["embedding_dim"]),
        session_dim=int(config["session_dim"]),
        artifact_dim=int(config["artifact_dim"]),
        dropout=float(config["dropout"]),
        adjacency=adjacency,
        sample_rate=float(config["sample_rate"]),
        window_seconds=tuple(config["window_seconds_multi"]),
        nuisance_residualization=bool(config.get("nuisance_residualization", False)),
    )
