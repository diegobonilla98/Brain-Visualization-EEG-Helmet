import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from brain_state_common import CHANNEL_NAMES, electrode_adjacency, normalize_window


SPECTRAL_BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 45.0)]


class MultiScaleLabeledEEGDataset(Dataset):
    def __init__(
        self,
        recordings,
        labels,
        recording_indices,
        max_window_samples,
        shortest_window_samples,
        step_samples,
        centers,
        scales,
        family_to_index,
        state_to_index,
        clip_value=12.0,
    ):
        self.recordings = recordings
        self.labels = labels
        self.max_window_samples = int(max_window_samples)
        self.shortest_window_samples = int(shortest_window_samples)
        self.step_samples = int(step_samples)
        self.centers = np.asarray(centers, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.family_to_index = family_to_index
        self.state_to_index = state_to_index
        self.clip_value = float(clip_value)
        self.indices = []
        self.family_indices = []
        self.recording_window_indices = []
        for recording_index in recording_indices:
            sample_count = recordings[recording_index].data.shape[1]
            first_end = self.max_window_samples + self.step_samples
            last_end = sample_count - self.step_samples
            for end in range(first_end, max(first_end, last_end + 1), self.step_samples):
                family = labels[recording_index]["task_family"]
                self.indices.append((recording_index, end))
                self.family_indices.append(family_to_index.get(family, -1))
                self.recording_window_indices.append(recording_index)

    def __len__(self):
        return len(self.indices)

    def label_at_end(self, recording_index, end):
        label_data = self.labels[recording_index]
        start = max(0, end - self.shortest_window_samples)
        valid = label_data["is_valid"][start:end]
        state_values = label_data["state_label"][start:end]
        usable = valid & (state_values != "unlabeled_transition") & (state_values != "invalid_stream_sample")
        if np.mean(usable) < 0.80:
            return -1, "unlabeled_transition", -1
        values, counts = np.unique(state_values[usable], return_counts=True)
        state_label = str(values[int(np.argmax(counts))])
        state_key = f"{label_data['task_family']}::{state_label}"
        segment_values = label_data["segment_index"][start:end][usable]
        segment_index = int(np.median(segment_values)) if len(segment_values) > 0 else -1
        return self.state_to_index.get(state_key, -1), state_label, segment_index

    def __getitem__(self, index):
        recording_index, end = self.indices[index]
        data = self.recordings[recording_index].data
        center = self.centers[recording_index]
        scale = self.scales[recording_index]
        previous_end = end - self.step_samples
        following_end = end + self.step_samples
        previous = normalize_window(data[:, previous_end - self.max_window_samples:previous_end], center, scale, self.clip_value)
        current = normalize_window(data[:, end - self.max_window_samples:end], center, scale, self.clip_value)
        following = normalize_window(data[:, following_end - self.max_window_samples:following_end], center, scale, self.clip_value)
        family_name = self.labels[recording_index]["task_family"]
        family_index = self.family_to_index.get(family_name, -1)
        previous_state, previous_label, previous_segment = self.label_at_end(recording_index, previous_end)
        current_state, current_label, current_segment = self.label_at_end(recording_index, end)
        following_state, following_label, following_segment = self.label_at_end(recording_index, following_end)
        smooth = (
            current_label not in {"unlabeled_transition", "invalid_stream_sample"}
            and previous_label == current_label == following_label
            and previous_segment == current_segment == following_segment
        )
        return (
            torch.from_numpy(previous),
            torch.from_numpy(current),
            torch.from_numpy(following),
            torch.tensor(family_index, dtype=torch.long),
            torch.tensor(current_state, dtype=torch.long),
            torch.tensor(smooth, dtype=torch.bool),
        )


class GraphFeatureBlockV2(nn.Module):
    def __init__(self, feature_count, dropout):
        super().__init__()
        self.local = nn.Linear(feature_count, feature_count)
        self.graph = nn.Linear(feature_count, feature_count)
        self.gate = nn.Linear(feature_count * 2, feature_count)
        self.norm = nn.LayerNorm(feature_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        spatial = torch.einsum("ij,bjft->bift", adjacency, x)
        local_features = x.permute(0, 1, 3, 2)
        graph_features = spatial.permute(0, 1, 3, 2)
        gate = torch.sigmoid(self.gate(torch.cat([local_features, graph_features], dim=-1)))
        mixed = self.local(local_features) + gate * self.graph(graph_features)
        output = self.norm(local_features + self.dropout(F.gelu(mixed)))
        return output.permute(0, 1, 3, 2)


class TemporalResidualBlockV2(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation, groups=width)
        self.norm = nn.GroupNorm(8, width)
        self.expand = nn.Conv1d(width, width * 2, 1)
        self.contract = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(self.depthwise(x))
        value, gate = self.expand(x).chunk(2, dim=1)
        x = self.contract(F.gelu(value) * torch.sigmoid(gate))
        return residual + self.dropout(x)


class ChannelAwareScaleEncoder(nn.Module):
    def __init__(self, channel_count, spatial_features, temporal_width, embedding_dim, dropout, adjacency, sample_rate):
        super().__init__()
        self.channel_count = int(channel_count)
        self.spatial_features = int(spatial_features)
        self.sample_rate = float(sample_rate)
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.channel_log_gain = nn.Parameter(torch.zeros(channel_count))
        self.channel_bias = nn.Parameter(torch.zeros(channel_count))
        self.channel_embedding = nn.Parameter(torch.randn(channel_count, spatial_features) * 0.02)
        self.channel_quality = nn.Sequential(
            nn.Linear(3, spatial_features),
            nn.GELU(),
            nn.Linear(spatial_features, 1),
        )
        self.temporal_stem = nn.Conv1d(channel_count, channel_count * spatial_features, 21, stride=2, padding=10, groups=channel_count)
        self.stem_norm = nn.GroupNorm(channel_count, channel_count * spatial_features)
        self.frequency_mlp = nn.Sequential(
            nn.LayerNorm(len(SPECTRAL_BANDS)),
            nn.Linear(len(SPECTRAL_BANDS), spatial_features * 2),
            nn.GELU(),
            nn.Linear(spatial_features * 2, spatial_features),
        )
        self.graph_blocks = nn.ModuleList([
            GraphFeatureBlockV2(spatial_features, dropout),
            GraphFeatureBlockV2(spatial_features, dropout),
            GraphFeatureBlockV2(spatial_features, dropout),
        ])
        self.channel_score = nn.Linear(spatial_features, 1)
        self.temporal_projection = nn.Conv1d(spatial_features * 2, temporal_width, 9, stride=2, padding=4)
        self.temporal_blocks = nn.ModuleList([
            TemporalResidualBlockV2(temporal_width, 1, dropout),
            TemporalResidualBlockV2(temporal_width, 2, dropout),
            TemporalResidualBlockV2(temporal_width, 4, dropout),
            TemporalResidualBlockV2(temporal_width, 8, dropout),
        ])
        self.temporal_score = nn.Conv1d(temporal_width, 1, 1)
        self.output = nn.Sequential(
            nn.Linear(temporal_width * 2 + spatial_features, temporal_width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_width * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def band_features(self, x):
        spectrum = torch.fft.rfft(x, dim=2)
        power = spectrum.real.square() + spectrum.imag.square()
        frequencies = torch.fft.rfftfreq(x.shape[2], d=1.0 / self.sample_rate).to(x.device)
        values = []
        for low, high in SPECTRAL_BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            values.append(torch.log1p(torch.mean(power[:, :, mask], dim=2) / x.shape[2]))
        return torch.stack(values, dim=2)

    def forward(self, x):
        gain = torch.exp(torch.clamp(self.channel_log_gain, -1.0, 1.0))[None, :, None]
        bias = self.channel_bias[None, :, None]
        x = x * gain + bias
        standard_deviation = torch.std(x, dim=2).clamp_min(1e-5)
        derivative_deviation = torch.std(torch.diff(x, dim=2), dim=2).clamp_min(1e-5)
        extreme = torch.amax(torch.abs(x), dim=2)
        quality_features = torch.stack([
            torch.log1p(standard_deviation),
            torch.log1p(derivative_deviation / standard_deviation),
            torch.log1p(extreme),
        ], dim=2)
        reliability = torch.sigmoid(self.channel_quality(quality_features)).unsqueeze(2)
        bands = self.band_features(x)
        frequency_features = self.frequency_mlp(bands)
        batch_size, channel_count, _ = x.shape
        features = F.gelu(self.stem_norm(self.temporal_stem(x)))
        features = features.reshape(batch_size, channel_count, self.spatial_features, features.shape[-1])
        features = features * reliability
        features = features + self.channel_embedding[None, :, :, None]
        features = features + frequency_features[:, :, :, None]
        for block in self.graph_blocks:
            features = block(features, self.adjacency)
        channel_logits = self.channel_score(features.permute(0, 1, 3, 2)).squeeze(-1)
        channel_logits = channel_logits + torch.log(reliability.squeeze(2).clamp_min(1e-4))
        channel_weights = torch.softmax(channel_logits, dim=1).unsqueeze(2)
        attended = torch.sum(features * channel_weights, dim=1)
        averaged = torch.mean(features, dim=1)
        temporal = self.temporal_projection(torch.cat([attended, averaged], dim=1))
        for block in self.temporal_blocks:
            temporal = block(temporal)
        temporal_weights = torch.softmax(self.temporal_score(temporal), dim=2)
        attended_time = torch.sum(temporal * temporal_weights, dim=2)
        averaged_time = torch.mean(temporal, dim=2)
        frequency_global = torch.sum(frequency_features * torch.mean(channel_weights, dim=3), dim=1)
        embedding = self.output(torch.cat([attended_time, averaged_time, frequency_global], dim=1))
        return embedding, bands, reliability.squeeze(2)


class MultiScaleBrainStateEncoder(nn.Module):
    def __init__(
        self,
        channel_count=32,
        spatial_features=10,
        temporal_width=112,
        embedding_dim=160,
        dropout=0.12,
        adjacency=None,
        sample_rate=125.0,
        window_seconds=(2.0, 4.0, 8.0),
    ):
        super().__init__()
        if adjacency is None:
            adjacency = electrode_adjacency(CHANNEL_NAMES[:channel_count])
        self.channel_count = int(channel_count)
        self.embedding_dim = int(embedding_dim)
        self.sample_rate = float(sample_rate)
        self.window_seconds = tuple(float(value) for value in window_seconds)
        self.window_samples = tuple(int(round(value * self.sample_rate)) for value in self.window_seconds)
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.scale_encoder = ChannelAwareScaleEncoder(
            channel_count,
            spatial_features,
            temporal_width,
            embedding_dim,
            dropout,
            adjacency,
            sample_rate,
        )
        self.scale_tokens = nn.Parameter(torch.randn(len(self.window_seconds), embedding_dim) * 0.02)
        self.scale_score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def encode_scale(self, x, scale_index):
        sample_count = self.window_samples[scale_index]
        embedding, bands, reliability = self.scale_encoder(x[:, :, -sample_count:])
        return embedding + self.scale_tokens[scale_index], bands, reliability

    def forward(self, x, return_scales=False):
        scale_embeddings = []
        band_rows = []
        reliability_rows = []
        for scale_index in range(len(self.window_samples)):
            embedding, bands, reliability = self.encode_scale(x, scale_index)
            scale_embeddings.append(embedding)
            band_rows.append(bands)
            reliability_rows.append(reliability)
        stacked = torch.stack(scale_embeddings, dim=1)
        weights = torch.softmax(self.scale_score(stacked).squeeze(-1), dim=1)
        attended = torch.sum(stacked * weights[:, :, None], dim=1)
        averaged = torch.mean(stacked, dim=1)
        fused = self.fusion(torch.cat([attended, averaged], dim=1))
        if return_scales:
            return fused, stacked, weights, band_rows, reliability_rows
        return fused

    def forward_longest(self, x):
        embedding, _, _ = self.encode_scale(x, len(self.window_samples) - 1)
        return embedding


class HybridBrainStateModel(nn.Module):
    def __init__(self, encoder, task_count, state_count, channel_count=32, band_count=7):
        super().__init__()
        self.encoder = encoder
        dimension = encoder.embedding_dim
        self.predictor = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension))
        self.spectral_decoder = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, channel_count * band_count))
        self.task_head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.10), nn.Linear(dimension, task_count))
        self.state_head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.10), nn.Linear(dimension, state_count))
        self.channel_count = int(channel_count)
        self.band_count = int(band_count)
        self.task_count = int(task_count)
        self.state_count = int(state_count)


def augment_eeg_v2(x, adjacency):
    output = x.clone()
    batch_size, channel_count, sample_count = output.shape
    global_gain = torch.exp(torch.randn(batch_size, 1, 1, device=x.device) * 0.03)
    channel_gain = torch.exp(torch.randn(batch_size, channel_count, 1, device=x.device) * 0.02)
    output = output * global_gain * channel_gain
    channel_mask = torch.rand(batch_size, channel_count, 1, device=x.device) < 0.04
    spatial_estimate = torch.einsum("ij,bjt->bit", adjacency, output)
    use_zero = torch.rand(batch_size, 1, 1, device=x.device) < 0.35
    replacement = torch.where(use_zero, torch.zeros_like(output), spatial_estimate)
    output = torch.where(channel_mask, replacement, output)
    mask_length = max(1, int(round(sample_count * 0.05)))
    starts = torch.randint(0, max(1, sample_count - mask_length + 1), (batch_size,), device=x.device)
    for index, start in enumerate(starts.tolist()):
        output[index, :, start:start + mask_length] = 0.0
    correlated_noise = torch.einsum("ij,bjt->bit", adjacency, torch.randn_like(output))
    independent_noise = torch.randn_like(output)
    output = output + 0.012 * correlated_noise + 0.006 * independent_noise
    return output


def off_diagonal_v2(matrix):
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss_v2(z1, z2):
    invariance = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    variance = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))
    centered1 = z1 - z1.mean(dim=0)
    centered2 = z2 - z2.mean(dim=0)
    denominator = max(1, z1.shape[0] - 1)
    covariance1 = centered1.T @ centered1 / denominator
    covariance2 = centered2.T @ centered2 / denominator
    covariance = (off_diagonal_v2(covariance1).square().sum() + off_diagonal_v2(covariance2).square().sum()) / z1.shape[1]
    return invariance, variance, covariance


def contrastive_prediction_loss_v2(predictor, source, target, temperature=0.16):
    predicted = F.normalize(predictor(source), dim=1)
    target = F.normalize(target.detach(), dim=1)
    logits = predicted @ target.T / float(temperature)
    labels = torch.arange(len(source), device=source.device)
    return F.cross_entropy(logits, labels)


def supervised_contrastive_loss(embedding, labels, temperature=0.12):
    valid = labels >= 0
    if torch.sum(valid) < 3:
        return embedding.new_tensor(0.0)
    embedding = F.normalize(embedding[valid], dim=1)
    labels = labels[valid]
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (labels[:, None] == labels[None, :]) & ~identity
    logits = similarity - torch.max(similarity, dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positive_count = positive.sum(dim=1)
    usable = positive_count > 0
    if torch.sum(usable) == 0:
        return embedding.new_tensor(0.0)
    mean_log_probability = (positive * log_probability).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_probability[usable].mean()


def hybrid_losses(model, previous, current, following, task_labels, state_labels, smooth_mask, weights):
    augmented = augment_eeg_v2(current, model.encoder.adjacency)
    z_current, scale_embeddings, scale_weights, band_rows, reliability_rows = model.encoder(current, return_scales=True)
    z_augmented = model.encoder(augmented)
    z_previous = model.encoder.forward_longest(previous)
    z_following = model.encoder.forward_longest(following)
    z_current_longest = scale_embeddings[:, -1]
    invariance, variance, covariance = vicreg_loss_v2(z_current, z_augmented)
    prediction = 0.5 * (
        contrastive_prediction_loss_v2(model.predictor, z_current_longest, z_following)
        + contrastive_prediction_loss_v2(model.predictor, z_current_longest, z_previous)
    )
    if torch.any(smooth_mask):
        velocity_before = z_current_longest[smooth_mask] - z_previous[smooth_mask]
        velocity_after = z_following[smooth_mask] - z_current_longest[smooth_mask]
        acceleration = F.smooth_l1_loss(velocity_after, velocity_before)
    else:
        acceleration = z_current.new_tensor(0.0)
    normalized_scales = F.normalize(scale_embeddings, dim=2)
    normalized_fused = F.normalize(z_current, dim=1)[:, None, :]
    scale_consistency = torch.mean(1.0 - torch.sum(normalized_scales * normalized_fused, dim=2))
    target_bands = band_rows[-1].detach()
    decoded = model.spectral_decoder(z_current).reshape(current.shape[0], model.channel_count, model.band_count)
    reconstruction = F.smooth_l1_loss(decoded, target_bands)
    task_valid = task_labels >= 0
    state_valid = state_labels >= 0
    task_classification = F.cross_entropy(model.task_head(z_current[task_valid]), task_labels[task_valid]) if torch.any(task_valid) else z_current.new_tensor(0.0)
    state_classification = F.cross_entropy(model.state_head(z_current[state_valid]), state_labels[state_valid]) if torch.any(state_valid) else z_current.new_tensor(0.0)
    scale_task_classification = z_current.new_tensor(0.0)
    if torch.any(task_valid):
        for scale_index in range(scale_embeddings.shape[1]):
            scale_task_classification = scale_task_classification + F.cross_entropy(
                model.task_head(scale_embeddings[task_valid, scale_index]),
                task_labels[task_valid],
            )
        scale_task_classification = scale_task_classification / scale_embeddings.shape[1]
    task_contrastive = supervised_contrastive_loss(z_current, task_labels)
    state_contrastive = supervised_contrastive_loss(z_current, state_labels)
    reliability = torch.stack(reliability_rows, dim=1)
    reliability_regularization = torch.mean((torch.mean(reliability, dim=2) - 0.75).square())
    scale_entropy = -torch.mean(torch.sum(scale_weights * torch.log(scale_weights.clamp_min(1e-8)), dim=1))
    total = (
        weights["invariance"] * invariance
        + weights["variance"] * variance
        + weights["covariance"] * covariance
        + weights["prediction"] * prediction
        + weights["acceleration"] * acceleration
        + weights["scale_consistency"] * scale_consistency
        + weights["reconstruction"] * reconstruction
        + weights["task_classification"] * task_classification
        + weights["state_classification"] * state_classification
        + weights["scale_task_classification"] * scale_task_classification
        + weights["task_contrastive"] * task_contrastive
        + weights["state_contrastive"] * state_contrastive
        + weights["reliability"] * reliability_regularization
        - weights["scale_entropy"] * scale_entropy
    )
    metrics = {
        "total": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "prediction": prediction,
        "acceleration": acceleration,
        "scale_consistency": scale_consistency,
        "reconstruction": reconstruction,
        "task_classification": task_classification,
        "state_classification": state_classification,
        "scale_task_classification": scale_task_classification,
        "task_contrastive": task_contrastive,
        "state_contrastive": state_contrastive,
        "reliability": reliability_regularization,
        "scale_entropy": scale_entropy,
    }
    return total, metrics


def build_v2_encoder_from_config(config, adjacency=None):
    return MultiScaleBrainStateEncoder(
        channel_count=int(config["channel_count"]),
        spatial_features=int(config["spatial_features"]),
        temporal_width=int(config["temporal_width"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
        adjacency=adjacency,
        sample_rate=float(config["sample_rate"]),
        window_seconds=tuple(config["window_seconds_multi"]),
    )
