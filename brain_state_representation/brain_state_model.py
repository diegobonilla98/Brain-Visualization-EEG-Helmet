import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from brain_state_common import CHANNEL_NAMES, electrode_adjacency, normalize_window
from brain_state_model_v2 import build_v2_encoder_from_config
from visualization_model import build_visualization_encoder_from_config


SPECTRAL_BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 45.0)]


class EEGTripletDataset(Dataset):
    def __init__(self, recordings, recording_indices, window_samples, step_samples, center, scale, clip_value=12.0):
        self.recordings = recordings
        self.window_samples = int(window_samples)
        self.step_samples = int(step_samples)
        centers = np.asarray(center, dtype=np.float32)
        scales = np.asarray(scale, dtype=np.float32)
        if centers.ndim == 1:
            centers = np.repeat(centers[None], len(recordings), axis=0)
            scales = np.repeat(scales[None], len(recordings), axis=0)
        self.center = centers
        self.scale = scales
        self.clip_value = float(clip_value)
        self.indices = []
        for recording_index in recording_indices:
            sample_count = recordings[recording_index].data.shape[1]
            first = self.step_samples
            last = sample_count - self.window_samples - self.step_samples
            for start in range(first, max(first, last + 1), self.step_samples):
                self.indices.append((recording_index, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        recording_index, start = self.indices[index]
        data = self.recordings[recording_index].data
        center = self.center[recording_index]
        scale = self.scale[recording_index]
        previous = normalize_window(data[:, start - self.step_samples:start - self.step_samples + self.window_samples], center, scale, self.clip_value)
        current = normalize_window(data[:, start:start + self.window_samples], center, scale, self.clip_value)
        following = normalize_window(data[:, start + self.step_samples:start + self.step_samples + self.window_samples], center, scale, self.clip_value)
        return torch.from_numpy(previous), torch.from_numpy(current), torch.from_numpy(following)


class GraphFeatureBlock(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.local = nn.Linear(feature_count, feature_count)
        self.graph = nn.Linear(feature_count, feature_count)
        self.norm = nn.LayerNorm(feature_count)

    def forward(self, x, adjacency):
        spatial = torch.einsum("ij,bjft->bift", adjacency, x)
        local = x.permute(0, 1, 3, 2)
        graph = spatial.permute(0, 1, 3, 2)
        output = self.norm(local + F.gelu(self.local(local) + self.graph(graph)))
        return output.permute(0, 1, 3, 2)


class TemporalResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation, groups=width)
        self.norm = nn.GroupNorm(8, width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(F.gelu(x))
        return residual + self.dropout(x)


class BrainStateEncoder(nn.Module):
    def __init__(self, channel_count=32, spatial_features=12, temporal_width=128, embedding_dim=128, dropout=0.10, adjacency=None):
        super().__init__()
        if adjacency is None:
            adjacency = electrode_adjacency(CHANNEL_NAMES[:channel_count])
        self.channel_count = int(channel_count)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.temporal_stem = nn.Conv1d(channel_count, channel_count * spatial_features, 25, stride=2, padding=12, groups=channel_count)
        self.stem_norm = nn.GroupNorm(channel_count, channel_count * spatial_features)
        self.graph_blocks = nn.ModuleList([GraphFeatureBlock(spatial_features), GraphFeatureBlock(spatial_features)])
        self.channel_score = nn.Linear(spatial_features, 1)
        self.temporal_projection = nn.Conv1d(spatial_features * 2, temporal_width, 9, stride=2, padding=4)
        self.temporal_blocks = nn.ModuleList([
            TemporalResidualBlock(temporal_width, 1, dropout),
            TemporalResidualBlock(temporal_width, 2, dropout),
            TemporalResidualBlock(temporal_width, 4, dropout),
            TemporalResidualBlock(temporal_width, 8, dropout),
        ])
        self.temporal_score = nn.Conv1d(temporal_width, 1, 1)
        self.embedding = nn.Sequential(
            nn.Linear(temporal_width * 2, temporal_width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_width * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        batch_size, channel_count, _ = x.shape
        x = F.gelu(self.stem_norm(self.temporal_stem(x)))
        x = x.reshape(batch_size, channel_count, -1, x.shape[-1])
        for block in self.graph_blocks:
            x = block(x, self.adjacency)
        channel_logits = self.channel_score(x.permute(0, 1, 3, 2)).squeeze(-1)
        channel_weights = torch.softmax(channel_logits, dim=1).unsqueeze(2)
        attended = torch.sum(x * channel_weights, dim=1)
        averaged = torch.mean(x, dim=1)
        x = self.temporal_projection(torch.cat([attended, averaged], dim=1))
        for block in self.temporal_blocks:
            x = block(x)
        temporal_weights = torch.softmax(self.temporal_score(x), dim=2)
        attended_time = torch.sum(x * temporal_weights, dim=2)
        averaged_time = torch.mean(x, dim=2)
        return self.embedding(torch.cat([attended_time, averaged_time], dim=1))


class BrainStateSSL(nn.Module):
    def __init__(self, encoder, channel_count=32, band_count=7):
        super().__init__()
        self.encoder = encoder
        dimension = encoder.embedding_dim
        self.predictor = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, dimension))
        self.spectral_decoder = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Linear(dimension * 2, channel_count * band_count))
        self.channel_count = int(channel_count)
        self.band_count = int(band_count)


def augment_eeg(x, adjacency, max_shift_samples=16, channel_drop_probability=0.12, time_mask_fraction=0.10, noise_scale=0.04):
    output = x.clone()
    batch_size, channel_count, sample_count = output.shape
    gains = torch.exp(torch.randn(batch_size, 1, 1, device=x.device) * 0.10)
    output = output * gains
    shifts = torch.randint(-max_shift_samples, max_shift_samples + 1, (batch_size,), device=x.device)
    for index, shift in enumerate(shifts.tolist()):
        output[index] = torch.roll(output[index], shift, dims=1)
    channel_mask = torch.rand(batch_size, channel_count, 1, device=x.device) < channel_drop_probability
    spatial_estimate = torch.einsum("ij,bjt->bit", adjacency, output)
    output = torch.where(channel_mask, spatial_estimate, output)
    mask_length = max(1, int(round(sample_count * time_mask_fraction)))
    starts = torch.randint(0, max(1, sample_count - mask_length + 1), (batch_size,), device=x.device)
    for index, start in enumerate(starts.tolist()):
        output[index, :, start:start + mask_length] = 0.0
    correlated_noise = torch.einsum("ij,bjt->bit", adjacency, torch.randn_like(output))
    output = output + noise_scale * correlated_noise
    return output


def spectral_targets(x, sample_rate):
    spectrum = torch.fft.rfft(x, dim=2)
    power = spectrum.real.square() + spectrum.imag.square()
    frequencies = torch.fft.rfftfreq(x.shape[2], d=1.0 / float(sample_rate)).to(x.device)
    targets = []
    for low, high in SPECTRAL_BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        targets.append(torch.log1p(torch.mean(power[:, :, mask], dim=2) / x.shape[2]))
    return torch.stack(targets, dim=2)


def off_diagonal(matrix):
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2):
    invariance = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    variance = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))
    centered1 = z1 - z1.mean(dim=0)
    centered2 = z2 - z2.mean(dim=0)
    denominator = max(1, z1.shape[0] - 1)
    covariance1 = centered1.T @ centered1 / denominator
    covariance2 = centered2.T @ centered2 / denominator
    covariance = (off_diagonal(covariance1).square().sum() + off_diagonal(covariance2).square().sum()) / z1.shape[1]
    return invariance, variance, covariance


def contrastive_prediction_loss(predictor, source, target, temperature=0.15):
    predicted = F.normalize(predictor(source), dim=1)
    target = F.normalize(target.detach(), dim=1)
    logits = predicted @ target.T / float(temperature)
    labels = torch.arange(len(source), device=source.device)
    return F.cross_entropy(logits, labels)


def ssl_losses(model, previous, current, following, sample_rate, weights):
    view1 = augment_eeg(current, model.encoder.adjacency)
    view2 = augment_eeg(current, model.encoder.adjacency)
    z1 = model.encoder(view1)
    z2 = model.encoder(view2)
    z_previous = model.encoder(previous)
    z_current = model.encoder(current)
    z_following = model.encoder(following)
    invariance, variance, covariance = vicreg_loss(z1, z2)
    prediction = 0.5 * (
        contrastive_prediction_loss(model.predictor, z_current, z_following)
        + contrastive_prediction_loss(model.predictor, z_current, z_previous)
    )
    velocity_before = z_current - z_previous
    velocity_after = z_following - z_current
    acceleration = F.smooth_l1_loss(velocity_after, velocity_before)
    decoded = model.spectral_decoder(z_current).reshape(current.shape[0], model.channel_count, model.band_count)
    reconstruction = F.smooth_l1_loss(decoded, spectral_targets(current, sample_rate))
    total = (
        weights["invariance"] * invariance
        + weights["variance"] * variance
        + weights["covariance"] * covariance
        + weights["prediction"] * prediction
        + weights["acceleration"] * acceleration
        + weights["reconstruction"] * reconstruction
    )
    metrics = {
        "total": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "prediction": prediction,
        "acceleration": acceleration,
        "reconstruction": reconstruction,
    }
    return total, metrics


def exponential_moving_average(target, source, decay):
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(decay).add_(source_parameter.data, alpha=1.0 - decay)
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(source_buffer)


def make_ema_encoder(encoder):
    output = copy.deepcopy(encoder)
    output.requires_grad_(False)
    output.eval()
    return output


def effective_rank(embeddings):
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular_values / max(float(np.sum(singular_values)), 1e-12)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return float(np.exp(entropy))


def build_encoder_from_config(config, adjacency=None):
    architecture = str(config.get("architecture", "v1"))
    if architecture in {"visualization_model_v1", "universal_tag_semantic_model_v1"}:
        return build_visualization_encoder_from_config(config, adjacency)
    if architecture == "channel_aware_multiscale_time_frequency_v2":
        return build_v2_encoder_from_config(config, adjacency)
    return BrainStateEncoder(
        channel_count=int(config["channel_count"]),
        spatial_features=int(config["spatial_features"]),
        temporal_width=int(config["temporal_width"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
        adjacency=adjacency,
    )
