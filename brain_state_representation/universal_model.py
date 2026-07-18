import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visualization_model import FactorizedBrainStateModel, FactorizedEEGDataset, factorized_losses


class TagAwareEEGDataset(FactorizedEEGDataset):
    def __init__(self, *args, tag_vectors, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag_vectors = np.asarray(tag_vectors, dtype=np.float32)

    def __getitem__(self, index):
        base = super().__getitem__(index)
        recording_index = self.indices[index][0]
        return base + (torch.from_numpy(self.tag_vectors[recording_index]),)


class UniversalBrainStateModel(FactorizedBrainStateModel):
    def __init__(self, *args, tag_count, tag_head_type="prototype", **kwargs):
        super().__init__(*args, **kwargs)
        dimension = self.encoder.embedding_dim
        self.tag_head_type = str(tag_head_type)
        if self.tag_head_type == "prototype":
            self.tag_projection = nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, dimension),
                nn.GELU(),
                nn.Dropout(0.10),
            )
            self.tag_prototypes = nn.Parameter(torch.empty(tag_count, dimension))
            self.tag_bias = nn.Parameter(torch.zeros(tag_count))
            nn.init.normal_(self.tag_prototypes, std=dimension ** -0.5)
        elif self.tag_head_type == "mlp":
            self.tag_head = nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, dimension),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(dimension, tag_count),
            )
        else:
            raise ValueError(f"Unknown tag head: {tag_head_type}")

    def tag_logits(self, state):
        if self.tag_head_type == "prototype":
            projected = F.normalize(self.tag_projection(state), dim=1)
            prototypes = F.normalize(self.tag_prototypes, dim=1)
            return 8.0 * projected @ prototypes.T + self.tag_bias
        return self.tag_head(state)


def build_universal_tag_model_from_bundle(bundle, encoder):
    if bundle.get("config", {}).get("architecture") != "universal_tag_semantic_model_v1":
        return None
    if "tag_model_state" not in bundle or "tag_vocabulary" not in bundle:
        return None
    model = UniversalBrainStateModel(
        encoder,
        len(bundle["task_to_index"]),
        len(bundle["state_to_index"]),
        len(bundle["session_to_index"]),
        len(bundle["artifact_to_index"]),
        len(bundle["config"]["future_horizon_seconds"]),
        len(bundle["channel_names"]),
        conditional_session_counts=bundle["conditional_session_counts"],
        tag_count=len(bundle["tag_vocabulary"]),
        tag_head_type=bundle["config"]["tag_head_type"],
    )
    state = model.state_dict()
    state.update(bundle["tag_model_state"])
    model.load_state_dict(state)
    return model


def multilabel_loss(logits, targets, positive_weights, label_smoothing=0.02):
    targets = targets * (1.0 - 2.0 * label_smoothing) + label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weights)


def tag_relational_loss(state, tags, recording_indices):
    if len(state) < 3:
        return state.new_tensor(0.0)
    normalized = F.normalize(state, dim=1)
    similarity = normalized @ normalized.T
    intersection = tags @ tags.T
    union = tags.sum(dim=1, keepdim=True) + tags.sum(dim=1, keepdim=True).T - intersection
    target = intersection / union.clamp_min(1.0)
    different_recordings = recording_indices[:, None] != recording_indices[None, :]
    upper = torch.triu(torch.ones_like(different_recordings), diagonal=1).bool()
    valid = different_recordings & upper & (union > 0)
    if not torch.any(valid):
        return state.new_tensor(0.0)
    return F.smooth_l1_loss(similarity[valid], target[valid])


def tag_prototype_alignment(model, state, tags):
    if model.tag_head_type != "prototype":
        return state.new_tensor(0.0)
    active = tags.sum(dim=1) > 0
    if not torch.any(active):
        return state.new_tensor(0.0)
    prototypes = F.normalize(model.tag_prototypes, dim=1)
    targets = tags[active] @ prototypes
    targets = targets / tags[active].sum(dim=1, keepdim=True).clamp_min(1.0)
    projected = F.normalize(model.tag_projection(state[active]), dim=1)
    return torch.mean(1.0 - F.cosine_similarity(projected, targets, dim=1))


def tag_augmentation_consistency(model, original_state, augmented_state):
    original = torch.sigmoid(model.tag_logits(original_state)).detach()
    augmented = torch.sigmoid(model.tag_logits(augmented_state))
    return F.smooth_l1_loss(augmented, original)


def session_style_augment(x, adjacency):
    batch_size, channel_count, sample_count = x.shape
    channel_gain = torch.exp(0.10 * torch.randn(batch_size, channel_count, 1, device=x.device))
    global_gain = torch.exp(0.06 * torch.randn(batch_size, 1, 1, device=x.device))
    output = x * channel_gain * global_gain
    spatial = torch.einsum("ij,bjt->bit", adjacency, output)
    mixing = 0.12 * torch.rand(batch_size, 1, 1, device=x.device)
    output = (1.0 - mixing) * output + mixing * spatial
    kernel = min(31, sample_count if sample_count % 2 == 1 else sample_count - 1)
    low_frequency = F.avg_pool1d(output, kernel_size=max(3, kernel), stride=1, padding=max(3, kernel) // 2)
    spectral_tilt = 0.08 * torch.randn(batch_size, channel_count, 1, device=x.device)
    output = output + spectral_tilt * low_frequency
    correlated_noise = torch.einsum("ij,bjt->bit", adjacency, torch.randn_like(output))
    return output + 0.015 * correlated_noise


def universal_losses(
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
    tag_targets,
    base_weights,
    universal_weights,
    positive_tag_weights,
    adversarial_strength,
):
    base_total, metrics, factors = factorized_losses(
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
        base_weights,
        adversarial_strength,
        return_factors=True,
    )
    state = factors["state"]
    tag_logits = model.tag_logits(state)
    tag_classification = multilabel_loss(tag_logits, tag_targets, positive_tag_weights)
    tag_relation = tag_relational_loss(state, tag_targets, recording_indices)
    prototype_alignment = tag_prototype_alignment(model, state, tag_targets)
    if universal_weights["tag_consistency"] > 0 or universal_weights["session_style_invariance"] > 0:
        style_state = model.encoder.encode_factors(session_style_augment(current, model.encoder.adjacency))["state"]
        tag_consistency = tag_augmentation_consistency(model, state, style_state)
        session_style_invariance = torch.mean(1.0 - F.cosine_similarity(state, style_state, dim=1))
    else:
        tag_consistency = state.new_tensor(0.0)
        session_style_invariance = state.new_tensor(0.0)
    total = (
        base_total
        + universal_weights["tag_classification"] * tag_classification
        + universal_weights["tag_relation"] * tag_relation
        + universal_weights["prototype_alignment"] * prototype_alignment
        + universal_weights["tag_consistency"] * tag_consistency
        + universal_weights["session_style_invariance"] * session_style_invariance
    )
    metrics.update({
        "tag_classification": tag_classification,
        "tag_relation": tag_relation,
        "prototype_alignment": prototype_alignment,
        "tag_consistency": tag_consistency,
        "session_style_invariance": session_style_invariance,
        "total": total,
    })
    return total, metrics
