import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from brain_state_common import CACHE_ROOT, MODEL_ROOT, StateDynamics, latest_pointer, load_preprocessed_recording, robust_normalization, save_joblib, state_dynamics_sequence, torch_device
from brain_state_model import build_encoder_from_config


MODEL_DIR = MODEL_ROOT
CACHE_DIR = CACHE_ROOT
MODEL_PATH = ""
OUTPUT_DIMENSIONS = 3
ENCODING_STEP_SECONDS = 0.25
BATCH_SIZE = 128
PCA_DIMENSIONS = 32
UMAP_NEIGHBORS = 45
UMAP_MIN_DISTANCE = 0.12
UMAP_MAX_FIT_POINTS = 15000
FAST_SMOOTH_SECONDS = 0.80
SLOW_SMOOTH_SECONDS = 3.0
COORDINATE_SMOOTH_SECONDS = 0.90
DEVICE = "auto"
RANDOM_SEED = 73
SHOW_INTERACTIVE = True


def selected_model_path():
    if len(str(MODEL_PATH).strip()) > 0:
        return Path(MODEL_PATH)
    universal_pointer = MODEL_DIR / "latest_universal_model_path.txt"
    if universal_pointer.exists():
        return Path(universal_pointer.read_text(encoding="utf-8").strip())
    return latest_pointer(MODEL_DIR, "latest_model_path.txt", "brain_state_model_*.pt")


def load_encoder(path, device):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    encoder = build_encoder_from_config(bundle["config"], bundle["adjacency"])
    encoder.load_state_dict(bundle["encoder_state"])
    encoder.to(device).eval()
    return bundle, encoder


def encode_recording(recording, bundle, encoder, device):
    config = bundle["config"]
    center, scale = robust_normalization([recording])
    window_samples = int(round(float(config["window_seconds"]) * float(config["sample_rate"])))
    step_samples = int(round(ENCODING_STEP_SECONDS * float(config["sample_rate"])))
    starts = np.arange(0, max(1, recording.data.shape[1] - window_samples + 1), max(1, step_samples), dtype=int)
    starts = starts[starts + window_samples <= recording.data.shape[1]]
    rows = []
    for batch_start in range(0, len(starts), BATCH_SIZE):
        batch_starts = starts[batch_start:batch_start + BATCH_SIZE]
        windows = np.stack([
            np.clip((recording.data[:, start:start + window_samples] - center[:, None]) / scale[:, None], -float(config["clip_value"]), float(config["clip_value"]))
            for start in batch_starts
        ]).astype(np.float32)
        with torch.inference_mode():
            rows.append(encoder(torch.from_numpy(windows).to(device)).cpu().numpy())
    embeddings = np.concatenate(rows, axis=0)
    times = (starts + window_samples / 2.0) / float(config["sample_rate"])
    return embeddings, times


def smooth_coordinates(coordinates, recording_ids):
    alpha = 1.0 - math.exp(-ENCODING_STEP_SECONDS / COORDINATE_SMOOTH_SECONDS)
    output = coordinates.copy()
    current_id = None
    state = None
    for index, recording_id in enumerate(recording_ids):
        if recording_id != current_id:
            state = coordinates[index].copy()
            current_id = recording_id
        else:
            state += alpha * (coordinates[index] - state)
        output[index] = state
    return output


def hue_values(hue_pca, dynamic_features):
    values = hue_pca.transform(dynamic_features)
    return np.mod(np.arctan2(values[:, 1], values[:, 0]) / (2.0 * np.pi) + 0.5, 1.0)


def plot_map(coordinates, hues, recording_ids, output_path):
    colors = plt.cm.hsv(hues)
    figure = plt.figure(figsize=(14, 10), facecolor="#050812")
    if coordinates.shape[1] == 3:
        axis = figure.add_subplot(111, projection="3d", facecolor="#050812")
        for recording_id in np.unique(recording_ids):
            indices = np.where(recording_ids == recording_id)[0]
            points = coordinates[indices]
            if len(points) > 1:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                collection = Line3DCollection(segments, colors=colors[indices[:-1]], linewidths=1.2, alpha=0.65)
                axis.add_collection3d(collection)
        axis.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=colors, s=5, alpha=0.35)
        axis.set_zlabel("state 3", color="white")
        axis.xaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        axis.yaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        axis.zaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
    else:
        axis = figure.add_subplot(111, facecolor="#050812")
        for recording_id in np.unique(recording_ids):
            indices = np.where(recording_ids == recording_id)[0]
            points = coordinates[indices]
            if len(points) > 1:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                collection = LineCollection(segments, colors=colors[indices[:-1]], linewidths=1.2, alpha=0.65)
                axis.add_collection(collection)
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=5, alpha=0.35)
    for dimension in range(coordinates.shape[1]):
        low, high = np.percentile(coordinates[:, dimension], [0.5, 99.5])
        padding = max(1e-6, (high - low) * 0.08)
        if dimension == 0:
            axis.set_xlim(low - padding, high + padding)
        elif dimension == 1:
            axis.set_ylim(low - padding, high + padding)
        else:
            axis.set_zlim(low - padding, high + padding)
    axis.set_xlabel("state 1", color="white")
    axis.set_ylabel("state 2", color="white")
    axis.set_title("Temporally coherent EEG state manifold", color="white", fontsize=17, pad=18)
    axis.tick_params(colors="#8290aa")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    if SHOW_INTERACTIVE:
        plt.show()
    plt.close(figure)


def main():
    device = torch_device(DEVICE)
    model_path = selected_model_path()
    bundle, encoder = load_encoder(model_path, device)
    source_paths = list(dict.fromkeys(bundle["training_recordings"] + bundle.get("validation_recordings", [])))
    recordings = [
        load_preprocessed_recording(
            source,
            CACHE_DIR,
            float(bundle["config"]["sample_rate"]),
            float(bundle["config"]["low_hz"]),
            float(bundle["config"]["high_hz"]),
            float(bundle["config"]["notch_hz"]),
        )
        for source in source_paths
    ]
    embedding_rows = []
    time_rows = []
    recording_rows = []
    for recording_index, recording in enumerate(recordings):
        print(f"Encoding {recording_index + 1}/{len(recordings)}: {recording.path}")
        embeddings, times = encode_recording(recording, bundle, encoder, device)
        embedding_rows.append(embeddings)
        time_rows.append(times)
        recording_rows.extend([str(recording.path.parent)] * len(embeddings))
    embeddings = np.concatenate(embedding_rows, axis=0)
    times = np.concatenate(time_rows, axis=0)
    recording_ids = np.asarray(recording_rows, dtype=object)
    dynamic_features = state_dynamics_sequence(embeddings, recording_ids, ENCODING_STEP_SECONDS, FAST_SMOOTH_SECONDS, SLOW_SMOOTH_SECONDS)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(dynamic_features)
    component_count = min(PCA_DIMENSIONS, scaled.shape[1], len(scaled) - 1)
    pca = PCA(n_components=component_count, whiten=True, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(scaled)
    fit_indices = np.linspace(0, len(reduced) - 1, min(len(reduced), UMAP_MAX_FIT_POINTS), dtype=int)
    manifold = umap.UMAP(
        n_components=OUTPUT_DIMENSIONS,
        n_neighbors=min(UMAP_NEIGHBORS, max(2, len(fit_indices) - 1)),
        min_dist=UMAP_MIN_DISTANCE,
        metric="cosine",
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
        low_memory=True,
    )
    manifold.fit(reduced[fit_indices])
    parametric_mapper = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="tanh",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=RANDOM_SEED,
    )
    parametric_mapper.fit(reduced[fit_indices], manifold.embedding_)
    raw_coordinates = parametric_mapper.predict(reduced)
    coordinates = smooth_coordinates(raw_coordinates, recording_ids)
    hue_pca = PCA(n_components=2, random_state=RANDOM_SEED)
    hue_pca.fit(dynamic_features)
    hues = hue_values(hue_pca, dynamic_features)
    bounds = np.percentile(coordinates, [0.5, 99.5], axis=0).T
    adjacent_distances = []
    accelerations = []
    for recording_id in np.unique(recording_ids):
        values = coordinates[recording_ids == recording_id]
        if len(values) > 2:
            adjacent_distances.extend(np.linalg.norm(np.diff(values, axis=0), axis=1).tolist())
            accelerations.extend(np.linalg.norm(np.diff(values, n=2, axis=0), axis=1).tolist())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    map_path = MODEL_DIR / f"brain_state_map_{stamp}.joblib"
    csv_path = MODEL_DIR / f"brain_state_embeddings_{stamp}.csv"
    figure_path = MODEL_DIR / f"brain_state_map_{stamp}.png"
    metrics_path = MODEL_DIR / f"brain_state_map_metrics_{stamp}.json"
    map_bundle = {
        "format_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path),
        "dimensions": OUTPUT_DIMENSIONS,
        "step_seconds": ENCODING_STEP_SECONDS,
        "fast_smooth_seconds": FAST_SMOOTH_SECONDS,
        "slow_smooth_seconds": SLOW_SMOOTH_SECONDS,
        "coordinate_smooth_seconds": COORDINATE_SMOOTH_SECONDS,
        "scaler": scaler,
        "pca": pca,
        "parametric_mapper": parametric_mapper,
        "hue_pca": hue_pca,
        "bounds": bounds,
    }
    save_joblib(map_path, map_bundle)
    output_columns = {"recording": recording_ids, "time_s": times, "hue": hues}
    for dimension in range(OUTPUT_DIMENSIONS):
        output_columns[f"state_{dimension + 1}"] = coordinates[:, dimension]
    for dimension in range(embeddings.shape[1]):
        output_columns[f"embedding_{dimension + 1:03d}"] = embeddings[:, dimension]
    output = pd.DataFrame(output_columns)
    output.to_csv(csv_path, index=False)
    metrics = {
        "model_path": str(model_path),
        "map_path": str(map_path),
        "embedding_rows": int(len(output)),
        "recording_count": int(len(recordings)),
        "dimensions": OUTPUT_DIMENSIONS,
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "parametric_teacher_rmse": float(np.sqrt(np.mean((parametric_mapper.predict(reduced[fit_indices]) - manifold.embedding_) ** 2))),
        "median_adjacent_map_distance": float(np.median(adjacent_distances)),
        "median_map_acceleration": float(np.median(accelerations)),
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    plot_map(coordinates, hues, recording_ids, figure_path)
    (MODEL_DIR / "latest_map_path.txt").write_text(str(map_path), encoding="utf-8")
    (MODEL_DIR / "latest_embeddings_path.txt").write_text(str(csv_path), encoding="utf-8")
    print("Saved map:", map_path)
    print("Saved embeddings:", csv_path)
    print("Saved figure:", figure_path)


if __name__ == "__main__":
    main()
