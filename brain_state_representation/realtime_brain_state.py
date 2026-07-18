import json
import math
import time
from collections import deque
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from brain_state_common import CHANNEL_NAMES, MODEL_ROOT, MODULE_ROOT, CausalEEGPreprocessor, StateDynamics, latest_pointer, load_joblib, torch_device
from brain_state_model import build_encoder_from_config
from universal_model import build_universal_tag_model_from_bundle
from eeg_quality_suite.brainaccess_stream import BrainAccessStream


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_DIR = MODEL_ROOT
MODEL_PATH = ""
MAP_PATH = ""
DEVICE = "auto"
INFERENCE_STEP_SECONDS = 0.25
BASELINE_SECONDS = 30.0
TAIL_SECONDS = 18.0
PLOT_UPDATE_MILLISECONDS = 50
BALL_SIZE = 125
TOP_TAG_COUNT = 7
TAG_SMOOTH_SECONDS = 1.5
OUTPUT_ROOT = MODULE_ROOT / "live_sessions"


def selected_paths():
    map_path = Path(MAP_PATH) if len(str(MAP_PATH).strip()) > 0 else latest_pointer(MODEL_DIR, "latest_map_path.txt", "brain_state_map_*.joblib")
    map_bundle = load_joblib(map_path)
    if len(str(MODEL_PATH).strip()) > 0:
        model_path = Path(MODEL_PATH)
    else:
        model_path = Path(map_bundle["model_path"])
    return model_path, map_path, map_bundle


def read_new_samples(recorder, consumed_chunks, row_indices):
    with recorder.lock:
        chunks = recorder.chunks[consumed_chunks:]
        consumed_chunks = len(recorder.chunks)
    rows = []
    for _, array, chunk_size in chunks:
        count = min(int(chunk_size), array.shape[1])
        if count > 0:
            rows.append(array[row_indices, :count])
    if len(rows) == 0:
        return consumed_chunks, None
    return consumed_chunks, np.concatenate(rows, axis=1)


def coordinate_color(hue):
    return plt.cm.hsv(float(hue) % 1.0)


class CoordinateFilter:
    def __init__(self, step_seconds, smoothing_seconds):
        self.alpha = 1.0 - math.exp(-float(step_seconds) / float(smoothing_seconds))
        self.value = None

    def update(self, value):
        value = np.asarray(value, dtype=float)
        if self.value is None:
            self.value = value.copy()
        else:
            self.value += self.alpha * (value - self.value)
        return self.value.copy()


class TagProbabilityFilter:
    def __init__(self, step_seconds, smoothing_seconds):
        self.alpha = 1.0 - math.exp(-float(step_seconds) / float(smoothing_seconds))
        self.value = None

    def update(self, value):
        value = np.asarray(value, dtype=float)
        if self.value is None:
            self.value = value.copy()
        else:
            self.value += self.alpha * (value - self.value)
        return self.value.copy()


def configure_tag_axis(axis, tag_count):
    axis.set_facecolor("#080d19")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.7, tag_count - 0.3)
    axis.invert_yaxis()
    axis.set_title("Predicted brain-state tags", color="white", fontsize=15, pad=18)
    axis.set_xlabel("smoothed confidence", color="#8d9ab2", fontsize=9)
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([])
    axis.tick_params(axis="x", colors="#63718a", labelsize=8)
    axis.grid(axis="x", alpha=0.10)
    for spine in axis.spines.values():
        spine.set_color("#1c2940")


def configure_axis(axis, dimensions, bounds):
    axis.set_facecolor("#050812")
    axis.set_xlabel("state 1", color="#aab7d1")
    axis.set_ylabel("state 2", color="#aab7d1")
    axis.tick_params(colors="#74829e")
    axis.grid(True, alpha=0.12)
    labels = ["x", "y", "z"]
    for dimension in range(dimensions):
        low, high = bounds[dimension]
        padding = max(1e-6, (high - low) * 0.10)
        getattr(axis, f"set_{labels[dimension]}lim")(low - padding, high + padding)
    if dimensions == 3:
        axis.set_zlabel("state 3", color="#aab7d1")
        axis.xaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        axis.yaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        axis.zaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))


def main():
    device = torch_device(DEVICE)
    model_path, map_path, map_bundle = selected_paths()
    model_bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    if list(model_bundle["channel_names"]) != CHANNEL_NAMES:
        raise RuntimeError("The trained channel order does not match the BrainAccess MAXI channel order")
    encoder = build_encoder_from_config(model_bundle["config"], model_bundle["adjacency"])
    encoder.load_state_dict(model_bundle["encoder_state"])
    encoder.to(device).eval()
    tag_model = build_universal_tag_model_from_bundle(model_bundle, encoder)
    if tag_model is not None:
        tag_model.to(device).eval()
    tag_vocabulary = list(model_bundle.get("tag_vocabulary", []))
    config = model_bundle["config"]
    window_samples = int(round(float(config["window_seconds"]) * float(config["sample_rate"])))
    dimensions = int(map_bundle["dimensions"])
    tail_count = max(3, int(round(TAIL_SECONDS / INFERENCE_STEP_SECONDS)))
    trajectory = deque(maxlen=tail_count)
    hue_tail = deque(maxlen=tail_count)
    output_rows = []
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"live_brain_state_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    figure = plt.figure(figsize=(15, 9), facecolor="#050812")
    grid = figure.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.95), wspace=0.16)
    if dimensions == 3:
        axis = figure.add_subplot(grid[0, :3], projection="3d")
        tail_artist = Line3DCollection([], linewidths=3.0, alpha=0.78)
        tail_on_axis = False
    else:
        axis = figure.add_subplot(grid[0, :3])
        tail_artist = LineCollection([], linewidths=3.0, alpha=0.78)
        tail_on_axis = True
        axis.add_collection(tail_artist)
    tag_axis = figure.add_subplot(grid[0, 3])
    configure_tag_axis(tag_axis, TOP_TAG_COUNT)
    tag_bars = tag_axis.barh(np.arange(TOP_TAG_COUNT), np.zeros(TOP_TAG_COUNT), height=0.64, color="#1f8fff", alpha=0.68)
    tag_labels = [tag_axis.text(0.025, index, "waiting for EEG", va="center", ha="left", color="white", fontsize=10, fontweight="semibold") for index in range(TOP_TAG_COUNT)]
    tag_scores = [tag_axis.text(0.975, index, "", va="center", ha="right", color="#dce8ff", fontsize=9) for index in range(TOP_TAG_COUNT)]
    if tag_model is None:
        tag_labels[0].set_text("No tag head in this model")
    configure_axis(axis, dimensions, np.asarray(map_bundle["bounds"]))
    axis.set_title("Live EEG state manifold", color="white", fontsize=18, pad=18)
    status = axis.text2D(0.02, 0.96, "Collecting the first window…", transform=axis.transAxes, color="#b8c6df", fontsize=11) if dimensions == 3 else axis.text(0.02, 0.96, "Collecting the first window…", transform=axis.transAxes, color="#b8c6df", fontsize=11)
    if dimensions == 3:
        glow = [axis.scatter([], [], [], s=size, alpha=alpha, depthshade=False) for size, alpha in [(BALL_SIZE * 3.5, 0.08), (BALL_SIZE * 2.0, 0.16), (BALL_SIZE, 0.95)]]
    else:
        glow = [axis.scatter([], [], s=size, alpha=alpha) for size, alpha in [(BALL_SIZE * 3.5, 0.08), (BALL_SIZE * 2.0, 0.16), (BALL_SIZE, 0.95)]]
    consumed_chunks = 0
    filtered_buffer = np.zeros((len(CHANNEL_NAMES), 0), dtype=np.float32)
    calibration_buffer = np.zeros((len(CHANNEL_NAMES), 0), dtype=np.float32)
    session_center = None
    session_scale = None
    last_inference_sample = 0
    started_at = time.perf_counter()
    with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
        recorder.start()
        channel_labels = [f"{name}_uV" for name in CHANNEL_NAMES]
        row_indices = [recorder.channel_indices[label] for label in channel_labels]
        preprocessor = CausalEEGPreprocessor(
            recorder.sample_frequency,
            float(config["sample_rate"]),
            len(CHANNEL_NAMES),
            float(config["low_hz"]),
            float(config["high_hz"]),
            float(config["notch_hz"]),
        )
        dynamics = StateDynamics(
            INFERENCE_STEP_SECONDS,
            float(map_bundle["fast_smooth_seconds"]),
            float(map_bundle["slow_smooth_seconds"]),
        )
        coordinate_filter = CoordinateFilter(INFERENCE_STEP_SECONDS, float(map_bundle["coordinate_smooth_seconds"]))
        tag_filter = TagProbabilityFilter(INFERENCE_STEP_SECONDS, TAG_SMOOTH_SECONDS)

        def update(_):
            nonlocal consumed_chunks, filtered_buffer, calibration_buffer, session_center, session_scale, last_inference_sample, tail_on_axis
            consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
            if new_samples is not None:
                filtered = preprocessor.process(new_samples)
                filtered_buffer = np.concatenate([filtered_buffer, filtered], axis=1)
                filtered_buffer = filtered_buffer[:, -window_samples:]
                if session_center is None:
                    calibration_buffer = np.concatenate([calibration_buffer, filtered], axis=1)
                last_inference_sample += filtered.shape[1]
            baseline_samples = int(round(BASELINE_SECONDS * float(config["sample_rate"])))
            if session_center is None and calibration_buffer.shape[1] >= baseline_samples:
                calibration = calibration_buffer[:, :baseline_samples]
                session_center = np.median(calibration, axis=1).astype(np.float32)
                session_scale = (1.4826 * np.median(np.abs(calibration - session_center[:, None]), axis=1)).astype(np.float32)
                positive = session_scale[session_scale > 1e-8]
                fallback = float(np.median(positive)) if len(positive) > 0 else 1.0
                session_scale[session_scale <= 1e-8] = fallback
                calibration_buffer = np.zeros((len(CHANNEL_NAMES), 0), dtype=np.float32)
            if session_center is None:
                remaining = max(0.0, (baseline_samples - calibration_buffer.shape[1]) / float(config["sample_rate"]))
                status.set_text(f"Eyes closed - fully relaxed - baseline calibration: {remaining:.1f}s")
                return []
            required_step = int(round(INFERENCE_STEP_SECONDS * float(config["sample_rate"])))
            if filtered_buffer.shape[1] < window_samples or last_inference_sample < required_step:
                remaining = max(0.0, (window_samples - filtered_buffer.shape[1]) / float(config["sample_rate"]))
                status.set_text(f"Collecting the first window: {remaining:.1f}s")
                return []
            last_inference_sample = 0
            window = np.clip((filtered_buffer - session_center[:, None]) / session_scale[:, None], -float(config["clip_value"]), float(config["clip_value"]))
            with torch.inference_mode():
                tensor = torch.from_numpy(window[None].astype(np.float32)).to(device)
                if str(config.get("architecture", "v1")) in {"channel_aware_multiscale_time_frequency_v2", "visualization_model_v1", "universal_tag_semantic_model_v1"}:
                    fused, scales, scale_weights, _, reliability = encoder(tensor, return_scales=True)
                    embedding = fused.cpu().numpy()[0]
                    normalized_scales = torch.nn.functional.normalize(scales, dim=2)
                    normalized_fused = torch.nn.functional.normalize(fused, dim=1)[:, None, :]
                    scale_disagreement = float((1.0 - torch.sum(normalized_scales * normalized_fused, dim=2)).mean().cpu())
                    scale_weight_values = scale_weights.cpu().numpy()[0]
                    reliability_values = torch.stack(reliability, dim=1).mean(dim=1).cpu().numpy()[0, :, 0]
                else:
                    embedding = encoder(tensor).cpu().numpy()[0]
                    scale_disagreement = None
                    scale_weight_values = None
                    reliability_values = None
                if tag_model is not None:
                    tag_probabilities = torch.sigmoid(tag_model.tag_logits(fused)).cpu().numpy()[0]
                else:
                    tag_probabilities = None
            dynamic_feature = dynamics.update(embedding)
            scaled = map_bundle["scaler"].transform(dynamic_feature[None])
            reduced = map_bundle["pca"].transform(scaled)
            raw_coordinate = map_bundle["parametric_mapper"].predict(reduced)[0]
            coordinate = coordinate_filter.update(raw_coordinate)
            hue_components = map_bundle["hue_pca"].transform(dynamic_feature[None])[0]
            hue = float(np.mod(np.arctan2(hue_components[1], hue_components[0]) / (2.0 * np.pi) + 0.5, 1.0))
            trajectory.append(coordinate)
            hue_tail.append(hue)
            elapsed = time.perf_counter() - started_at
            row = {"pc_time_perf_counter_s": time.perf_counter(), "t_from_start_s": elapsed, "hue": hue}
            for dimension, value in enumerate(coordinate):
                row[f"state_{dimension + 1}"] = float(value)
            for dimension, value in enumerate(embedding):
                row[f"embedding_{dimension + 1:03d}"] = float(value)
            if tag_probabilities is not None:
                smoothed_tag_probabilities = tag_filter.update(tag_probabilities)
                top_tag_indices = np.argsort(smoothed_tag_probabilities)[::-1][:TOP_TAG_COUNT]
                for tag_index, tag_name in enumerate(tag_vocabulary):
                    row[f"tag_probability_{tag_name}"] = float(smoothed_tag_probabilities[tag_index])
                for rank, tag_index in enumerate(top_tag_indices):
                    probability = float(smoothed_tag_probabilities[tag_index])
                    label = tag_vocabulary[tag_index].replace("_", " ")
                    tag_bars[rank].set_width(probability)
                    tag_bars[rank].set_color(plt.cm.viridis(0.20 + 0.75 * probability))
                    tag_labels[rank].set_text(label)
                    tag_scores[rank].set_text(f"{probability:0.0%}")
                    row[f"top_tag_{rank + 1}"] = tag_vocabulary[tag_index]
                    row[f"top_tag_{rank + 1}_probability"] = probability
            if scale_disagreement is not None:
                row["multiwindow_disagreement"] = scale_disagreement
                for scale_index, value in enumerate(scale_weight_values):
                    row[f"multiwindow_weight_{scale_index + 1}"] = float(value)
                row["channel_reliability_mean"] = float(np.mean(reliability_values))
                row["channel_reliability_min"] = float(np.min(reliability_values))
            output_rows.append(row)
            points = np.asarray(trajectory)
            colors = plt.cm.hsv(np.asarray(hue_tail))
            if len(points) > 1:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                fade = np.linspace(0.08, 0.90, len(segments))
                segment_colors = colors[:-1].copy()
                segment_colors[:, 3] = fade
                if dimensions == 3 and not tail_on_axis:
                    tail_artist.set_segments(segments)
                    tail_artist.set_color(segment_colors)
                    axis.add_collection3d(tail_artist)
                    tail_on_axis = True
                else:
                    tail_artist.set_segments(segments)
                    tail_artist.set_color(segment_colors)
            ball_color = coordinate_color(hue)
            for artist in glow:
                artist.set_facecolor(ball_color)
                artist.set_edgecolor("none")
                if dimensions == 3:
                    artist._offsets3d = ([coordinate[0]], [coordinate[1]], [coordinate[2]])
                else:
                    artist.set_offsets(coordinate[:2][None])
            status.set_text(f"t = {elapsed:7.1f}s   hue = {hue:0.2f}   {device}")
            return []

        animation = FuncAnimation(figure, update, interval=PLOT_UPDATE_MILLISECONDS, blit=False, cache_frame_data=False)
        figure.tight_layout()
        plt.show()
    trajectory_path = output_dir / "trajectory.csv"
    metadata_path = output_dir / "metadata.json"
    pd.DataFrame(output_rows).to_csv(trajectory_path, index=False)
    metadata = {
        "model_path": str(model_path),
        "map_path": str(map_path),
        "device_name": DEVICE_NAME,
        "inference_step_seconds": INFERENCE_STEP_SECONDS,
        "baseline_seconds": BASELINE_SECONDS,
        "tail_seconds": TAIL_SECONDS,
        "tag_model_enabled": tag_model is not None,
        "tag_vocabulary": tag_vocabulary,
        "tag_smoothing_seconds": TAG_SMOOTH_SECONDS,
        "trajectory_rows": len(output_rows),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("Saved live trajectory:", trajectory_path)


if __name__ == "__main__":
    main()
