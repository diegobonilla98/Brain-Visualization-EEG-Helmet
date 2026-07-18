import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfilt_zi, tf2sos


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = MODULE_ROOT / "models"
CACHE_ROOT = MODULE_ROOT / "cache"
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO3", "POz", "PO4", "O1", "Oz", "O2", "Iz",
]
ELECTRODE_XY = {
    "Fp1": (-0.32, 0.95), "Fp2": (0.32, 0.95),
    "F7": (-0.88, 0.58), "F3": (-0.48, 0.58), "Fz": (0.0, 0.62), "F4": (0.48, 0.58), "F8": (0.88, 0.58),
    "FC5": (-0.68, 0.32), "FC1": (-0.25, 0.31), "FC2": (0.25, 0.31), "FC6": (0.68, 0.32),
    "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0.0, 0.0), "C4": (0.5, 0.0), "T8": (1.0, 0.0),
    "CP5": (-0.68, -0.30), "CP1": (-0.25, -0.30), "CP2": (0.25, -0.30), "CP6": (0.68, -0.30),
    "P7": (-0.88, -0.55), "P3": (-0.48, -0.56), "Pz": (0.0, -0.60), "P4": (0.48, -0.56), "P8": (0.88, -0.55),
    "PO3": (-0.36, -0.79), "POz": (0.0, -0.82), "PO4": (0.36, -0.79),
    "O1": (-0.30, -0.96), "Oz": (0.0, -1.0), "O2": (0.30, -0.96), "Iz": (0.0, -1.12),
}


@dataclass
class RecordingArray:
    path: Path
    data: np.ndarray
    sample_rate: float


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def discover_recordings(roots):
    paths = []
    for root_value in roots:
        root = resolve_path(root_value)
        if not root.exists():
            continue
        if root.is_file() and root.name == "eeg_samples.csv":
            paths.append(root.resolve())
        else:
            paths.extend(path.resolve() for path in root.rglob("eeg_samples.csv"))
    return sorted(set(paths), key=str)


def metadata_sample_rate(csv_path):
    metadata_path = csv_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if "sample_frequency_hz" in metadata:
            return float(metadata["sample_frequency_hz"])
    return 250.0


def filter_sos(sample_rate, low_hz=0.5, high_hz=45.0, notch_hz=50.0):
    high = min(float(high_hz), float(sample_rate) / 2.0 - 1.0)
    sections = [butter(4, [float(low_hz), high], btype="bandpass", fs=float(sample_rate), output="sos")]
    if notch_hz > 0 and sample_rate > notch_hz * 2.2:
        b, a = iirnotch(float(notch_hz), 30.0, float(sample_rate))
        sections.insert(0, tf2sos(b, a))
    return np.concatenate(sections, axis=0)


class CausalEEGPreprocessor:
    def __init__(self, input_rate, output_rate, channel_count, low_hz=0.5, high_hz=45.0, notch_hz=50.0):
        self.input_rate = float(input_rate)
        self.output_rate = float(output_rate)
        self.channel_count = int(channel_count)
        self.sos = filter_sos(self.input_rate, low_hz, high_hz, notch_hz)
        self.zi = None
        self.sample_index = 0
        ratio = self.input_rate / self.output_rate
        self.integer_decimation = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-8 else None

    def process(self, samples):
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != self.channel_count:
            raise ValueError(f"Expected ({self.channel_count}, samples), received {data.shape}")
        if data.shape[1] == 0:
            return np.zeros((self.channel_count, 0), dtype=np.float32)
        data = data - np.mean(data, axis=0, keepdims=True)
        if self.zi is None:
            base = sosfilt_zi(self.sos)
            self.zi = base[:, None, :] * data[None, :, 0, None]
        filtered, self.zi = sosfilt(self.sos, data, axis=1, zi=self.zi)
        if self.integer_decimation is not None:
            indices = np.arange(filtered.shape[1]) + self.sample_index
            keep = indices % self.integer_decimation == 0
            output = filtered[:, keep]
        else:
            up = int(round(self.output_rate))
            down = int(round(self.input_rate))
            divisor = math.gcd(up, down)
            output = resample_poly(filtered, up // divisor, down // divisor, axis=1)
        self.sample_index += data.shape[1]
        return output.astype(np.float32)


def load_csv_eeg(csv_path):
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    lower = {column.lower(): column for column in header}
    columns = []
    for name in CHANNEL_NAMES:
        column = lower.get(f"{name.lower()}_uv")
        if column is None:
            raise ValueError(f"{csv_path} does not contain the required channel {name}_uV")
        columns.append(column)
    validity_column = lower.get("valid_eeg_sample") or lower.get("streaming")
    use_columns = columns + ([validity_column] if validity_column is not None else [])
    frame = pd.read_csv(csv_path, usecols=use_columns)
    data = frame[columns].to_numpy(dtype=np.float64).T
    if validity_column is not None:
        values = frame[validity_column]
        if values.dtype == object:
            valid = values.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        else:
            valid = values.to_numpy(dtype=float) > 0.5
        data[:, ~valid] = np.nan
    sample_axis = np.arange(data.shape[1])
    for channel_index in range(data.shape[0]):
        finite = np.isfinite(data[channel_index])
        if np.all(finite):
            continue
        if np.sum(finite) < 2:
            data[channel_index] = 0.0
        else:
            data[channel_index] = np.interp(sample_axis, sample_axis[finite], data[channel_index, finite])
    return data


def cache_key(csv_path, output_rate, low_hz, high_hz, notch_hz):
    stat = csv_path.stat()
    value = f"{csv_path}|{stat.st_size}|{stat.st_mtime_ns}|{output_rate}|{low_hz}|{high_hz}|{notch_hz}|v2"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def load_preprocessed_recording(csv_path, cache_root=CACHE_ROOT, output_rate=125.0, low_hz=0.5, high_hz=45.0, notch_hz=50.0):
    csv_path = Path(csv_path).resolve()
    cache_root = resolve_path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{cache_key(csv_path, output_rate, low_hz, high_hz, notch_hz)}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return RecordingArray(csv_path, cached["data"], float(cached["sample_rate"]))
    input_rate = metadata_sample_rate(csv_path)
    raw = load_csv_eeg(csv_path)
    preprocessor = CausalEEGPreprocessor(input_rate, output_rate, len(CHANNEL_NAMES), low_hz, high_hz, notch_hz)
    data = preprocessor.process(raw)
    np.savez(cache_path, data=data, sample_rate=np.asarray(output_rate, dtype=np.float64))
    return RecordingArray(csv_path, data, float(output_rate))


def load_recordings(roots, cache_root=CACHE_ROOT, output_rate=125.0, low_hz=0.5, high_hz=45.0, notch_hz=50.0):
    paths = discover_recordings(roots)
    if len(paths) == 0:
        raise FileNotFoundError(f"No eeg_samples.csv files found under {roots}")
    recordings = []
    for index, path in enumerate(paths):
        print(f"Loading recording {index + 1}/{len(paths)}: {path}")
        recordings.append(load_preprocessed_recording(path, cache_root, output_rate, low_hz, high_hz, notch_hz))
    return recordings


def robust_normalization(recordings, subsample_step=10):
    sampled = np.concatenate([recording.data[:, ::subsample_step] for recording in recordings], axis=1).astype(np.float64)
    center = np.median(sampled, axis=1)
    scale = 1.4826 * np.median(np.abs(sampled - center[:, None]), axis=1)
    positive = scale[scale > 1e-8]
    fallback = float(np.median(positive)) if len(positive) > 0 else 1.0
    scale[scale <= 1e-8] = fallback
    return center.astype(np.float32), scale.astype(np.float32)


def recording_normalizations(recordings, subsample_step=10):
    centers = []
    scales = []
    for recording in recordings:
        center, scale = robust_normalization([recording], subsample_step)
        centers.append(center)
        scales.append(scale)
    return np.stack(centers), np.stack(scales)


def normalize_window(window, center, scale, clip_value=12.0):
    normalized = (np.asarray(window, dtype=np.float32) - center[:, None]) / scale[:, None]
    return np.clip(normalized, -float(clip_value), float(clip_value))


def electrode_adjacency(channel_names=CHANNEL_NAMES, neighbor_count=5, sigma=0.42):
    coordinates = np.asarray([ELECTRODE_XY[name] for name in channel_names], dtype=np.float32)
    distances = np.sqrt(np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=2))
    adjacency = np.zeros_like(distances, dtype=np.float32)
    for index in range(len(channel_names)):
        neighbors = np.argsort(distances[index])[:neighbor_count + 1]
        adjacency[index, neighbors] = np.exp(-(distances[index, neighbors] ** 2) / (2.0 * sigma ** 2))
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(len(channel_names), dtype=np.float32)
    degree = np.sum(adjacency, axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    return inverse[:, None] * adjacency * inverse[None, :]


def latest_pointer(directory, pointer_name, pattern):
    directory = resolve_path(directory)
    pointer = directory / pointer_name
    if pointer.exists():
        path = Path(pointer.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    candidates = sorted(directory.glob(pattern))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No matching artifact in {directory}")
    return candidates[-1]


def save_joblib(path, value):
    joblib.dump(value, path, compress=3)


def load_joblib(path):
    return joblib.load(path)


class StateDynamics:
    def __init__(self, step_seconds, fast_seconds=0.55, slow_seconds=2.5):
        self.fast_alpha = 1.0 - math.exp(-float(step_seconds) / float(fast_seconds))
        self.slow_alpha = 1.0 - math.exp(-float(step_seconds) / float(slow_seconds))
        self.fast = None
        self.slow = None
        self.previous_fast = None

    def update(self, embedding):
        value = np.asarray(embedding, dtype=np.float64)
        if self.fast is None:
            self.fast = value.copy()
            self.slow = value.copy()
            self.previous_fast = value.copy()
        else:
            self.previous_fast = self.fast.copy()
            self.fast += self.fast_alpha * (value - self.fast)
            self.slow += self.slow_alpha * (value - self.slow)
        velocity = self.fast - self.previous_fast
        return np.concatenate([self.fast, 0.35 * self.slow, 0.12 * velocity])


def state_dynamics_sequence(embeddings, recording_ids, step_seconds, fast_seconds=0.55, slow_seconds=2.5):
    output = np.zeros((len(embeddings), embeddings.shape[1] * 3), dtype=np.float64)
    current_id = None
    dynamics = None
    for index, (embedding, recording_id) in enumerate(zip(embeddings, recording_ids)):
        if recording_id != current_id:
            dynamics = StateDynamics(step_seconds, fast_seconds, slow_seconds)
            current_id = recording_id
        output[index] = dynamics.update(embedding)
    return output


def torch_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
