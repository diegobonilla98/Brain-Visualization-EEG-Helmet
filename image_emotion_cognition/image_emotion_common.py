import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample, sosfiltfilt, welch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = MODULE_ROOT.parent / "sessions"
PROTOCOL_VERSION = "image_emotion_passive_viewing_v1"
MODEL_ROOT = MODULE_ROOT / "models"
DEFAULT_EMOSET_ROOT = Path("F:/EmoSet-118K")
EMOTION_LABELS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]
EMOTION_TO_INDEX = {label: index for index, label in enumerate(EMOTION_LABELS)}
DEFAULT_MAXI_32_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO3", "POz", "PO4", "O1", "Oz", "O2", "Iz",
]
FEATURE_BANDS = [
    (1.0, 4.0, "delta"),
    (4.0, 8.0, "theta"),
    (8.0, 12.0, "alpha"),
    (12.0, 16.0, "low_beta"),
    (16.0, 24.0, "mid_beta"),
    (24.0, 32.0, "high_beta"),
    (32.0, 45.0, "low_gamma"),
]
FEATURE_GROUPS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "left_frontal": ["Fp1", "F7", "F3", "FC5", "FC1"],
    "right_frontal": ["Fp2", "F8", "F4", "FC6", "FC2"],
    "temporal": ["T7", "T8", "F7", "F8"],
    "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["PO3", "POz", "PO4", "O1", "Oz", "O2"],
    "left_posterior": ["CP5", "CP1", "P7", "P3", "PO3", "O1"],
    "right_posterior": ["CP6", "CP2", "P8", "P4", "PO4", "O2"],
    "midline": ["Fz", "Cz", "Pz", "POz", "Oz"],
}
ASYMMETRY_PAIRS = [
    ("Fp1", "Fp2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FC5", "FC6"),
    ("FC1", "FC2"),
    ("C3", "C4"),
    ("CP5", "CP6"),
    ("CP1", "CP2"),
    ("P3", "P4"),
    ("PO3", "PO4"),
    ("O1", "O2"),
]


def resolve_project_path(path_value):
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


def split_file_from_dataset(dataset_root, split_name):
    root = Path(dataset_root)
    if root.is_file():
        return root.parent, root
    direct_path = root / f"{split_name}.json"
    if direct_path.exists():
        return root, direct_path
    subset_path = root / split_name / "train.json"
    if subset_path.exists():
        return root, subset_path
    nested_path = root / "train.json"
    if nested_path.exists():
        return root, nested_path
    raise FileNotFoundError(f"No split JSON found for {split_name} under {root}")


def load_emoset_manifest(dataset_root=DEFAULT_EMOSET_ROOT, split_name="train"):
    root, split_path = split_file_from_dataset(dataset_root, split_name)
    rows = read_json(split_path)
    records = []
    for row in rows:
        emotion = str(row[0])
        image_relative_path = str(row[1])
        annotation_relative_path = str(row[2]) if len(row) > 2 else ""
        image_path = root / image_relative_path
        annotation_path = root / annotation_relative_path if len(annotation_relative_path) > 0 else Path("")
        image_id = Path(image_relative_path).stem
        records.append({
            "emotion": emotion,
            "emotion_index": EMOTION_TO_INDEX.get(emotion, -1),
            "image_relative_path": image_relative_path,
            "annotation_relative_path": annotation_relative_path,
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "image_id": image_id,
            "split_file": str(split_path),
        })
    manifest = pd.DataFrame(records)
    manifest = manifest[manifest["emotion"].isin(EMOTION_LABELS)].reset_index(drop=True)
    return manifest


def read_annotation_values(annotation_path):
    path = Path(annotation_path)
    if not path.exists():
        return {}
    data = read_json(path)
    output = {}
    for key in ["brightness", "colorfulness", "facial_expression"]:
        if key in data:
            output[key] = data[key]
    return output


def collected_image_ids(data_root=DATA_ROOT):
    root = Path(data_root)
    image_ids = set()
    if not root.exists():
        return image_ids
    for schedule_path in root.glob("*/session_schedule.csv"):
        df = pd.read_csv(schedule_path)
        if "image_id" in df.columns:
            image_ids.update(df["image_id"].astype(str).tolist())
    return image_ids


def balanced_emotion_schedule(manifest, images_per_session, emotions, random_seed, data_root=DATA_ROOT, avoid_previous=True):
    rng = np.random.default_rng(random_seed + int(time.time()))
    selected_emotions = list(emotions) if len(emotions) > 0 else EMOTION_LABELS.copy()
    available = manifest[manifest["emotion"].isin(selected_emotions)].copy()
    if avoid_previous:
        seen = collected_image_ids(data_root)
        unseen = available[~available["image_id"].astype(str).isin(seen)].copy()
        if len(unseen) >= images_per_session:
            available = unseen
    quota_base = images_per_session // len(selected_emotions)
    quota_extra = images_per_session % len(selected_emotions)
    emotion_order = selected_emotions.copy()
    rng.shuffle(emotion_order)
    quotas = {emotion: quota_base for emotion in selected_emotions}
    for emotion in emotion_order[:quota_extra]:
        quotas[emotion] += 1
    frames = []
    for emotion in selected_emotions:
        pool = available[available["emotion"] == emotion]
        if len(pool) < quotas[emotion]:
            pool = manifest[manifest["emotion"] == emotion]
        if len(pool) == 0:
            raise RuntimeError(f"No images available for emotion {emotion}")
        replace = len(pool) < quotas[emotion]
        indices = rng.choice(pool.index.to_numpy(), size=quotas[emotion], replace=replace)
        frames.append(manifest.loc[indices].copy())
    schedule = pd.concat(frames, ignore_index=True)
    for shuffle_index in range(2000):
        schedule = schedule.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        adjacent = schedule["emotion"].eq(schedule["emotion"].shift()).any()
        if not adjacent:
            break
    schedule.insert(0, "trial_index", np.arange(1, len(schedule) + 1, dtype=int))
    annotation_rows = []
    for annotation_path in schedule["annotation_path"].tolist():
        annotation_rows.append(read_annotation_values(annotation_path))
    annotation_frame = pd.DataFrame(annotation_rows)
    for column in ["brightness", "colorfulness", "facial_expression"]:
        if column not in annotation_frame.columns:
            annotation_frame[column] = None
        schedule[column] = annotation_frame[column].to_numpy(dtype=object)
    return schedule


def read_metadata(path):
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    return read_json(metadata_path)


def eeg_columns(df):
    return [column for column in df.columns if column.endswith("_uV")]


def clean_channel_name(column):
    return column.replace("_uV", "")


def ordered_eeg_columns(df):
    columns = eeg_columns(df)
    lower_map = {clean_channel_name(column).lower(): column for column in columns}
    ordered = []
    for name in DEFAULT_MAXI_32_NAMES:
        column = lower_map.get(name.lower())
        if column is not None:
            ordered.append(column)
    for column in columns:
        if column not in ordered:
            ordered.append(column)
    return ordered


def select_channel_columns(df, wanted_names, fallback_count=3):
    columns = eeg_columns(df)
    selected = []
    lower_map = {clean_channel_name(column).lower(): column for column in columns}
    for name in wanted_names:
        column = lower_map.get(str(name).lower())
        if column is not None and column not in selected:
            selected.append(column)
    if len(selected) == 0:
        selected = columns[:fallback_count]
    return selected


def sampling_frequency_from_df(df, metadata):
    if "sample_frequency_hz" in metadata:
        return float(metadata["sample_frequency_hz"])
    if "sample_time_est_s" in df.columns and len(df) > 2:
        dt = np.median(np.diff(df["sample_time_est_s"].to_numpy(dtype=float)))
        if dt > 0:
            return float(1.0 / dt)
    return 250.0


def get_eeg_array(df, columns):
    return df[columns].to_numpy(dtype=float).T


def valid_eeg_sample_mask(df):
    if "valid_eeg_sample" in df.columns:
        values = df["valid_eeg_sample"]
        if values.dtype == object:
            text = values.astype(str).str.lower()
            return text.isin(["true", "1", "yes"]).to_numpy()
        return values.to_numpy(dtype=bool)
    if "streaming" in df.columns:
        return df["streaming"].to_numpy(dtype=float) > 0.5
    return np.ones(len(df), dtype=bool)


def mark_invalid_eeg_samples(df):
    output = df.copy()
    valid = valid_eeg_sample_mask(output)
    output["valid_eeg_sample"] = valid
    columns = eeg_columns(output)
    if len(columns) > 0 and not np.all(valid):
        output.loc[~valid, columns] = np.nan
    return output


def common_average_reference(data):
    array = np.asarray(data, dtype=float)
    reference = np.zeros((1, array.shape[1]), dtype=float)
    valid = np.any(np.isfinite(array), axis=0)
    if np.any(valid):
        reference[:, valid] = np.nanmean(array[:, valid], axis=0, keepdims=True)
    return array - reference


def interpolate_missing_samples(data):
    output = np.asarray(data, dtype=float).copy()
    if output.ndim != 2:
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.arange(output.shape[1])
    for index in range(output.shape[0]):
        row = output[index]
        finite = np.isfinite(row)
        if np.all(finite):
            continue
        if np.sum(finite) == 0:
            output[index] = 0.0
        elif np.sum(finite) == 1:
            output[index] = row[finite][0]
        else:
            output[index] = np.interp(x, x[finite], row[finite])
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def bandpass(data, fs, low, high, order=4):
    nyquist = fs / 2.0
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch50(data, fs):
    b, a = iirnotch(50.0, 30.0, fs)
    return filtfilt(b, a, data, axis=-1)


def safe_filter(data, fs, low=1.0, high=45.0, notch=True):
    filtered = interpolate_missing_samples(data)
    if notch and fs > 120:
        filtered = notch50(filtered, fs)
    high_value = min(high, fs / 2.0 - 1.0)
    if high_value > low and filtered.shape[-1] >= 64:
        filtered = bandpass(filtered, fs, low, high_value)
    return filtered


def bandpower(data, fs, fmin, fmax):
    nperseg = min(data.shape[-1], int(fs * 2))
    if nperseg < 8:
        return np.zeros(data.shape[0], dtype=float)
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, axis=-1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(data.shape[0], dtype=float)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=-1)


def make_windows(df, fs, window_s=2.0, step_s=0.25):
    n = len(df)
    window_n = int(round(window_s * fs))
    step_n = int(round(step_s * fs))
    starts = list(range(0, max(1, n - window_n + 1), max(1, step_n)))
    return [(start, start + window_n) for start in starts if start + window_n <= n]


def majority_label(values):
    series = pd.Series(values).dropna()
    if len(series) == 0:
        return "unknown"
    return str(series.value_counts().idxmax())


def assign_image_labels(eeg_df, events_df):
    df = eeg_df.copy()
    df["state_label"] = "unlabeled"
    df["state_phase"] = "unlabeled"
    df["emotion_label"] = "unknown"
    df["emotion_index"] = -1
    df["trial_index"] = -1
    df["image_id"] = "unknown"
    df["image_path"] = ""
    df["state_elapsed_s"] = np.nan
    if events_df is None or len(events_df) == 0 or "sample_time_est_s" not in df.columns:
        return df
    if "event_type" not in events_df.columns or "pc_time_perf_counter_s" not in events_df.columns:
        return df
    starts = events_df[events_df["event_type"] == "state_start"].copy()
    if len(starts) == 0:
        return df
    starts = starts.sort_values("pc_time_perf_counter_s").reset_index(drop=True)
    defaults = {
        "state_label": "unlabeled",
        "state_phase": "unlabeled",
        "emotion_label": "unknown",
        "emotion_index": -1,
        "trial_index": -1,
        "image_id": "unknown",
        "image_path": "",
    }
    for column, default_value in defaults.items():
        if column not in starts.columns:
            starts[column] = default_value
    starts["emotion_index"] = pd.to_numeric(starts["emotion_index"], errors="coerce").fillna(-1).astype(int)
    starts["trial_index"] = pd.to_numeric(starts["trial_index"], errors="coerce").fillna(-1).astype(int)
    start_times = starts["pc_time_perf_counter_s"].to_numpy(dtype=float)
    sample_times = df["sample_time_est_s"].to_numpy(dtype=float)
    indices = np.searchsorted(start_times, sample_times, side="right") - 1
    valid = indices >= 0
    if not np.any(valid):
        return df
    valid_indices = indices[valid]
    for column in ["state_label", "state_phase", "emotion_label", "image_id", "image_path"]:
        values = starts[column].astype(str).to_numpy()
        assigned = np.full(len(df), df[column].iloc[0], dtype=object)
        assigned[valid] = values[valid_indices]
        df[column] = assigned
    for column in ["emotion_index", "trial_index"]:
        values = starts[column].to_numpy(dtype=int)
        assigned = np.full(len(df), -1, dtype=int)
        assigned[valid] = values[valid_indices]
        df[column] = assigned
    elapsed = np.full(len(df), np.nan, dtype=float)
    elapsed[valid] = sample_times[valid] - start_times[valid_indices]
    df["state_elapsed_s"] = elapsed
    return df


def save_labeled_image_samples(session_dir):
    path = Path(session_dir)
    eeg_path = path / "eeg_samples.csv"
    events_path = path / "events.csv"
    df = pd.read_csv(eeg_path)
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    labeled = assign_image_labels(df, events_df)
    labeled = mark_invalid_eeg_samples(labeled)
    labeled.to_csv(path / "eeg_samples_labeled.csv", index=False)
    return labeled


def load_image_session(session_dir):
    path = Path(session_dir)
    labeled_path = path / "eeg_samples_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
    else:
        df = save_labeled_image_samples(path)
    events_path = path / "events.csv"
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    metadata = read_metadata(path)
    df = mark_invalid_eeg_samples(df)
    return df, events_df, metadata


def find_image_session_dirs(data_root=DATA_ROOT):
    root = Path(data_root)
    if not root.exists():
        return []
    sessions = []
    for session_dir in sorted(root.glob("session_*")):
        metadata = read_metadata(session_dir)
        if (session_dir / "eeg_samples.csv").exists() and (session_dir / "events.csv").exists() and metadata.get("protocol_version") == PROTOCOL_VERSION:
            sessions.append(session_dir)
    return sessions


def finite_feature_dict(row):
    output = {}
    for key, value in row.items():
        number = float(value)
        if not np.isfinite(number):
            number = 0.0
        output[key] = number
    return output


def group_band_logpowers(segment_df, fs, group_columns):
    if len(group_columns) == 0:
        return {}
    data = get_eeg_array(segment_df, group_columns)
    data = common_average_reference(data)
    data = safe_filter(data, fs, 1.0, min(45.0, fs / 2.0 - 1.0))
    output = {}
    for fmin, fmax, band_name in FEATURE_BANDS:
        if fmax >= fs / 2.0:
            continue
        output[band_name] = np.log10(bandpower(data, fs, fmin, fmax) + 1e-18)
    output["rms"] = np.sqrt(np.mean(data * data, axis=1))
    output["sample_std"] = np.std(data, axis=1)
    return output


def compute_emotion_feature_dict(segment_df, baseline_df, fs):
    row = {}
    group_band_values = {}
    baseline_band_values = {}
    for group_name, names in FEATURE_GROUPS.items():
        columns = select_channel_columns(segment_df, names, fallback_count=0)
        values = group_band_logpowers(segment_df, fs, columns)
        base_values = group_band_logpowers(baseline_df, fs, columns) if baseline_df is not None and len(baseline_df) >= int(fs * 0.5) else {}
        group_band_values[group_name] = values
        baseline_band_values[group_name] = base_values
        for band_name, powers in values.items():
            row[f"{group_name}_{band_name}_mean"] = float(np.mean(powers))
            row[f"{group_name}_{band_name}_std"] = float(np.std(powers))
            row[f"{group_name}_{band_name}_p25"] = float(np.percentile(powers, 25))
            row[f"{group_name}_{band_name}_p75"] = float(np.percentile(powers, 75))
            if band_name in base_values:
                row[f"{group_name}_{band_name}_baseline_delta"] = float(np.mean(powers) - np.mean(base_values[band_name]))
    columns = select_channel_columns(segment_df, DEFAULT_MAXI_32_NAMES, fallback_count=0)
    if len(columns) > 0:
        data = get_eeg_array(segment_df, columns)
        data = common_average_reference(data)
        data = safe_filter(data, fs, 1.0, min(45.0, fs / 2.0 - 1.0))
        names = [clean_channel_name(column) for column in columns]
        channel_powers = {}
        for fmin, fmax, band_name in FEATURE_BANDS:
            if fmax >= fs / 2.0:
                continue
            powers = np.log10(bandpower(data, fs, fmin, fmax) + 1e-18)
            channel_powers[band_name] = {name: float(power) for name, power in zip(names, powers)}
            for name, power in zip(names, powers):
                row[f"channel_{name}_{band_name}"] = float(power)
        for left_name, right_name in ASYMMETRY_PAIRS:
            for _, _, band_name in FEATURE_BANDS:
                values = channel_powers.get(band_name, {})
                if left_name in values and right_name in values:
                    row[f"asym_{left_name}_{right_name}_{band_name}"] = values[left_name] - values[right_name]
        if "alpha" in channel_powers and "low_gamma" in channel_powers:
            alpha = channel_powers["alpha"]
            gamma = channel_powers["low_gamma"]
            shared = [name for name in names if name in alpha and name in gamma]
            if len(shared) > 0:
                row["global_low_gamma_minus_alpha"] = float(np.mean([gamma[name] - alpha[name] for name in shared]))
    for band_name in ["theta", "alpha", "low_beta", "mid_beta", "low_gamma"]:
        left = group_band_values.get("left_frontal", {}).get(band_name)
        right = group_band_values.get("right_frontal", {}).get(band_name)
        if left is not None and right is not None:
            row[f"frontal_left_minus_right_{band_name}"] = float(np.mean(left) - np.mean(right))
        posterior = group_band_values.get("occipital", {}).get(band_name)
        frontal = group_band_values.get("frontal", {}).get(band_name)
        if posterior is not None and frontal is not None:
            row[f"frontal_minus_occipital_{band_name}"] = float(np.mean(frontal) - np.mean(posterior))
    return finite_feature_dict(row)


def tensor_from_segment(segment_df, fs, channel_columns, output_sample_count):
    data = get_eeg_array(segment_df, channel_columns)
    data = common_average_reference(data)
    data = safe_filter(data, fs, 1.0, min(45.0, fs / 2.0 - 1.0))
    data = data - np.mean(data, axis=1, keepdims=True)
    if output_sample_count is not None and data.shape[1] != output_sample_count:
        data = resample(data, int(output_sample_count), axis=1)
    return data.astype(np.float32)


def build_session_training_rows(session_dir, window_seconds, step_seconds, onset_skip_seconds, min_valid_fraction, min_label_fraction, tensor_sample_count):
    df, events_df, metadata = load_image_session(session_dir)
    fs = sampling_frequency_from_df(df, metadata)
    valid_mask = valid_eeg_sample_mask(df)
    channel_columns = ordered_eeg_columns(df)
    feature_rows = []
    label_rows = []
    group_rows = []
    metadata_rows = []
    tensor_rows = []
    image_trials = sorted(pd.to_numeric(df.loc[df["state_phase"].astype(str) == "image", "trial_index"], errors="coerce").dropna().astype(int).unique().tolist())
    for trial_index in image_trials:
        trial_df = df[df["trial_index"].astype(int) == int(trial_index)]
        image_df = trial_df[trial_df["state_phase"].astype(str) == "image"].copy()
        fixation_df = trial_df[trial_df["state_phase"].astype(str) == "fixation"].copy()
        if len(image_df) == 0:
            continue
        if "state_elapsed_s" in image_df.columns:
            elapsed = pd.to_numeric(image_df["state_elapsed_s"], errors="coerce").to_numpy(dtype=float)
            image_df = image_df[elapsed >= onset_skip_seconds]
        if len(image_df) < int(window_seconds * fs):
            continue
        emotion = majority_label(image_df["emotion_label"])
        if emotion not in EMOTION_LABELS:
            continue
        windows = make_windows(image_df, fs, window_seconds, step_seconds)
        image_original_indices = image_df.index.to_numpy(dtype=int)
        for start, end in windows:
            original_start = int(image_original_indices[start])
            original_end = int(image_original_indices[end - 1]) + 1
            valid_fraction = float(np.mean(valid_mask[original_start:original_end])) if original_end > original_start else 0.0
            if valid_fraction < min_valid_fraction:
                continue
            window_df = image_df.iloc[start:end]
            phase_fraction = float(np.mean(window_df["state_phase"].astype(str).to_numpy() == "image"))
            emotion_fraction = float(np.mean(window_df["emotion_label"].astype(str).to_numpy() == emotion))
            if phase_fraction < min_label_fraction or emotion_fraction < min_label_fraction:
                continue
            features = compute_emotion_feature_dict(window_df, fixation_df, fs)
            feature_rows.append(features)
            label_rows.append(emotion)
            image_id = majority_label(window_df["image_id"])
            session_path = Path(session_dir)
            group = f"{session_path.name}|trial_{int(trial_index):03d}|{image_id}"
            group_rows.append(group)
            metadata_rows.append({
                "session_dir": str(session_path),
                "session_id": session_path.name,
                "trial_index": int(trial_index),
                "image_id": image_id,
                "image_path": majority_label(window_df["image_path"]),
                "emotion_label": emotion,
                "emotion_index": int(EMOTION_TO_INDEX[emotion]),
                "window_start_sample": original_start,
                "window_end_sample": original_end,
                "time_s": float(window_df["t_from_stream_start_s"].mean()) if "t_from_stream_start_s" in window_df.columns else float(original_start / fs),
                "valid_eeg_fraction": valid_fraction,
            })
            tensor_rows.append(tensor_from_segment(window_df, fs, channel_columns, tensor_sample_count))
    return feature_rows, label_rows, group_rows, metadata_rows, tensor_rows, [clean_channel_name(column) for column in channel_columns]


def build_training_table(session_dirs, window_seconds, step_seconds, onset_skip_seconds, min_valid_fraction, min_label_fraction, tensor_sample_count):
    feature_rows = []
    label_rows = []
    group_rows = []
    metadata_rows = []
    tensor_rows = []
    channel_names = []
    for session_dir in session_dirs:
        session_features, session_labels, session_groups, session_metadata, session_tensors, session_channels = build_session_training_rows(
            session_dir,
            window_seconds,
            step_seconds,
            onset_skip_seconds,
            min_valid_fraction,
            min_label_fraction,
            tensor_sample_count,
        )
        if len(channel_names) == 0:
            channel_names = session_channels
        feature_rows.extend(session_features)
        label_rows.extend(session_labels)
        group_rows.extend(session_groups)
        metadata_rows.extend(session_metadata)
        tensor_rows.extend(session_tensors)
    if len(feature_rows) == 0:
        return pd.DataFrame(), np.asarray([], dtype=object), np.asarray([], dtype=object), pd.DataFrame(), np.zeros((0, 0, 0), dtype=np.float32), channel_names
    feature_frame = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame = feature_frame.reindex(sorted(feature_frame.columns), axis=1)
    labels = np.asarray(label_rows, dtype=object)
    groups = np.asarray(group_rows, dtype=object)
    metadata_frame = pd.DataFrame(metadata_rows)
    tensors = np.stack(tensor_rows, axis=0).astype(np.float32)
    return feature_frame, labels, groups, metadata_frame, tensors, channel_names
