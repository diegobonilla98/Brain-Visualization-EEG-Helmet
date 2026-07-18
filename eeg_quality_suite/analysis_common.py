import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch, spectrogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OCCIPITAL_NAMES = ["O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz"]
FRONTAL_NAMES = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8", "F3", "F4", "Fz"]
TEMPORAL_NAMES = ["T7", "T8", "FT7", "FT8", "F7", "F8", "TP7", "TP8"]
CENTRAL_NAMES = ["C3", "Cz", "C4", "CP1", "CP2", "FC1", "FC2"]


def find_latest_recording(root, prefix):
    root_path = Path(root)
    candidates = sorted(path for path in root_path.iterdir() if path.is_dir() and (path.name.startswith(f"{prefix}_") or f"_{prefix}" in path.name))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No recording found for {prefix} in {root}")
    return candidates[-1]


def read_metadata(path):
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def eeg_columns(df):
    return [column for column in df.columns if column.endswith("_uV")]


def clean_channel_name(column):
    return column.replace("_uV", "")


def select_channel_columns(df, wanted_names, fallback_count=3):
    columns = eeg_columns(df)
    selected = []
    lower_map = {clean_channel_name(column).lower(): column for column in columns}
    for name in wanted_names:
        column = lower_map.get(name.lower())
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
            return 1.0 / dt
    return 250.0


def get_eeg_array(df, columns):
    return df[columns].to_numpy(dtype=float).T


def common_average_reference(data):
    array = np.asarray(data, dtype=float)
    reference = np.zeros((1, array.shape[1]), dtype=float)
    valid = np.any(np.isfinite(array), axis=0)
    if np.any(valid):
        reference[:, valid] = np.nanmean(array[:, valid], axis=0, keepdims=True)
    return array - reference


def robust_zscore(data, axis=1):
    median = np.nanmedian(data, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(data - median), axis=axis, keepdims=True)
    return (data - median) / (1.4826 * mad + 1e-9)


def bandpass(data, fs, low, high, order=4):
    nyquist = fs / 2.0
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch50(data, fs):
    b, a = iirnotch(50.0, 30.0, fs)
    return filtfilt(b, a, data, axis=-1)


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


def safe_filter(data, fs, low=1.0, high=45.0, notch=True):
    filtered = interpolate_missing_samples(data)
    if notch and fs > 120:
        filtered = notch50(filtered, fs)
    high_value = min(high, fs / 2.0 - 1.0)
    if high_value > low:
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


def narrowband_power(data, fs, target_hz, width=0.5):
    return bandpower(data, fs, max(0.1, target_hz - width), target_hz + width)


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


def state_labels_for_windows(df, windows, column="state_label"):
    if column not in df.columns:
        return ["unknown" for _ in windows]
    labels = []
    for start, end in windows:
        labels.append(majority_label(df[column].iloc[start:end]))
    return labels


def assign_state_labels(eeg_df, events_df):
    df = eeg_df.copy()
    df["state_label"] = "unlabeled"
    if events_df is None or len(events_df) == 0:
        return df
    state_events = events_df[events_df["event_type"].isin(["state_start", "state_end"])]
    if len(state_events) == 0:
        return df
    starts = state_events[state_events["event_type"] == "state_start"].copy()
    starts = starts.sort_values("pc_time_perf_counter_s")
    if len(starts) == 0:
        return df
    times = df["sample_time_est_s"].to_numpy(dtype=float)
    start_times = starts["pc_time_perf_counter_s"].to_numpy(dtype=float)
    labels = starts["state_label"].astype(str).to_numpy()
    indices = np.searchsorted(start_times, times, side="right") - 1
    valid = indices >= 0
    assigned = np.full(len(df), "unlabeled", dtype=object)
    assigned[valid] = labels[indices[valid]]
    df["state_label"] = assigned
    return df


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


def window_valid_fractions(df, windows):
    valid = valid_eeg_sample_mask(df)
    return np.asarray([np.mean(valid[start:end]) for start, end in windows], dtype=float)


def recording_quality_summary(df, windows=None):
    valid = valid_eeg_sample_mask(df)
    summary = {
        "valid_sample_fraction": float(np.mean(valid)) if len(valid) > 0 else 0.0,
        "invalid_sample_count": int(np.sum(~valid)) if len(valid) > 0 else 0,
        "total_sample_count": int(len(valid)),
    }
    if windows is not None and len(windows) > 0:
        fractions = window_valid_fractions(df, windows)
        summary.update({
            "window_valid_fraction_min": float(np.min(fractions)),
            "window_valid_fraction_median": float(np.median(fractions)),
            "window_valid_fraction_max": float(np.max(fractions)),
        })
    return summary


def save_labeled_samples(recording_dir):
    path = Path(recording_dir)
    eeg_path = path / "eeg_samples.csv"
    events_path = path / "events.csv"
    df = pd.read_csv(eeg_path)
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    labeled = assign_state_labels(df, events_df)
    labeled.to_csv(path / "eeg_samples_labeled.csv", index=False)
    return labeled


def load_labeled(recording_dir):
    path = Path(recording_dir)
    labeled_path = path / "eeg_samples_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
    else:
        df = save_labeled_samples(path)
    events_path = path / "events.csv"
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    metadata = read_metadata(path)
    df = mark_invalid_eeg_samples(df)
    return df, events_df, metadata


def normalize_traces(data, mode="per_channel"):
    if mode == "global":
        center = np.nanmedian(data)
        scale = np.nanmedian(np.abs(data - center)) * 1.4826 + 1e-9
        return (data - center) / scale
    return robust_zscore(data, axis=1)


def plot_recording_summary(recording_dir, normalization="per_channel"):
    path = Path(recording_dir)
    df, events_df, metadata = load_labeled(path)
    fs = sampling_frequency_from_df(df, metadata)
    columns = eeg_columns(df)
    data = get_eeg_array(df, columns)
    data_car = common_average_reference(data)
    data_norm = normalize_traces(data_car, normalization)
    t = df["t_from_stream_start_s"].to_numpy(dtype=float) if "t_from_stream_start_s" in df.columns else np.arange(len(df)) / fs
    occ_cols = select_channel_columns(df, OCCIPITAL_NAMES, fallback_count=3)
    occ = common_average_reference(get_eeg_array(df, occ_cols))
    occ_mean = np.mean(safe_filter(occ, fs, 1, min(45, fs / 2 - 1)), axis=0)
    frequencies, psd = welch(occ_mean, fs=fs, nperseg=min(len(occ_mean), int(fs * 4)))
    spec_f, spec_t, spec_s = spectrogram(occ_mean, fs=fs, nperseg=min(len(occ_mean), int(fs * 2)), noverlap=int(fs))
    max_channels = min(12, data_norm.shape[0])
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(3, 1, 1)
    offsets = np.arange(max_channels) * 6.0
    ax1.plot(t, data_norm[:max_channels].T + offsets)
    ax1.set_title("Normalized EEG traces")
    ax1.set_xlabel("seconds")
    ax1.set_yticks(offsets)
    ax1.set_yticklabels([clean_channel_name(col) for col in columns[:max_channels]])
    ax2 = plt.subplot(3, 1, 2)
    ax2.semilogy(frequencies, psd + 1e-18)
    ax2.set_xlim(1, min(60, fs / 2))
    ax2.set_title("Occipital PSD")
    ax2.set_xlabel("Hz")
    ax3 = plt.subplot(3, 1, 3)
    mask = spec_f <= min(60, fs / 2)
    ax3.pcolormesh(spec_t, spec_f[mask], 10 * np.log10(spec_s[mask] + 1e-18), shading="auto")
    ax3.set_title("Occipital spectrogram")
    ax3.set_xlabel("seconds")
    ax3.set_ylabel("Hz")
    plt.tight_layout()
    output = path / "summary.png"
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def build_basic_features(df, fs, windows, channel_group_columns):
    features = []
    for start, end in windows:
        row = []
        for columns in channel_group_columns:
            data = get_eeg_array(df.iloc[start:end], columns)
            data = common_average_reference(data)
            data = safe_filter(data, fs, 1, min(100, fs / 2 - 1))
            bands = [(1, 4), (4, 8), (8, 12), (12, 30), (30, min(95, fs / 2 - 1))]
            for fmin, fmax in bands:
                if fmax > fmin:
                    row.append(float(np.mean(np.log10(bandpower(data, fs, fmin, fmax) + 1e-18))))
            row.append(float(np.mean(np.std(data, axis=1))))
            row.append(float(np.max(np.abs(robust_zscore(data, axis=1)))))
        features.append(row)
    return np.asarray(features, dtype=float)


def train_ml_classifier(features, labels):
    labels = np.asarray(labels, dtype=object)
    valid = labels != "unknown"
    features = features[valid]
    labels = labels[valid]
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) < 8:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return model, None, None, None
    stratify = labels if min(pd.Series(labels).value_counts()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.35, random_state=7, stratify=stratify)
    model = RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(unique)).tolist(),
        "labels": list(unique),
    }
    model.fit(features, labels)
    all_pred = model.predict(features)
    return model, all_pred, labels, metrics


def threshold_from_labels(scores, labels, positive_label):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=object)
    positive = scores[labels == positive_label]
    negative = scores[(labels != positive_label) & (labels != "unknown")]
    if len(positive) > 0 and len(negative) > 0:
        return float((np.median(positive) + np.median(negative)) / 2.0)
    return float(np.median(scores))


def threshold_rule_from_labels(scores, labels, positive_label):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=object)
    positive = scores[labels == positive_label]
    negative = scores[(labels != positive_label) & (labels != "unknown")]
    if len(positive) > 0 and len(negative) > 0:
        positive_median = float(np.median(positive))
        negative_median = float(np.median(negative))
        return float((positive_median + negative_median) / 2.0), bool(positive_median > negative_median)
    return float(np.median(scores)), True


def apply_binary_threshold(scores, threshold, positive_above, positive_label, negative_label):
    predictions = []
    for score in scores:
        is_positive = score > threshold
        if is_positive == positive_above:
            predictions.append(positive_label)
        else:
            predictions.append(negative_label)
    return predictions


def save_predictions(recording_dir, name, result_df, metrics, figure_path=None):
    path = Path(recording_dir)
    out_csv = path / f"{name}_predictions.csv"
    out_json = path / f"{name}_metrics.json"
    result_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return out_csv, out_json, figure_path


def plot_score_predictions(recording_dir, name, result_df, score_column, true_column, pred_column):
    path = Path(recording_dir)
    plt.figure(figsize=(14, 5))
    plt.plot(result_df["time_s"], result_df[score_column])
    plt.xlabel("seconds")
    plt.ylabel(score_column)
    plt.title(name)
    unique_labels = list(pd.Series(result_df[true_column]).dropna().unique())
    for label in unique_labels:
        mask = result_df[true_column] == label
        if np.any(mask):
            xs = result_df.loc[mask, "time_s"]
            if len(xs) > 0:
                plt.scatter(xs, result_df.loc[mask, score_column], s=8, label=f"true {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    output = path / f"{name}_score.png"
    plt.savefig(output, dpi=160)
    plt.close()
    return output
