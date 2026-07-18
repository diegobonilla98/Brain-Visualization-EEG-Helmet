import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_common import (
    OCCIPITAL_NAMES,
    FRONTAL_NAMES,
    TEMPORAL_NAMES,
    build_basic_features,
    common_average_reference,
    find_latest_recording,
    get_eeg_array,
    load_labeled,
    make_windows,
    robust_zscore,
    safe_filter,
    bandpower,
    sampling_frequency_from_df,
    select_channel_columns,
    state_labels_for_windows,
    train_ml_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOW_SECONDS = 2.0
STEP_SECONDS = 0.25
CLASS_LABELS = ["closed_eyes", "open_eyes", "strong_blinks"]
BAND_LABELS = ["delta", "theta", "alpha", "beta", "gamma"]


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def compact_label(label):
    value = str(label)
    if value in CLASS_LABELS:
        return value
    return "unknown"


def build_feature_names(occ_cols, frontal_cols, temporal_cols):
    names = []
    for group_name, columns in [
        ("occ", occ_cols),
        ("frontal", frontal_cols),
        ("temporal", temporal_cols),
    ]:
        for band_name in BAND_LABELS:
            names.append(f"{group_name}_{band_name}_logpower")
        names.append(f"{group_name}_channel_std_mean")
        names.append(f"{group_name}_robust_zmax")
    names.append("alpha_signal_score")
    names.append("blink_signal_score")
    return names


def alpha_signal_score(segment_df, occ_cols, fs):
    occ = get_eeg_array(segment_df, occ_cols)
    occ = common_average_reference(occ)
    occ = safe_filter(occ, fs, 1, min(45, fs / 2 - 1))
    p_alpha = np.mean(bandpower(occ, fs, 8, 12))
    p_broad = np.mean(bandpower(occ, fs, 1, min(40, fs / 2 - 1)))
    return float(np.log10((p_alpha + 1e-18) / (p_broad + 1e-18)))


def blink_signal_score(segment_df, frontal_cols, fs):
    frontal = get_eeg_array(segment_df, frontal_cols)
    frontal = common_average_reference(frontal)
    frontal = safe_filter(frontal, fs, 0.5, min(12, fs / 2 - 1), notch=False)
    return float(np.max(np.abs(robust_zscore(frontal, axis=1))))



def feature_vector_from_segment(segment_df, fs, occ_cols, frontal_cols, temporal_cols):
    if len(segment_df) < 4:
        return None
    windows = [(0, len(segment_df))]
    base_features = build_basic_features(
        segment_df,
        fs,
        windows,
        [occ_cols, frontal_cols, temporal_cols],
    )[0]
    alpha_score = alpha_signal_score(segment_df, occ_cols, fs)
    blink_score = blink_signal_score(segment_df, frontal_cols, fs)
    return np.hstack([base_features, [alpha_score, blink_score]])


def feature_frame_from_segment(segment_df, fs, bundle):
    vector = feature_vector_from_segment(
        segment_df,
        fs,
        bundle["occ_cols"],
        bundle["frontal_cols"],
        bundle["temporal_cols"],
    )
    if vector is None:
        return None
    return pd.DataFrame([vector], columns=bundle["feature_names"])


def grouped_feature_names(feature_names):
    groups = []
    current_group = None
    current_names = []
    for name in feature_names:
        group_name = name.split("_", 1)[0]
        if current_group is None:
            current_group = group_name
        if group_name != current_group:
            groups.append((current_group, current_names))
            current_group = group_name
            current_names = []
        current_names.append(name)
    if len(current_names) > 0:
        groups.append((current_group, current_names))
    return groups


def build_training_table(recording_dir):
    recording_path = resolve_project_path(recording_dir)
    df, events_df, metadata = load_labeled(recording_path)
    fs = sampling_frequency_from_df(df, metadata)
    windows = make_windows(df, fs, WINDOW_SECONDS, STEP_SECONDS)
    labels = [compact_label(label) for label in state_labels_for_windows(df, windows)]
    occ_cols = select_channel_columns(df, OCCIPITAL_NAMES, fallback_count=3)
    frontal_cols = select_channel_columns(df, FRONTAL_NAMES, fallback_count=4)
    temporal_cols = select_channel_columns(df, TEMPORAL_NAMES, fallback_count=4)
    feature_names = build_feature_names(occ_cols, frontal_cols, temporal_cols)
    rows = []
    for window_index, (start, end) in enumerate(windows):
        segment_df = df.iloc[start:end].copy()
        vector = feature_vector_from_segment(segment_df, fs, occ_cols, frontal_cols, temporal_cols)
        if vector is None:
            continue
        row = {
            "window_start_sample": start,
            "window_end_sample": end,
            "label": labels[window_index],
        }
        for name, value in zip(feature_names, vector):
            row[name] = value
        rows.append(row)
    feature_frame = pd.DataFrame(rows)
    return feature_frame, metadata, fs, occ_cols, frontal_cols, temporal_cols, feature_names


def train_eyes_open_closed_model(recording_dir):
    feature_frame, metadata, fs, occ_cols, frontal_cols, temporal_cols, feature_names = build_training_table(recording_dir)
    labels = feature_frame["label"].to_numpy(dtype=object)
    features = feature_frame[feature_names].to_numpy(dtype=float)
    model, _, _, metrics = train_ml_classifier(features, labels)
    if metrics is None:
        raise RuntimeError("Not enough labeled windows to train the eyes model.")
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "class_labels": list(metrics["labels"]),
        "occ_cols": occ_cols,
        "frontal_cols": frontal_cols,
        "temporal_cols": temporal_cols,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "training_recording_dir": str(resolve_project_path(recording_dir)),
        "sample_frequency_hz": fs,
    }
    training_windows = feature_frame.copy()
    return bundle, metrics, training_windows, metadata


def probabilities_from_model(model, features, class_names):
    raw = model.predict_proba(features)
    output = np.zeros((len(features), len(class_names)), dtype=float)
    classes = list(model.classes_)
    for class_index, class_name in enumerate(class_names):
        if class_name in classes:
            output[:, class_index] = raw[:, classes.index(class_name)]
    return output


def predict_bundle(bundle, feature_frame):
    frame = feature_frame.reindex(columns=bundle["feature_names"], fill_value=0.0)
    features = frame.to_numpy(dtype=float)
    predictions = bundle["model"].predict(features)
    probabilities = probabilities_from_model(bundle["model"], features, bundle["class_labels"])
    result = pd.DataFrame({"prediction": predictions})
    probability_frame = pd.DataFrame(
        probabilities,
        columns=[f"probability_{label}" for label in bundle["class_labels"]],
    )
    return pd.concat([result, probability_frame], axis=1)


def save_model_bundle(bundle, model_output_dir, metrics, training_windows):
    output_dir = resolve_project_path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"eyes_model_{stamp}.pkl"
    metrics_path = output_dir / f"eyes_metrics_{stamp}.json"
    windows_path = output_dir / f"eyes_training_windows_{stamp}.csv"
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    training_windows.to_csv(windows_path, index=False)
    (output_dir / "latest_model_path.txt").write_text(str(model_path), encoding="utf-8")
    return model_path, metrics_path, windows_path


def load_model_bundle(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


def latest_model_path(model_dir):
    output_dir = resolve_project_path(model_dir)
    pointer = output_dir / "latest_model_path.txt"
    if pointer.exists():
        path = Path(pointer.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    candidates = sorted(output_dir.glob("eyes_model_*.pkl"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No trained eyes model found in {output_dir}")
    return candidates[-1]
