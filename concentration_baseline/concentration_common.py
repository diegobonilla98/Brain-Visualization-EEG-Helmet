import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eeg_quality_suite"))

from analysis_common import bandpower, clean_channel_name, common_average_reference, eeg_columns, get_eeg_array, make_windows, majority_label, mark_invalid_eeg_samples, safe_filter, sampling_frequency_from_df, select_channel_columns, valid_eeg_sample_mask


PROTOCOL_VERSION = "concentration_baseline_v1"
CLASS_LABELS = ["not_concentrated", "concentrated"]
FEATURE_BANDS = [
    (1.0, 4.0, "delta"),
    (4.0, 8.0, "theta"),
    (8.0, 12.0, "alpha"),
    (12.0, 16.0, "low_beta"),
    (16.0, 24.0, "mid_beta"),
    (24.0, 32.0, "high_beta"),
    (32.0, 45.0, "low_gamma"),
]
CHANNEL_GROUPS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6"],
    "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["PO3", "POz", "PO4", "O1", "Oz", "O2"],
    "temporal": ["T7", "T8", "F7", "F8"],
    "all": [],
}


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_metadata(path):
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def assign_concentration_labels(eeg_df, events_df):
    df = eeg_df.copy()
    defaults = {
        "state_label": "unlabeled",
        "state_phase": "unlabeled",
        "class_label": "unknown",
        "block_index": -1,
        "state_elapsed_s": np.nan,
    }
    for column, value in defaults.items():
        df[column] = value
    df["block_uid"] = "unknown"
    if events_df is None or len(events_df) == 0 or "sample_time_est_s" not in df.columns:
        return df
    if "event_type" not in events_df.columns or "pc_time_perf_counter_s" not in events_df.columns:
        return df
    starts = events_df[events_df["event_type"] == "state_start"].copy()
    if len(starts) == 0:
        return df
    starts = starts.sort_values("pc_time_perf_counter_s").reset_index(drop=True)
    for column, value in defaults.items():
        if column not in starts.columns:
            starts[column] = value
    starts["block_index"] = pd.to_numeric(starts["block_index"], errors="coerce").fillna(-1).astype(int)
    start_times = starts["pc_time_perf_counter_s"].to_numpy(dtype=float)
    sample_times = df["sample_time_est_s"].to_numpy(dtype=float)
    indices = np.searchsorted(start_times, sample_times, side="right") - 1
    valid = indices >= 0
    if not np.any(valid):
        return df
    valid_indices = indices[valid]
    for column in ["state_label", "state_phase", "class_label"]:
        values = starts[column].astype(str).to_numpy()
        assigned = np.full(len(df), str(defaults[column]), dtype=object)
        assigned[valid] = values[valid_indices]
        df[column] = assigned
    values = starts["block_index"].to_numpy(dtype=int)
    assigned_blocks = np.full(len(df), -1, dtype=int)
    assigned_blocks[valid] = values[valid_indices]
    df["block_index"] = assigned_blocks
    elapsed = np.full(len(df), np.nan, dtype=float)
    elapsed[valid] = sample_times[valid] - start_times[valid_indices]
    df["state_elapsed_s"] = elapsed
    df["block_uid"] = "block_" + df["block_index"].astype(str)
    return df


def save_labeled_concentration_samples(session_dir):
    path = Path(session_dir)
    eeg_path = path / "eeg_samples.csv"
    events_path = path / "events.csv"
    df = pd.read_csv(eeg_path)
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    labeled = assign_concentration_labels(df, events_df)
    labeled = mark_invalid_eeg_samples(labeled)
    labeled.to_csv(path / "eeg_samples_labeled.csv", index=False)
    return labeled


def load_concentration_session(session_dir):
    path = Path(session_dir)
    labeled_path = path / "eeg_samples_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
    else:
        df = save_labeled_concentration_samples(path)
    events_path = path / "events.csv"
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    metadata = read_metadata(path)
    df = mark_invalid_eeg_samples(df)
    return df, events_df, metadata


def find_concentration_session_dirs(data_root):
    root = resolve_project_path(data_root)
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


def feature_dict_from_segment(segment_df, fs):
    row = {}
    group_band_means = {}
    for group_name, names in CHANNEL_GROUPS.items():
        if len(names) == 0:
            columns = eeg_columns(segment_df)
        else:
            columns = select_channel_columns(segment_df, names, fallback_count=0)
        if len(columns) == 0:
            continue
        data = get_eeg_array(segment_df, columns)
        data = common_average_reference(data)
        data = safe_filter(data, fs, 1.0, min(45.0, fs / 2.0 - 1.0))
        group_band_means[group_name] = {}
        for fmin, fmax, band_name in FEATURE_BANDS:
            if fmax >= fs / 2.0:
                continue
            powers = np.log10(bandpower(data, fs, fmin, fmax) + 1e-18)
            group_band_means[group_name][band_name] = float(np.mean(powers))
            row[f"{group_name}_{band_name}_mean"] = float(np.mean(powers))
            row[f"{group_name}_{band_name}_std"] = float(np.std(powers))
            row[f"{group_name}_{band_name}_p25"] = float(np.percentile(powers, 25))
            row[f"{group_name}_{band_name}_p75"] = float(np.percentile(powers, 75))
        row[f"{group_name}_rms_mean"] = float(np.mean(np.sqrt(np.mean(data * data, axis=1))))
        row[f"{group_name}_sample_std_mean"] = float(np.mean(np.std(data, axis=1)))
    frontal = group_band_means.get("frontal", {})
    parietal = group_band_means.get("parietal", {})
    occipital = group_band_means.get("occipital", {})
    for band_name in ["theta", "alpha", "low_beta", "mid_beta"]:
        if band_name in frontal and band_name in parietal:
            row[f"frontal_minus_parietal_{band_name}"] = frontal[band_name] - parietal[band_name]
        if band_name in frontal and band_name in occipital:
            row[f"frontal_minus_occipital_{band_name}"] = frontal[band_name] - occipital[band_name]
    if "theta" in frontal and "low_beta" in frontal:
        row["frontal_theta_minus_low_beta"] = frontal["theta"] - frontal["low_beta"]
    if "alpha" in occipital and "theta" in frontal:
        row["frontal_theta_minus_occipital_alpha"] = frontal["theta"] - occipital["alpha"]
    return finite_feature_dict(row)


def feature_frame_from_segment(segment_df, fs, feature_names=None):
    frame = pd.DataFrame([feature_dict_from_segment(segment_df, fs)])
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_names is not None:
        frame = frame.reindex(columns=list(feature_names), fill_value=0.0)
    return frame


def build_session_feature_rows(session_dir, window_seconds, step_seconds, min_valid_fraction, min_label_fraction, onset_skip_seconds):
    df, events_df, metadata = load_concentration_session(session_dir)
    fs = sampling_frequency_from_df(df, metadata)
    windows = make_windows(df, fs, window_seconds, step_seconds)
    valid_mask = valid_eeg_sample_mask(df)
    feature_rows = []
    label_rows = []
    group_rows = []
    metadata_rows = []
    session_name = Path(session_dir).name
    for start, end in windows:
        window_df = df.iloc[start:end]
        valid_fraction = float(np.mean(valid_mask[start:end])) if end > start else 0.0
        if valid_fraction < min_valid_fraction:
            continue
        phase = majority_label(window_df["state_phase"]) if "state_phase" in window_df.columns else "unknown"
        if phase != "task":
            continue
        phase_fraction = float(np.mean(window_df["state_phase"].astype(str).to_numpy() == "task"))
        if phase_fraction < min_label_fraction:
            continue
        class_label = majority_label(window_df["class_label"])
        if class_label not in CLASS_LABELS:
            continue
        label_fraction = float(np.mean(window_df["class_label"].astype(str).to_numpy() == class_label))
        if label_fraction < min_label_fraction:
            continue
        elapsed_values = pd.to_numeric(window_df["state_elapsed_s"], errors="coerce").to_numpy(dtype=float)
        if len(elapsed_values) > 0 and np.nanmin(elapsed_values) < onset_skip_seconds:
            continue
        block_index = int(pd.to_numeric(window_df["block_index"], errors="coerce").dropna().mode().iloc[0])
        feature_rows.append(feature_dict_from_segment(window_df, fs))
        label_rows.append(class_label)
        group_rows.append(f"{session_name}|block_{block_index}")
        metadata_rows.append({
            "session_dir": str(Path(session_dir)),
            "session_id": session_name,
            "block_index": block_index,
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "time_s": float(window_df["t_from_stream_start_s"].mean()) if "t_from_stream_start_s" in window_df.columns else float(start / fs),
            "valid_eeg_fraction": valid_fraction,
            "class_label": class_label,
        })
    return feature_rows, label_rows, group_rows, metadata_rows


def build_training_table(session_dirs, window_seconds, step_seconds, min_valid_fraction, min_label_fraction, onset_skip_seconds):
    feature_rows = []
    label_rows = []
    group_rows = []
    metadata_rows = []
    multiple_sessions = len(session_dirs) >= 2
    for session_dir in session_dirs:
        session_features, session_labels, session_groups, session_metadata = build_session_feature_rows(
            session_dir,
            window_seconds,
            step_seconds,
            min_valid_fraction,
            min_label_fraction,
            onset_skip_seconds,
        )
        if multiple_sessions:
            session_groups = [str(session_dir) for _ in session_groups]
        feature_rows.extend(session_features)
        label_rows.extend(session_labels)
        group_rows.extend(session_groups)
        metadata_rows.extend(session_metadata)
    if len(feature_rows) == 0:
        return pd.DataFrame(), np.asarray([], dtype=object), np.asarray([], dtype=object), pd.DataFrame()
    feature_frame = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame = feature_frame.reindex(sorted(feature_frame.columns), axis=1)
    return feature_frame, np.asarray(label_rows, dtype=object), np.asarray(group_rows, dtype=object), pd.DataFrame(metadata_rows)


def make_estimator(random_state):
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.35, class_weight="balanced", random_state=random_state))
    svm = make_pipeline(StandardScaler(), SVC(C=0.8, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=random_state + 1))
    forest = RandomForestClassifier(n_estimators=450, min_samples_leaf=6, max_features="sqrt", class_weight="balanced_subsample", random_state=random_state + 2, n_jobs=-1)
    trees = ExtraTreesClassifier(n_estimators=550, min_samples_leaf=6, max_features="sqrt", class_weight="balanced", random_state=random_state + 3, n_jobs=-1)
    return VotingClassifier(
        estimators=[("logistic", logistic), ("svm", svm), ("forest", forest), ("trees", trees)],
        voting="soft",
        weights=[1.1, 1.2, 1.0, 1.0],
    )


def choose_splitter(groups, cv_splits):
    unique = np.unique(groups)
    if len(unique) <= 1:
        return None
    if len(unique) <= cv_splits:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=min(cv_splits, len(unique)))


def probabilities_from_model(model, features, class_names):
    raw = model.predict_proba(features)
    output = np.zeros((len(features), len(class_names)), dtype=float)
    classes = list(model.classes_)
    for class_index, class_name in enumerate(class_names):
        if class_name in classes:
            output[:, class_index] = raw[:, classes.index(class_name)]
    return output


def train_concentration_model(feature_frame, labels, groups, random_state=20260621, cv_splits=5):
    features = feature_frame.to_numpy(dtype=float)
    labels = np.asarray(labels, dtype=object)
    class_names = [label for label in CLASS_LABELS if label in set(labels.tolist())]
    splitter = choose_splitter(groups, cv_splits)
    oof_probabilities = np.full((len(labels), len(class_names)), np.nan, dtype=float)
    oof_predictions = np.full(len(labels), "unknown", dtype=object)
    fold_summaries = []
    if splitter is not None:
        for fold_index, (train_indices, valid_indices) in enumerate(splitter.split(features, labels, groups)):
            if len(np.unique(labels[train_indices])) < 2 or len(np.unique(labels[valid_indices])) < 1:
                continue
            model = make_estimator(random_state + fold_index * 31)
            model.fit(features[train_indices], labels[train_indices])
            probabilities = probabilities_from_model(model, features[valid_indices], class_names)
            predictions = np.asarray(class_names, dtype=object)[np.argmax(probabilities, axis=1)]
            oof_probabilities[valid_indices] = probabilities
            oof_predictions[valid_indices] = predictions
            fold_summaries.append({
                "fold": int(fold_index + 1),
                "train_windows": int(len(train_indices)),
                "valid_windows": int(len(valid_indices)),
                "train_groups": int(len(np.unique(groups[train_indices]))),
                "valid_groups": int(len(np.unique(groups[valid_indices]))),
                "balanced_accuracy": float(balanced_accuracy_score(labels[valid_indices], predictions)),
                "macro_f1": float(f1_score(labels[valid_indices], predictions, average="macro", zero_division=0)),
            })
    model = make_estimator(random_state + 1009)
    model.fit(features, labels)
    metrics = {
        "window_count": int(len(labels)),
        "feature_count": int(feature_frame.shape[1]),
        "group_count": int(len(np.unique(groups))),
        "label_counts": {label: int(np.sum(labels == label)) for label in CLASS_LABELS},
        "cross_validation": {
            "available": bool(np.all(np.isfinite(oof_probabilities))),
            "folds": fold_summaries,
        },
    }
    if np.all(np.isfinite(oof_probabilities)):
        metrics["cross_validation"].update({
            "accuracy": float(accuracy_score(labels, oof_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, oof_predictions)),
            "macro_f1": float(f1_score(labels, oof_predictions, average="macro", zero_division=0)),
            "classification_report": classification_report(labels, oof_predictions, labels=CLASS_LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, oof_predictions, labels=CLASS_LABELS).tolist(),
            "classes": CLASS_LABELS,
        })
    bundle = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": PROTOCOL_VERSION,
        "class_labels": CLASS_LABELS,
        "feature_names": list(feature_frame.columns),
        "model": model,
        "metrics": metrics,
    }
    return bundle, metrics, oof_probabilities, oof_predictions


def predict_bundle(bundle, feature_frame):
    frame = feature_frame.reindex(columns=bundle["feature_names"], fill_value=0.0)
    features = frame.to_numpy(dtype=float)
    predictions = bundle["model"].predict(features)
    probabilities = probabilities_from_model(bundle["model"], features, bundle["class_labels"])
    result = pd.DataFrame({"prediction": predictions})
    probability_frame = pd.DataFrame(probabilities, columns=[f"probability_{label}" for label in bundle["class_labels"]])
    return pd.concat([result, probability_frame], axis=1)


def save_model_bundle(bundle, model_output_dir, metrics, training_windows):
    output_dir = resolve_project_path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"concentration_model_{stamp}.pkl"
    metrics_path = output_dir / f"concentration_metrics_{stamp}.json"
    windows_path = output_dir / f"concentration_training_windows_{stamp}.csv"
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    write_json(metrics_path, metrics)
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
    candidates = sorted(output_dir.glob("concentration_model_*.pkl"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No trained concentration model found in {output_dir}")
    return candidates[-1]
