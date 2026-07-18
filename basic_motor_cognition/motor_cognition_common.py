import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = "motor_imagery_erd_v3"
CLASS_LABELS = ["rest", "left_hand", "right_hand", "both_feet", "both_hands"]
ACTIVE_CLASS_LABELS = ["left_hand", "right_hand", "both_feet", "both_hands"]
TARGET_COLUMNS = ["left_arm", "right_arm", "left_leg", "right_leg"]
TARGET_DISPLAY_NAMES = {
    "left_arm": "left hand",
    "right_arm": "right hand",
    "left_leg": "left foot",
    "right_leg": "right foot",
}
CONDITIONS = [
    {
        "condition_name": "rest",
        "screen_text": "REST",
        "instruction": "Stay relaxed. Keep attention steady. Do not imagine movement.",
        "class_label": "rest",
        "active_intent": 0,
        "left_arm": 0,
        "right_arm": 0,
        "left_leg": 0,
        "right_leg": 0,
    },
    {
        "condition_name": "left_hand",
        "screen_text": "LEFT HAND",
        "instruction": "Imagine repeatedly opening and closing the left hand from inside the body.",
        "class_label": "left_hand",
        "active_intent": 1,
        "left_arm": 1,
        "right_arm": 0,
        "left_leg": 0,
        "right_leg": 0,
    },
    {
        "condition_name": "right_hand",
        "screen_text": "RIGHT HAND",
        "instruction": "Imagine repeatedly opening and closing the right hand from inside the body.",
        "class_label": "right_hand",
        "active_intent": 1,
        "left_arm": 0,
        "right_arm": 1,
        "left_leg": 0,
        "right_leg": 0,
    },
    {
        "condition_name": "both_feet",
        "screen_text": "BOTH FEET",
        "instruction": "Imagine both feet pressing pedals or starting to walk. Keep legs relaxed.",
        "class_label": "both_feet",
        "active_intent": 1,
        "left_arm": 0,
        "right_arm": 0,
        "left_leg": 1,
        "right_leg": 1,
    },
    {
        "condition_name": "both_hands",
        "screen_text": "BOTH HANDS",
        "instruction": "Imagine both hands opening and closing together without tensing.",
        "class_label": "both_hands",
        "active_intent": 1,
        "left_arm": 1,
        "right_arm": 1,
        "left_leg": 0,
        "right_leg": 0,
    },
]
FEATURE_BANDS = [
    (8.0, 12.0, "mu"),
    (12.0, 16.0, "low_beta"),
    (16.0, 24.0, "mid_beta"),
    (24.0, 32.0, "high_beta"),
]
CHANNEL_GROUPS = {
    "left_motor": ["FC5", "FC1", "C3", "CP5", "CP1"],
    "right_motor": ["FC2", "FC6", "C4", "CP2", "CP6"],
    "midline_motor": ["FC1", "FC2", "Cz", "CP1", "CP2", "Pz"],
    "all_motor": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"],
    "artifact_control": ["Fp1", "Fp2", "F7", "F8", "T7", "T8"],
}
PAIR_CHANNELS = [("C3", "C4"), ("FC1", "FC2"), ("CP1", "CP2"), ("FC5", "FC6"), ("CP5", "CP6")]


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


def condition_targets(condition):
    return {target: int(condition[target]) for target in TARGET_COLUMNS}


def target_mask(targets):
    mask = 0
    for index, target in enumerate(TARGET_COLUMNS):
        if int(targets.get(target, 0)) == 1:
            mask += 1 << index
    return mask


def eeg_columns(df):
    return [column for column in df.columns if column.endswith("_uV")]


def clean_channel_name(column):
    return column.replace("_uV", "")


def select_channel_columns(df, wanted_names, fallback_count=0):
    columns = eeg_columns(df)
    selected = []
    lower_map = {clean_channel_name(column).lower(): column for column in columns}
    for name in wanted_names:
        column = lower_map.get(str(name).lower())
        if column is not None and column not in selected:
            selected.append(column)
    if len(selected) == 0 and fallback_count > 0:
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


def get_eeg_array(df, columns):
    return df[columns].to_numpy(dtype=float).T


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


def safe_filter(data, fs, low=4.0, high=40.0, notch=True):
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


def make_windows(df, fs, window_s=3.0, step_s=0.25):
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


def assign_motor_labels(eeg_df, events_df):
    df = eeg_df.copy()
    default_columns = {
        "state_label": "unlabeled",
        "state_phase": "unlabeled",
        "condition_name": "unknown",
        "class_label": "unknown",
        "active_intent": 0,
        "trial_index": -1,
        "session_index": -1,
        "trial_uid": "unknown",
        "state_elapsed_s": np.nan,
    }
    for column, value in default_columns.items():
        df[column] = value
    for target in TARGET_COLUMNS:
        df[target] = 0
    if events_df is None or len(events_df) == 0 or "sample_time_est_s" not in df.columns:
        return df
    starts = events_df[events_df["event_type"] == "state_start"].copy()
    if len(starts) == 0:
        return df
    starts = starts.sort_values("pc_time_perf_counter_s").reset_index(drop=True)
    for column, value in default_columns.items():
        if column not in starts.columns:
            starts[column] = value
    for target in TARGET_COLUMNS:
        if target not in starts.columns:
            starts[target] = 0
        starts[target] = pd.to_numeric(starts[target], errors="coerce").fillna(0).astype(int)
    starts["active_intent"] = pd.to_numeric(starts["active_intent"], errors="coerce").fillna(0).astype(int)
    start_times = starts["pc_time_perf_counter_s"].to_numpy(dtype=float)
    sample_times = df["sample_time_est_s"].to_numpy(dtype=float)
    indices = np.searchsorted(start_times, sample_times, side="right") - 1
    valid = indices >= 0
    if not np.any(valid):
        return df
    valid_indices = indices[valid]
    for column in ["state_label", "state_phase", "condition_name", "class_label"]:
        values = starts[column].astype(str).to_numpy()
        assigned = np.full(len(df), str(default_columns[column]), dtype=object)
        assigned[valid] = values[valid_indices]
        df[column] = assigned
    for column in ["trial_index", "session_index", "active_intent"]:
        values = pd.to_numeric(starts[column], errors="coerce").fillna(default_columns[column]).astype(int).to_numpy()
        assigned = np.full(len(df), int(default_columns[column]), dtype=int)
        assigned[valid] = values[valid_indices]
        df[column] = assigned
    for target in TARGET_COLUMNS:
        values = starts[target].to_numpy(dtype=int)
        assigned = np.zeros(len(df), dtype=int)
        assigned[valid] = values[valid_indices]
        df[target] = assigned
    elapsed = np.full(len(df), np.nan, dtype=float)
    elapsed[valid] = sample_times[valid] - start_times[valid_indices]
    df["state_elapsed_s"] = elapsed
    df["trial_uid"] = df["session_index"].astype(str) + "_" + df["trial_index"].astype(str)
    return df


def save_labeled_motor_samples(session_dir):
    path = Path(session_dir)
    eeg_path = path / "eeg_samples.csv"
    events_path = path / "events.csv"
    df = pd.read_csv(eeg_path)
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    labeled = assign_motor_labels(df, events_df)
    labeled = mark_invalid_eeg_samples(labeled)
    labeled.to_csv(path / "eeg_samples_labeled.csv", index=False)
    labeled.to_csv(path / "eeg_samples_motor_labeled.csv", index=False)
    return labeled


def load_motor_session(session_dir):
    path = Path(session_dir)
    labeled_path = path / "eeg_samples_motor_labeled.csv"
    if labeled_path.exists():
        df = pd.read_csv(labeled_path)
    else:
        df = save_labeled_motor_samples(path)
    events_path = path / "events.csv"
    events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    metadata = read_metadata(path)
    df = mark_invalid_eeg_samples(df)
    return df, events_df, metadata


def find_motor_session_dirs(data_root):
    root = resolve_project_path(data_root)
    sessions = []
    for session_dir in sorted(root.glob("session_*")):
        metadata = read_metadata(session_dir)
        if (session_dir / "eeg_samples.csv").exists() and (session_dir / "events.csv").exists() and metadata.get("protocol_version") == PROTOCOL_VERSION:
            sessions.append(session_dir)
    return sessions


def feature_dict_from_segment(segment_df, fs):
    row = {}
    channel_band_values = {}
    for group_name, names in CHANNEL_GROUPS.items():
        columns = select_channel_columns(segment_df, names)
        if len(columns) == 0:
            continue
        data = get_eeg_array(segment_df, columns)
        data = common_average_reference(data)
        data = safe_filter(data, fs, 4.0, min(40.0, fs / 2.0 - 1.0))
        for fmin, fmax, band_name in FEATURE_BANDS:
            powers = np.log10(bandpower(data, fs, fmin, fmax) + 1e-18)
            row[f"{group_name}_{band_name}_mean"] = float(np.mean(powers))
            row[f"{group_name}_{band_name}_std"] = float(np.std(powers))
            row[f"{group_name}_{band_name}_p25"] = float(np.percentile(powers, 25))
            row[f"{group_name}_{band_name}_p75"] = float(np.percentile(powers, 75))
            if group_name == "all_motor":
                for column, value in zip(columns, powers):
                    channel_band_values[(clean_channel_name(column), band_name)] = float(value)
                    row[f"{clean_channel_name(column)}_{band_name}"] = float(value)
    for left, right in PAIR_CHANNELS:
        for _, _, band_name in FEATURE_BANDS:
            left_value = channel_band_values.get((left, band_name))
            right_value = channel_band_values.get((right, band_name))
            if left_value is not None and right_value is not None:
                row[f"asym_{left}_{right}_{band_name}"] = left_value - right_value
    row = {key: 0.0 if not np.isfinite(float(value)) else float(value) for key, value in row.items()}
    return row


def feature_frame_from_segment(segment_df, fs, feature_names=None):
    frame = pd.DataFrame([feature_dict_from_segment(segment_df, fs)])
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_names is not None:
        frame = frame.reindex(columns=list(feature_names), fill_value=0.0)
    return frame


def trial_baseline_feature(df, fs, trial_index, baseline_skip_s):
    trial_df = df[(df["trial_index"] == trial_index) & (df["state_phase"] == "baseline")].copy()
    if "state_elapsed_s" in trial_df.columns:
        trial_df = trial_df[pd.to_numeric(trial_df["state_elapsed_s"], errors="coerce") >= baseline_skip_s]
    if len(trial_df) < int(fs):
        return None
    return feature_dict_from_segment(trial_df, fs)


def subtract_baseline(feature_row, baseline_row):
    output = {}
    for key, value in feature_row.items():
        output[f"erd_{key}"] = float(value) - float(baseline_row.get(key, 0.0))
    return output


def build_session_feature_rows(session_dir, window_seconds, step_seconds, min_valid_fraction, min_label_fraction, transition_skip_seconds, baseline_skip_seconds):
    df, events_df, metadata = load_motor_session(session_dir)
    fs = sampling_frequency_from_df(df, metadata)
    windows = make_windows(df, fs, window_seconds, step_seconds)
    valid_mask = valid_eeg_sample_mask(df)
    feature_rows = []
    labels = []
    active_labels = []
    group_rows = []
    metadata_rows = []
    baseline_by_trial = {}
    trial_values = sorted(pd.Series(df["trial_index"]).dropna().astype(int).unique())
    for trial_index in trial_values:
        if trial_index < 0:
            continue
        baseline = trial_baseline_feature(df, fs, trial_index, baseline_skip_seconds)
        if baseline is not None:
            baseline_by_trial[trial_index] = baseline
    for start, end in windows:
        window_df = df.iloc[start:end]
        valid_fraction = float(np.mean(valid_mask[start:end])) if end > start else 0.0
        if valid_fraction < min_valid_fraction:
            continue
        phase = majority_label(window_df["state_phase"]) if "state_phase" in window_df.columns else "unknown"
        if phase != "imagery":
            continue
        phase_fraction = float(np.mean(window_df["state_phase"].astype(str).to_numpy() == phase))
        if phase_fraction < min_label_fraction:
            continue
        elapsed_values = pd.to_numeric(window_df["state_elapsed_s"], errors="coerce").to_numpy(dtype=float)
        if len(elapsed_values) > 0 and np.nanmin(elapsed_values) < transition_skip_seconds:
            continue
        trial_index = int(pd.to_numeric(window_df["trial_index"], errors="coerce").dropna().mode().iloc[0])
        if trial_index not in baseline_by_trial:
            continue
        class_label = majority_label(window_df["class_label"])
        if class_label not in CLASS_LABELS:
            continue
        feature_row = feature_dict_from_segment(window_df, fs)
        erd_row = subtract_baseline(feature_row, baseline_by_trial[trial_index])
        feature_rows.append(erd_row)
        labels.append(class_label)
        active_labels.append(class_label if class_label != "rest" else "rest")
        session_name = Path(session_dir).parent.name
        group_rows.append(f"{session_name}|trial_{trial_index}")
        metadata_rows.append({
            "session_dir": str(Path(session_dir)),
            "session_name": session_name,
            "trial_index": trial_index,
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "time_s": float(window_df["t_from_stream_start_s"].mean()) if "t_from_stream_start_s" in window_df.columns else float(start / fs),
            "valid_eeg_fraction": valid_fraction,
            "condition_name": majority_label(window_df["condition_name"]),
            "class_label": class_label,
            "active_intent": int(class_label != "rest"),
        })
    return feature_rows, labels, active_labels, group_rows, metadata_rows


def build_training_table(session_dirs, window_seconds, step_seconds, min_valid_fraction, min_label_fraction, transition_skip_seconds, baseline_skip_seconds):
    feature_rows = []
    labels = []
    active_labels = []
    group_rows = []
    metadata_rows = []
    multiple_sessions = len(session_dirs) >= 2
    for session_dir in session_dirs:
        session_features, session_labels, session_active_labels, session_groups, session_metadata = build_session_feature_rows(
            session_dir,
            window_seconds,
            step_seconds,
            min_valid_fraction,
            min_label_fraction,
            transition_skip_seconds,
            baseline_skip_seconds,
        )
        if multiple_sessions:
            session_groups = [str(session_dir) for _ in session_groups]
        feature_rows.extend(session_features)
        labels.extend(session_labels)
        active_labels.extend(session_active_labels)
        group_rows.extend(session_groups)
        metadata_rows.extend(session_metadata)
    if len(feature_rows) == 0:
        return pd.DataFrame(), pd.Series(dtype=object), np.asarray([], dtype=object), pd.DataFrame()
    feature_frame = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame = feature_frame.reindex(sorted(feature_frame.columns), axis=1)
    label_series = pd.Series(labels, name="class_label")
    groups = np.asarray(group_rows, dtype=object)
    metadata_frame = pd.DataFrame(metadata_rows)
    return feature_frame, label_series, groups, metadata_frame


def make_intent_estimator(random_state):
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.45, class_weight="balanced", random_state=random_state))
    forest = RandomForestClassifier(n_estimators=350, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", random_state=random_state + 1, n_jobs=-1)
    trees = ExtraTreesClassifier(n_estimators=450, min_samples_leaf=8, max_features="sqrt", class_weight="balanced", random_state=random_state + 2, n_jobs=-1)
    return VotingClassifier(
        estimators=[("logistic", logistic), ("forest", forest), ("trees", trees)],
        voting="soft",
        weights=[1.2, 1.0, 1.0],
    )


def make_active_estimator(random_state):
    svm = make_pipeline(StandardScaler(), SVC(C=0.7, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=random_state))
    forest = RandomForestClassifier(n_estimators=450, min_samples_leaf=6, max_features="sqrt", class_weight="balanced_subsample", random_state=random_state + 1, n_jobs=-1)
    trees = ExtraTreesClassifier(n_estimators=550, min_samples_leaf=6, max_features="sqrt", class_weight="balanced", random_state=random_state + 2, n_jobs=-1)
    return VotingClassifier(
        estimators=[("svm", svm), ("forest", forest), ("trees", trees)],
        voting="soft",
        weights=[1.3, 1.0, 1.0],
    )


def positive_probability(model, features):
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, classes.index(1)]


def best_binary_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.linspace(0.25, 0.75, 101)
    scores = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        scores.append(f1_score(labels, predictions, zero_division=0))
    return float(thresholds[int(np.argmax(scores))])


def choose_splitter(groups, cv_splits):
    unique = np.unique(groups)
    if len(unique) <= 1:
        return None
    if len(unique) <= cv_splits:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=min(cv_splits, len(unique)))


def train_motor_model(feature_frame, labels, groups, random_state=37, cv_splits=5):
    features = feature_frame.to_numpy(dtype=float)
    labels_array = labels.astype(str).to_numpy()
    intent_labels = (labels_array != "rest").astype(int)
    splitter = choose_splitter(groups, cv_splits)
    oof_intent_probability = np.full(len(labels_array), np.nan, dtype=float)
    oof_final_label = np.full(len(labels_array), "unknown", dtype=object)
    fold_summaries = []
    if splitter is not None:
        for fold_index, (train_indices, test_indices) in enumerate(splitter.split(features, labels_array, groups)):
            intent_model = make_intent_estimator(random_state + fold_index * 17)
            intent_model.fit(features[train_indices], intent_labels[train_indices])
            oof_intent_probability[test_indices] = positive_probability(intent_model, features[test_indices])
            active_train = train_indices[labels_array[train_indices] != "rest"]
            if len(np.unique(labels_array[active_train])) >= 2:
                active_model = make_active_estimator(random_state + fold_index * 29)
                active_model.fit(features[active_train], labels_array[active_train])
                active_predictions = active_model.predict(features[test_indices])
            else:
                active_predictions = np.full(len(test_indices), "rest", dtype=object)
            fold_summaries.append({
                "fold": int(fold_index + 1),
                "train_windows": int(len(train_indices)),
                "test_windows": int(len(test_indices)),
                "train_groups": int(len(np.unique(groups[train_indices]))),
                "test_groups": int(len(np.unique(groups[test_indices]))),
            })
            oof_final_label[test_indices] = active_predictions
    intent_threshold = best_binary_threshold(intent_labels[np.isfinite(oof_intent_probability)], oof_intent_probability[np.isfinite(oof_intent_probability)]) if np.any(np.isfinite(oof_intent_probability)) else 0.5
    if np.all(np.isfinite(oof_intent_probability)):
        oof_final_label[oof_intent_probability < intent_threshold] = "rest"
    intent_model = make_intent_estimator(random_state)
    intent_model.fit(features, intent_labels)
    active_indices = np.where(labels_array != "rest")[0]
    active_model = make_active_estimator(random_state + 1)
    active_model.fit(features[active_indices], labels_array[active_indices])
    metrics = {
        "window_count": int(len(labels_array)),
        "feature_count": int(feature_frame.shape[1]),
        "group_count": int(len(np.unique(groups))),
        "label_counts": {label: int(np.sum(labels_array == label)) for label in CLASS_LABELS},
        "intent_threshold": float(intent_threshold),
        "cross_validation": {
            "available": bool(np.all(np.isfinite(oof_intent_probability))),
            "folds": fold_summaries,
        },
    }
    if np.all(np.isfinite(oof_intent_probability)):
        intent_predictions = (oof_intent_probability >= intent_threshold).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(intent_labels, intent_predictions, average="binary", zero_division=0)
        metrics["cross_validation"].update({
            "final_accuracy": float(accuracy_score(labels_array, oof_final_label)),
            "final_balanced_accuracy": float(balanced_accuracy_score(labels_array, oof_final_label)),
            "final_classification_report": classification_report(labels_array, oof_final_label, labels=CLASS_LABELS, output_dict=True, zero_division=0),
            "final_confusion_matrix": confusion_matrix(labels_array, oof_final_label, labels=CLASS_LABELS).tolist(),
            "intent_metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support_positive": int(np.sum(intent_labels == 1)),
                "support_negative": int(np.sum(intent_labels == 0)),
                "roc_auc": float(roc_auc_score(intent_labels, oof_intent_probability)) if len(np.unique(intent_labels)) == 2 else None,
            },
        })
    bundle = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": PROTOCOL_VERSION,
        "class_labels": CLASS_LABELS,
        "active_class_labels": ACTIVE_CLASS_LABELS,
        "feature_names": list(feature_frame.columns),
        "intent_threshold": float(intent_threshold),
        "intent_model": intent_model,
        "active_model": active_model,
        "metrics": metrics,
    }
    return bundle, metrics, oof_intent_probability, oof_final_label


def predict_bundle(bundle, feature_frame):
    features = feature_frame.reindex(columns=bundle["feature_names"], fill_value=0.0).to_numpy(dtype=float)
    intent_probability = positive_probability(bundle["intent_model"], features)
    active_prediction = bundle["active_model"].predict(features)
    final_prediction = active_prediction.astype(object)
    final_prediction[intent_probability < float(bundle["intent_threshold"])] = "rest"
    active_probability_frame = pd.DataFrame(bundle["active_model"].predict_proba(features), columns=[f"active_probability_{label}" for label in bundle["active_model"].classes_])
    result = pd.DataFrame({
        "intent_probability": intent_probability,
        "prediction": final_prediction,
        "active_prediction": active_prediction,
    })
    return pd.concat([result, active_probability_frame], axis=1)


def save_model_bundle(bundle, model_output_dir, metrics, training_windows):
    output_dir = resolve_project_path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"motor_cognition_model_{stamp}.pkl"
    metrics_path = output_dir / f"motor_cognition_metrics_{stamp}.json"
    windows_path = output_dir / f"motor_cognition_training_windows_{stamp}.csv"
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
    candidates = sorted(output_dir.glob("motor_cognition_model_*.pkl"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No trained motor cognition model found in {output_dir}")
    return candidates[-1]
