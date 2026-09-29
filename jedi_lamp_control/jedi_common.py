import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eeg_quality_suite"))

from analysis_common import bandpower, clean_channel_name, common_average_reference, eeg_columns, get_eeg_array, make_windows, majority_label, mark_invalid_eeg_samples, safe_filter, sampling_frequency_from_df, select_channel_columns, valid_eeg_sample_mask


PROTOCOL_VERSION = "jedi_lamp_will_v1"
CLASS_LABELS = ["nothing", "the_will"]
FEATURE_BANDS = [
    (1.0, 4.0, "delta"),
    (4.0, 8.0, "theta"),
    (8.0, 12.0, "alpha"),
    (12.0, 18.0, "low_beta"),
    (18.0, 26.0, "mid_beta"),
    (26.0, 36.0, "high_beta"),
    (36.0, 45.0, "low_gamma"),
]
CHANNEL_GROUPS = {
    "frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6"],
    "central": ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "left_motor": ["FC5", "FC1", "C3", "CP5", "CP1"],
    "right_motor": ["FC2", "FC6", "C4", "CP2", "CP6"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["PO3", "POz", "PO4", "O1", "Oz", "O2"],
    "temporal": ["T7", "T8", "F7", "F8"],
}


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def read_metadata(session_dir):
    path = Path(session_dir) / "metadata.json"
    return read_json(path) if path.exists() else {}


def assign_labels(eeg_df, events_df):
    frame = eeg_df.copy()
    frame["state_label"] = "unlabeled"
    frame["state_phase"] = "unlabeled"
    frame["class_label"] = "unknown"
    frame["block_index"] = -1
    frame["negative_variant"] = ""
    frame["state_elapsed_s"] = np.nan
    frame["block_uid"] = "unknown"
    if len(events_df) == 0 or "sample_time_est_s" not in frame.columns:
        return mark_invalid_eeg_samples(frame)
    starts = events_df[events_df["event_type"].astype(str) == "state_start"].copy()
    if len(starts) == 0:
        return mark_invalid_eeg_samples(frame)
    starts = starts.sort_values("pc_time_perf_counter_s").reset_index(drop=True)
    for column, default in [("state_label", "unlabeled"), ("state_phase", "unlabeled"), ("class_label", "unknown"), ("block_index", -1), ("negative_variant", "")]:
        if column not in starts.columns:
            starts[column] = default
    start_times = starts["pc_time_perf_counter_s"].to_numpy(dtype=float)
    sample_times = frame["sample_time_est_s"].to_numpy(dtype=float)
    indices = np.searchsorted(start_times, sample_times, side="right") - 1
    valid = indices >= 0
    selected = indices[valid]
    for column in ["state_label", "state_phase", "class_label", "negative_variant"]:
        output = np.full(len(frame), "", dtype=object)
        output[valid] = starts[column].astype(str).to_numpy()[selected]
        frame[column] = output
    blocks = np.full(len(frame), -1, dtype=int)
    blocks[valid] = pd.to_numeric(starts["block_index"], errors="coerce").fillna(-1).astype(int).to_numpy()[selected]
    frame["block_index"] = blocks
    elapsed = np.full(len(frame), np.nan, dtype=float)
    elapsed[valid] = sample_times[valid] - start_times[selected]
    frame["state_elapsed_s"] = elapsed
    frame["block_uid"] = "block_" + frame["block_index"].astype(str)
    return mark_invalid_eeg_samples(frame)


def save_labeled_samples(session_dir):
    path = Path(session_dir)
    eeg = pd.read_csv(path / "eeg_samples.csv")
    events = pd.read_csv(path / "events.csv") if (path / "events.csv").exists() else pd.DataFrame()
    labeled = assign_labels(eeg, events)
    labeled.to_csv(path / "eeg_samples_labeled.csv", index=False)
    return labeled


def find_session_dirs(data_root):
    root = resolve_project_path(data_root)
    if not root.exists():
        return []
    output = []
    for path in sorted(root.glob("session_*")):
        if (path / "eeg_samples.csv").exists() and (path / "events.csv").exists() and read_metadata(path).get("protocol_version") == PROTOCOL_VERSION:
            output.append(path)
    return output


def finite_row(row):
    return {name: float(value) if np.isfinite(float(value)) else 0.0 for name, value in row.items()}


def feature_dict_from_segment(segment_df, sample_rate):
    columns = eeg_columns(segment_df)
    data = get_eeg_array(segment_df, columns)
    data = common_average_reference(data)
    filtered = safe_filter(data, sample_rate, 1.0, min(45.0, sample_rate / 2.0 - 1.0))
    row = {}
    channel_names = [clean_channel_name(column) for column in columns]
    channel_bandpower = {}
    for minimum, maximum, band_name in FEATURE_BANDS:
        if maximum >= sample_rate / 2.0:
            continue
        powers = np.log10(bandpower(filtered, sample_rate, minimum, maximum) + 1e-18)
        channel_bandpower[band_name] = powers
        for index, channel_name in enumerate(channel_names):
            row[f"channel_{channel_name}_{band_name}_logpower"] = powers[index]
    for group_name, requested_names in CHANNEL_GROUPS.items():
        group_columns = select_channel_columns(segment_df, requested_names, fallback_count=0)
        indices = [columns.index(column) for column in group_columns if column in columns]
        if not indices:
            continue
        group_data = filtered[indices]
        row[f"{group_name}_rms"] = np.mean(np.sqrt(np.mean(group_data * group_data, axis=1)))
        row[f"{group_name}_std"] = np.mean(np.std(group_data, axis=1))
        row[f"{group_name}_line_length"] = np.mean(np.mean(np.abs(np.diff(group_data, axis=1)), axis=1))
        row[f"{group_name}_peak_to_peak"] = np.mean(np.ptp(group_data, axis=1))
        for band_name, powers in channel_bandpower.items():
            values = powers[indices]
            row[f"{group_name}_{band_name}_mean"] = np.mean(values)
            row[f"{group_name}_{band_name}_std"] = np.std(values)
    for band_name, powers in channel_bandpower.items():
        lookup = {channel_name: powers[index] for index, channel_name in enumerate(channel_names)}
        for left, right, pair_name in [("C3", "C4", "motor"), ("FC1", "FC2", "frontocentral"), ("CP1", "CP2", "centroparietal"), ("F3", "F4", "frontal"), ("P3", "P4", "parietal")]:
            if left in lookup and right in lookup:
                row[f"{pair_name}_{band_name}_left_minus_right"] = lookup[left] - lookup[right]
    row["global_rms"] = np.mean(np.sqrt(np.mean(filtered * filtered, axis=1)))
    row["global_line_length"] = np.mean(np.mean(np.abs(np.diff(filtered, axis=1)), axis=1))
    row["global_peak_to_peak"] = np.mean(np.ptp(filtered, axis=1))
    return finite_row(row)


def feature_frame_from_segment(segment_df, sample_rate, feature_names=None):
    frame = pd.DataFrame([feature_dict_from_segment(segment_df, sample_rate)]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.reindex(columns=feature_names, fill_value=0.0) if feature_names is not None else frame


def build_session_rows(session_dir, window_seconds, step_seconds, minimum_valid_fraction, minimum_label_fraction, onset_skip_seconds):
    path = Path(session_dir)
    labeled_path = path / "eeg_samples_labeled.csv"
    frame = pd.read_csv(labeled_path) if labeled_path.exists() else save_labeled_samples(path)
    metadata = read_metadata(path)
    sample_rate = sampling_frequency_from_df(frame, metadata)
    valid_mask = valid_eeg_sample_mask(frame)
    features = []
    labels = []
    groups = []
    rows = []
    for start, end in make_windows(frame, sample_rate, window_seconds, step_seconds):
        window = frame.iloc[start:end]
        valid_fraction = float(np.mean(valid_mask[start:end]))
        if valid_fraction < minimum_valid_fraction:
            continue
        phase = majority_label(window["state_phase"])
        label = majority_label(window["class_label"])
        phase_fraction = float(np.mean(window["state_phase"].astype(str).to_numpy() == "task"))
        label_fraction = float(np.mean(window["class_label"].astype(str).to_numpy() == label))
        if phase != "task" or label not in CLASS_LABELS or phase_fraction < minimum_label_fraction or label_fraction < minimum_label_fraction:
            continue
        elapsed = pd.to_numeric(window["state_elapsed_s"], errors="coerce").to_numpy(dtype=float)
        if len(elapsed) and np.nanmin(elapsed) < onset_skip_seconds:
            continue
        block_index = int(pd.to_numeric(window["block_index"], errors="coerce").dropna().mode().iloc[0])
        features.append(feature_dict_from_segment(window, sample_rate))
        labels.append(label)
        groups.append(f"{path.name}|block_{block_index}")
        rows.append({
            "session_id": path.name,
            "session_dir": str(path),
            "block_index": block_index,
            "class_label": label,
            "negative_variant": majority_label(window["negative_variant"]),
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "valid_fraction": valid_fraction,
        })
    return features, labels, groups, rows


def build_training_table(session_dirs, window_seconds, step_seconds, minimum_valid_fraction, minimum_label_fraction, onset_skip_seconds):
    feature_rows = []
    labels = []
    groups = []
    metadata_rows = []
    multiple_sessions = len(session_dirs) >= 2
    for session_dir in session_dirs:
        session_features, session_labels, session_groups, session_metadata = build_session_rows(session_dir, window_seconds, step_seconds, minimum_valid_fraction, minimum_label_fraction, onset_skip_seconds)
        if multiple_sessions:
            session_groups = [Path(session_dir).name] * len(session_groups)
        feature_rows.extend(session_features)
        labels.extend(session_labels)
        groups.extend(session_groups)
        metadata_rows.extend(session_metadata)
    frame = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frame):
        frame = frame.reindex(sorted(frame.columns), axis=1)
    return frame, np.asarray(labels, dtype=object), np.asarray(groups, dtype=object), pd.DataFrame(metadata_rows)


def selected_feature_count(feature_count):
    return min(feature_count, max(24, int(round(np.sqrt(feature_count) * 6))))


def candidate_models(random_state, feature_count):
    k = selected_feature_count(feature_count)
    logistic = make_pipeline(SimpleImputer(), StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(C=0.35, class_weight="balanced", max_iter=5000, random_state=random_state))
    elastic = make_pipeline(SimpleImputer(), RobustScaler(), SelectKBest(f_classif, k=k), LogisticRegression(C=0.25, penalty="elasticnet", solver="saga", l1_ratio=0.25, class_weight="balanced", max_iter=6000, random_state=random_state + 1))
    linear_svm = make_pipeline(SimpleImputer(), StandardScaler(), SelectKBest(f_classif, k=k), SVC(C=0.35, kernel="linear", class_weight="balanced", probability=True, random_state=random_state + 2))
    rbf_svm = make_pipeline(SimpleImputer(), RobustScaler(), SelectKBest(f_classif, k=k), SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=random_state + 3))
    extra = ExtraTreesClassifier(n_estimators=650, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=random_state + 4, n_jobs=-1)
    forest = RandomForestClassifier(n_estimators=550, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", random_state=random_state + 5, n_jobs=-1)
    vote = VotingClassifier(estimators=[("logistic", clone(logistic)), ("rbf", clone(rbf_svm)), ("extra", clone(extra))], voting="soft", weights=[1.2, 1.1, 1.0], n_jobs=-1)
    broad_vote = VotingClassifier(estimators=[("logistic", clone(logistic)), ("elastic", clone(elastic)), ("linear", clone(linear_svm)), ("rbf", clone(rbf_svm)), ("extra", clone(extra)), ("forest", clone(forest))], voting="soft", weights=[1.2, 0.9, 0.8, 1.1, 1.0, 0.9], n_jobs=-1)
    return {
        "logistic_selected": logistic,
        "elastic_logistic_selected": elastic,
        "linear_svm_selected": linear_svm,
        "rbf_svm_selected": rbf_svm,
        "extra_trees": extra,
        "random_forest": forest,
        "compact_soft_vote": vote,
        "broad_soft_vote": broad_vote,
    }


def probability_for_will(model, features):
    probabilities = model.predict_proba(features)
    class_index = list(model.classes_).index("the_will")
    return probabilities[:, class_index]


def validation_splitter(labels, groups, random_state, folds):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("Need at least two independent sessions or blocks for grouped validation.")
    session_level = all("|block_" not in str(group) for group in unique_groups)
    if session_level and len(unique_groups) <= folds:
        return LeaveOneGroupOut()
    group_labels = {group: majority_label(labels[groups == group]) for group in unique_groups}
    groups_per_class = [sum(label == class_label for label in group_labels.values()) for class_label in CLASS_LABELS]
    split_count = min(folds, len(unique_groups), min(groups_per_class))
    if split_count < 2:
        raise RuntimeError("Need at least two independent groups for each class.")
    return StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=random_state)


def metric_summary(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=object)
    true_will = labels == "the_will"
    predictions = np.where(probabilities >= threshold, "the_will", "nothing")
    matrix = confusion_matrix(labels, predictions, labels=CLASS_LABELS)
    false_positive_rate = float(matrix[0, 1] / max(1, matrix[0].sum()))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "will_precision": float(precision_score(true_will, probabilities >= threshold, zero_division=0)),
        "will_recall": float(recall_score(true_will, probabilities >= threshold, zero_division=0)),
        "will_average_precision": float(average_precision_score(true_will, probabilities)),
        "roc_auc": float(roc_auc_score(true_will, probabilities)),
        "false_will_rate": false_positive_rate,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(labels, predictions, labels=CLASS_LABELS, output_dict=True, zero_division=0),
    }


def conservative_threshold(labels, probabilities, maximum_false_will_rate=0.10):
    candidates = np.linspace(0.50, 0.90, 81)
    rows = []
    for threshold in candidates:
        metrics = metric_summary(labels, probabilities, float(threshold))
        rows.append((float(threshold), metrics))
    acceptable = [row for row in rows if row[1]["false_will_rate"] <= maximum_false_will_rate]
    pool = acceptable if acceptable else rows
    pool.sort(key=lambda row: (row[1]["macro_f1"], row[1]["balanced_accuracy"], -row[1]["false_will_rate"]), reverse=True)
    return pool[0][0], pool[0][1]


def train_model_search(feature_frame, labels, groups, random_state, folds):
    features = feature_frame.to_numpy(dtype=float)
    splitter = validation_splitter(labels, groups, random_state, folds)
    candidates = candidate_models(random_state, feature_frame.shape[1])
    results = []
    best = None
    for candidate_index, (name, estimator) in enumerate(candidates.items()):
        oof = np.full(len(labels), np.nan, dtype=float)
        fold_rows = []
        for fold_index, (train_indices, valid_indices) in enumerate(splitter.split(features, labels, groups)):
            if len(np.unique(labels[train_indices])) < 2 or len(np.unique(labels[valid_indices])) < 2:
                continue
            model = clone(estimator)
            model.fit(features[train_indices], labels[train_indices])
            probabilities = probability_for_will(model, features[valid_indices])
            oof[valid_indices] = probabilities
            fold_metrics = metric_summary(labels[valid_indices], probabilities, 0.5)
            fold_rows.append({"fold": fold_index + 1, "train_windows": len(train_indices), "validation_windows": len(valid_indices), "validation_groups": sorted(np.unique(groups[valid_indices]).tolist()), **fold_metrics})
        valid = np.isfinite(oof)
        if not np.all(valid):
            continue
        threshold, metrics = conservative_threshold(labels, oof)
        selection_score = metrics["macro_f1"] + 0.5 * metrics["balanced_accuracy"] - 0.35 * metrics["false_will_rate"]
        result = {"name": name, "selection_score": selection_score, "activation_threshold": threshold, "metrics": metrics, "folds": fold_rows, "oof_probabilities": oof}
        results.append(result)
        if best is None or result["selection_score"] > best["selection_score"]:
            best = result
    if best is None:
        raise RuntimeError("Grouped validation could not evaluate any model. Collect more complete blocks or sessions.")
    final_model = clone(candidates[best["name"]])
    final_model.fit(features, labels)
    serializable_results = [{key: value for key, value in result.items() if key != "oof_probabilities"} for result in results]
    serializable_results.sort(key=lambda row: row["selection_score"], reverse=True)
    bundle = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": PROTOCOL_VERSION,
        "class_labels": CLASS_LABELS,
        "feature_names": list(feature_frame.columns),
        "model_name": best["name"],
        "model": final_model,
        "activation_threshold": float(best["activation_threshold"]),
        "validation_metrics": best["metrics"],
    }
    return bundle, serializable_results, best["oof_probabilities"]


def predict_will_probability(bundle, feature_frame):
    frame = feature_frame.reindex(columns=bundle["feature_names"], fill_value=0.0)
    return float(probability_for_will(bundle["model"], frame.to_numpy(dtype=float))[0])


def save_model(bundle, output_dir, metrics, training_windows):
    root = resolve_project_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = root / f"jedi_will_model_{stamp}.pkl"
    metrics_path = root / f"jedi_will_metrics_{stamp}.json"
    windows_path = root / f"jedi_will_training_windows_{stamp}.csv"
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    write_json(metrics_path, metrics)
    training_windows.to_csv(windows_path, index=False)
    (root / "latest_model_path.txt").write_text(str(model_path), encoding="utf-8")
    return model_path, metrics_path, windows_path


def latest_model_path(model_dir):
    root = resolve_project_path(model_dir)
    pointer = root / "latest_model_path.txt"
    if pointer.exists():
        path = Path(pointer.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    candidates = sorted(root.glob("jedi_will_model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No trained Jedi will model found under {root}")
    return candidates[-1]


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)
