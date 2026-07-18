import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from analysis_common import (
    OCCIPITAL_NAMES,
    FRONTAL_NAMES,
    TEMPORAL_NAMES,
    find_latest_recording,
    load_labeled,
    sampling_frequency_from_df,
    select_channel_columns,
    get_eeg_array,
    common_average_reference,
    safe_filter,
    bandpower,
    make_windows,
    state_labels_for_windows,
    build_basic_features,
    train_ml_classifier,
    threshold_rule_from_labels,
    window_valid_fractions,
    recording_quality_summary,
    robust_zscore,
    save_predictions,
    plot_score_predictions,
)


INPUT_DIR = ""
RECORDINGS_ROOT = "sessions"
WINDOW_SECONDS = 2.0
STEP_SECONDS = 0.25


def compact_label(label):
    value = str(label)
    if value == "closed_eyes":
        return "closed_eyes"
    if value == "open_eyes":
        return "open_eyes"
    if value == "strong_blinks":
        return "strong_blinks"
    return "unknown"


def main():
    recording_dir = Path(INPUT_DIR) if INPUT_DIR else find_latest_recording(RECORDINGS_ROOT, "eyes")
    df, events_df, metadata = load_labeled(recording_dir)
    fs = sampling_frequency_from_df(df, metadata)
    windows = make_windows(df, fs, WINDOW_SECONDS, STEP_SECONDS)
    true_labels = [compact_label(label) for label in state_labels_for_windows(df, windows)]
    occ_cols = select_channel_columns(df, OCCIPITAL_NAMES, fallback_count=3)
    frontal_cols = select_channel_columns(df, FRONTAL_NAMES, fallback_count=4)
    temporal_cols = select_channel_columns(df, TEMPORAL_NAMES, fallback_count=4)
    alpha_scores = []
    blink_scores = []
    times = []
    for start, end in windows:
        occ = get_eeg_array(df.iloc[start:end], occ_cols)
        occ = common_average_reference(occ)
        occ = safe_filter(occ, fs, 1, min(45, fs / 2 - 1))
        p_alpha = np.mean(bandpower(occ, fs, 8, 12))
        p_broad = np.mean(bandpower(occ, fs, 1, min(40, fs / 2 - 1)))
        alpha_scores.append(float(np.log10((p_alpha + 1e-18) / (p_broad + 1e-18))))
        frontal = get_eeg_array(df.iloc[start:end], frontal_cols)
        frontal = common_average_reference(frontal)
        frontal = safe_filter(frontal, fs, 0.5, min(12, fs / 2 - 1), notch=False)
        blink_scores.append(float(np.max(np.abs(robust_zscore(frontal, axis=1)))))
        times.append(float(df["t_from_stream_start_s"].iloc[start:end].mean()))
    valid_fractions = window_valid_fractions(df, windows)
    alpha_threshold, alpha_positive_above = threshold_rule_from_labels(alpha_scores, true_labels, "closed_eyes")
    blink_threshold, blink_positive_above = threshold_rule_from_labels(blink_scores, true_labels, "strong_blinks")
    signal_pred = []
    for alpha_score, blink_score in zip(alpha_scores, blink_scores):
        blink_positive = (blink_score > blink_threshold) == blink_positive_above
        alpha_positive = (alpha_score > alpha_threshold) == alpha_positive_above
        if blink_positive:
            signal_pred.append("strong_blinks")
        elif alpha_positive:
            signal_pred.append("closed_eyes")
        else:
            signal_pred.append("open_eyes")
    valid = np.asarray(true_labels) != "unknown"
    signal_metrics = {}
    if np.any(valid):
        signal_metrics = {
            "accuracy": float(accuracy_score(np.asarray(true_labels)[valid], np.asarray(signal_pred)[valid])),
            "classification_report": classification_report(np.asarray(true_labels)[valid], np.asarray(signal_pred)[valid], output_dict=True, zero_division=0),
            "alpha_threshold": alpha_threshold,
            "alpha_positive_above_threshold": alpha_positive_above,
            "blink_threshold": blink_threshold,
            "blink_positive_above_threshold": blink_positive_above,
            "occipital_columns": occ_cols,
            "frontal_columns": frontal_cols,
        }
    base_features = build_basic_features(df, fs, windows, [occ_cols, frontal_cols, temporal_cols])
    added = np.asarray([alpha_scores, blink_scores], dtype=float).T
    features = np.hstack([base_features, added])
    ml_model, ml_pred_valid, ml_labels_valid, ml_metrics = train_ml_classifier(features, true_labels)
    ml_pred = np.full(len(windows), "unknown", dtype=object)
    valid_indices = np.where(valid)[0]
    if ml_pred_valid is not None:
        ml_pred[valid_indices] = ml_pred_valid
    result = pd.DataFrame({
        "time_s": times,
        "window_start_sample": [start for start, end in windows],
        "window_end_sample": [end for start, end in windows],
        "valid_eeg_fraction": valid_fractions,
        "true_label": true_labels,
        "alpha_signal_score": alpha_scores,
        "blink_signal_score": blink_scores,
        "signal_prediction": signal_pred,
        "ml_prediction": ml_pred,
    })
    metrics = {
        "recording_dir": str(recording_dir),
        "recording_quality": recording_quality_summary(df, windows),
        "method_signal_science": signal_metrics,
        "method_ml": ml_metrics,
    }
    figure = plot_score_predictions(recording_dir, "eyes_alpha", result, "alpha_signal_score", "true_label", "signal_prediction")
    out_csv, out_json, _ = save_predictions(recording_dir, "eyes", result, metrics, figure)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Saved:", figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
