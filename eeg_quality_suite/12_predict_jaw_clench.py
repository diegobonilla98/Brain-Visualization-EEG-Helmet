import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from analysis_common import (
    TEMPORAL_NAMES,
    FRONTAL_NAMES,
    CENTRAL_NAMES,
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
    apply_binary_threshold,
    window_valid_fractions,
    recording_quality_summary,
    save_predictions,
    plot_score_predictions,
)


INPUT_DIR = ""
RECORDINGS_ROOT = "sessions"
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.20


def compact_label(label):
    value = str(label)
    if value == "jaw_clench":
        return "jaw_clench"
    if value == "relaxed_jaw":
        return "relaxed_jaw"
    return "unknown"


def main():
    recording_dir = Path(INPUT_DIR) if INPUT_DIR else find_latest_recording(RECORDINGS_ROOT, "jaw")
    df, events_df, metadata = load_labeled(recording_dir)
    fs = sampling_frequency_from_df(df, metadata)
    windows = make_windows(df, fs, WINDOW_SECONDS, STEP_SECONDS)
    true_labels = [compact_label(label) for label in state_labels_for_windows(df, windows)]
    temporal_cols = select_channel_columns(df, TEMPORAL_NAMES, fallback_count=4)
    frontal_cols = select_channel_columns(df, FRONTAL_NAMES, fallback_count=4)
    central_cols = select_channel_columns(df, CENTRAL_NAMES, fallback_count=3)
    jaw_scores = []
    times = []
    for start, end in windows:
        temporal = get_eeg_array(df.iloc[start:end], temporal_cols)
        temporal = common_average_reference(temporal)
        temporal = safe_filter(temporal, fs, 2, min(100, fs / 2 - 1))
        p_emg = np.mean(bandpower(temporal, fs, 35, min(95, fs / 2 - 1)))
        p_eeg = np.mean(bandpower(temporal, fs, 8, 30))
        jaw_scores.append(float(np.log10((p_emg + 1e-18) / (p_eeg + 1e-18))))
        times.append(float(df["t_from_stream_start_s"].iloc[start:end].mean()))
    valid_fractions = window_valid_fractions(df, windows)
    threshold, positive_above = threshold_rule_from_labels(jaw_scores, true_labels, "jaw_clench")
    signal_pred = apply_binary_threshold(jaw_scores, threshold, positive_above, "jaw_clench", "relaxed_jaw")
    valid = np.asarray(true_labels) != "unknown"
    signal_metrics = {}
    if np.any(valid):
        signal_metrics = {
            "accuracy": float(accuracy_score(np.asarray(true_labels)[valid], np.asarray(signal_pred)[valid])),
            "classification_report": classification_report(np.asarray(true_labels)[valid], np.asarray(signal_pred)[valid], output_dict=True, zero_division=0),
            "threshold": threshold,
            "positive_above_threshold": positive_above,
            "temporal_columns": temporal_cols,
        }
    base_features = build_basic_features(df, fs, windows, [temporal_cols, frontal_cols, central_cols])
    added = np.asarray(jaw_scores, dtype=float).reshape(-1, 1)
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
        "jaw_signal_score": jaw_scores,
        "signal_prediction": signal_pred,
        "ml_prediction": ml_pred,
    })
    metrics = {
        "recording_dir": str(recording_dir),
        "recording_quality": recording_quality_summary(df, windows),
        "method_signal_science": signal_metrics,
        "method_ml": ml_metrics,
    }
    figure = plot_score_predictions(recording_dir, "jaw", result, "jaw_signal_score", "true_label", "signal_prediction")
    out_csv, out_json, _ = save_predictions(recording_dir, "jaw", result, metrics, figure)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Saved:", figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
