import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from analysis_common import (
    OCCIPITAL_NAMES,
    CENTRAL_NAMES,
    find_latest_recording,
    load_labeled,
    sampling_frequency_from_df,
    select_channel_columns,
    get_eeg_array,
    common_average_reference,
    safe_filter,
    bandpower,
    narrowband_power,
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
WINDOW_SECONDS = 2.5
STEP_SECONDS = 0.25
TARGET_WIDTH_HZ = 0.6


def binary_label(label):
    value = str(label)
    if value.startswith("flicker"):
        return "flicker"
    if value.startswith("rest") or value.startswith("baseline"):
        return "rest"
    return "unknown"


def main():
    recording_dir = Path(INPUT_DIR) if INPUT_DIR else find_latest_recording(RECORDINGS_ROOT, "ssvep")
    df, events_df, metadata = load_labeled(recording_dir)
    fs = sampling_frequency_from_df(df, metadata)
    target_hz = float(metadata.get("actual_flicker_hz") or metadata.get("target_flicker_hz") or 10.0)
    windows = make_windows(df, fs, WINDOW_SECONDS, STEP_SECONDS)
    true_labels = [binary_label(label) for label in state_labels_for_windows(df, windows)]
    occ_cols = select_channel_columns(df, OCCIPITAL_NAMES, fallback_count=3)
    central_cols = select_channel_columns(df, CENTRAL_NAMES, fallback_count=3)
    signal_scores = []
    times = []
    for start, end in windows:
        segment = get_eeg_array(df.iloc[start:end], occ_cols)
        segment = common_average_reference(segment)
        segment = safe_filter(segment, fs, 1, min(45, fs / 2 - 1))
        p_target = np.mean(narrowband_power(segment, fs, target_hz, TARGET_WIDTH_HZ))
        p_h2 = np.mean(narrowband_power(segment, fs, min(target_hz * 2.0, fs / 2 - 2), TARGET_WIDTH_HZ))
        p_alpha = np.mean(bandpower(segment, fs, 8, 12))
        p_broad = np.mean(bandpower(segment, fs, 5, min(40, fs / 2 - 1)))
        score = np.log10((p_target + 0.6 * p_h2 + 1e-18) / (p_broad + 0.25 * p_alpha + 1e-18))
        signal_scores.append(float(score))
        times.append(float(df["t_from_stream_start_s"].iloc[start:end].mean()))
    valid_fractions = window_valid_fractions(df, windows)
    threshold, positive_above = threshold_rule_from_labels(signal_scores, true_labels, "flicker")
    signal_pred = apply_binary_threshold(signal_scores, threshold, positive_above, "flicker", "rest")
    valid = np.asarray(true_labels) != "unknown"
    signal_metrics = {}
    if np.any(valid):
        signal_metrics = {
            "accuracy": float(accuracy_score(np.asarray(true_labels)[valid], np.asarray(signal_pred)[valid])),
            "classification_report": classification_report(np.asarray(true_labels)[valid], np.asarray(signal_pred)[valid], output_dict=True, zero_division=0),
            "threshold": threshold,
            "positive_above_threshold": positive_above,
            "target_hz": target_hz,
            "occipital_columns": occ_cols,
        }
    base_features = build_basic_features(df, fs, windows, [occ_cols, central_cols])
    target_features = np.asarray(signal_scores, dtype=float).reshape(-1, 1)
    features = np.hstack([base_features, target_features])
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
        "ssvep_signal_score": signal_scores,
        "signal_prediction": signal_pred,
        "ml_prediction": ml_pred,
    })
    metrics = {
        "recording_dir": str(recording_dir),
        "recording_quality": recording_quality_summary(df, windows),
        "method_signal_science": signal_metrics,
        "method_ml": ml_metrics,
    }
    figure = plot_score_predictions(recording_dir, "ssvep", result, "ssvep_signal_score", "true_label", "signal_prediction")
    out_csv, out_json, _ = save_predictions(recording_dir, "ssvep", result, metrics, figure)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Saved:", figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
