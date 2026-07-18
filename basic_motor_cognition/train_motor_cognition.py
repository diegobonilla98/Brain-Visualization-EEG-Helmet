import json

import numpy as np

from motor_cognition_common import (
    PROTOCOL_VERSION,
    build_training_table,
    find_motor_session_dirs,
    read_metadata,
    save_model_bundle,
    train_motor_model,
)


DATA_ROOT = "sessions"
MODEL_OUTPUT_DIR = "basic_motor_cognition/models"
WINDOW_SECONDS = 3.0
STEP_SECONDS = 0.25
MIN_VALID_FRACTION = 0.95
MIN_LABEL_FRACTION = 0.90
TRANSITION_SKIP_SECONDS = 1.50
BASELINE_SKIP_SECONDS = 1.00
RANDOM_STATE = 37
CV_SPLITS = 5
MIN_TRAINING_WINDOWS = 100


def select_protocol_sessions(session_dirs):
    versions = {}
    selected = []
    for session_dir in session_dirs:
        metadata = read_metadata(session_dir)
        version = str(metadata.get("protocol_version", "legacy"))
        versions[str(session_dir)] = version
        if version == PROTOCOL_VERSION:
            selected.append(session_dir)
    return selected, versions


def main():
    all_session_dirs = find_motor_session_dirs(DATA_ROOT)
    session_dirs, protocol_versions = select_protocol_sessions(all_session_dirs)
    if len(session_dirs) == 0:
        raise FileNotFoundError(f"No {PROTOCOL_VERSION} sessions found under {DATA_ROOT}. Record new sessions with record_motor_cognition.py.")
    feature_frame, labels, groups, window_metadata = build_training_table(
        session_dirs,
        WINDOW_SECONDS,
        STEP_SECONDS,
        MIN_VALID_FRACTION,
        MIN_LABEL_FRACTION,
        TRANSITION_SKIP_SECONDS,
        BASELINE_SKIP_SECONDS,
    )
    if len(feature_frame) < MIN_TRAINING_WINDOWS:
        raise RuntimeError(f"Only {len(feature_frame)} usable windows found. Collect more complete sessions before training.")
    bundle, metrics, oof_intent_probability, oof_final_label = train_motor_model(
        feature_frame,
        labels,
        groups,
        random_state=RANDOM_STATE,
        cv_splits=CV_SPLITS,
    )
    validation_group_level = "session" if len(session_dirs) >= 2 else "trial"
    metrics.update({
        "session_dirs": [str(path) for path in session_dirs],
        "protocol_versions": protocol_versions,
        "protocol_filter": PROTOCOL_VERSION,
        "training_settings": {
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "min_label_fraction": MIN_LABEL_FRACTION,
            "transition_skip_seconds": TRANSITION_SKIP_SECONDS,
            "baseline_skip_seconds": BASELINE_SKIP_SECONDS,
            "random_state": RANDOM_STATE,
            "cv_splits": CV_SPLITS,
            "validation_group_level": validation_group_level,
        },
    })
    training_windows = window_metadata.copy()
    training_windows["true_class_label"] = labels.to_numpy(dtype=object)
    training_windows["oof_intent_probability"] = oof_intent_probability
    training_windows["oof_prediction"] = oof_final_label
    training_windows["oof_available"] = np.isfinite(oof_intent_probability)
    model_path, metrics_path, windows_path = save_model_bundle(bundle, MODEL_OUTPUT_DIR, metrics, training_windows)
    print("Sessions:")
    for session_dir in session_dirs:
        print(session_dir)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved training windows:", windows_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
