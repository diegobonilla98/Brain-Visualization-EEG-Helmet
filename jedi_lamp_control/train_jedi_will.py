import json

import numpy as np
import pandas as pd

from jedi_common import CLASS_LABELS, PROTOCOL_VERSION, build_training_table, find_session_dirs, save_model, train_model_search


DATA_ROOT = "sessions"
MODEL_OUTPUT_DIR = "jedi_lamp_control/models"
WINDOW_SECONDS = 3.0
STEP_SECONDS = 0.40
MINIMUM_VALID_FRACTION = 0.95
MINIMUM_LABEL_FRACTION = 0.90
ONSET_SKIP_SECONDS = 1.50
RANDOM_STATE = 20260719
CV_FOLDS = 5
MINIMUM_WINDOWS = 80
MINIMUM_BLOCKS_PER_CLASS = 4


def main():
    session_dirs = find_session_dirs(DATA_ROOT)
    if not session_dirs:
        raise FileNotFoundError(f"No {PROTOCOL_VERSION} sessions found. Run record_jedi_will.py in multiple sessions first.")
    features, labels, groups, window_metadata = build_training_table(
        session_dirs,
        WINDOW_SECONDS,
        STEP_SECONDS,
        MINIMUM_VALID_FRACTION,
        MINIMUM_LABEL_FRACTION,
        ONSET_SKIP_SECONDS,
    )
    if len(features) < MINIMUM_WINDOWS:
        raise RuntimeError(f"Only {len(features)} usable windows were found. Collect more complete blocks before training.")
    counts = pd.Series(labels).value_counts()
    if any(int(counts.get(label, 0)) == 0 for label in CLASS_LABELS):
        raise RuntimeError("Both NOTHING and THE WILL are required.")
    block_counts = window_metadata.groupby("class_label")["block_index"].nunique()
    if any(int(block_counts.get(label, 0)) < MINIMUM_BLOCKS_PER_CLASS for label in CLASS_LABELS):
        raise RuntimeError(f"Need at least {MINIMUM_BLOCKS_PER_CLASS} completed blocks per class.")
    bundle, candidates, oof_probabilities = train_model_search(features, labels, groups, RANDOM_STATE, CV_FOLDS)
    threshold = float(bundle["activation_threshold"])
    oof_predictions = np.where(oof_probabilities >= threshold, "the_will", "nothing")
    window_metadata = window_metadata.copy()
    window_metadata["true_class_label"] = labels
    window_metadata["validation_group"] = groups
    window_metadata["oof_probability_the_will"] = oof_probabilities
    window_metadata["oof_prediction"] = oof_predictions
    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_model": bundle["model_name"],
        "activation_threshold": threshold,
        "selected_validation_metrics": bundle["validation_metrics"],
        "candidate_ranking": candidates,
        "session_count": len(session_dirs),
        "sessions": [str(path) for path in session_dirs],
        "window_count": len(features),
        "feature_count": features.shape[1],
        "class_counts": {label: int(np.sum(labels == label)) for label in CLASS_LABELS},
        "validation_group_level": "session" if len(session_dirs) >= 2 else "block",
        "training_settings": {
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
            "minimum_label_fraction": MINIMUM_LABEL_FRACTION,
            "onset_skip_seconds": ONSET_SKIP_SECONDS,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        },
        "interpretation": "Session-level validation is the meaningful estimate when at least two sessions exist. Block validation from one session is provisional and can overestimate real-time generalization.",
    }
    bundle["training_settings"] = metrics["training_settings"]
    bundle["session_ids"] = [path.name for path in session_dirs]
    model_path, metrics_path, windows_path = save_model(bundle, MODEL_OUTPUT_DIR, metrics, window_metadata)
    print("Selected model:", bundle["model_name"])
    print("Activation threshold:", threshold)
    print("Validation:", json.dumps(bundle["validation_metrics"], indent=2))
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved windows:", windows_path)


if __name__ == "__main__":
    main()
