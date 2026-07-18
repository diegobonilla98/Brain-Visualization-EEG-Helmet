import json

from analysis_common import find_latest_recording
from eyes_open_closed_common import (
    resolve_project_path,
    save_model_bundle,
    train_eyes_open_closed_model,
)


INPUT_DIR = ""
RECORDINGS_ROOT = "recordings"
MODEL_DIR = "eeg_quality_suite/models"


def main():
    recording_dir = resolve_project_path(INPUT_DIR) if INPUT_DIR else find_latest_recording(RECORDINGS_ROOT, "eyes")
    bundle, metrics, training_windows, metadata = train_eyes_open_closed_model(recording_dir)
    model_path, metrics_path, windows_path = save_model_bundle(bundle, MODEL_DIR, metrics, training_windows)
    print("Training recording:", recording_dir)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved training windows:", windows_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
