import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import event, visual

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))

from brainaccess_stream import BrainAccessStream
from stimulus_common import estimate_refresh_rate, make_window
from motor_cognition_common import feature_dict_from_segment, latest_model_path, load_model_bundle, predict_bundle, resolve_project_path, subtract_baseline


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_DIR = "basic_motor_cognition/models"
MODEL_PATH = ""
OUTPUT_ROOT = "sessions"
OUTPUT_PREFIX = "basic_motor_cognition_inference"
BASELINE_SECONDS = 20.0
WINDOW_SECONDS = 3.0
UPDATE_SECONDS = 0.25
SCREEN_REFRESH_SLEEP_SECONDS = 0.01


def selected_model_path():
    if len(str(MODEL_PATH).strip()) > 0:
        return resolve_project_path(MODEL_PATH)
    return latest_model_path(MODEL_DIR)


def make_output_dir():
    root = resolve_project_path(OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"session_{stamp}_{OUTPUT_PREFIX}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_new_samples(recorder, consumed_chunks, row_indices):
    with recorder.lock:
        new_chunks = recorder.chunks[consumed_chunks:]
        consumed_chunks = len(recorder.chunks)
    arrays = []
    for _, array, chunk_size in new_chunks:
        count = min(int(chunk_size), array.shape[1])
        if count > 0:
            arrays.append(array[row_indices, :count])
    if len(arrays) == 0:
        return consumed_chunks, None
    return consumed_chunks, np.concatenate(arrays, axis=1)


def draw_text(win, text_stim, text):
    text_stim.text = text
    text_stim.draw()
    win.flip()


def format_prediction(prediction):
    rows = [
        "LIVE MOTOR IMAGERY",
        "",
        f"Prediction: {prediction['prediction']}",
        f"Intent probability: {prediction['intent_probability']:.2f}",
        f"Active classifier: {prediction['active_prediction']}",
        "",
    ]
    probability_columns = [column for column in prediction.index if column.startswith("active_probability_")]
    for column in probability_columns:
        label = column.replace("active_probability_", "")
        rows.append(f"{label}: {float(prediction[column]):.2f}")
    rows.extend(["", "Think only. Keep muscles relaxed.", "Press ESC to stop."])
    return "\n".join(rows)


def main():
    model_path = selected_model_path()
    bundle = load_model_bundle(model_path)
    output_dir = make_output_dir()
    predictions = []
    win = make_window("black")
    estimate_refresh_rate(win)
    text_stim = visual.TextStim(win, text="", height=0.052, color="white", pos=(0, 0), wrapWidth=1.55, alignText="center")
    try:
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            sample_rate = float(recorder.sample_frequency)
            window_samples = int(round(WINDOW_SECONDS * sample_rate))
            channel_labels = [label for label in recorder.channel_labels if label.endswith("_uV")]
            row_indices = [recorder.channel_indices[label] for label in channel_labels]
            consumed_chunks = 0
            baseline_buffer = np.zeros((len(channel_labels), 0), dtype=float)
            baseline_start = time.perf_counter()
            while time.perf_counter() - baseline_start < BASELINE_SECONDS:
                consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
                if new_samples is not None:
                    baseline_buffer = np.concatenate([baseline_buffer, new_samples], axis=1)
                remaining = max(0.0, BASELINE_SECONDS - (time.perf_counter() - baseline_start))
                draw_text(win, text_stim, f"BASELINE CALIBRATION\n\nRelax. Do not imagine movement.\n\n{remaining:.1f}s")
                if "escape" in event.getKeys():
                    break
                time.sleep(SCREEN_REFRESH_SLEEP_SECONDS)
            if baseline_buffer.shape[1] < window_samples:
                raise RuntimeError("Not enough baseline samples collected for inference.")
            baseline_df = pd.DataFrame(baseline_buffer.T, columns=channel_labels)
            baseline_features = feature_dict_from_segment(baseline_df, sample_rate)
            buffer = np.zeros((len(channel_labels), 0), dtype=float)
            last_prediction_time = 0.0
            started_at = time.perf_counter()
            while True:
                consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
                if new_samples is not None:
                    buffer = np.concatenate([buffer, new_samples], axis=1)
                    buffer = buffer[:, -window_samples:]
                now = time.perf_counter()
                if buffer.shape[1] >= window_samples and now - last_prediction_time >= UPDATE_SECONDS:
                    segment_df = pd.DataFrame(buffer[:, -window_samples:].T, columns=channel_labels)
                    segment_features = feature_dict_from_segment(segment_df, sample_rate)
                    erd_features = subtract_baseline(segment_features, baseline_features)
                    feature_frame = pd.DataFrame([erd_features]).reindex(columns=bundle["feature_names"], fill_value=0.0)
                    prediction = predict_bundle(bundle, feature_frame).iloc[0]
                    row = {
                        "pc_time_perf_counter_s": now,
                        "t_from_inference_start_s": now - started_at,
                    }
                    row.update(prediction.to_dict())
                    predictions.append(row)
                    draw_text(win, text_stim, format_prediction(prediction))
                    last_prediction_time = now
                if "escape" in event.getKeys():
                    break
                time.sleep(SCREEN_REFRESH_SLEEP_SECONDS)
    finally:
        win.close()
        predictions_path = output_dir / "predictions.csv"
        metadata_path = output_dir / "metadata.json"
        pd.DataFrame(predictions).to_csv(predictions_path, index=False)
        metadata = {
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "baseline_seconds": BASELINE_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "update_seconds": UPDATE_SECONDS,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print("Saved predictions:", predictions_path)
        print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
