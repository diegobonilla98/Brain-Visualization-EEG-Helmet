import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import event, visual

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainaccess_stream import BrainAccessStream
from concentration_common import feature_frame_from_segment, latest_model_path, load_model_bundle, predict_bundle, resolve_project_path
from stimulus_common import estimate_refresh_rate, make_window


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_DIR = "concentration_baseline/models"
MODEL_PATH = ""
OUTPUT_ROOT = "sessions"
OUTPUT_PREFIX = "concentration_inference"
SETTLE_SECONDS = 8.0
WINDOW_SECONDS = 4.0
UPDATE_SECONDS = 0.5
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


def draw_text(win, title_stim, detail_stim, title, detail, color="white"):
    title_stim.text = title
    title_stim.color = color
    detail_stim.text = detail
    title_stim.draw()
    detail_stim.draw()
    win.flip()


def format_detail(prediction):
    concentrated = float(prediction.get("probability_concentrated", 0.0))
    not_concentrated = float(prediction.get("probability_not_concentrated", 0.0))
    return f"concentrated: {concentrated:.2f}\nnot concentrated: {not_concentrated:.2f}\n\nPress ESC to stop."


def main():
    model_path = selected_model_path()
    bundle = load_model_bundle(model_path)
    output_dir = make_output_dir()
    predictions = []
    win = make_window("#111111")
    estimate_refresh_rate(win)
    title_stim = visual.TextStim(win, text="", height=0.075, color="white", pos=(0.0, 0.12), wrapWidth=1.55, alignText="center")
    detail_stim = visual.TextStim(win, text="", height=0.042, color="#d6d6d6", pos=(0.0, -0.18), wrapWidth=1.45, alignText="center")
    try:
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            sample_rate = float(recorder.sample_frequency)
            window_samples = int(round(WINDOW_SECONDS * sample_rate))
            channel_labels = [label for label in recorder.channel_labels if label.endswith("_uV")]
            row_indices = [recorder.channel_indices[label] for label in channel_labels]
            consumed_chunks = 0
            settle_start = time.perf_counter()
            while time.perf_counter() - settle_start < SETTLE_SECONDS:
                consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
                remaining = max(0.0, SETTLE_SECONDS - (time.perf_counter() - settle_start))
                draw_text(win, title_stim, detail_stim, "SETTLE", f"Relax and look at the screen.\nStarting in {remaining:.1f}s", "#f0f0f0")
                if "escape" in event.getKeys():
                    break
                time.sleep(SCREEN_REFRESH_SLEEP_SECONDS)
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
                    feature_frame = feature_frame_from_segment(segment_df, sample_rate, bundle["feature_names"])
                    prediction = predict_bundle(bundle, feature_frame).iloc[0]
                    label = str(prediction["prediction"])
                    confidence = max(float(prediction.get("probability_concentrated", 0.0)), float(prediction.get("probability_not_concentrated", 0.0)))
                    row = {
                        "pc_time_perf_counter_s": now,
                        "t_from_inference_start_s": now - started_at,
                        "prediction": label,
                        "confidence": confidence,
                    }
                    row.update(prediction.to_dict())
                    predictions.append(row)
                    color = "#77d27a" if label == "concentrated" else "#e2c36a"
                    draw_text(win, title_stim, detail_stim, label.replace("_", " ").upper(), format_detail(prediction), color)
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
            "settle_seconds": SETTLE_SECONDS,
            "window_seconds": WINDOW_SECONDS,
            "update_seconds": UPDATE_SECONDS,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print("Saved predictions:", predictions_path)
        print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
