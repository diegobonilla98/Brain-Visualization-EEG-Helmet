import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import event, visual

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from box_motion_common import DIRECTION_LABELS, feature_dict_from_segment, latest_model_path, load_model_bundle, predict_bundle, resolve_project_path, subtract_baseline
from brainaccess_stream import BrainAccessStream
from stimulus_common import estimate_refresh_rate, make_window


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_DIR = "box_motion_cognition/models"
MODEL_PATH = ""
OUTPUT_ROOT = "sessions"
OUTPUT_PREFIX = "box_motion_inference"
BASELINE_SECONDS = 15.0
WINDOW_SECONDS = 3.0
UPDATE_SECONDS = 0.35
SCREEN_REFRESH_SLEEP_SECONDS = 0.01
CONFIDENCE_THRESHOLD = 0.30
MOVE_STEP = 0.035
POSITION_LIMIT = 0.36


DIRECTION_VECTORS = {
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


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


class LiveBoxRenderer:
    def __init__(self, win):
        self.win = win
        self.box = visual.Rect(win, width=0.14, height=0.14, fillColor="#ededed", lineColor="#ededed", pos=(0.0, 0.0))
        self.boundary = visual.Rect(win, width=0.86, height=0.86, fillColor=None, lineColor="#444a54", lineWidth=2, pos=(0.0, 0.0))
        self.title = visual.TextStim(win, text="", height=0.056, color="white", pos=(0.0, 0.46), wrapWidth=1.55, alignText="center")
        self.detail = visual.TextStim(win, text="", height=0.036, color="#d8d8d8", pos=(0.0, -0.48), wrapWidth=1.45, alignText="center")

    def draw(self, box_position, title, detail):
        self.win.color = "#101014"
        self.box.pos = box_position
        self.boundary.draw()
        self.box.draw()
        self.title.text = title
        self.detail.text = detail
        self.title.draw()
        self.detail.draw()
        self.win.flip()


def clamp_position(position):
    x_value = float(np.clip(position[0], -POSITION_LIMIT, POSITION_LIMIT))
    y_value = float(np.clip(position[1], -POSITION_LIMIT, POSITION_LIMIT))
    return x_value, y_value


def update_position(position, direction, confidence):
    if direction not in DIRECTION_VECTORS or confidence < CONFIDENCE_THRESHOLD:
        return position
    dx, dy = DIRECTION_VECTORS[direction]
    return clamp_position((position[0] + dx * MOVE_STEP, position[1] + dy * MOVE_STEP))


def format_detail(prediction, confidence):
    rows = [f"confidence: {confidence:.2f}", ""]
    for label in DIRECTION_LABELS:
        rows.append(f"{label}: {float(prediction.get(f'probability_{label}', 0.0)):.2f}")
    rows.extend(["", "Think the direction. Do not move muscles. Press ESC to stop."])
    return "\n".join(rows)


def main():
    model_path = selected_model_path()
    bundle = load_model_bundle(model_path)
    output_dir = make_output_dir()
    predictions = []
    box_position = (0.0, 0.0)
    win = make_window("#101014")
    estimate_refresh_rate(win)
    renderer = LiveBoxRenderer(win)
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
                renderer.draw((0.0, 0.0), "BASELINE", f"Relax. Do not think about moving the box.\n{remaining:.1f}s")
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
                    delta_features = subtract_baseline(segment_features, baseline_features)
                    feature_frame = pd.DataFrame([delta_features]).reindex(columns=bundle["feature_names"], fill_value=0.0)
                    prediction = predict_bundle(bundle, feature_frame).iloc[0]
                    direction = str(prediction["prediction"])
                    confidence = max([float(prediction.get(f"probability_{label}", 0.0)) for label in DIRECTION_LABELS])
                    box_position = update_position(box_position, direction, confidence)
                    row = {
                        "pc_time_perf_counter_s": now,
                        "t_from_inference_start_s": now - started_at,
                        "prediction": direction,
                        "confidence": confidence,
                        "box_x": box_position[0],
                        "box_y": box_position[1],
                    }
                    row.update(prediction.to_dict())
                    predictions.append(row)
                    renderer.draw(box_position, direction.upper(), format_detail(prediction, confidence))
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
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "move_step": MOVE_STEP,
            "position_limit": POSITION_LIMIT,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print("Saved predictions:", predictions_path)
        print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
