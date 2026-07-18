import json
import time

import numpy as np
import pandas as pd
from psychopy import event, visual

from brainaccess_stream import BrainAccessStream, DEFAULT_MAXI_32_NAMES
from eyes_open_closed_common import (
    feature_frame_from_segment,
    grouped_feature_names,
    latest_model_path,
    load_model_bundle,
    predict_bundle,
    resolve_project_path,
)
from stimulus_common import estimate_refresh_rate, make_window


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_DIR = "eeg_quality_suite/models"
MODEL_PATH = ""
OUTPUT_ROOT = "sessions"
OUTPUT_PREFIX = "eyes_inference"
SETTLE_SECONDS = 3.0
WINDOW_SECONDS = 2.0
UPDATE_SECONDS = 0.25
PREDICTION_SMOOTH_SECONDS = 1.5
SCREEN_REFRESH_SLEEP_SECONDS = 0.01
EEG_DRAW_SECONDS = 0.05
EEG_MAX_POINTS = 500

PANEL_BOTTOM = -0.44
PANEL_TOP = 0.30
EEG_LEFT = -0.98
EEG_RIGHT = -0.02
HEATMAP_LEFT = 0.02
HEATMAP_RIGHT = 0.98

DISPLAY = {
    "closed_eyes": {
        "title": "CLOSED EYES",
        "color": "#6ea8ff",
        "detail": "Posterior alpha is elevated.",
    },
    "open_eyes": {
        "title": "OPEN EYES",
        "color": "#7dff9a",
        "detail": "Eyes appear open.",
    },
    "strong_blinks": {
        "title": "BLINK",
        "color": "#ffd56e",
        "detail": "Large frontal blink artifact detected.",
    },
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


def clean_channel_name(label):
    return str(label).replace("_uV", "")


def display_channel_names(channel_labels):
    names = [clean_channel_name(label) for label in channel_labels]
    if len(names) == len(DEFAULT_MAXI_32_NAMES):
        return DEFAULT_MAXI_32_NAMES
    return names


def rgb_norm(red, green, blue):
    return (
        (float(red) / 127.5) - 1.0,
        (float(green) / 127.5) - 1.0,
        (float(blue) / 127.5) - 1.0,
    )


def short_feature_name(name):
    text = str(name).replace("_logpower", "").replace("_signal_score", "")
    return text.replace("_", " ")


def value_color_norm(value, vmin, vmax):
    if not np.isfinite(value):
        return rgb_norm(70, 70, 78)
    if vmax - vmin < 1e-9:
        scaled = 0.5
    else:
        scaled = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    red = 42 + (255 - 42) * scaled
    green = 74 + (118 - 74) * scaled
    blue = 130 + (84 - 130) * scaled
    return rgb_norm(red, green, blue)


def importance_color_norm(importance, max_importance):
    if max_importance < 1e-9:
        scaled = 0.0
    else:
        scaled = float(np.clip(importance / max_importance, 0.0, 1.0))
    return rgb_norm(40 + 180 * scaled, 40 + 130 * scaled, 48 + 40 * scaled)


def subsample_row(row, max_points=EEG_MAX_POINTS):
    row = np.asarray(row, dtype=float)
    if len(row) <= max_points:
        return row
    indices = np.linspace(0, len(row) - 1, max_points).astype(int)
    return row[indices]


class PredictionSmoother:
    def __init__(self, class_labels, smooth_seconds):
        self.class_labels = list(class_labels)
        self.smooth_seconds = float(smooth_seconds)
        self.probabilities = None
        self.last_update_time = None

    def update(self, prediction_row, now):
        current = np.array(
            [float(prediction_row.get(f"probability_{label}", 0.0)) for label in self.class_labels],
            dtype=float,
        )
        if self.probabilities is None or self.last_update_time is None:
            self.probabilities = current.copy()
        else:
            dt = max(1e-3, now - self.last_update_time)
            blend = 1.0 - np.exp(-dt / self.smooth_seconds)
            self.probabilities = (1.0 - blend) * self.probabilities + blend * current
        self.last_update_time = now
        best_index = int(np.argmax(self.probabilities))
        result = {
            "prediction": self.class_labels[best_index],
            "raw_prediction": str(prediction_row["prediction"]),
        }
        for index, label in enumerate(self.class_labels):
            result[f"probability_{label}"] = float(self.probabilities[index])
            result[f"raw_probability_{label}"] = float(
                prediction_row.get(f"probability_{label}", 0.0)
            )
        return result


class EegPanel:
    def __init__(self, win, channel_names):
        self.win = win
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.trace_left = EEG_LEFT + 0.08
        self.trace_right = EEG_RIGHT - 0.01
        self.panel_center_x = 0.5 * (EEG_LEFT + EEG_RIGHT)
        self.has_data = False

        self.frame = visual.Rect(
            win,
            width=EEG_RIGHT - EEG_LEFT,
            height=PANEL_TOP - PANEL_BOTTOM,
            pos=(self.panel_center_x, 0.5 * (PANEL_BOTTOM + PANEL_TOP)),
            lineColor=rgb_norm(58, 66, 84),
            fillColor=rgb_norm(16, 19, 26),
            units="norm",
            lineWidth=2,
        )
        self.title = visual.TextStim(
            win,
            text="32-channel EEG (per-channel scaled)",
            height=0.022,
            color="#d2d2d2",
            pos=(self.panel_center_x, PANEL_TOP - 0.025),
            units="norm",
        )
        self.row_backgrounds = []
        self.channel_labels = []
        self.traces = []
        row_height = (PANEL_TOP - PANEL_BOTTOM - 0.05) / max(self.n_channels, 1)
        for channel_index, channel_name in enumerate(channel_names):
            y_center = PANEL_TOP - 0.05 - (channel_index + 0.5) * row_height
            self.row_backgrounds.append(
                visual.Rect(
                    win,
                    width=EEG_RIGHT - EEG_LEFT - 0.02,
                    height=row_height * 0.92,
                    pos=(self.panel_center_x, y_center),
                    lineWidth=0,
                    fillColor=rgb_norm(20, 23, 30) if channel_index % 2 == 0 else rgb_norm(18, 21, 28),
                    units="norm",
                )
            )
            self.channel_labels.append(
                visual.TextStim(
                    win,
                    text=channel_name,
                    height=min(0.016, row_height * 0.55),
                    color="#97a3b3",
                    pos=(EEG_LEFT + 0.04, y_center),
                    alignText="right",
                    anchorHoriz="right",
                    units="norm",
                )
            )
            self.traces.append(
                visual.ShapeStim(
                    win,
                    vertices=[(self.trace_left, y_center), (self.trace_right, y_center)],
                    closeShape=False,
                    lineWidth=1.6,
                    lineColor=rgb_norm(70, 220, 255),
                    fillColor=None,
                    units="norm",
                )
            )
        self.placeholder = visual.TextStim(
            win,
            text="Waiting for EEG...",
            height=0.028,
            color="#9aa3b2",
            pos=(self.panel_center_x, 0.5 * (PANEL_BOTTOM + PANEL_TOP)),
            units="norm",
        )

    def update(self, channel_data):
        channel_data = np.asarray(channel_data, dtype=float)
        if channel_data.ndim != 2 or channel_data.shape[1] < 2:
            return
        self.has_data = True
        row_height = (PANEL_TOP - PANEL_BOTTOM - 0.05) / max(self.n_channels, 1)
        for channel_index in range(self.n_channels):
            row = subsample_row(channel_data[channel_index])
            y_center = PANEL_TOP - 0.05 - (channel_index + 0.5) * row_height
            y_half = row_height * 0.38
            if np.any(np.isfinite(row)):
                low = float(np.nanpercentile(row, 2))
                high = float(np.nanpercentile(row, 98))
                if high - low < 1.0:
                    high = low + 1.0
                scaled = np.clip((row - low) / (high - low), 0.0, 1.0)
            else:
                scaled = np.zeros(len(row), dtype=float)
            xs = np.linspace(self.trace_left, self.trace_right, len(row))
            ys = y_center + (scaled - 0.5) * 2.0 * y_half
            self.traces[channel_index].vertices = list(zip(xs.tolist(), ys.tolist()))

    def draw(self):
        self.frame.draw()
        self.title.draw()
        if not self.has_data:
            self.placeholder.draw()
            return
        for background in self.row_backgrounds:
            background.draw()
        for label in self.channel_labels:
            label.draw()
        for trace in self.traces:
            trace.draw()


class FeatureHeatmapPanel:
    def __init__(self, win, bundle):
        self.win = win
        self.bundle = bundle
        self.feature_names = list(bundle["feature_names"])
        self.importances = np.asarray(bundle["model"].feature_importances_, dtype=float)
        self.max_importance = float(np.max(self.importances)) if len(self.importances) else 1.0
        self.groups = grouped_feature_names(self.feature_names)
        self.panel_center_x = 0.5 * (HEATMAP_LEFT + HEATMAP_RIGHT)
        self.has_data = False
        self.value_left = HEATMAP_LEFT + 0.34
        self.value_right = HEATMAP_LEFT + 0.62
        self.importance_left = HEATMAP_LEFT + 0.64
        self.importance_right = HEATMAP_RIGHT - 0.02

        self.frame = visual.Rect(
            win,
            width=HEATMAP_RIGHT - HEATMAP_LEFT,
            height=PANEL_TOP - PANEL_BOTTOM,
            pos=(self.panel_center_x, 0.5 * (PANEL_BOTTOM + PANEL_TOP)),
            lineColor=rgb_norm(58, 66, 84),
            fillColor=rgb_norm(16, 19, 26),
            units="norm",
            lineWidth=2,
        )
        self.title = visual.TextStim(
            win,
            text="Model whitebox: current features + RF importance",
            height=0.022,
            color="#d2d2d2",
            pos=(self.panel_center_x, PANEL_TOP - 0.025),
            units="norm",
        )
        self.header_labels = [
            visual.TextStim(
                win,
                text=text,
                height=0.016,
                color="#969696",
                pos=pos,
                alignText="left",
                anchorHoriz="left",
                units="norm",
            )
            for text, pos in [
                ("feature", (HEATMAP_LEFT + 0.02, PANEL_TOP - 0.055)),
                ("value", (self.value_left, PANEL_TOP - 0.055)),
                ("importance", (self.importance_left, PANEL_TOP - 0.055)),
            ]
        ]
        self.placeholder = visual.TextStim(
            win,
            text="Waiting for model window...",
            height=0.028,
            color="#9aa3b2",
            pos=(self.panel_center_x, 0.5 * (PANEL_BOTTOM + PANEL_TOP)),
            units="norm",
        )
        self.rows = []
        self._build_rows()

    def _build_rows(self):
        row_slots = []
        for group_name, names in self.groups:
            row_slots.append(("group", group_name))
            for name in names:
                row_slots.append(("feature", name))
        usable_height = (PANEL_TOP - PANEL_BOTTOM) - 0.08
        row_height = usable_height / max(len(row_slots), 1)
        text_height = min(0.015, row_height * 0.55)
        y = PANEL_TOP - 0.08
        for slot_type, slot_name in row_slots:
            y_center = y - 0.5 * row_height
            if slot_type == "group":
                self.rows.append(
                    {
                        "type": "group",
                        "band": visual.Rect(
                            self.win,
                            width=HEATMAP_RIGHT - HEATMAP_LEFT - 0.02,
                            height=row_height * 0.92,
                            pos=(self.panel_center_x, y_center),
                            lineWidth=0,
                            fillColor=rgb_norm(28, 32, 42),
                            units="norm",
                        ),
                        "label": visual.TextStim(
                            self.win,
                            text=str(slot_name).upper(),
                            height=text_height,
                            color="#78beff",
                            pos=(HEATMAP_LEFT + 0.02, y_center),
                            alignText="left",
                            anchorHoriz="left",
                            units="norm",
                        ),
                    }
                )
            else:
                self.rows.append(
                    {
                        "type": "feature",
                        "name": slot_name,
                        "band": visual.Rect(
                            self.win,
                            width=HEATMAP_RIGHT - HEATMAP_LEFT - 0.02,
                            height=row_height * 0.92,
                            pos=(self.panel_center_x, y_center),
                            lineColor=rgb_norm(40, 44, 56),
                            fillColor=rgb_norm(24, 26, 34),
                            units="norm",
                            lineWidth=1,
                        ),
                        "label": visual.TextStim(
                            self.win,
                            text=short_feature_name(slot_name),
                            height=text_height,
                            color="#b9bec8",
                            pos=(HEATMAP_LEFT + 0.02, y_center),
                            alignText="left",
                            anchorHoriz="left",
                            units="norm",
                        ),
                        "value_bg": visual.Rect(
                            self.win,
                            width=self.value_right - self.value_left,
                            height=row_height * 0.55,
                            pos=(0.5 * (self.value_left + self.value_right), y_center),
                            lineWidth=0,
                            fillColor=rgb_norm(34, 36, 44),
                            units="norm",
                        ),
                        "value_fill": visual.Rect(
                            self.win,
                            width=0.001,
                            height=row_height * 0.55,
                            pos=(self.value_left + 0.0005, y_center),
                            lineWidth=0,
                            fillColor=rgb_norm(70, 120, 200),
                            units="norm",
                        ),
                        "importance_bg": visual.Rect(
                            self.win,
                            width=self.importance_right - self.importance_left,
                            height=row_height * 0.55,
                            pos=(0.5 * (self.importance_left + self.importance_right), y_center),
                            lineWidth=0,
                            fillColor=rgb_norm(34, 36, 44),
                            units="norm",
                        ),
                        "importance_fill": visual.Rect(
                            self.win,
                            width=0.001,
                            height=row_height * 0.55,
                            pos=(self.importance_left + 0.0005, y_center),
                            lineWidth=0,
                            fillColor=rgb_norm(90, 170, 90),
                            units="norm",
                        ),
                        "value_text": visual.TextStim(
                            self.win,
                            text="",
                            height=text_height,
                            color="#aab0ba",
                            pos=(self.value_right + 0.01, y_center),
                            alignText="left",
                            anchorHoriz="left",
                            units="norm",
                        ),
                    }
                )
            y -= row_height

    def update(self, feature_frame):
        feature_names = self.feature_names
        values = feature_frame.iloc[0][feature_names].to_numpy(dtype=float)
        group_ranges = {}
        for group_name, names in self.groups:
            indices = [feature_names.index(name) for name in names]
            group_values = values[indices]
            group_ranges[group_name] = (
                float(np.min(group_values)),
                float(np.max(group_values)),
            )
        value_span = self.value_right - self.value_left
        importance_span = self.importance_right - self.importance_left
        for row in self.rows:
            if row["type"] != "feature":
                continue
            feature_index = feature_names.index(row["name"])
            value = float(values[feature_index])
            importance = float(self.importances[feature_index])
            group_name = row["name"].split("_", 1)[0]
            vmin, vmax = group_ranges[group_name]
            if vmax - vmin < 1e-9:
                value_fraction = 0.5
            else:
                value_fraction = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
            importance_fraction = float(importance / max(self.max_importance, 1e-9))
            fill_width = max(0.001, value_span * value_fraction)
            row["value_fill"].width = fill_width
            row["value_fill"].pos = (self.value_left + fill_width * 0.5, row["value_fill"].pos[1])
            row["value_fill"].fillColor = value_color_norm(value, vmin, vmax)
            importance_width = max(0.001, importance_span * importance_fraction)
            row["importance_fill"].width = importance_width
            row["importance_fill"].pos = (
                self.importance_left + importance_width * 0.5,
                row["importance_fill"].pos[1],
            )
            row["importance_fill"].fillColor = importance_color_norm(importance, self.max_importance)
            row["value_text"].text = f"{value:+.2f}"
        self.has_data = True

    def draw(self):
        self.frame.draw()
        self.title.draw()
        if not self.has_data:
            self.placeholder.draw()
            return
        for header in self.header_labels:
            header.draw()
        for row in self.rows:
            row["band"].draw()
            if row["type"] == "group":
                row["label"].draw()
            else:
                row["label"].draw()
                row["value_bg"].draw()
                row["value_fill"].draw()
                row["importance_bg"].draw()
                row["importance_fill"].draw()
                row["value_text"].draw()


class InferenceDashboard:
    def __init__(self, win, channel_names, bundle):
        self.win = win
        self.prediction_label = "..."
        self.prediction_color = "#f0f0f0"
        self.detail_text = "Streaming..."
        self.eeg_panel = EegPanel(win, channel_names)
        self.heatmap_panel = FeatureHeatmapPanel(win, bundle)
        self.title_stim = visual.TextStim(
            win,
            text="EYES OPEN / CLOSED",
            height=0.05,
            color="white",
            pos=(0.0, 0.43),
            wrapWidth=1.9,
            alignText="center",
            units="norm",
        )
        self.detail_stim = visual.TextStim(
            win,
            text="",
            height=0.026,
            color="#d0d0d0",
            pos=(0.0, 0.36),
            wrapWidth=1.9,
            alignText="center",
            units="norm",
        )
        self.footer_stim = visual.TextStim(
            win,
            text="Press ESC to stop.",
            height=0.022,
            color="#888888",
            pos=(0.0, -0.47),
            units="norm",
        )

    def set_status(self, title, color, detail):
        self.prediction_label = title
        self.prediction_color = color
        self.detail_text = detail

    def set_prediction(self, prediction_row):
        label = str(prediction_row["prediction"])
        display = DISPLAY.get(label, {"title": label.upper(), "color": "white", "detail": ""})
        closed_prob = float(prediction_row.get("probability_closed_eyes", 0.0))
        open_prob = float(prediction_row.get("probability_open_eyes", 0.0))
        blink_prob = float(prediction_row.get("probability_strong_blinks", 0.0))
        self.set_status(
            display["title"],
            display["color"],
            (
                f"{display['detail']}   "
                f"closed {closed_prob:.2f}   open {open_prob:.2f}   blink {blink_prob:.2f}"
            ),
        )

    def update_eeg(self, channel_data):
        self.eeg_panel.update(channel_data)

    def update_heatmap(self, feature_frame):
        self.heatmap_panel.update(feature_frame)

    def draw(self):
        self.title_stim.text = self.prediction_label
        self.title_stim.color = self.prediction_color
        self.detail_stim.text = self.detail_text
        self.eeg_panel.draw()
        self.heatmap_panel.draw()
        self.title_stim.draw()
        self.detail_stim.draw()
        self.footer_stim.draw()
        self.win.flip()


def main():
    model_path = selected_model_path()
    bundle = load_model_bundle(model_path)
    window_seconds = float(bundle.get("window_seconds", WINDOW_SECONDS))
    output_dir = make_output_dir()
    predictions = []
    win = make_window("#0b0d12")
    estimate_refresh_rate(win)
    try:
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            sample_rate = float(recorder.sample_frequency)
            window_samples = int(round(window_seconds * sample_rate))
            channel_labels = [label for label in recorder.channel_labels if label.endswith("_uV")]
            channel_names = display_channel_names(channel_labels)
            row_indices = [recorder.channel_indices[label] for label in channel_labels]
            dashboard = InferenceDashboard(win, channel_names, bundle)
            smoother = PredictionSmoother(bundle["class_labels"], PREDICTION_SMOOTH_SECONDS)
            consumed_chunks = 0
            settle_start = time.perf_counter()
            while time.perf_counter() - settle_start < SETTLE_SECONDS:
                consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
                remaining = max(0.0, SETTLE_SECONDS - (time.perf_counter() - settle_start))
                dashboard.set_status(
                    "EYES OPEN / CLOSED",
                    "#f0f0f0",
                    f"Streaming EEG... live prediction starts in {remaining:.1f}s",
                )
                dashboard.draw()
                if "escape" in event.getKeys():
                    break
                time.sleep(SCREEN_REFRESH_SLEEP_SECONDS)
            buffer = np.zeros((len(channel_labels), 0), dtype=float)
            last_prediction_time = 0.0
            last_eeg_draw_time = 0.0
            started_at = time.perf_counter()
            while True:
                consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
                if new_samples is not None:
                    buffer = np.concatenate([buffer, new_samples], axis=1)
                    buffer = buffer[:, -window_samples:]
                now = time.perf_counter()
                if buffer.shape[1] >= 2 and now - last_eeg_draw_time >= EEG_DRAW_SECONDS:
                    dashboard.update_eeg(buffer)
                    last_eeg_draw_time = now
                if buffer.shape[1] >= window_samples and now - last_prediction_time >= UPDATE_SECONDS:
                    segment_df = pd.DataFrame(buffer[:, -window_samples:].T, columns=channel_labels)
                    feature_frame = feature_frame_from_segment(segment_df, sample_rate, bundle)
                    if feature_frame is not None:
                        raw_prediction = predict_bundle(bundle, feature_frame).iloc[0]
                        smoothed = smoother.update(raw_prediction, now)
                        row = {
                            "pc_time_perf_counter_s": now,
                            "t_from_inference_start_s": now - started_at,
                        }
                        row.update(smoothed)
                        predictions.append(row)
                        dashboard.set_prediction(pd.Series(smoothed))
                        dashboard.update_heatmap(feature_frame)
                        last_prediction_time = now
                dashboard.draw()
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
            "window_seconds": window_seconds,
            "update_seconds": UPDATE_SECONDS,
            "settle_seconds": SETTLE_SECONDS,
            "prediction_smooth_seconds": PREDICTION_SMOOTH_SECONDS,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print("Saved predictions:", predictions_path)
        print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
