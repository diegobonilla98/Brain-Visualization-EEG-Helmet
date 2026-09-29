import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import serial
from psychopy import event, visual
from serial.tools import list_ports


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eeg_quality_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainaccess_stream import BrainAccessStream
from jedi_common import feature_frame_from_segment, latest_model_path, load_model, predict_will_probability
from stimulus_common import estimate_refresh_rate, make_window


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_DIR = "jedi_lamp_control/models"
MODEL_PATH = ""
SERIAL_PORT = ""
SERIAL_BAUD = 115200
RELAY_ENABLED = True
TURN_LAMP_OFF_ON_EXIT = True
OUTPUT_ROOT = PROJECT_ROOT / "jedi_lamp_control" / "live_runs"
SETTLE_SECONDS = 5.0
NOTHING_CALIBRATION_SECONDS = 20.0
UPDATE_SECONDS = 0.25
PROBABILITY_SMOOTH_SECONDS = 0.55
USE_MANUAL_ACTIVATION_THRESHOLD = True
MANUAL_ACTIVATION_THRESHOLD = 0.55
MINIMUM_ACTIVATION_THRESHOLD = 0.70
CALIBRATION_THRESHOLD_MARGIN = 0.12
ACTIVATION_HISTORY = 6
ACTIVATION_AGREEMENT = 4
RELEASE_HISTORY = 8
RELEASE_AGREEMENT = 6
RELEASE_THRESHOLD = 0.35
REFRACTORY_SECONDS = 2.0
SCREEN_LOOP_SLEEP_SECONDS = 0.01


def selected_model_path():
    return Path(MODEL_PATH) if str(MODEL_PATH).strip() else latest_model_path(MODEL_DIR)


def resolve_activation_threshold(bundle, calibration_probabilities):
    if USE_MANUAL_ACTIVATION_THRESHOLD:
        return float(MANUAL_ACTIVATION_THRESHOLD)
    learned_threshold = float(bundle["activation_threshold"])
    session_threshold = float(np.quantile(calibration_probabilities, 0.95) + CALIBRATION_THRESHOLD_MARGIN) if calibration_probabilities else learned_threshold
    return min(0.95, max(MINIMUM_ACTIVATION_THRESHOLD, learned_threshold, session_threshold))


def selected_serial_port():
    if str(SERIAL_PORT).strip():
        return SERIAL_PORT
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports were found. Connect the Arduino or set SERIAL_PORT.")
    preferred_terms = ["arduino", "ch340", "usb serial", "wch", "cp210"]
    preferred = [port for port in ports if any(term in f"{port.description} {port.manufacturer}".lower() for term in preferred_terms)]
    return (preferred or ports)[0].device


class RelayController:
    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.connection = None
        self.port = "disabled"
        self.is_on = False

    def connect(self):
        if not self.enabled:
            return
        self.port = selected_serial_port()
        self.connection = serial.Serial(self.port, SERIAL_BAUD, timeout=0.15, write_timeout=0.5)
        time.sleep(2.0)
        self.set_state(False)

    def set_state(self, is_on):
        command = b"Y" if is_on else b"N"
        if self.connection is not None:
            self.connection.write(command)
            self.connection.flush()
        self.is_on = bool(is_on)

    def toggle(self):
        self.set_state(not self.is_on)
        return self.is_on

    def read_reply(self):
        if self.connection is None or self.connection.in_waiting <= 0:
            return ""
        return self.connection.readline().decode("utf-8", errors="replace").strip()

    def close(self):
        if self.connection is not None:
            if TURN_LAMP_OFF_ON_EXIT:
                self.set_state(False)
            self.connection.close()
            self.connection = None


class WillGate:
    def __init__(self, activation_threshold):
        self.activation_threshold = float(activation_threshold)
        self.alpha = 1.0 - math.exp(-UPDATE_SECONDS / PROBABILITY_SMOOTH_SECONDS)
        self.smoothed = None
        self.activation_votes = deque(maxlen=ACTIVATION_HISTORY)
        self.release_votes = deque(maxlen=RELEASE_HISTORY)
        self.armed = True
        self.last_toggle_time = -np.inf

    def update(self, probability, now):
        if self.smoothed is None:
            self.smoothed = float(probability)
        else:
            self.smoothed += self.alpha * (float(probability) - self.smoothed)
        self.activation_votes.append(float(probability) >= self.activation_threshold)
        self.release_votes.append(float(probability) <= RELEASE_THRESHOLD)
        activation_count = int(sum(self.activation_votes))
        release_count = int(sum(self.release_votes))
        toggled = False
        if self.armed and now - self.last_toggle_time >= REFRACTORY_SECONDS:
            if len(self.activation_votes) == ACTIVATION_HISTORY and activation_count >= ACTIVATION_AGREEMENT and self.smoothed >= self.activation_threshold:
                toggled = True
                self.armed = False
                self.last_toggle_time = now
                self.release_votes.clear()
        elif not self.armed and now - self.last_toggle_time >= REFRACTORY_SECONDS:
            if len(self.release_votes) == RELEASE_HISTORY and release_count >= RELEASE_AGREEMENT and self.smoothed <= RELEASE_THRESHOLD:
                self.armed = True
                self.activation_votes.clear()
        return {
            "raw_probability": float(probability),
            "smoothed_probability": float(self.smoothed),
            "activation_votes": activation_count,
            "activation_vote_capacity": ACTIVATION_HISTORY,
            "release_votes": release_count,
            "release_vote_capacity": RELEASE_HISTORY,
            "armed": self.armed,
            "toggled": toggled,
            "refractory_remaining_s": max(0.0, REFRACTORY_SECONDS - (now - self.last_toggle_time)),
        }


def read_new_samples(recorder, consumed_chunks, row_indices):
    with recorder.lock:
        chunks = recorder.chunks[consumed_chunks:]
        consumed_chunks = len(recorder.chunks)
    rows = []
    for _, array, chunk_size in chunks:
        count = min(int(chunk_size), array.shape[1])
        if count > 0:
            rows.append(array[row_indices, :count])
    return consumed_chunks, np.concatenate(rows, axis=1) if rows else None


def draw_screen(window, title_stim, detail_stim, meter, title, detail, probability, color):
    title_stim.text = title
    title_stim.color = color
    detail_stim.text = detail
    meter.width = max(0.001, min(0.90, 0.90 * float(probability)))
    meter.fillColor = color
    meter.lineColor = color
    title_stim.draw()
    detail_stim.draw()
    meter.draw()
    window.flip()


def probability_from_buffer(buffer, channel_labels, sample_rate, bundle, window_samples):
    segment = pd.DataFrame(buffer[:, -window_samples:].T, columns=channel_labels)
    features = feature_frame_from_segment(segment, sample_rate, bundle["feature_names"])
    return predict_will_probability(bundle, features)


def make_output_dir():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"jedi_live_{time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main():
    model_path = selected_model_path()
    bundle = load_model(model_path)
    output_dir = make_output_dir()
    relay = RelayController(RELAY_ENABLED)
    predictions = []
    toggles = []
    activation_threshold = None
    window = make_window("#050812")
    estimate_refresh_rate(window)
    title_stim = visual.TextStim(window, text="", height=0.090, color="white", pos=(0.0, 0.19), wrapWidth=1.55, alignText="center")
    detail_stim = visual.TextStim(window, text="", height=0.040, color="#cbd5e1", pos=(0.0, -0.06), wrapWidth=1.55, alignText="center")
    meter = visual.Rect(window, width=0.001, height=0.035, pos=(-0.45, -0.38), anchor="center-left", fillColor="#64748b", lineColor="#64748b")
    try:
        relay.connect()
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            sample_rate = float(recorder.sample_frequency)
            window_seconds = float(bundle.get("training_settings", {}).get("window_seconds", 3.0))
            window_samples = int(round(window_seconds * sample_rate))
            channel_labels = [label for label in recorder.channel_labels if label.endswith("_uV")]
            row_indices = [recorder.channel_indices[label] for label in channel_labels]
            consumed_chunks = 0
            buffer = np.zeros((len(channel_labels), 0), dtype=float)
            settle_start = time.perf_counter()
            while time.perf_counter() - settle_start < SETTLE_SECONDS:
                consumed_chunks, samples = read_new_samples(recorder, consumed_chunks, row_indices)
                remaining = SETTLE_SECONDS - (time.perf_counter() - settle_start)
                draw_screen(window, title_stim, detail_stim, meter, "CONNECTING", f"Relay: {relay.port}\nEEG settling: {remaining:.1f}s", 0.0, "#94a3b8")
                if "escape" in event.getKeys():
                    return
                time.sleep(SCREEN_LOOP_SLEEP_SECONDS)
            calibration_probabilities = []
            calibration_start = time.perf_counter()
            last_prediction = 0.0
            while time.perf_counter() - calibration_start < NOTHING_CALIBRATION_SECONDS:
                consumed_chunks, samples = read_new_samples(recorder, consumed_chunks, row_indices)
                if samples is not None:
                    buffer = np.concatenate([buffer, samples], axis=1)[:, -window_samples:]
                now = time.perf_counter()
                if buffer.shape[1] >= window_samples and now - last_prediction >= UPDATE_SECONDS:
                    calibration_probabilities.append(probability_from_buffer(buffer, channel_labels, sample_rate, bundle, window_samples))
                    last_prediction = now
                remaining = NOTHING_CALIBRATION_SECONDS - (now - calibration_start)
                current = calibration_probabilities[-1] if calibration_probabilities else 0.0
                draw_screen(window, title_stim, detail_stim, meter, "DO NOTHING", f"Live false-trigger calibration: {remaining:.1f}s\nThink, speak, or behave normally. Do not perform the Jedi ritual.", current, "#cbd5e1")
                if "escape" in event.getKeys():
                    return
                time.sleep(SCREEN_LOOP_SLEEP_SECONDS)
            activation_threshold = resolve_activation_threshold(bundle, calibration_probabilities)
            gate = WillGate(activation_threshold)
            started_at = time.perf_counter()
            last_prediction = 0.0
            while True:
                consumed_chunks, samples = read_new_samples(recorder, consumed_chunks, row_indices)
                if samples is not None:
                    buffer = np.concatenate([buffer, samples], axis=1)[:, -window_samples:]
                now = time.perf_counter()
                if buffer.shape[1] >= window_samples and now - last_prediction >= UPDATE_SECONDS:
                    probability = probability_from_buffer(buffer, channel_labels, sample_rate, bundle, window_samples)
                    gate_state = gate.update(probability, now)
                    if gate_state["toggled"]:
                        lamp_on = relay.toggle()
                        toggles.append({"pc_time_perf_counter_s": now, "t_from_start_s": now - started_at, "lamp_on": lamp_on, "probability": probability, "smoothed_probability": gate_state["smoothed_probability"]})
                    reply = relay.read_reply()
                    lamp_text = "LAMP ON" if relay.is_on else "LAMP OFF"
                    if gate_state["armed"]:
                        if gate_state["activation_votes"] > 0:
                            title = "HOLD THE WILL"
                            color = "#fbbf24"
                        else:
                            title = "ARMED · DO NOTHING"
                            color = "#60a5fa"
                        instruction = f"{lamp_text}\nWill: {gate_state['smoothed_probability']:.0%} · agreement {gate_state['activation_votes']}/{ACTIVATION_HISTORY} · threshold {activation_threshold:.0%}"
                    else:
                        title = "RELEASE · RETURN TO NOTHING"
                        color = "#22c55e" if gate_state["refractory_remaining_s"] <= 0 else "#a78bfa"
                        instruction = f"{lamp_text}\nRelease agreement {gate_state['release_votes']}/{RELEASE_HISTORY} · refractory {gate_state['refractory_remaining_s']:.1f}s"
                    draw_screen(window, title_stim, detail_stim, meter, title, instruction + "\n\nESC safely turns the lamp off and exits.", gate_state["smoothed_probability"], color)
                    predictions.append({"pc_time_perf_counter_s": now, "t_from_start_s": now - started_at, "lamp_on": relay.is_on, "activation_threshold": activation_threshold, "arduino_reply": reply, **gate_state})
                    last_prediction = now
                if "escape" in event.getKeys():
                    break
                time.sleep(SCREEN_LOOP_SLEEP_SECONDS)
            recorder.stop()
    finally:
        relay.close()
        window.close()
        pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False)
        pd.DataFrame(toggles).to_csv(output_dir / "lamp_toggles.csv", index=False)
        metadata = {
            "model_path": str(model_path),
            "relay_enabled": RELAY_ENABLED,
            "serial_port": relay.port,
            "serial_baud": SERIAL_BAUD,
            "settle_seconds": SETTLE_SECONDS,
            "nothing_calibration_seconds": NOTHING_CALIBRATION_SECONDS,
            "update_seconds": UPDATE_SECONDS,
            "activation_threshold_mode": "manual" if USE_MANUAL_ACTIVATION_THRESHOLD else "learned",
            "activation_threshold": activation_threshold,
            "manual_activation_threshold": MANUAL_ACTIVATION_THRESHOLD,
            "minimum_activation_threshold": MINIMUM_ACTIVATION_THRESHOLD,
            "calibration_threshold_margin": CALIBRATION_THRESHOLD_MARGIN,
            "anti_jumping": {
                "activation_history": ACTIVATION_HISTORY,
                "activation_agreement": ACTIVATION_AGREEMENT,
                "release_history": RELEASE_HISTORY,
                "release_agreement": RELEASE_AGREEMENT,
                "release_threshold": RELEASE_THRESHOLD,
                "refractory_seconds": REFRACTORY_SECONDS,
            },
            "prediction_rows": len(predictions),
            "toggle_count": len(toggles),
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print("Saved live run:", output_dir)


if __name__ == "__main__":
    main()
