import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from brain_music_common import CHANNEL_NAMES, DEFAULT_MODEL_ROOT, DEFAULT_SAMPLE_ROOT, BrainMusicComposer, BrainStateProjector, PianoSampleLibrary, PolyphonicPianoEngine, latest_pointer, make_session_dir
from brain_music_visualizer import BrainMusicDashboard
from brainaccess_stream import BrainAccessStream
from brain_state_common import CausalEEGPreprocessor


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
MODEL_ROOT = DEFAULT_MODEL_ROOT
SAMPLE_ROOT = DEFAULT_SAMPLE_ROOT
MODEL_PATH = ""
MAP_PATH = ""
DEVICE = "auto"
BASELINE_SECONDS = 30.0
BASELINE_INSTRUCTION = "CLOSE EYES - RELAX FACE, JAW, NECK, HANDS AND FEET"
INFERENCE_STEP_SECONDS = 0.25
TAIL_SECONDS = 20.0
FRAME_INTERVAL_MS = 33
AUDIO_ENABLED = True
SHOW_TRAINING_MANIFOLD = True


def read_new_samples(recorder, consumed_chunks, row_indices):
    with recorder.lock:
        chunks = recorder.chunks[consumed_chunks:]
        consumed_chunks = len(recorder.chunks)
    rows = []
    for _, array, chunk_size in chunks:
        count = min(int(chunk_size), array.shape[1])
        if count > 0:
            rows.append(array[row_indices, :count])
    if len(rows) == 0:
        return consumed_chunks, None
    return consumed_chunks, np.concatenate(rows, axis=1)


def context_coordinates(model_root, map_bundle):
    if not SHOW_TRAINING_MANIFOLD:
        return None
    embeddings_path = latest_pointer(model_root, "latest_embeddings_path.txt", "brain_state_embeddings_*.csv")
    columns = [f"state_{index + 1}" for index in range(int(map_bundle["dimensions"]))]
    return pd.read_csv(embeddings_path, usecols=columns).to_numpy(dtype=float)


def main():
    projector = BrainStateProjector(MODEL_ROOT, MODEL_PATH, MAP_PATH, DEVICE, INFERENCE_STEP_SECONDS)
    library = PianoSampleLibrary(SAMPLE_ROOT)
    audio = PolyphonicPianoEngine(library, enabled=AUDIO_ENABLED)
    composer = BrainMusicComposer(audio, projector.map_bundle["bounds"], INFERENCE_STEP_SECONDS)
    training_context = context_coordinates(MODEL_ROOT, projector.map_bundle)
    output_dir = make_session_dir("live")
    states = []
    consumed_chunks = 0
    filtered_buffer = np.zeros((len(CHANNEL_NAMES), 0), dtype=np.float32)
    calibration_buffer = np.zeros((len(CHANNEL_NAMES), 0), dtype=np.float32)
    session_center = None
    session_scale = None
    samples_since_inference = 0
    current_coordinate = np.mean(np.asarray(projector.map_bundle["bounds"], dtype=float), axis=1)
    current_hue = 0.55
    state_ready = False
    started_at = time.perf_counter()
    with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
        recorder.start()
        labels = [f"{name}_uV" for name in CHANNEL_NAMES]
        row_indices = [recorder.channel_indices[label] for label in labels]
        preprocessor = CausalEEGPreprocessor(
            recorder.sample_frequency,
            float(projector.config["sample_rate"]),
            len(CHANNEL_NAMES),
            float(projector.config["low_hz"]),
            float(projector.config["high_hz"]),
            float(projector.config["notch_hz"]),
        )
        dashboard = BrainMusicDashboard(projector.map_bundle, training_context, TAIL_SECONDS, INFERENCE_STEP_SECONDS)
        status = dashboard.state_axis.text2D(0.02, 0.96, "Preparing helmet...", transform=dashboard.state_axis.transAxes, color="#8ce9ff", fontsize=10) if int(projector.map_bundle["dimensions"]) == 3 else dashboard.state_axis.text(0.02, 0.96, "Preparing helmet...", transform=dashboard.state_axis.transAxes, color="#8ce9ff", fontsize=10)
        audio.start()
        print("Baseline calibration:")
        print("Close your eyes and relax your face, jaw, tongue, neck, shoulders, hands, and feet.")
        print("Remain still and breathe naturally. Music starts after calibration.")

        def update(_):
            nonlocal consumed_chunks, filtered_buffer, calibration_buffer, session_center, session_scale, samples_since_inference, current_coordinate, current_hue, state_ready
            now = time.perf_counter()
            consumed_chunks, new_samples = read_new_samples(recorder, consumed_chunks, row_indices)
            if new_samples is not None:
                filtered = preprocessor.process(new_samples)
                filtered_buffer = np.concatenate([filtered_buffer, filtered], axis=1)[:, -projector.window_samples:]
                if session_center is None:
                    calibration_buffer = np.concatenate([calibration_buffer, filtered], axis=1)
                samples_since_inference += filtered.shape[1]
            baseline_samples = int(round(BASELINE_SECONDS * float(projector.config["sample_rate"])))
            if session_center is None and calibration_buffer.shape[1] >= baseline_samples:
                calibration = calibration_buffer[:, :baseline_samples]
                session_center = np.median(calibration, axis=1).astype(np.float32)
                session_scale = (1.4826 * np.median(np.abs(calibration - session_center[:, None]), axis=1)).astype(np.float32)
                positive = session_scale[session_scale > 1e-8]
                fallback = float(np.median(positive)) if len(positive) > 0 else 1.0
                session_scale[session_scale <= 1e-8] = fallback
                calibration_buffer = np.zeros((len(CHANNEL_NAMES), 0), dtype=np.float32)
                samples_since_inference = 0
            if session_center is None:
                remaining = max(0.0, (baseline_samples - calibration_buffer.shape[1]) / float(projector.config["sample_rate"]))
                status.set_text(f"{BASELINE_INSTRUCTION}\nCALIBRATING - {remaining:04.1f}s - MUSIC MUTED")
                return []
            required_step = int(round(INFERENCE_STEP_SECONDS * float(projector.config["sample_rate"])))
            new_state = False
            if filtered_buffer.shape[1] >= projector.window_samples and samples_since_inference >= required_step:
                embedding, current_coordinate, current_hue = projector.encode(filtered_buffer, session_center, session_scale)
                composer.update_brain_state(current_coordinate, current_hue, embedding)
                elapsed = now - started_at
                row = {"pc_time_perf_counter_s": now, "t_from_start_s": elapsed, "hue": current_hue}
                for dimension, value in enumerate(current_coordinate):
                    row[f"state_{dimension + 1}"] = float(value)
                for dimension, value in enumerate(embedding):
                    row[f"embedding_{dimension + 1:03d}"] = float(value)
                if projector.last_scale_disagreement is not None:
                    row["multiwindow_disagreement"] = projector.last_scale_disagreement
                    for scale_index, value in enumerate(projector.last_scale_weights):
                        row[f"multiwindow_weight_{scale_index + 1}"] = float(value)
                    row["channel_reliability_mean"] = float(np.mean(projector.last_channel_reliability))
                    row["channel_reliability_min"] = float(np.min(projector.last_channel_reliability))
                states.append(row)
                samples_since_inference = 0
                state_ready = True
                new_state = True
            if state_ready:
                composer.tick(now)
                status.set_text(f"LIVE - {composer.bpm:03.0f} BPM - {len(composer.events):04d} NOTES - {projector.device}")
                return dashboard.update(current_coordinate, current_hue, composer, now, new_state)
            status.set_text("CALIBRATED - OPEN EYES WHEN READY\nCOLLECTING FIRST 4-SECOND STATE - MUSIC MUTED")
            return []

        animation = FuncAnimation(dashboard.figure, update, interval=FRAME_INTERVAL_MS, blit=False, cache_frame_data=False)
        dashboard.figure.tight_layout(rect=(0, 0, 1, 0.97))
        plt.show()
    audio.stop()
    states_path = output_dir / "brain_states.csv"
    pd.DataFrame(states).to_csv(states_path, index=False)
    metadata = {
        "mode": "live_brain_music",
        "device_name": DEVICE_NAME,
        "model_path": str(projector.model_path),
        "map_path": str(projector.map_path),
        "sample_root": str(SAMPLE_ROOT),
        "baseline_seconds": BASELINE_SECONDS,
        "inference_step_seconds": INFERENCE_STEP_SECONDS,
        "audio_enabled": AUDIO_ENABLED,
        "state_count": len(states),
        "event_count": len(composer.events),
    }
    events_path, metadata_path = composer.save_events(output_dir, metadata)
    print("Saved brain states:", states_path)
    print("Saved musical events:", events_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
