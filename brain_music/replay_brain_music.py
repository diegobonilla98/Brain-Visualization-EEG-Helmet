import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from brain_music_common import DEFAULT_MODEL_ROOT, DEFAULT_SAMPLE_ROOT, BrainMusicComposer, PianoSampleLibrary, PolyphonicPianoEngine, latest_pointer, load_joblib, make_session_dir
from brain_music_visualizer import BrainMusicDashboard


MODEL_ROOT = DEFAULT_MODEL_ROOT
SAMPLE_ROOT = DEFAULT_SAMPLE_ROOT
MAP_PATH = ""
EMBEDDINGS_PATH = ""
RECORDING_FILTER = ""
REPLAY_SPEED = 1.0
LOOP = True
AUDIO_ENABLED = True
TAIL_SECONDS = 20.0
FRAME_INTERVAL_MS = 33


def selected_paths():
    map_path = Path(MAP_PATH) if len(str(MAP_PATH).strip()) > 0 else latest_pointer(MODEL_ROOT, "latest_map_path.txt", "brain_state_map_*.joblib")
    embeddings_path = Path(EMBEDDINGS_PATH) if len(str(EMBEDDINGS_PATH).strip()) > 0 else latest_pointer(MODEL_ROOT, "latest_embeddings_path.txt", "brain_state_embeddings_*.csv")
    return map_path, embeddings_path


def main():
    map_path, embeddings_path = selected_paths()
    map_bundle = load_joblib(map_path)
    dimensions = int(map_bundle["dimensions"])
    state_columns = [f"state_{index + 1}" for index in range(dimensions)]
    frame = pd.read_csv(embeddings_path, usecols=["recording", "time_s", "hue"] + state_columns)
    recordings = frame["recording"].drop_duplicates().tolist()
    if len(str(RECORDING_FILTER).strip()) > 0:
        matches = [recording for recording in recordings if str(RECORDING_FILTER).lower() in recording.lower()]
        if len(matches) == 0:
            raise ValueError(f"No recording matched {RECORDING_FILTER}")
        recording = matches[-1]
    else:
        recording = recordings[-1]
    replay_frame = frame[frame["recording"] == recording].reset_index(drop=True)
    coordinates = replay_frame[state_columns].to_numpy(dtype=float)
    hues = replay_frame["hue"].to_numpy(dtype=float)
    context_coordinates = frame[state_columns].to_numpy(dtype=float)
    library = PianoSampleLibrary(SAMPLE_ROOT)
    audio = PolyphonicPianoEngine(library, enabled=AUDIO_ENABLED)
    composer = BrainMusicComposer(audio, map_bundle["bounds"], float(map_bundle["step_seconds"]))
    dashboard = BrainMusicDashboard(map_bundle, context_coordinates, TAIL_SECONDS, float(map_bundle["step_seconds"]))
    output_dir = make_session_dir("replay")
    started_at = time.perf_counter()
    state = {"last_index": -1, "cycle": 0}
    audio.start()
    print("Replaying:", recording)

    def update(_):
        now = time.perf_counter()
        elapsed = (now - started_at) * REPLAY_SPEED
        raw_index = int(elapsed / float(map_bundle["step_seconds"]))
        if LOOP:
            index = raw_index % len(coordinates)
            cycle = raw_index // len(coordinates)
        else:
            index = min(raw_index, len(coordinates) - 1)
            cycle = 0
        new_state = index != state["last_index"] or cycle != state["cycle"]
        if new_state:
            composer.update_brain_state(coordinates[index], hues[index])
            state["last_index"] = index
            state["cycle"] = cycle
        composer.tick(now)
        return dashboard.update(coordinates[index], hues[index], composer, now, new_state)

    animation = FuncAnimation(dashboard.figure, update, interval=FRAME_INTERVAL_MS, blit=False, cache_frame_data=False)
    dashboard.figure.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    audio.stop()
    metadata = {
        "mode": "recording_replay",
        "recording": recording,
        "map_path": str(map_path),
        "embeddings_path": str(embeddings_path),
        "sample_root": str(SAMPLE_ROOT),
        "replay_speed": REPLAY_SPEED,
        "audio_enabled": AUDIO_ENABLED,
        "event_count": len(composer.events),
    }
    events_path, metadata_path = composer.save_events(output_dir, metadata)
    print("Saved musical events:", events_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
