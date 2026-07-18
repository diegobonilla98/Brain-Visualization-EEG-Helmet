from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from brain_state_common import MODEL_ROOT, latest_pointer, load_joblib


MODEL_DIR = MODEL_ROOT
EMBEDDINGS_PATH = ""
MAP_PATH = ""
RECORDING_FILTER = ""
REPLAY_SPEED = 1.0
TAIL_SECONDS = 18.0
LOOP = True
BALL_SIZE = 125


def selected_paths():
    embeddings_path = Path(EMBEDDINGS_PATH) if len(str(EMBEDDINGS_PATH).strip()) > 0 else latest_pointer(MODEL_DIR, "latest_embeddings_path.txt", "brain_state_embeddings_*.csv")
    map_path = Path(MAP_PATH) if len(str(MAP_PATH).strip()) > 0 else latest_pointer(MODEL_DIR, "latest_map_path.txt", "brain_state_map_*.joblib")
    return embeddings_path, map_path


def main():
    embeddings_path, map_path = selected_paths()
    map_bundle = load_joblib(map_path)
    frame = pd.read_csv(embeddings_path)
    recordings = frame["recording"].drop_duplicates().tolist()
    if len(str(RECORDING_FILTER).strip()) > 0:
        matches = [recording for recording in recordings if str(RECORDING_FILTER).lower() in recording.lower()]
        if len(matches) == 0:
            raise ValueError(f"No recording matched {RECORDING_FILTER}")
        selected_recording = matches[-1]
    else:
        selected_recording = recordings[-1]
    frame = frame[frame["recording"] == selected_recording].reset_index(drop=True)
    dimensions = int(map_bundle["dimensions"])
    coordinate_columns = [f"state_{index + 1}" for index in range(dimensions)]
    coordinates = frame[coordinate_columns].to_numpy(dtype=float)
    hues = frame["hue"].to_numpy(dtype=float)
    step_seconds = float(map_bundle["step_seconds"])
    interval_ms = max(1, int(round(step_seconds * 1000.0 / max(REPLAY_SPEED, 1e-6))))
    tail_count = max(3, int(round(TAIL_SECONDS / step_seconds)))
    tail = deque(maxlen=tail_count)
    tail_hues = deque(maxlen=tail_count)
    figure = plt.figure(figsize=(12, 9), facecolor="#050812")
    if dimensions == 3:
        axis = figure.add_subplot(111, projection="3d")
        tail_artist = Line3DCollection([], linewidths=3.0, alpha=0.78)
        axis.add_collection3d(tail_artist)
        glow = [axis.scatter([], [], [], s=size, alpha=alpha, depthshade=False) for size, alpha in [(BALL_SIZE * 3.5, 0.08), (BALL_SIZE * 2.0, 0.16), (BALL_SIZE, 0.95)]]
    else:
        axis = figure.add_subplot(111)
        tail_artist = LineCollection([], linewidths=3.0, alpha=0.78)
        axis.add_collection(tail_artist)
        glow = [axis.scatter([], [], s=size, alpha=alpha) for size, alpha in [(BALL_SIZE * 3.5, 0.08), (BALL_SIZE * 2.0, 0.16), (BALL_SIZE, 0.95)]]
    axis.set_facecolor("#050812")
    axis.set_xlabel("state 1", color="#aab7d1")
    axis.set_ylabel("state 2", color="#aab7d1")
    axis.tick_params(colors="#74829e")
    axis.grid(True, alpha=0.12)
    labels = ["x", "y", "z"]
    for dimension in range(dimensions):
        low, high = map_bundle["bounds"][dimension]
        padding = max(1e-6, (high - low) * 0.10)
        getattr(axis, f"set_{labels[dimension]}lim")(low - padding, high + padding)
    if dimensions == 3:
        axis.set_zlabel("state 3", color="#aab7d1")
        axis.xaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        axis.yaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        axis.zaxis.pane.set_facecolor((0.02, 0.03, 0.07, 1.0))
        status = axis.text2D(0.02, 0.96, "", transform=axis.transAxes, color="#b8c6df", fontsize=11)
    else:
        status = axis.text(0.02, 0.96, "", transform=axis.transAxes, color="#b8c6df", fontsize=11)
    axis.set_title("Recorded EEG state replay", color="white", fontsize=18, pad=18)
    print("Replaying:", selected_recording)

    def update(frame_index):
        index = frame_index % len(coordinates)
        if index == 0 and len(tail) > 0:
            tail.clear()
            tail_hues.clear()
        coordinate = coordinates[index]
        hue = hues[index]
        tail.append(coordinate)
        tail_hues.append(hue)
        points = np.asarray(tail)
        colors = plt.cm.hsv(np.asarray(tail_hues))
        if len(points) > 1:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            segment_colors = colors[:-1].copy()
            segment_colors[:, 3] = np.linspace(0.08, 0.90, len(segments))
            tail_artist.set_segments(segments)
            tail_artist.set_color(segment_colors)
        color = plt.cm.hsv(hue)
        for artist in glow:
            artist.set_facecolor(color)
            artist.set_edgecolor("none")
            if dimensions == 3:
                artist._offsets3d = ([coordinate[0]], [coordinate[1]], [coordinate[2]])
            else:
                artist.set_offsets(coordinate[:2][None])
        status.set_text(f"t = {frame.loc[index, 'time_s']:7.1f}s   hue = {hue:0.2f}   speed = {REPLAY_SPEED:0.1f}x")
        return []

    frame_count = None if LOOP else len(coordinates)
    animation = FuncAnimation(figure, update, frames=frame_count, interval=interval_ms, blit=False, cache_frame_data=False)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
