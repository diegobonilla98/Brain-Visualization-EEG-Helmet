import math
import runpy
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from brain_music_common import NOTE_NAMES


class BrainMusicDashboard:
    def __init__(self, map_bundle, context_coordinates=None, tail_seconds=20.0, step_seconds=0.25):
        self.map_bundle = map_bundle
        self.dimensions = int(map_bundle["dimensions"])
        self.tail = deque(maxlen=max(4, int(round(tail_seconds / step_seconds))))
        self.tail_hues = deque(maxlen=self.tail.maxlen)
        self.last_coordinate = None
        self.figure = plt.figure(figsize=(16, 10), facecolor="#030611")
        grid = self.figure.add_gridspec(4, 5, width_ratios=[1.0, 1.0, 1.0, 0.80, 0.80], height_ratios=[1.0, 1.0, 0.30, 0.36], hspace=0.12, wspace=0.10)
        if self.dimensions == 3:
            self.state_axis = self.figure.add_subplot(grid[:3, :4], projection="3d")
            self.tail_artist = Line3DCollection([np.zeros((2, 3), dtype=float)], linewidths=3.0, alpha=0.86)
            self.state_axis.add_collection3d(self.tail_artist)
            self.glow = [self.state_axis.scatter([], [], [], s=size, alpha=alpha, depthshade=False) for size, alpha in [(800, 0.05), (430, 0.10), (220, 0.22), (95, 1.0)]]
            self.orbit_lines = [self.state_axis.plot([], [], [], linewidth=1.2, alpha=0.45)[0] for _ in range(3)]
        else:
            self.state_axis = self.figure.add_subplot(grid[:3, :4])
            self.tail_artist = LineCollection([], linewidths=3.0, alpha=0.86)
            self.state_axis.add_collection(self.tail_artist)
            self.glow = [self.state_axis.scatter([], [], s=size, alpha=alpha) for size, alpha in [(800, 0.05), (430, 0.10), (220, 0.22), (95, 1.0)]]
            self.orbit_lines = []
        self.info_axis = self.figure.add_subplot(grid[:2, 4])
        self.wheel_axis = self.figure.add_subplot(grid[2, 4])
        self.keyboard_axis = self.figure.add_subplot(grid[3, :])
        self._configure_state_axis(context_coordinates)
        self._configure_info_axis()
        self._configure_wheel()
        self._configure_keyboard()
        self.figure.suptitle("B R A I N   S T A T E   S O N A T A", color="#eaf3ff", fontsize=21, fontweight="light", y=0.985)

    def _configure_state_axis(self, context_coordinates):
        axis = self.state_axis
        axis.set_facecolor("#030611")
        axis.tick_params(colors="#53627d", labelsize=8)
        axis.grid(True, alpha=0.10)
        axis.set_xlabel("STATE I", color="#7586a3", labelpad=10)
        axis.set_ylabel("STATE II", color="#7586a3", labelpad=10)
        labels = ["x", "y", "z"]
        for dimension in range(self.dimensions):
            low, high = self.map_bundle["bounds"][dimension]
            padding = max(1e-6, (high - low) * 0.10)
            getattr(axis, f"set_{labels[dimension]}lim")(low - padding, high + padding)
        if self.dimensions == 3:
            axis.set_zlabel("STATE III", color="#7586a3", labelpad=10)
            axis.xaxis.pane.set_facecolor((0.01, 0.02, 0.05, 1.0))
            axis.yaxis.pane.set_facecolor((0.01, 0.02, 0.05, 1.0))
            axis.zaxis.pane.set_facecolor((0.01, 0.02, 0.05, 1.0))
        if context_coordinates is not None and len(context_coordinates) > 0:
            points = np.asarray(context_coordinates, dtype=float)
            stride = max(1, len(points) // 3500)
            points = points[::stride]
            if self.dimensions == 3:
                axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, color="#6684b5", alpha=0.045, depthshade=False)
            else:
                axis.scatter(points[:, 0], points[:, 1], s=2, color="#6684b5", alpha=0.045)

    def _configure_info_axis(self):
        self.info_axis.set_facecolor("#030611")
        self.info_axis.set_xlim(0, 1)
        self.info_axis.set_ylim(0, 1)
        self.info_axis.axis("off")
        self.info_axis.text(0.04, 0.96, "NEURAL COMPOSER", color="#6fdcff", fontsize=10, fontweight="bold", va="top")
        self.key_text = self.info_axis.text(0.04, 0.84, "C", color="white", fontsize=34, fontweight="light", va="top")
        self.mode_text = self.info_axis.text(0.04, 0.71, "IONIAN", color="#91a8ca", fontsize=12, va="top")
        self.chord_text = self.info_axis.text(0.04, 0.60, "C", color="#ff63c8", fontsize=24, va="top")
        self.tempo_text = self.info_axis.text(0.04, 0.47, "076 BPM", color="#dce8ff", fontsize=14, va="top")
        self.notes_text = self.info_axis.text(0.04, 0.34, "", color="#8bf5cf", fontsize=10, va="top", wrap=True)
        self.speed_bar = Rectangle((0.04, 0.20), 0.0, 0.025, facecolor="#4bc8ff", edgecolor="none")
        self.curve_bar = Rectangle((0.04, 0.13), 0.0, 0.025, facecolor="#ff5fc8", edgecolor="none")
        self.info_axis.add_patch(self.speed_bar)
        self.info_axis.add_patch(self.curve_bar)
        self.info_axis.text(0.04, 0.235, "MOTION", color="#63738f", fontsize=8)
        self.info_axis.text(0.04, 0.165, "CURVATURE", color="#63738f", fontsize=8)
        self.info_axis.text(0.04, 0.05, "EEG → GRAPH ENCODER → MANIFOLD → HARMONY", color="#43516b", fontsize=7)

    def _configure_wheel(self):
        axis = self.wheel_axis
        axis.set_facecolor("#030611")
        axis.set_aspect("equal")
        axis.set_xlim(-1.25, 1.25)
        axis.set_ylim(-1.25, 1.25)
        axis.axis("off")
        angles = np.linspace(math.pi / 2.0, math.pi / 2.0 - 2.0 * math.pi, 12, endpoint=False)
        self.wheel_positions = np.column_stack([np.cos(angles), np.sin(angles)])
        self.wheel_dots = axis.scatter(self.wheel_positions[:, 0], self.wheel_positions[:, 1], s=65, color="#172138", edgecolor="#44516a", linewidth=0.6)
        for index, (x, y) in enumerate(self.wheel_positions):
            axis.text(x * 1.18, y * 1.18, NOTE_NAMES[index].replace("♯", "#"), color="#65738d", fontsize=7, ha="center", va="center")
        axis.add_patch(Circle((0, 0), 0.54, fill=False, edgecolor="#202c45", linewidth=0.8))

    def _configure_keyboard(self):
        axis = self.keyboard_axis
        axis.set_facecolor("#030611")
        axis.set_xlim(-0.2, 22.2)
        axis.set_ylim(0, 1.15)
        axis.axis("off")
        white_pitch_classes = {0, 2, 4, 5, 7, 9, 11}
        self.key_patches = {}
        white_index = 0
        for midi in range(48, 85):
            if midi % 12 in white_pitch_classes:
                patch = Rectangle((white_index, 0.02), 0.94, 1.02, facecolor="#d8e1ec", edgecolor="#101726", linewidth=0.7, zorder=1)
                axis.add_patch(patch)
                self.key_patches[midi] = patch
                white_index += 1
        white_index = 0
        for midi in range(48, 85):
            pitch_class = midi % 12
            if pitch_class in white_pitch_classes:
                white_index += 1
                continue
            patch = Rectangle((white_index - 0.30, 0.47), 0.58, 0.57, facecolor="#0b1020", edgecolor="#25314a", linewidth=0.6, zorder=3)
            axis.add_patch(patch)
            self.key_patches[midi] = patch

    def update(self, coordinate, hue, composer, now, new_state=True):
        coordinate = np.asarray(coordinate, dtype=float)
        if new_state or self.last_coordinate is None:
            self.tail.append(coordinate.copy())
            self.tail_hues.append(float(hue))
            self.last_coordinate = coordinate.copy()
        points = np.asarray(self.tail)
        colors = plt.cm.hsv(np.asarray(self.tail_hues))
        if len(points) > 1:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            segment_colors = colors[:-1].copy()
            segment_colors[:, 3] = np.linspace(0.05, 0.95, len(segments))
            self.tail_artist.set_segments(segments)
            self.tail_artist.set_color(segment_colors)
        ball_color = plt.cm.hsv(float(hue) % 1.0)
        for artist in self.glow:
            artist.set_facecolor(ball_color)
            artist.set_edgecolor("none")
            if self.dimensions == 3:
                artist._offsets3d = ([coordinate[0]], [coordinate[1]], [coordinate[2]])
            else:
                artist.set_offsets(coordinate[:2][None])
        if self.dimensions == 3:
            phase = (now * composer.bpm / 60.0) % 1.0
            radius = 0.10 + 0.12 * (1.0 - phase)
            theta = np.linspace(0, 2.0 * math.pi, 80)
            rings = [
                (coordinate[0] + radius * np.cos(theta), coordinate[1] + radius * np.sin(theta), np.full_like(theta, coordinate[2])),
                (coordinate[0] + radius * np.cos(theta), np.full_like(theta, coordinate[1]), coordinate[2] + radius * np.sin(theta)),
                (np.full_like(theta, coordinate[0]), coordinate[1] + radius * np.cos(theta), coordinate[2] + radius * np.sin(theta)),
            ]
            for line, values in zip(self.orbit_lines, rings):
                line.set_data(values[0], values[1])
                line.set_3d_properties(values[2])
                line.set_color(ball_color)
                line.set_alpha(0.40 * (1.0 - phase))
        snapshot = composer.snapshot()
        self.key_text.set_text(snapshot["key"])
        self.key_text.set_color(ball_color)
        self.mode_text.set_text(snapshot["mode"])
        self.chord_text.set_text(snapshot["chord"])
        self.tempo_text.set_text(f"{snapshot['bpm']:03.0f} BPM")
        self.notes_text.set_text(snapshot["recent_notes"])
        self.speed_bar.set_width(0.78 * float(np.clip(snapshot["speed"] * 2.2, 0.0, 1.0)))
        self.curve_bar.set_width(0.78 * float(np.clip(snapshot["curvature"] * 0.65, 0.0, 1.0)))
        wheel_colors = []
        for pitch_class in range(12):
            if pitch_class in snapshot["chord_pitch_classes"]:
                wheel_colors.append("#ff61c7")
            elif pitch_class in snapshot["scale_pitch_classes"]:
                wheel_colors.append("#48d8ff")
            else:
                wheel_colors.append("#172138")
        self.wheel_dots.set_facecolors(wheel_colors)
        active = set(composer.active_notes(now))
        chord = set(composer.chord_midis)
        for midi, patch in self.key_patches.items():
            is_black = midi % 12 in {1, 3, 6, 8, 10}
            if midi in active:
                patch.set_facecolor(ball_color)
            elif midi in chord:
                patch.set_facecolor("#7d3d83" if is_black else "#e9a7e2")
            else:
                patch.set_facecolor("#0b1020" if is_black else "#d8e1ec")
        return []


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "replay_brain_music.py"), run_name="__main__")
