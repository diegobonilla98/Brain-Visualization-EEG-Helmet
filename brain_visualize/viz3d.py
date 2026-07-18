import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import open3d as o3d
import matplotlib
import mne
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))
from brainaccess_stream import BrainAccessStream, DEFAULT_MAXI_32_NAMES


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
USE_BIAS = True
BIAS_CHANNEL_NAME = "Iz"
USE_SIMULATED_DATA = False
NORMALIZE_MODE = "global"
ACTIVE_VIEW_NAME = "activation"
SPECTRAL_WINDOW_SECONDS = 3.0
BASELINE_WARMUP_SECONDS = 8.0
BASELINE_WARMUP_ADAPT_RATE = 0.18
BASELINE_ADAPT_RATE = 0.012
NOTCH_50_HZ = True
USE_COMMON_AVERAGE_REFERENCE = True
HISTORY_SECONDS = 8.0
TARGET_FPS = 30.0
COLORMAP_NAME = "turbo"
TEXTURE_SIZE = 256
INTERP_SIGMA = 0.20
DISK_RADIUS_MAX = 1.35
SHOW_ELECTRODES = True
ELECTRODE_RADIUS = 0.028
ELECTRODE_SURFACE_OFFSET = 0.035
OUTER_SURFACE_PERCENTILE = 85.0
OUTER_SURFACE_RADIUS_WEIGHT = 0.12
MIRROR_LEFT_RIGHT = False
LIGHT_ON = True
BACKGROUND_COLOR = np.array([0.04, 0.05, 0.08])
SIM_SAMPLE_RATE = 250
ROBUST_LOW_PERCENTILE = 4.0
ROBUST_HIGH_PERCENTILE = 96.0

CHANNEL_NAMES = DEFAULT_MAXI_32_NAMES

FREQUENCY_BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("low_beta", 12.0, 16.0),
    ("beta", 16.0, 30.0),
    ("low_gamma", 30.0, 45.0),
    ("broadband", 1.0, 45.0),
]

VIEW_KEY_ORDER = ["activation", "delta", "theta", "alpha", "low_beta", "beta", "low_gamma", "broadband"]

ACTIVATION_WEIGHTS = {
    "theta": 0.30,
    "alpha": -0.45,
    "low_beta": 0.20,
    "beta": 0.30,
    "low_gamma": 0.25,
}

FRONT_VIEW_FRONT = np.array([-1.0, 0.0, 0.0])
FRONT_VIEW_UP = np.array([0.0, 0.0, 1.0])
FRONT_VIEW_ZOOM = 0.62

PATH_STL_FILE = Path(__file__).resolve().parent.parent / "assets" / "brain.stl"


def head_components(points, center):
    delta = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    superior = delta[..., 2]
    anterior = -delta[..., 0]
    right = delta[..., 1] * (-1.0 if MIRROR_LEFT_RIGHT else 1.0)
    return anterior, right, superior


def project_to_disk(anterior, right, superior):
    norm = np.sqrt(anterior * anterior + right * right + superior * superior)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    superior_unit = np.clip(superior / norm, -1.0, 1.0)
    polar = np.arccos(superior_unit)
    azimuth = np.arctan2(right, anterior)
    radius = polar / (np.pi / 2.0)
    disk_x = radius * np.sin(azimuth)
    disk_y = radius * np.cos(azimuth)
    return np.stack([disk_x, disk_y], axis=-1)


def montage_disk_coords(channel_names):
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    raw = np.array([positions[name] for name in channel_names], dtype=np.float64)
    center = raw.mean(axis=0)
    anterior = raw[:, 1] - center[1]
    right = (raw[:, 0] - center[0]) * (-1.0 if MIRROR_LEFT_RIGHT else 1.0)
    superior = raw[:, 2] - center[2]
    disk = project_to_disk(anterior, right, superior)
    return disk, raw


def disk_to_texel(disk_xy):
    column = (disk_xy[..., 0] / DISK_RADIUS_MAX * 0.5 + 0.5) * (TEXTURE_SIZE - 1)
    row = (1.0 - (disk_xy[..., 1] / DISK_RADIUS_MAX * 0.5 + 0.5)) * (TEXTURE_SIZE - 1)
    column = np.clip(np.rint(column), 0, TEXTURE_SIZE - 1).astype(np.int64)
    row = np.clip(np.rint(row), 0, TEXTURE_SIZE - 1).astype(np.int64)
    return row, column


def build_grid_weight_matrix(electrode_disk):
    axis = np.linspace(-DISK_RADIUS_MAX, DISK_RADIUS_MAX, TEXTURE_SIZE)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_y = grid_y[::-1, :]
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    diff = grid_points[:, None, :] - electrode_disk[None, :, :]
    distance = np.sqrt((diff * diff).sum(axis=2))
    weights = np.exp(-(distance / INTERP_SIGMA) ** 2)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum = np.where(weights_sum < 1e-12, 1e-12, weights_sum)
    return weights / weights_sum


def cast_electrodes_to_surface(mesh, center, electrode_raw):
    anterior = electrode_raw[:, 1] - electrode_raw[:, 1].mean()
    right = (electrode_raw[:, 0] - electrode_raw[:, 0].mean()) * (-1.0 if MIRROR_LEFT_RIGHT else 1.0)
    superior = electrode_raw[:, 2] - electrode_raw[:, 2].mean()
    directions = np.stack([-anterior, right, superior], axis=1)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    vertices = np.asarray(mesh.vertices)
    centered_vertices = vertices - center[None, :]
    radius = np.linalg.norm(centered_vertices, axis=1)
    unit_vertices = centered_vertices / np.maximum(radius[:, None], 1e-12)
    score = unit_vertices @ directions.T
    score = score + OUTER_SURFACE_RADIUS_WEIGHT * (radius[:, None] / np.max(radius))
    score[radius < np.percentile(radius, OUTER_SURFACE_PERCENTILE), :] -= 10.0
    points = vertices[np.argmax(score, axis=0)]
    points = points + directions * ELECTRODE_SURFACE_OFFSET
    return points


def build_electrode_mesh(points):
    combined = o3d.geometry.TriangleMesh()
    owner = []
    for index, point in enumerate(points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=ELECTRODE_RADIUS, resolution=10)
        sphere.translate(point)
        owner.extend([index] * len(sphere.vertices))
        combined += sphere
    combined.compute_vertex_normals()
    return combined, np.array(owner, dtype=np.int64)


def build_filter_bank(sample_rate):
    nyquist = 0.5 * sample_rate
    filters = {}
    for name, low, high in FREQUENCY_BANDS:
        high_value = min(float(high), nyquist - 1.0)
        if high_value > low:
            filters[name] = butter(4, [float(low) / nyquist, high_value / nyquist], btype="bandpass", output="sos")
    return filters


def preprocess_window(window, sample_rate):
    data = np.asarray(window, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = data - data.mean(axis=1, keepdims=True)
    if USE_COMMON_AVERAGE_REFERENCE:
        data = data - data.mean(axis=0, keepdims=True)
    if NOTCH_50_HZ and sample_rate > 120 and data.shape[1] >= 64:
        b, a = iirnotch(50.0, 30.0, sample_rate)
        data = filtfilt(b, a, data, axis=1)
    return data


def compute_band_log_power(buffer, sample_rate, filter_bank):
    window_samples = max(64, int(SPECTRAL_WINDOW_SECONDS * sample_rate))
    window = buffer[:, -window_samples:]
    data = preprocess_window(window, sample_rate)
    output = {}
    for name, sos in filter_bank.items():
        filtered = sosfiltfilt(sos, data, axis=1)
        power = np.mean(filtered * filtered, axis=1)
        output[name] = 10.0 * np.log10(power + 1e-18)
    return output


def update_baseline(current_power, baseline_power, adapt_rate):
    if baseline_power is None:
        return {name: values.copy() for name, values in current_power.items()}
    updated = {}
    for name, values in current_power.items():
        previous = baseline_power.get(name)
        if previous is None:
            updated[name] = values.copy()
        else:
            updated[name] = (1.0 - adapt_rate) * previous + adapt_rate * values
    return updated


def band_deltas(current_power, baseline_power):
    return {name: current_power[name] - baseline_power[name] for name in current_power if name in baseline_power}


def activation_index(deltas):
    score = np.zeros(len(CHANNEL_NAMES), dtype=np.float64)
    weight_sum = 0.0
    for name, weight in ACTIVATION_WEIGHTS.items():
        if name in deltas:
            score += float(weight) * deltas[name]
            weight_sum += abs(float(weight))
    if weight_sum > 0:
        score = score / weight_sum
    return score


def activation_for_view(current_power, baseline_power, view_name):
    deltas = band_deltas(current_power, baseline_power)
    if view_name == "activation":
        return activation_index(deltas)
    if view_name in deltas:
        return deltas[view_name]
    return activation_index(deltas)


def normalize_activation(activation, history):
    history.append(activation.copy())
    stacked = np.array(history)
    if NORMALIZE_MODE == "global":
        low = np.percentile(stacked, ROBUST_LOW_PERCENTILE)
        high = np.percentile(stacked, ROBUST_HIGH_PERCENTILE)
    else:
        low = np.percentile(stacked, ROBUST_LOW_PERCENTILE, axis=0)
        high = np.percentile(stacked, ROBUST_HIGH_PERCENTILE, axis=0)
    span = np.maximum(high - low, 1e-9)
    return np.clip((activation - low) / span, 0.0, 1.0)


def simulated_chunk(channel_disk, elapsed, sample_count, sample_rate):
    times = elapsed + np.arange(sample_count) / sample_rate
    beta_hotspot = np.array([0.55 * np.cos(elapsed * 0.7), 0.55 * np.sin(elapsed * 0.5)])
    alpha_hotspot = np.array([-0.45 * np.cos(elapsed * 0.35), 0.45 * np.sin(elapsed * 0.42)])
    beta_distance = np.linalg.norm(channel_disk - beta_hotspot[None, :], axis=1)
    alpha_distance = np.linalg.norm(channel_disk - alpha_hotspot[None, :], axis=1)
    beta_amplitude = 4.0 + 28.0 * np.exp(-(beta_distance / 0.38) ** 2)
    alpha_amplitude = 16.0 - 11.0 * np.exp(-(alpha_distance / 0.42) ** 2)
    beta = beta_amplitude[:, None] * np.sin(2.0 * np.pi * 20.0 * times)[None, :]
    alpha = alpha_amplitude[:, None] * np.sin(2.0 * np.pi * 10.0 * times)[None, :]
    theta = 4.0 * np.sin(2.0 * np.pi * 6.0 * times)[None, :]
    noise = np.random.normal(0.0, 2.0, size=(len(channel_disk), sample_count))
    offset = 100000.0
    return offset + alpha + beta + theta + noise


class RealSource:
    def __init__(self):
        self.recorder = BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME, use_bias=USE_BIAS, bias_channel_name=BIAS_CHANNEL_NAME)
        self.row_indices = None
        self.consumed = 0
        self.sample_rate = None

    def __enter__(self):
        self.recorder.connect()
        self.recorder.start()
        self.sample_rate = self.recorder.sample_frequency
        self.row_indices = [self.recorder.channel_indices[f"{name}_uV"] for name in CHANNEL_NAMES]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.recorder.close()

    def read(self):
        with self.recorder.lock:
            new_chunks = self.recorder.chunks[self.consumed:]
            self.consumed = len(self.recorder.chunks)
        columns = []
        for _, array, chunk_size in new_chunks:
            count = min(chunk_size, array.shape[1])
            if count > 0:
                columns.append(array[self.row_indices, :count])
        if not columns:
            return None
        return np.concatenate(columns, axis=1)


class SimulatedSource:
    def __init__(self, channel_disk):
        self.channel_disk = channel_disk
        self.sample_rate = SIM_SAMPLE_RATE
        self.start_time = None
        self.last_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        now = time.perf_counter()
        sample_count = int((now - self.last_time) * self.sample_rate)
        if sample_count <= 0:
            return None
        chunk = simulated_chunk(self.channel_disk, now - self.start_time, sample_count, self.sample_rate)
        self.last_time = now
        return chunk


def configure_view(vis):
    control = vis.get_view_control()
    control.set_front(FRONT_VIEW_FRONT)
    control.set_up(FRONT_VIEW_UP)
    control.set_zoom(FRONT_VIEW_ZOOM)


def main():
    global NORMALIZE_MODE, ACTIVE_VIEW_NAME

    mesh = o3d.io.read_triangle_mesh(str(PATH_STL_FILE))
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)

    channel_disk, electrode_raw = montage_disk_coords(CHANNEL_NAMES)
    grid_weights = build_grid_weight_matrix(channel_disk)

    vertex_anterior, vertex_right, vertex_superior = head_components(vertices, center)
    vertex_disk = project_to_disk(vertex_anterior, vertex_right, vertex_superior)
    vertex_rows, vertex_cols = disk_to_texel(vertex_disk)

    colormap = matplotlib.colormaps[COLORMAP_NAME]

    electrode_points = cast_electrodes_to_surface(mesh, center, electrode_raw)
    electrode_mesh, electrode_owner = build_electrode_mesh(electrode_points)

    state = {"paused": False, "history": None}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("EEG Real-Time Heatmap", 1280, 800)
    vis.add_geometry(mesh)
    if SHOW_ELECTRODES:
        vis.add_geometry(electrode_mesh)
    render_option = vis.get_render_option()
    render_option.background_color = BACKGROUND_COLOR
    render_option.light_on = LIGHT_ON
    render_option.mesh_show_back_face = True
    configure_view(vis)

    def toggle_normalize(_):
        global NORMALIZE_MODE
        NORMALIZE_MODE = "global" if NORMALIZE_MODE == "individual" else "individual"
        if state["history"] is not None:
            state["history"].clear()
        print("Normalize mode:", NORMALIZE_MODE)
        return False

    def toggle_pause(_):
        state["paused"] = not state["paused"]
        print("Paused:", state["paused"])
        return False

    def set_active_view(view_name):
        def callback(_):
            global ACTIVE_VIEW_NAME
            ACTIVE_VIEW_NAME = view_name
            if state["history"] is not None:
                state["history"].clear()
            print("Active view:", ACTIVE_VIEW_NAME)
            return False
        return callback

    vis.register_key_callback(ord("N"), toggle_normalize)
    vis.register_key_callback(ord(" "), toggle_pause)
    vis.register_key_callback(ord("A"), set_active_view("activation"))
    for index, view_name in enumerate(VIEW_KEY_ORDER, start=1):
        vis.register_key_callback(ord(str(index)), set_active_view(view_name))

    source_factory = SimulatedSource(channel_disk) if USE_SIMULATED_DATA else RealSource()
    frame_interval = 1.0 / TARGET_FPS

    print("Keys: space pause, N normalize global/individual, A or 1 activation, 2 delta, 3 theta, 4 alpha, 5 low_beta, 6 beta, 7 low_gamma, 8 broadband")
    print("Activation view combines theta/beta/low-gamma increases with alpha suppression relative to the adaptive baseline.")

    with source_factory as source:
        sample_rate = source.sample_rate
        filter_bank = build_filter_bank(sample_rate)
        buffer_length = max(int(max(SPECTRAL_WINDOW_SECONDS, 1.0) * sample_rate), 128)
        buffer = np.zeros((len(CHANNEL_NAMES), buffer_length), dtype=np.float64)
        history = deque(maxlen=max(2, int(HISTORY_SECONDS * TARGET_FPS)))
        state["history"] = history
        baseline_power = None
        visual_start_time = time.perf_counter()

        running = True
        while running:
            loop_start = time.perf_counter()
            if not state["paused"]:
                new_samples = source.read()
                if new_samples is not None and new_samples.shape[1] > 0:
                    buffer = np.concatenate([buffer, new_samples], axis=1)[:, -buffer_length:]

                current_power = compute_band_log_power(buffer, sample_rate, filter_bank)
                warmup_active = time.perf_counter() - visual_start_time < BASELINE_WARMUP_SECONDS
                adapt_rate = BASELINE_WARMUP_ADAPT_RATE if warmup_active else BASELINE_ADAPT_RATE
                if baseline_power is None:
                    baseline_power = update_baseline(current_power, baseline_power, adapt_rate)
                activation = activation_for_view(current_power, baseline_power, ACTIVE_VIEW_NAME)
                baseline_power = update_baseline(current_power, baseline_power, adapt_rate)
                normalized = normalize_activation(activation, history)

                texture = (grid_weights @ normalized).reshape(TEXTURE_SIZE, TEXTURE_SIZE)
                texture_rgb = colormap(texture)[:, :, :3]

                vertex_colors = texture_rgb[vertex_rows, vertex_cols]
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                vis.update_geometry(mesh)

                if SHOW_ELECTRODES:
                    electrode_rgb = colormap(normalized)[:, :3]
                    electrode_mesh.vertex_colors = o3d.utility.Vector3dVector(electrode_rgb[electrode_owner])
                    vis.update_geometry(electrode_mesh)

            running = vis.poll_events()
            vis.update_renderer()

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    vis.destroy_window()


if __name__ == "__main__":
    main()
