import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode


device_name = "BA MAXI 034"
sample_rate = 250
window_seconds = 8
gain_mode = GainMode.X8
plot_update_ms = 40
vertical_spacing = 8.0

buffer_size = sample_rate * window_seconds
data_buffer = deque(maxlen=buffer_size)
data_lock = threading.Lock()
running = True
mgr_holder = {"mgr": None}


def callback(chunk, chunk_size):
    chunk_array = np.asarray(chunk, dtype=np.float64)

    if chunk_array.ndim != 2:
        return

    samples = chunk_array.T

    with data_lock:
        for sample in samples:
            data_buffer.append(sample.copy())


def acquisition_thread():
    global running

    core.init()

    devices = core.scan()
    device_names = [device.name for device in devices]
    print("Found devices:", device_names)

    with EEGManager() as mgr:
        mgr_holder["mgr"] = mgr

        status = mgr.connect(device_name)
        print("Connection status:", status)

        if status == 2:
            raise RuntimeError("Firmware stream incompatible. Update firmware from BrainAccess Board.")

        if status != 0:
            raise RuntimeError(f"Connection failed with status {status}")

        features = mgr.get_device_features()
        eeg_channels = features.electrode_count()

        print("Connected:", mgr.is_connected())
        print("Battery:", mgr.get_battery_info().level)
        print("Sample frequency:", mgr.get_sample_frequency())
        print("EEG channels:", eeg_channels)

        for channel_index in range(eeg_channels):
            channel_id = eeg_channel.ELECTRODE_MEASUREMENT + channel_index
            mgr.set_channel_enabled(channel_id, True)
            mgr.set_channel_gain(channel_id, gain_mode)

        if eeg_channels > 0:
            bias_channel = eeg_channel.ELECTRODE_MEASUREMENT + eeg_channels - 1
            mgr.set_channel_bias(bias_channel, True)

        mgr.set_callback_chunk(callback)
        mgr.load_config()

        mgr.start_stream()
        print("Streaming... Close the plot window to stop.")

        while running:
            time.sleep(0.05)

        mgr.stop_stream()
        print("Stopped.")

    core.close()


thread = threading.Thread(target=acquisition_thread, daemon=True)
thread.start()

time.sleep(3)

fig, ax = plt.subplots(figsize=(16, 10))
channel_count = 32
x_axis = np.arange(buffer_size) / sample_rate
lines = []

for channel_index in range(channel_count):
    line, = ax.plot(x_axis, np.zeros(buffer_size) + channel_index * vertical_spacing, linewidth=0.8)
    lines.append(line)

ax.set_xlim(0, window_seconds)
ax.set_ylim(-vertical_spacing, channel_count * vertical_spacing)
ax.set_xlabel("Time in rolling window (s)")
ax.set_ylabel("Channels")
ax.set_title("BrainAccess MAXI 32-channel real-time EEG")
ax.set_yticks([i * vertical_spacing for i in range(channel_count)])
ax.set_yticklabels([f"EEG_{i + 1:02d}" for i in range(channel_count)])
ax.grid(True, alpha=0.25)


def update_plot(frame):
    with data_lock:
        if len(data_buffer) == 0:
            return lines

        data = np.asarray(data_buffer, dtype=np.float64)

    if data.shape[0] < 2:
        return lines

    if data.shape[1] != channel_count:
        return lines

    padded = np.zeros((buffer_size, channel_count), dtype=np.float64)

    if data.shape[0] >= buffer_size:
        padded[:, :] = data[-buffer_size:, :]
    else:
        padded[-data.shape[0]:, :] = data

    demeaned = padded - np.mean(padded, axis=0, keepdims=True)
    std = np.std(demeaned, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    normalized = demeaned / std

    for channel_index, line in enumerate(lines):
        y = normalized[:, channel_index] + channel_index * vertical_spacing
        line.set_ydata(y)

    return lines


def on_close(event):
    global running
    running = False
    time.sleep(0.5)


fig.canvas.mpl_connect("close_event", on_close)

animation = FuncAnimation(
    fig,
    update_plot,
    interval=plot_update_ms,
    blit=False,
    cache_frame_data=False,
)

plt.tight_layout()
plt.show()

running = False
thread.join(timeout=3)