import time
import threading
import numpy as np
import pandas as pd

from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode

device_name = "BA MAXI 034"
duration_seconds = 10
output_csv = "brainaccess_maxi_32ch.csv"
output_npz = "brainaccess_maxi_32ch.npz"

data_lock = threading.Lock()
data_chunks = []

def callback(chunk, chunk_size):
    chunk_array = np.asarray(chunk, dtype=np.float64)
    with data_lock:
        data_chunks.append(chunk_array.copy())

core.init()

devices = core.scan()
device_names = [device.name for device in devices]
print("Found devices:", device_names)

with EEGManager() as mgr:
    status = mgr.connect(device_name)
    print("Connection status:", status)

    if status == 2:
        raise RuntimeError("Firmware stream incompatible. Update firmware from BrainAccess Board.")

    if status != 0:
        raise RuntimeError(f"Connection failed with status {status}")

    features = mgr.get_device_features()
    eeg_channels = features.electrode_count()
    sample_rate = mgr.get_sample_frequency()

    print("Connected:", mgr.is_connected())
    print("Battery:", mgr.get_battery_info().level)
    print("Sample frequency:", sample_rate)
    print("EEG channels:", eeg_channels)

    for channel_index in range(eeg_channels):
        channel_id = eeg_channel.ELECTRODE_MEASUREMENT + channel_index
        mgr.set_channel_enabled(channel_id, True)
        mgr.set_channel_gain(channel_id, GainMode.X8)

    if eeg_channels > 0:
        bias_channel = eeg_channel.ELECTRODE_MEASUREMENT + eeg_channels - 1
        mgr.set_channel_bias(bias_channel, True)

    mgr.set_callback_chunk(callback)
    mgr.load_config()

    mgr.start_stream()
    print("Streaming...")
    time.sleep(duration_seconds)
    mgr.stop_stream()
    print("Stopped.")

core.close()

if len(data_chunks) == 0:
    raise RuntimeError("No data chunks received.")

data = np.concatenate(data_chunks, axis=1)

channel_names = [f"EEG_{i + 1:02d}_uV" for i in range(data.shape[0])]
df = pd.DataFrame(data.T, columns=channel_names)

df.to_csv(output_csv, index=False)
np.savez(
    output_npz,
    data=data,
    channel_names=np.array(channel_names),
    sample_rate=sample_rate,
)

print("Data shape:", data.shape)
print("Saved:", output_csv)
print("Saved:", output_npz)
print(df.head())