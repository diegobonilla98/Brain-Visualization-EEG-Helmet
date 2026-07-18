import time
import threading

import numpy as np

from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode

DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
DURATION_SECONDS = 30.0
PROGRESS_EVERY_SECONDS = 5.0

MAXI_32_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO3", "POz", "PO4", "O1", "Oz", "O2", "Iz",
]

FLAT_FRACTION_DEAD = 0.85
ZERO_FRACTION_DEAD = 0.50
FLAT_STD_DEAD_UV = 150.0
FLAT_RANGE_DEAD_UV = 1000.0
WEAK_STD_FRACTION_OF_MEDIAN = 0.08
WEAK_STD_MIN_UV = 400.0

data_lock = threading.Lock()
data_chunks = []


def callback(chunk, chunk_size):
    chunk_array = np.asarray(chunk, dtype=np.float64)
    with data_lock:
        data_chunks.append(chunk_array.copy())


def channel_labels(channel_count):
    if channel_count == len(MAXI_32_NAMES):
        return MAXI_32_NAMES.copy()
    return [f"EEG_{index + 1:02d}" for index in range(channel_count)]


def classify_channel(samples):
    values = np.asarray(samples, dtype=np.float64)
    sample_count = len(values)
    if sample_count < 2:
        return {
            "status": "dead_zero",
            "std_uV": 0.0,
            "range_uV": 0.0,
            "flat_fraction": 1.0,
            "zero_fraction": 1.0,
            "mean_uV": float(values[0]) if sample_count else 0.0,
        }

    diffs = np.diff(values)
    zero_fraction = float(np.mean(values == 0.0))
    flat_fraction = float(np.mean(diffs == 0.0))
    std_uV = float(np.std(values))
    range_uV = float(np.max(values) - np.min(values))
    mean_uV = float(np.mean(values))

    if zero_fraction >= ZERO_FRACTION_DEAD:
        status = "dead_zero"
    elif flat_fraction >= FLAT_FRACTION_DEAD and (std_uV <= FLAT_STD_DEAD_UV or range_uV <= FLAT_RANGE_DEAD_UV):
        status = "dead_flat"
    else:
        status = "ok"

    return {
        "status": status,
        "std_uV": std_uV,
        "range_uV": range_uV,
        "flat_fraction": flat_fraction,
        "zero_fraction": zero_fraction,
        "mean_uV": mean_uV,
    }


def apply_weak_labels(channel_metrics):
    ok_stds = [metrics["std_uV"] for metrics in channel_metrics if metrics["status"] == "ok"]
    if len(ok_stds) == 0:
        return
    median_std = float(np.median(ok_stds))
    weak_limit = max(WEAK_STD_MIN_UV, median_std * WEAK_STD_FRACTION_OF_MEDIAN)
    for metrics in channel_metrics:
        if metrics["status"] != "ok":
            continue
        if metrics["std_uV"] < weak_limit:
            metrics["status"] = "weak"


def print_diagnosis(labels, channel_metrics, sample_rate, sample_count, duration_seconds):
    dead = [labels[index] for index, metrics in enumerate(channel_metrics) if metrics["status"].startswith("dead")]
    weak = [labels[index] for index, metrics in enumerate(channel_metrics) if metrics["status"] == "weak"]
    ok = [labels[index] for index, metrics in enumerate(channel_metrics) if metrics["status"] == "ok"]

    print()
    print("=" * 72)
    print("ELECTRODE DIAGNOSIS")
    print("=" * 72)
    print(f"Device: {DEVICE_NAME}")
    print(f"Duration: {duration_seconds:.1f} s")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples per channel: {sample_count}")
    print(f"OK: {len(ok)}   Weak: {len(weak)}   Dead: {len(dead)}")
    print()

    print("Per-channel metrics:")
    print(f"{'Electrode':<8} {'Status':<12} {'Std uV':>10} {'Range uV':>12} {'Flat%':>8} {'Zero%':>8} {'Mean uV':>12}")
    for label, metrics in zip(labels, channel_metrics):
        print(
            f"{label:<8} {metrics['status']:<12} "
            f"{metrics['std_uV']:10.1f} {metrics['range_uV']:12.1f} "
            f"{metrics['flat_fraction'] * 100:7.1f} {metrics['zero_fraction'] * 100:7.1f} "
            f"{metrics['mean_uV']:12.1f}"
        )

    print()
    if len(dead) > 0:
        print("Dead electrodes:", ", ".join(dead))
    else:
        print("Dead electrodes: none")

    if len(weak) > 0:
        print("Weak electrodes:", ", ".join(weak))
    else:
        print("Weak electrodes: none")

    if len(ok) > 0:
        print("Healthy electrodes:", ", ".join(ok))

    dead_zero_count = sum(1 for metrics in channel_metrics if metrics["status"] == "dead_zero")
    if dead_zero_count >= max(4, len(channel_metrics) // 4):
        print()
        print(
            "Note: many channels are mostly zeros. That often means Bluetooth stream loss "
            "or firmware/channel parsing issues, not only bad electrode contact."
        )

    if len(dead) >= len(channel_metrics) // 2:
        print(
            "Note: more than half the cap looks bad. Check gel/contact, bias on Iz, "
            "and rerun with the helmet still and BrainAccess Board closed."
        )

    print("=" * 72)


core.init()

devices = core.scan()
device_names = [device.name for device in devices]
print("Found devices:", device_names)

matches = [name for name in device_names if DEVICE_NAME in name]
if len(matches) == 0:
    if len(device_names) == 0:
        core.close()
        raise RuntimeError("No BrainAccess devices found")
    target_name = device_names[0]
else:
    target_name = matches[0]

print("Connecting to:", target_name)

with EEGManager() as mgr:
    status = mgr.connect(target_name)
    print("Connection status:", status)

    if status == 2:
        raise RuntimeError("Firmware stream incompatible. Update firmware from BrainAccess Board.")

    if status != 0:
        raise RuntimeError(f"Connection failed with status {status}")

    features = mgr.get_device_features()
    eeg_channels = int(features.electrode_count())
    sample_rate = int(mgr.get_sample_frequency())
    labels = channel_labels(eeg_channels)

    print("Connected:", mgr.is_connected())
    print("Battery:", mgr.get_battery_info().level)
    print("Sample frequency:", sample_rate)
    print("EEG channels:", eeg_channels)

    for channel_index in range(eeg_channels):
        channel_id = eeg_channel.ELECTRODE_MEASUREMENT + channel_index
        mgr.set_channel_enabled(channel_id, True)
        mgr.set_channel_gain(channel_id, GainMode.X8)

    bias_channel_name = "Iz"
    bias_index = labels.index(bias_channel_name) if bias_channel_name in labels else eeg_channels - 1
    mgr.set_channel_bias(eeg_channel.ELECTRODE_MEASUREMENT + bias_index, True)

    mgr.set_callback_chunk(callback)
    mgr.load_config()

    mgr.start_stream()
    print(f"Recording {DURATION_SECONDS:.0f} s for electrode diagnostics...")
    stream_start = time.perf_counter()
    next_progress = stream_start + PROGRESS_EVERY_SECONDS
    while True:
        now = time.perf_counter()
        elapsed = now - stream_start
        if elapsed >= DURATION_SECONDS:
            break
        if now >= next_progress:
            print(f"  {elapsed:4.0f} / {DURATION_SECONDS:.0f} s")
            next_progress += PROGRESS_EVERY_SECONDS
        time.sleep(0.05)
    mgr.stop_stream()
    print("Recording finished.")

core.close()

if len(data_chunks) == 0:
    raise RuntimeError("No data chunks received.")

data = np.concatenate(data_chunks, axis=1)
if data.shape[0] != eeg_channels:
    raise RuntimeError(f"Expected {eeg_channels} EEG channels, got {data.shape[0]}")

channel_metrics = [classify_channel(data[channel_index, :]) for channel_index in range(eeg_channels)]
apply_weak_labels(channel_metrics)
print_diagnosis(
    labels=labels,
    channel_metrics=channel_metrics,
    sample_rate=sample_rate,
    sample_count=data.shape[1],
    duration_seconds=DURATION_SECONDS,
)
