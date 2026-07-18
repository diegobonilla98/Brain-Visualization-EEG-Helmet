import json
import os
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd

import brainaccess
from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel
from brainaccess.core.gain_mode import GainMode
from brainaccess.core.polarity import Polarity


CONNECT_RETRIES = 5
CONNECT_RETRY_SECONDS = 2.0


DEFAULT_MAXI_32_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO3", "POz", "PO4", "O1", "Oz", "O2", "Iz",
]


class BrainAccessStream:
    def __init__(self, device_name="BA MAXI 034", channel_names=None, gain_name="X8", use_sample_number=True, use_streaming=True, enabled_channel_names=None, use_bias=True, bias_channel_name="Iz"):
        self.device_name = device_name
        self.channel_names = channel_names
        self.gain_name = gain_name
        self.use_sample_number = use_sample_number
        self.use_streaming = use_streaming
        self.enabled_channel_names = enabled_channel_names
        self.use_bias = use_bias
        self.bias_channel_name = bias_channel_name
        self.mgr = None
        self.sample_frequency = None
        self.eeg_channel_count = None
        self.target_name = None
        self.connected = False
        self.streaming = False
        self.lock = threading.Lock()
        self.chunks = []
        self.stream_start_perf_s = None
        self.stream_stop_perf_s = None
        self.channel_ids = []
        self.channel_labels = []
        self.channel_indices = {}
        self.assumed_labels = []
        self.enabled_electrode_indices = []
        self.bias_channel_name_used = None
        self.device_info = None
        self.battery_info = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def connect(self):
        last_error = None
        for attempt in range(CONNECT_RETRIES):
            if attempt > 0:
                time.sleep(CONNECT_RETRY_SECONDS)
                print(f"Retrying BrainAccess connection ({attempt + 1}/{CONNECT_RETRIES})...")
            try:
                return self._connect_once()
            except Exception as error:
                last_error = error
                self.close()
        raise last_error

    def _connect_once(self):
        core.init()
        devices = core.scan()
        names = []
        for device in devices:
            name = device.name if hasattr(device, "name") else str(device)
            names.append(name)
        print("Found devices:", names)
        matches = [name for name in names if self.device_name in name]
        if len(matches) == 0 and len(names) > 0:
            matches = [names[0]]
        if len(matches) == 0:
            raise RuntimeError(f"No BrainAccess device found for {self.device_name}")
        self.target_name = matches[0]
        self.mgr = EEGManager()
        status = self.mgr.connect(self.target_name)
        if status == 1:
            raise RuntimeError("BrainAccess connection failed")
        if status == 2:
            raise RuntimeError("BrainAccess stream is incompatible. Update firmware")
        self.connected = True
        print("Connection status:", status)
        print("Connected:", self.mgr.is_connected())
        self.device_info = self.mgr.get_device_info()
        self.battery_info = self.mgr.get_battery_info()
        print("Battery:", self.battery_info.level)
        features = self.mgr.get_device_features()
        self.eeg_channel_count = int(features.electrode_count())
        self.sample_frequency = int(self.mgr.get_sample_frequency())
        print("Sample frequency:", self.sample_frequency)
        print("EEG channels:", self.eeg_channel_count)
        if self.channel_names is None:
            if self.eeg_channel_count == 32:
                self.channel_names = DEFAULT_MAXI_32_NAMES.copy()
            else:
                self.channel_names = [f"EEG_{i + 1:02d}" for i in range(self.eeg_channel_count)]
        if len(self.channel_names) != self.eeg_channel_count:
            self.channel_names = [f"EEG_{i + 1:02d}" for i in range(self.eeg_channel_count)]
        self.configure_channels()
        return self

    def selected_electrode_indices(self):
        if self.enabled_channel_names is None:
            return list(range(self.eeg_channel_count))
        lower_map = {name.lower(): index for index, name in enumerate(self.channel_names)}
        indices = []
        for name in self.enabled_channel_names:
            index = lower_map.get(str(name).lower())
            if index is not None and index not in indices:
                indices.append(index)
        if len(indices) == 0:
            raise RuntimeError("enabled_channel_names did not match any configured EEG channel")
        return indices

    def configure_bias(self):
        if not self.use_bias:
            self.bias_channel_name_used = None
            return
        bias_index = self.eeg_channel_count - 1
        if self.bias_channel_name is not None:
            wanted = str(self.bias_channel_name).lower()
            for index, name in enumerate(self.channel_names):
                if str(name).lower() == wanted:
                    bias_index = index
                    break
        channel_id = eeg_channel.ELECTRODE_MEASUREMENT + bias_index
        self.mgr.set_channel_bias(channel_id, Polarity.BOTH)
        self.bias_channel_name_used = self.channel_names[bias_index]

    def configure_channels(self):
        gain = getattr(GainMode, self.gain_name)
        self.channel_ids = []
        self.channel_labels = []
        self.enabled_electrode_indices = self.selected_electrode_indices()
        if self.use_sample_number:
            self.mgr.set_channel_enabled(eeg_channel.SAMPLE_NUMBER, True)
            self.channel_ids.append(eeg_channel.SAMPLE_NUMBER)
            self.channel_labels.append("device_sample")
        for index in self.enabled_electrode_indices:
            channel_id = eeg_channel.ELECTRODE_MEASUREMENT + index
            self.mgr.set_channel_enabled(channel_id, True)
            self.mgr.set_channel_gain(channel_id, gain)
            self.channel_ids.append(channel_id)
            self.channel_labels.append(f"{self.channel_names[index]}_uV")
        self.configure_bias()
        if self.use_streaming:
            self.mgr.set_channel_enabled(eeg_channel.STREAMING, True)
            self.channel_ids.append(eeg_channel.STREAMING)
            self.channel_labels.append("streaming")
        self.mgr.load_config()
        self.assumed_labels = self.channel_labels.copy()
        print("Configured channels:", len(self.channel_labels))
        print("Bias channel:", self.bias_channel_name_used)

    def start(self):
        self.chunks = []
        self.stream_start_perf_s = time.perf_counter()
        self.mgr.set_callback_chunk(self._callback)
        self.mgr.start_stream()
        self.streaming = True
        time.sleep(0.25)
        self.resolve_channel_indices()
        print("Stream started")
        return self.stream_start_perf_s

    def stop(self):
        if self.streaming:
            self.stream_stop_perf_s = time.perf_counter()
            self.mgr.stop_stream()
            self.streaming = False
            print("Stream stopped")
        return self.stream_stop_perf_s

    def close(self):
        if self.mgr is not None:
            if self.streaming:
                self.stop()
            self.mgr.destroy()
            self.mgr = None
        try:
            core.close()
        except Exception:
            pass
        self.connected = False

    def annotate(self, text):
        if self.mgr is not None and self.streaming:
            safe_text = str(text).replace(" ", "_")[:64]
            if len(safe_text) > 0:
                self.mgr.annotate(safe_text)

    def get_annotations(self):
        if self.mgr is None:
            return {"annotations": [], "timestamps": []}
        return self.mgr.get_annotations()

    def resolve_channel_indices(self):
        self.channel_indices = {}
        for channel_id, label in zip(self.channel_ids, self.channel_labels):
            try:
                self.channel_indices[label] = int(self.mgr.get_channel_index(channel_id))
            except Exception:
                self.channel_indices[label] = len(self.channel_indices)
        print("Channel indices:", self.channel_indices)

    def _callback(self, chunk, chunk_size):
        receive_perf_s = time.perf_counter()
        array = np.asarray(chunk, dtype=np.float64)
        if array.ndim != 2:
            return
        if array.shape[0] != len(self.channel_labels) and array.shape[1] == len(self.channel_labels):
            array = array.T
        with self.lock:
            self.chunks.append((receive_perf_s, array.copy(), int(chunk_size)))

    def dataframe(self):
        with self.lock:
            chunks = list(self.chunks)
        rows = []
        sample_counter = 0
        first_sample_time_est_s = None
        for receive_perf_s, array, chunk_size in chunks:
            if array.ndim != 2:
                continue
            n_channels = array.shape[0]
            n_samples = array.shape[1]
            used_chunk_size = min(chunk_size, n_samples)
            if first_sample_time_est_s is None:
                first_sample_time_est_s = receive_perf_s - ((used_chunk_size - 1) / float(self.sample_frequency))
            for sample_offset in range(used_chunk_size):
                sample_time_from_chunk_s = receive_perf_s - ((used_chunk_size - 1 - sample_offset) / float(self.sample_frequency))
                row = {
                    "sample_index": sample_counter,
                    "sample_offset_in_chunk": sample_offset,
                    "pc_time_received_s": receive_perf_s,
                    "sample_time_est_s": first_sample_time_est_s + (sample_counter / float(self.sample_frequency)),
                    "sample_time_est_from_chunk_s": sample_time_from_chunk_s,
                    "chunk_timing_error_s": sample_time_from_chunk_s - (first_sample_time_est_s + (sample_counter / float(self.sample_frequency))),
                }
                if self.stream_start_perf_s is not None:
                    row["t_from_stream_start_s"] = row["sample_time_est_s"] - self.stream_start_perf_s
                for label in self.channel_labels:
                    index = self.channel_indices.get(label, None)
                    if index is None or index >= n_channels:
                        continue
                    row[label] = array[index, sample_offset]
                rows.append(row)
                sample_counter += 1
        return pd.DataFrame(rows)

    def metadata(self):
        device_info = self.device_info
        sample_per_packet = int(device_info.sample_per_packet) if device_info is not None else None
        if sample_per_packet is not None and sample_per_packet > 10000:
            sample_per_packet = None
        return {
            "device_name_requested": self.device_name,
            "device_name_connected": self.target_name,
            "brainaccess_sdk_version": getattr(brainaccess, "__version__", None),
            "device_model": device_info.device_model.name if device_info is not None else None,
            "hardware_version": repr(device_info.hardware_version) if device_info is not None else None,
            "firmware_version": repr(device_info.firmware_version) if device_info is not None else None,
            "serial_number": int(device_info.serial_number) if device_info is not None else None,
            "sample_per_packet": sample_per_packet,
            "battery_level": int(self.battery_info.level) if self.battery_info is not None else None,
            "sample_frequency_hz": self.sample_frequency,
            "eeg_channel_count": self.eeg_channel_count,
            "channel_names": self.channel_names,
            "enabled_channel_names": [self.channel_names[index] for index in self.enabled_electrode_indices],
            "enabled_eeg_channel_count": len(self.enabled_electrode_indices),
            "gain_name": self.gain_name,
            "use_sample_number": self.use_sample_number,
            "use_streaming": self.use_streaming,
            "use_bias": self.use_bias,
            "bias_channel_name": self.bias_channel_name_used,
            "stream_start_perf_s": self.stream_start_perf_s,
            "stream_stop_perf_s": self.stream_stop_perf_s,
        }


def make_output_dir(output_root, prefix):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"session_{stamp}_{prefix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_metadata(path, metadata):
    with open(Path(path) / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
