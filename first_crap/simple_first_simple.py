import time
from brainaccess import core
from brainaccess.core.eeg_manager import EEGManager

core.init()

devices = core.scan()
device_names = [device.name for device in devices]
print("Found devices:", device_names)

device_name = next(name for name in device_names if "BA MAXI" in name)
print("Connecting to:", device_name)

with EEGManager() as mgr:
    status = mgr.connect(device_name)
    print("Connection status:", status)

    if status == 2:
        raise RuntimeError("Connected, but firmware stream is incompatible. Update firmware from BrainAccess Board.")

    if status != 0:
        raise RuntimeError(f"Connection failed with status {status}")

    print("Connected:", mgr.is_connected())
    print("Battery:", mgr.get_battery_info().level)
    print("Sample frequency:", mgr.get_sample_frequency())
    print("Features:", mgr.get_device_features())

core.close()