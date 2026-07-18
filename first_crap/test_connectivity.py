from brainaccess import core

core.init()
print("version:", core.get_version())

try:
    core.set_config_fields(
        enable_logs=True,
        log_path="brainaccess_core.log",
        log_level=0,
        adapter_index=0,
        append_logs=False,
        autoflush=True,
    )

    print("before scan")
    devices = core.scan()
    print("after scan")
    print([device.name for device in devices])

finally:
    core.close()