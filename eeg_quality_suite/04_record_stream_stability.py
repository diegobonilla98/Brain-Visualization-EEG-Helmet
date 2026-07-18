import json
import time
from pathlib import Path

import pandas as pd

from brainaccess_stream import BrainAccessStream, make_output_dir


DEVICE_NAME = "BA MAXI 034"
OUTPUT_ROOT = "sessions"
DURATION_SECONDS = 30.0
GAIN_NAME = "X8"
USE_SAMPLE_NUMBER = False
USE_STREAMING = True
USE_BIAS = True
BIAS_CHANNEL_NAME = "Iz"
SAVE_RAW_CSV = True

CHANNEL_GROUPS = [
    {
        "name": "all_32_no_sample_bias",
        "channels": None,
        "use_sample_number": False,
        "use_bias": True,
    },
    {
        "name": "all_32_sample_bias",
        "channels": None,
        "use_sample_number": True,
        "use_bias": True,
    },
    {
        "name": "all_32_sample_no_bias",
        "channels": None,
        "use_sample_number": True,
        "use_bias": False,
    },
    {
        "name": "posterior_8",
        "channels": ["P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz"],
        "use_sample_number": False,
        "use_bias": True,
    },
    {
        "name": "temporal_frontal_8",
        "channels": ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "T7", "T8"],
        "use_sample_number": False,
        "use_bias": True,
    },
    {
        "name": "minimal_4",
        "channels": ["O1", "Oz", "T7", "T8"],
        "use_sample_number": False,
        "use_bias": True,
    },
]


def summarize_stream(df):
    if "streaming" in df.columns:
        valid = df["streaming"].to_numpy(dtype=float) > 0.5
    else:
        valid = pd.Series([True] * len(df)).to_numpy(dtype=bool)
    invalid = ~valid
    transitions = pd.Series(invalid).astype(int).diff().fillna(0).ne(0).to_numpy()
    run_starts = [0] + list(pd.Series(transitions).loc[lambda series: series].index)
    run_starts = sorted(set(int(value) for value in run_starts if int(value) < len(df)))
    run_ends = run_starts[1:] + [len(df)]
    invalid_lengths = []
    for start, end in zip(run_starts, run_ends):
        if invalid[start]:
            invalid_lengths.append(end - start)
    return {
        "samples": int(len(df)),
        "valid_fraction": float(valid.mean()) if len(valid) > 0 else 0.0,
        "invalid_samples": int(invalid.sum()),
        "invalid_run_count": int(len(invalid_lengths)),
        "invalid_run_max_samples": int(max(invalid_lengths)) if len(invalid_lengths) > 0 else 0,
        "invalid_run_median_samples": float(pd.Series(invalid_lengths).median()) if len(invalid_lengths) > 0 else 0.0,
    }


def main():
    output_dir = make_output_dir(OUTPUT_ROOT, "stream_stability")
    rows = []
    for group in CHANNEL_GROUPS:
        name = group["name"]
        channels = group["channels"]
        use_sample_number = group.get("use_sample_number", USE_SAMPLE_NUMBER)
        use_bias = group.get("use_bias", USE_BIAS)
        print("Testing:", name)
        with BrainAccessStream(
            device_name=DEVICE_NAME,
            gain_name=GAIN_NAME,
            use_sample_number=use_sample_number,
            use_streaming=USE_STREAMING,
            enabled_channel_names=channels,
            use_bias=use_bias,
            bias_channel_name=BIAS_CHANNEL_NAME,
        ) as recorder:
            recorder.start()
            time.sleep(DURATION_SECONDS)
            recorder.stop()
            df = recorder.dataframe()
            metadata = recorder.metadata()
        summary = summarize_stream(df)
        row = {
            "group": name,
            "duration_seconds": DURATION_SECONDS,
            "enabled_channels": "all" if channels is None else ",".join(channels),
            "use_sample_number": use_sample_number,
            "use_bias": use_bias,
            **summary,
        }
        rows.append(row)
        if SAVE_RAW_CSV:
            df.to_csv(output_dir / f"{name}_eeg_samples.csv", index=False)
        metadata.update({
            "test": "stream_stability",
            "channel_group": name,
            "duration_seconds": DURATION_SECONDS,
            "stream_summary": summary,
        })
        with open(output_dir / f"{name}_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "stream_stability_summary.csv", index=False)
    with open(output_dir / "stream_stability_summary.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    print(summary_df.to_string(index=False))
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
