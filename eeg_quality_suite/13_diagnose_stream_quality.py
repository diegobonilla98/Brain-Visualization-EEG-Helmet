import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


RECORDINGS_ROOT = "sessions"
OUTPUT_JSON = "recordings/eeg_stream_quality_report.json"
OUTPUT_CHANNEL_CSV = "recordings/eeg_channel_health_report.csv"
IMPUTATION_REFERENCE = "eyes"
IMPUTATION_MASK_SOURCE = "ssvep"
IMPUTATION_SAMPLES = 30000
EEG_RAIL_UV = 550000.0
DEAD_RANGE98_UV = 50.0


def eeg_columns(df):
    return [column for column in df.columns if column.endswith("_uV")]


def valid_mask(df):
    if "streaming" in df.columns:
        return df["streaming"].to_numpy(dtype=float) > 0.5
    return np.ones(len(df), dtype=bool)


def contiguous_runs(mask):
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return []
    transitions = np.flatnonzero(np.diff(values.astype(int)) != 0) + 1
    boundaries = np.r_[0, transitions, len(values)]
    runs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        runs.append({
            "value": bool(values[start]),
            "start": int(start),
            "end": int(end),
            "samples": int(end - start),
        })
    return runs


def run_length_summary(runs, fs):
    lengths = np.asarray([run["samples"] for run in runs], dtype=float)
    if len(lengths) == 0:
        return {
            "count": 0,
            "samples_median": 0.0,
            "samples_p95": 0.0,
            "samples_max": 0.0,
            "ms_median": 0.0,
            "ms_p95": 0.0,
            "ms_max": 0.0,
        }
    return {
        "count": int(len(lengths)),
        "samples_median": float(np.median(lengths)),
        "samples_p95": float(np.percentile(lengths, 95)),
        "samples_max": float(np.max(lengths)),
        "ms_median": float(np.median(lengths) / fs * 1000.0),
        "ms_p95": float(np.percentile(lengths, 95) / fs * 1000.0),
        "ms_max": float(np.max(lengths) / fs * 1000.0),
    }


def channel_health(df, recording_name):
    columns = eeg_columns(df)
    valid = valid_mask(df)
    rows = []
    for column in columns:
        values = df.loc[valid, column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            rows.append({
                "recording": recording_name,
                "channel": column,
                "valid_samples": 0,
                "median_uV": None,
                "robust_sigma_uV": None,
                "range98_uV": None,
                "flat_fraction": None,
                "rail_fraction": None,
                "dead_like": True,
            })
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        p01 = float(np.percentile(values, 1))
        p99 = float(np.percentile(values, 99))
        range98 = p99 - p01
        flat_fraction = float(np.mean(np.abs(np.diff(values)) < 1e-12)) if len(values) > 1 else 0.0
        rail_fraction = float(np.mean(np.abs(values) >= EEG_RAIL_UV))
        rows.append({
            "recording": recording_name,
            "channel": column,
            "valid_samples": int(len(values)),
            "median_uV": median,
            "robust_sigma_uV": 1.4826 * mad,
            "range98_uV": range98,
            "flat_fraction": flat_fraction,
            "rail_fraction": rail_fraction,
            "dead_like": bool(range98 < DEAD_RANGE98_UV or flat_fraction > 0.99),
        })
    return rows


def recording_summary(recording_dir):
    eeg_path = recording_dir / "eeg_samples.csv"
    metadata_path = recording_dir / "metadata.json"
    labeled_path = recording_dir / "eeg_samples_labeled.csv"
    df = pd.read_csv(eeg_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    labeled_df = pd.read_csv(labeled_path) if labeled_path.exists() else df
    fs = float(metadata.get("sample_frequency_hz", 250.0))
    valid = valid_mask(df)
    invalid_runs = [run for run in contiguous_runs(~valid) if run["value"]]
    valid_runs = [run for run in contiguous_runs(valid) if run["value"]]
    columns = eeg_columns(df)
    zero_all = (df[columns].fillna(0.0).abs().sum(axis=1).to_numpy(dtype=float) == 0.0) if len(columns) > 0 else np.zeros(len(df), dtype=bool)
    label_invalid = {}
    if "state_label" in labeled_df.columns:
        label_data = pd.DataFrame({
            "state_label": labeled_df["state_label"].astype(str),
            "invalid": ~valid,
        })
        label_invalid = label_data.groupby("state_label")["invalid"].agg(["count", "mean"]).to_dict(orient="index")
    invalid_lengths = [run["samples"] for run in invalid_runs]
    modulo_9_fraction = float(np.mean((np.asarray(invalid_lengths) % 9) == 0)) if len(invalid_lengths) > 0 else 0.0
    chunk_rows = df.groupby("pc_time_received_s")["sample_index"].count() if "pc_time_received_s" in df.columns else pd.Series(dtype=int)
    return {
        "recording": recording_dir.name,
        "rows": int(len(df)),
        "duration_s": float(len(df) / fs),
        "fs_hz": fs,
        "valid_fraction": float(np.mean(valid)) if len(valid) > 0 else 0.0,
        "invalid_fraction": float(np.mean(~valid)) if len(valid) > 0 else 0.0,
        "invalid_samples": int(np.sum(~valid)),
        "invalid_duration_s": float(np.sum(~valid) / fs),
        "invalid_run_summary": run_length_summary(invalid_runs, fs),
        "valid_run_summary": run_length_summary(valid_runs, fs),
        "invalid_run_lengths_top": [int(value) for value in sorted(invalid_lengths, reverse=True)[:12]],
        "invalid_run_lengths_modulo_9_fraction": modulo_9_fraction,
        "zero_all_equals_invalid": bool(np.array_equal(zero_all, ~valid)),
        "chunk_size_counts_top": {str(int(index)): int(value) for index, value in chunk_rows.value_counts().head(10).items()},
        "label_invalid_fraction": label_invalid,
    }


def interpolate_linear(data, observed):
    output = np.asarray(data, dtype=float).copy()
    x = np.arange(output.shape[0])
    for column_index in range(output.shape[1]):
        y = output[:, column_index]
        if np.sum(observed) == 0:
            output[:, column_index] = 0.0
        elif np.sum(observed) == 1:
            output[:, column_index] = y[observed][0]
        else:
            output[:, column_index] = np.interp(x, x[observed], y[observed])
    return output


def band_power_ratio(original, reconstructed, fs, fmin, fmax):
    nperseg = min(len(original), int(fs * 4))
    ratios = []
    for column_index in range(original.shape[1]):
        freqs, original_psd = welch(original[:, column_index], fs=fs, nperseg=nperseg)
        _, reconstructed_psd = welch(reconstructed[:, column_index], fs=fs, nperseg=nperseg)
        band = (freqs >= fmin) & (freqs <= fmax)
        original_power = np.trapezoid(original_psd[band], freqs[band])
        reconstructed_power = np.trapezoid(reconstructed_psd[band], freqs[band])
        ratios.append(float(reconstructed_power / (original_power + 1e-18)))
    return float(np.median(ratios))


def imputation_stress_test(recording_dirs):
    reference_dir = next((path for path in recording_dirs if path.name.startswith(IMPUTATION_REFERENCE) or f"_{IMPUTATION_REFERENCE}" in path.name), None)
    mask_dir = next((path for path in recording_dirs if path.name.startswith(IMPUTATION_MASK_SOURCE) or f"_{IMPUTATION_MASK_SOURCE}" in path.name), None)
    if reference_dir is None or mask_dir is None:
        return {}
    reference_df = pd.read_csv(reference_dir / "eeg_samples.csv")
    mask_df = pd.read_csv(mask_dir / "eeg_samples.csv")
    columns = eeg_columns(reference_df)
    fs = 250.0
    reference_valid = valid_mask(reference_df)
    clean_runs = [run for run in contiguous_runs(reference_valid) if run["value"] and run["samples"] >= IMPUTATION_SAMPLES]
    if len(clean_runs) == 0:
        return {}
    clean_start = clean_runs[0]["start"]
    clean_end = clean_start + IMPUTATION_SAMPLES
    true_data = reference_df.iloc[clean_start:clean_end][columns].to_numpy(dtype=float)
    mask = valid_mask(mask_df)[:IMPUTATION_SAMPLES]
    if len(mask) < IMPUTATION_SAMPLES:
        return {}
    if np.all(mask):
        return {
            "reference_recording": reference_dir.name,
            "mask_recording": mask_dir.name,
            "samples": IMPUTATION_SAMPLES,
            "mask_valid_fraction": 1.0,
            "note": "No missing samples in the selected mask recording; imputation stress test skipped.",
        }
    zero_fill = true_data.copy()
    zero_fill[~mask, :] = 0.0
    linear_data = true_data.copy()
    linear_data[~mask, :] = np.nan
    linear_fill = interpolate_linear(linear_data, mask)
    centered_true = true_data - np.mean(true_data, axis=0, keepdims=True)
    centered_zero = zero_fill - np.mean(zero_fill, axis=0, keepdims=True)
    centered_linear = linear_fill - np.mean(linear_fill, axis=0, keepdims=True)
    sigma = np.std(centered_true, axis=0, keepdims=True) + 1e-18
    missing = ~mask
    zero_nrmse_missing = float(np.sqrt(np.mean(((centered_zero[missing] - centered_true[missing]) / sigma) ** 2)))
    linear_nrmse_missing = float(np.sqrt(np.mean(((centered_linear[missing] - centered_true[missing]) / sigma) ** 2)))
    zero_corr = float(np.corrcoef(centered_true.reshape(-1), centered_zero.reshape(-1))[0, 1])
    linear_corr = float(np.corrcoef(centered_true.reshape(-1), centered_linear.reshape(-1))[0, 1])
    return {
        "reference_recording": reference_dir.name,
        "mask_recording": mask_dir.name,
        "samples": IMPUTATION_SAMPLES,
        "mask_valid_fraction": float(np.mean(mask)),
        "zero_fill_nrmse_on_missing": zero_nrmse_missing,
        "linear_fill_nrmse_on_missing": linear_nrmse_missing,
        "zero_fill_correlation_all": zero_corr,
        "linear_fill_correlation_all": linear_corr,
        "linear_bandpower_ratio_1_4_hz": band_power_ratio(centered_true, centered_linear, fs, 1, 4),
        "linear_bandpower_ratio_8_12_hz": band_power_ratio(centered_true, centered_linear, fs, 8, 12),
        "linear_bandpower_ratio_30_95_hz": band_power_ratio(centered_true, centered_linear, fs, 30, 95),
    }


def main():
    root = Path(RECORDINGS_ROOT)
    recording_dirs = sorted([path for path in root.iterdir() if path.is_dir() and (path / "eeg_samples.csv").exists()])
    summaries = []
    channel_rows = []
    for recording_dir in recording_dirs:
        df = pd.read_csv(recording_dir / "eeg_samples.csv")
        summaries.append(recording_summary(recording_dir))
        channel_rows.extend(channel_health(df, recording_dir.name))
    report = {
        "recordings": summaries,
        "imputation_stress_test": imputation_stress_test(recording_dirs),
    }
    output_json = Path(OUTPUT_JSON)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(channel_rows).to_csv(OUTPUT_CHANNEL_CSV, index=False)
    print(json.dumps(report, indent=2))
    print("Saved:", OUTPUT_JSON)
    print("Saved:", OUTPUT_CHANNEL_CSV)


if __name__ == "__main__":
    main()
