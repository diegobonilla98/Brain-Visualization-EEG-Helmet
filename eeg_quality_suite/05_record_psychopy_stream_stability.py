import json

import pandas as pd
from psychopy import event, visual

from brainaccess_stream import BrainAccessStream, make_output_dir
from stimulus_common import make_window, estimate_refresh_rate, frames_for_seconds, perf_time_after_flip, draw_center_text


DEVICE_NAME = "BA MAXI 034"
OUTPUT_ROOT = "sessions"
GAIN_NAME = "X8"
USE_SAMPLE_NUMBER = True
USE_STREAMING = True
USE_BIAS = False
TARGET_FLICKER_HZ = 10.0
BASELINE_SECONDS = 8.0
FLICKER_SECONDS = 4.0
REST_SECONDS = 8.0
REPETITIONS = 8


def summarize_stream(df):
    valid = df["streaming"].to_numpy(dtype=float) > 0.5 if "streaming" in df.columns else pd.Series([True] * len(df)).to_numpy(dtype=bool)
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


def draw_flicker_frame(win, refresh_rate_hz, frame_index):
    frames_per_cycle = max(2, int(round(refresh_rate_hz / TARGET_FLICKER_HZ)))
    bright_frames = max(1, frames_per_cycle // 2)
    phase = frame_index % frames_per_cycle
    bright = phase < bright_frames
    color = "white" if bright else "black"
    text_color = "black" if bright else "white"
    rect = visual.Rect(win, width=3.0, height=3.0, fillColor=color, lineColor=color)
    cross = visual.TextStim(win, text="+", height=0.10, color=text_color)
    rect.draw()
    cross.draw()


def draw_rest_frame(win, seconds_left):
    draw_center_text(win, f"REST\n{int(seconds_left) + 1}s\nPress ESC to stop", height=0.07)


def run_state(win, events, refresh_rate_hz, seconds, state_label, frame_counter):
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    for local_frame in range(total_frames):
        seconds_left = seconds - (local_frame / refresh_rate_hz)
        if state_label == "flicker":
            draw_flicker_frame(win, refresh_rate_hz, local_frame)
        else:
            draw_rest_frame(win, seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        events.append({
            "frame_index": frame_counter,
            "local_frame_index": local_frame,
            "pc_time_perf_counter_s": perf_time,
            "flip_time_psychopy_s": flip_time,
            "state_label": state_label,
        })
        frame_counter += 1
        if "escape" in event.getKeys():
            return frame_counter, True
    return frame_counter, False


def main():
    output_dir = make_output_dir(OUTPUT_ROOT, "psychopy_stream_stability")
    events = []
    win = make_window("black")
    refresh_rate_hz = estimate_refresh_rate(win)
    duration_seconds = BASELINE_SECONDS + (FLICKER_SECONDS + REST_SECONDS) * REPETITIONS
    frame_counter = 0
    try:
        with BrainAccessStream(
            device_name=DEVICE_NAME,
            gain_name=GAIN_NAME,
            use_sample_number=USE_SAMPLE_NUMBER,
            use_streaming=USE_STREAMING,
            use_bias=USE_BIAS,
        ) as recorder:
            recorder.start()
            frame_counter, stopped = run_state(win, events, refresh_rate_hz, BASELINE_SECONDS, "baseline_rest", frame_counter)
            if not stopped:
                for repetition in range(REPETITIONS):
                    frame_counter, stopped = run_state(win, events, refresh_rate_hz, FLICKER_SECONDS, "flicker", frame_counter)
                    if stopped:
                        break
                    frame_counter, stopped = run_state(win, events, refresh_rate_hz, REST_SECONDS, f"rest_{repetition + 1}", frame_counter)
                    if stopped:
                        break
            recorder.stop()
            df = recorder.dataframe()
            metadata = recorder.metadata()
    finally:
        win.close()
    summary = summarize_stream(df)
    metadata.update({
        "test": "psychopy_stream_stability",
        "duration_seconds": duration_seconds,
        "baseline_seconds": BASELINE_SECONDS,
        "flicker_seconds": FLICKER_SECONDS,
        "rest_seconds": REST_SECONDS,
        "repetitions": REPETITIONS,
        "target_flicker_hz": TARGET_FLICKER_HZ,
        "display_refresh_rate_hz": refresh_rate_hz,
        "stream_summary": summary,
    })
    df.to_csv(output_dir / "eeg_samples.csv", index=False)
    pd.DataFrame(events).to_csv(output_dir / "events.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with open(output_dir / "stream_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
