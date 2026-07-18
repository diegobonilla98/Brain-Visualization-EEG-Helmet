import time
from pathlib import Path

import pandas as pd
from psychopy import visual, event

from brainaccess_stream import BrainAccessStream, make_output_dir, save_metadata
from stimulus_common import make_window, estimate_refresh_rate, countdown, frames_for_seconds, flicker_pattern, perf_time_after_flip, draw_center_text, wait_with_text
from analysis_common import save_labeled_samples, plot_recording_summary


DEVICE_NAME = "BA MAXI 034"
OUTPUT_ROOT = "sessions"
TARGET_FLICKER_HZ = 10.0
BASELINE_SECONDS = 12.0
FLICKER_SECONDS = 20.0
REST_SECONDS = 10.0
REPETITIONS = 4
FLASH_FULL_SCREEN = True
FLASH_PATCH_HEIGHT = 0.65
SUMMARY_NORMALIZATION = "per_channel"
GAIN_NAME = "X8"


def add_event(events, event_type, state_label, pc_time_perf_counter_s, flip_time_psychopy_s=None, frame_index=None, value=None):
    events.append({
        "event_type": event_type,
        "state_label": state_label,
        "pc_time_perf_counter_s": pc_time_perf_counter_s,
        "flip_time_psychopy_s": flip_time_psychopy_s,
        "frame_index": frame_index,
        "value": value,
    })


def present_rest(win, events, recorder, refresh_rate_hz, seconds, label, frame_counter):
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(label)
    started = False
    for local_frame in range(total_frames):
        draw_center_text(win, f"{label}\nLook at the cross\nPress ESC to stop", height=0.055)
        draw_center_text(win, "+", height=0.12, pos=(0, -0.18))
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", label, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", label, perf_time, flip_time, frame_counter, "rest")
        frame_counter += 1
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
    add_event(events, "state_end", label, time.perf_counter(), None, frame_counter, None)
    return frame_counter


def present_flicker(win, events, recorder, refresh_rate_hz, seconds, target_hz, frame_counter):
    frames_per_cycle, bright_frames, dark_frames, actual_hz = flicker_pattern(refresh_rate_hz, target_hz)
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    label = f"flicker_{actual_hz:.4f}Hz"
    recorder.annotate(label)
    if FLASH_FULL_SCREEN:
        flash_rect = visual.Rect(win, width=3.0, height=3.0, fillColor="black", lineColor="black")
    else:
        flash_rect = visual.Rect(win, width=FLASH_PATCH_HEIGHT, height=FLASH_PATCH_HEIGHT, fillColor="black", lineColor="black")
    fixation = visual.TextStim(win, text="+", height=0.10, color="white")
    started = False
    for local_frame in range(total_frames):
        phase = local_frame % frames_per_cycle
        bright = phase < bright_frames
        flash_rect.fillColor = "white" if bright else "black"
        flash_rect.lineColor = "white" if bright else "black"
        fixation.color = "black" if bright else "white"
        flash_rect.draw()
        fixation.draw()
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", label, perf_time, flip_time, frame_counter, actual_hz)
            started = True
        add_event(events, "frame", label, perf_time, flip_time, frame_counter, "white" if bright else "black")
        frame_counter += 1
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
    add_event(events, "state_end", label, time.perf_counter(), None, frame_counter, actual_hz)
    return frame_counter, actual_hz, frames_per_cycle, bright_frames, dark_frames


def main():
    output_dir = make_output_dir(OUTPUT_ROOT, "ssvep")
    events = []
    win = make_window("black")
    refresh_rate_hz = estimate_refresh_rate(win)
    frame_counter = 0
    actual_hz = None
    frames_per_cycle = None
    bright_frames = None
    dark_frames = None
    try:
        wait_with_text(win, "SSVEP FLICKER TEST\nSit still. Relax jaw. Look at the cross.\nDo not run this if flicker bothers you.\nPress ESC to abort.", 5.0)
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            countdown(win, 3)
            frame_counter = present_rest(win, events, recorder, refresh_rate_hz, BASELINE_SECONDS, "baseline_open_eyes", frame_counter)
            for repetition in range(REPETITIONS):
                frame_counter, actual_hz, frames_per_cycle, bright_frames, dark_frames = present_flicker(win, events, recorder, refresh_rate_hz, FLICKER_SECONDS, TARGET_FLICKER_HZ, frame_counter)
                frame_counter = present_rest(win, events, recorder, refresh_rate_hz, REST_SECONDS, f"rest_{repetition + 1}", frame_counter)
            recorder.stop()
            eeg_df = recorder.dataframe()
            metadata = recorder.metadata()
            metadata.update({
                "test": "ssvep_flicker",
                "display_refresh_rate_hz": refresh_rate_hz,
                "target_flicker_hz": TARGET_FLICKER_HZ,
                "actual_flicker_hz": actual_hz,
                "frames_per_cycle": frames_per_cycle,
                "bright_frames": bright_frames,
                "dark_frames": dark_frames,
                "software_timing_note": "Stimulus state is logged after PsychoPy win.flip and aligned to EEG callback receive timing using time.perf_counter.",
            })
            eeg_df.to_csv(output_dir / "eeg_samples.csv", index=False)
            pd.DataFrame(events).to_csv(output_dir / "events.csv", index=False)
            annotations = recorder.get_annotations()
            pd.DataFrame(annotations).to_csv(output_dir / "brainaccess_annotations.csv", index=False)
            save_metadata(output_dir, metadata)
        save_labeled_samples(output_dir)
        plot_recording_summary(output_dir, SUMMARY_NORMALIZATION)
        wait_with_text(win, f"DONE\nSaved to:\n{Path(output_dir).resolve()}", 3.0)
    finally:
        win.close()


if __name__ == "__main__":
    main()
