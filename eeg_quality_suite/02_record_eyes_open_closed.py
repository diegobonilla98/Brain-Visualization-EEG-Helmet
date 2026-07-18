import time
from pathlib import Path

import pandas as pd
from psychopy import event

from brainaccess_stream import BrainAccessStream, make_output_dir, save_metadata
from stimulus_common import make_window, estimate_refresh_rate, countdown, frames_for_seconds, perf_time_after_flip, draw_center_text, wait_with_text, play_beep
from analysis_common import save_labeled_samples, plot_recording_summary


DEVICE_NAME = "BA MAXI 034"
OUTPUT_ROOT = "sessions"
CLOSE_SECONDS = 25.0
OPEN_SECONDS = 25.0
REPETITIONS = 5
INCLUDE_BLINK_BLOCK = True
BLINK_BLOCK_SECONDS = 24.0
BLINK_EVERY_SECONDS = 2.0
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


def present_text_state(win, events, recorder, refresh_rate_hz, seconds, label, screen_text, frame_counter, beep=False):
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(label)
    if beep:
        play_beep(1200, 250)
    started = False
    for local_frame in range(total_frames):
        draw_center_text(win, f"{screen_text}\n\n{int(seconds - local_frame / refresh_rate_hz) + 1}s\nPress ESC to stop", height=0.07)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", label, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", label, perf_time, flip_time, frame_counter, screen_text)
        frame_counter += 1
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
    add_event(events, "state_end", label, time.perf_counter(), None, frame_counter, None)
    return frame_counter


def present_blink_block(win, events, recorder, refresh_rate_hz, seconds, every_seconds, frame_counter):
    label = "strong_blinks"
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    every_frames = max(1, frames_for_seconds(refresh_rate_hz, every_seconds))
    recorder.annotate(label)
    started = False
    for local_frame in range(total_frames):
        should_beep = local_frame % every_frames == 0
        if should_beep:
            play_beep(1000, 120)
        draw_center_text(win, "STRONG BLINKS\nBlink hard on every beep\nThen keep face relaxed", height=0.065)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", label, perf_time, flip_time, frame_counter, None)
            started = True
        if should_beep:
            add_event(events, "blink_beep", label, perf_time, flip_time, frame_counter, "blink_now")
        add_event(events, "frame", label, perf_time, flip_time, frame_counter, "blink_block")
        frame_counter += 1
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
    add_event(events, "state_end", label, time.perf_counter(), None, frame_counter, None)
    return frame_counter


def main():
    output_dir = make_output_dir(OUTPUT_ROOT, "eyes")
    events = []
    win = make_window("black")
    refresh_rate_hz = estimate_refresh_rate(win)
    frame_counter = 0
    try:
        wait_with_text(win, "EYES OPEN VS CLOSED TEST\nWhen OPEN EYES appears you will hear a beep.\nSit still. Jaw relaxed. No deliberate movement.\nPress ESC to abort.", 5.0)
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            countdown(win, 3)
            for repetition in range(REPETITIONS):
                frame_counter = present_text_state(win, events, recorder, refresh_rate_hz, CLOSE_SECONDS, "closed_eyes", "CLOSE EYES", frame_counter, beep=False)
                frame_counter = present_text_state(win, events, recorder, refresh_rate_hz, OPEN_SECONDS, "open_eyes", "OPEN EYES", frame_counter, beep=True)
            if INCLUDE_BLINK_BLOCK:
                frame_counter = present_blink_block(win, events, recorder, refresh_rate_hz, BLINK_BLOCK_SECONDS, BLINK_EVERY_SECONDS, frame_counter)
            recorder.stop()
            eeg_df = recorder.dataframe()
            metadata = recorder.metadata()
            metadata.update({
                "test": "eyes_open_closed_and_blinks",
                "display_refresh_rate_hz": refresh_rate_hz,
                "close_seconds": CLOSE_SECONDS,
                "open_seconds": OPEN_SECONDS,
                "repetitions": REPETITIONS,
                "include_blink_block": INCLUDE_BLINK_BLOCK,
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
