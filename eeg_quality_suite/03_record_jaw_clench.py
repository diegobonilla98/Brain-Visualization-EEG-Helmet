import time
from pathlib import Path

import pandas as pd
from psychopy import event

from brainaccess_stream import BrainAccessStream, make_output_dir, save_metadata
from stimulus_common import make_window, estimate_refresh_rate, countdown, frames_for_seconds, perf_time_after_flip, draw_center_text, wait_with_text, play_beep
from analysis_common import save_labeled_samples, plot_recording_summary


DEVICE_NAME = "BA MAXI 034"
OUTPUT_ROOT = "sessions"
RELAX_SECONDS = 12.0
CLENCH_SECONDS = 5.0
REPETITIONS = 10
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


def present_state(win, events, recorder, refresh_rate_hz, seconds, label, text, frame_counter, beep=False):
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(label)
    if beep:
        play_beep(850, 160)
    started = False
    for local_frame in range(total_frames):
        draw_center_text(win, f"{text}\n\n{int(seconds - local_frame / refresh_rate_hz) + 1}s\nPress ESC to stop", height=0.07)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", label, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", label, perf_time, flip_time, frame_counter, text)
        frame_counter += 1
        if "escape" in event.getKeys():
            raise KeyboardInterrupt
    add_event(events, "state_end", label, time.perf_counter(), None, frame_counter, None)
    return frame_counter


def main():
    output_dir = make_output_dir(OUTPUT_ROOT, "jaw")
    events = []
    win = make_window("black")
    refresh_rate_hz = estimate_refresh_rate(win)
    frame_counter = 0
    try:
        wait_with_text(win, "JAW CLENCH TEST\nAlternate relaxed jaw and strong jaw clench.\nDo not move the helmet. Do not touch electrodes.\nPress ESC to abort.", 5.0)
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            countdown(win, 3)
            for repetition in range(REPETITIONS):
                frame_counter = present_state(win, events, recorder, refresh_rate_hz, RELAX_SECONDS, "relaxed_jaw", "RELAX JAW", frame_counter, beep=False)
                frame_counter = present_state(win, events, recorder, refresh_rate_hz, CLENCH_SECONDS, "jaw_clench", "CLENCH JAW", frame_counter, beep=True)
            frame_counter = present_state(win, events, recorder, refresh_rate_hz, RELAX_SECONDS, "relaxed_jaw", "RELAX JAW", frame_counter, beep=False)
            recorder.stop()
            eeg_df = recorder.dataframe()
            metadata = recorder.metadata()
            metadata.update({
                "test": "jaw_clench",
                "display_refresh_rate_hz": refresh_rate_hz,
                "relax_seconds": RELAX_SECONDS,
                "clench_seconds": CLENCH_SECONDS,
                "repetitions": REPETITIONS,
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
