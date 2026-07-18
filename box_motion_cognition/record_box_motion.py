import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import event, visual

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_common import plot_recording_summary
from box_motion_common import DIRECTION_LABELS, PROTOCOL_VERSION, resolve_project_path, save_labeled_box_motion_samples
from brainaccess_stream import BrainAccessStream, save_metadata
from stimulus_common import countdown, estimate_refresh_rate, frames_for_seconds, make_window, perf_time_after_flip, play_beep, wait_with_text


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
OUTPUT_ROOT = "sessions"
SESSION_PREFIX = "session"
SAMPLES_PER_DIRECTION = 6
BASELINE_SECONDS = 4.0
CUE_SECONDS = 2.0
IMAGERY_SECONDS = 7.0
COOLDOWN_SECONDS = 2.0
RANDOM_SEED = 20260621
SUMMARY_NORMALIZATION = "per_channel"
BACKGROUND_COLOR = "#101014"
BOX_COLOR = "#ededed"
TARGET_COLOR = "#5ba8ff"


DIRECTION_TEXT = {
    "up": "Think the box upward.",
    "down": "Think the box downward.",
    "left": "Think the box left.",
    "right": "Think the box right.",
}
DIRECTION_OFFSET = {
    "up": (0.0, 0.22),
    "down": (0.0, -0.22),
    "left": (-0.32, 0.0),
    "right": (0.32, 0.0),
}


class BoxMotionRenderer:
    def __init__(self, win):
        self.win = win
        self.box = visual.Rect(win, width=0.14, height=0.14, fillColor=BOX_COLOR, lineColor=BOX_COLOR, pos=(0.0, 0.0))
        self.target = visual.Rect(win, width=0.16, height=0.16, fillColor=None, lineColor=TARGET_COLOR, lineWidth=3, pos=(0.0, 0.0))
        self.title = visual.TextStim(win, text="", height=0.056, color="white", pos=(0.0, 0.34), wrapWidth=1.5, alignText="center")
        self.detail = visual.TextStim(win, text="", height=0.038, color="#d8d8d8", pos=(0.0, -0.35), wrapWidth=1.45, alignText="center")
        self.arrow = visual.TextStim(win, text="", height=0.095, color=TARGET_COLOR, pos=(0.0, 0.0), wrapWidth=1.0, alignText="center")

    def draw_baseline(self, trial_index, total_trials, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.title.text = "BASELINE"
        self.detail.text = f"Trial {trial_index}/{total_trials}\nRelax. Do not think about moving the box.\n{int(seconds_left) + 1}s"
        self.box.pos = (0.0, 0.0)
        self.box.draw()
        self.title.draw()
        self.detail.draw()

    def draw_cue(self, direction, trial_index, total_trials, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.title.text = direction.upper()
        self.detail.text = f"Trial {trial_index}/{total_trials}\nGet ready. Do not move muscles.\n{int(seconds_left) + 1}s"
        self.box.pos = (0.0, 0.0)
        self.target.pos = DIRECTION_OFFSET[direction]
        self.arrow.text = self.arrow_for_direction(direction)
        self.arrow.pos = DIRECTION_OFFSET[direction]
        self.box.draw()
        self.target.draw()
        self.arrow.draw()
        self.title.draw()
        self.detail.draw()

    def draw_imagery(self, direction, trial_index, total_trials, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.title.text = direction.upper()
        self.detail.text = f"{DIRECTION_TEXT[direction]}\nNo actual movement. Jaw and face relaxed.\n{int(seconds_left) + 1}s"
        self.box.pos = (0.0, 0.0)
        self.target.pos = DIRECTION_OFFSET[direction]
        self.arrow.text = self.arrow_for_direction(direction)
        self.arrow.pos = DIRECTION_OFFSET[direction]
        self.box.draw()
        self.target.draw()
        self.arrow.draw()
        self.title.draw()
        self.detail.draw()

    def draw_cooldown(self, trial_index, total_trials, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.title.text = "RESET"
        self.detail.text = f"Trial {trial_index}/{total_trials}\nRelease the image. Blink if needed.\n{int(seconds_left) + 1}s"
        self.box.pos = (0.0, 0.0)
        self.box.draw()
        self.title.draw()
        self.detail.draw()

    def arrow_for_direction(self, direction):
        if direction == "up":
            return "^"
        if direction == "down":
            return "v"
        if direction == "left":
            return "<"
        return ">"


def make_session_output_dir():
    root = resolve_project_path(OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"session_{stamp}_{SESSION_PREFIX}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def make_schedule():
    rng = np.random.default_rng(RANDOM_SEED + int(time.time()))
    labels = []
    for direction in DIRECTION_LABELS:
        labels.extend([direction] * SAMPLES_PER_DIRECTION)
    rng.shuffle(labels)
    rows = []
    for trial_index, direction in enumerate(labels, start=1):
        rows.append({
            "trial_index": trial_index,
            "direction_label": direction,
            "class_label": direction,
        })
    return rows


def add_event(events, event_type, state_label, state_phase, direction_label, trial_index, pc_time_perf_counter_s, flip_time_psychopy_s, frame_index, value):
    events.append({
        "event_type": event_type,
        "state_label": state_label,
        "state_phase": state_phase,
        "direction_label": direction_label,
        "class_label": direction_label,
        "trial_index": int(trial_index),
        "pc_time_perf_counter_s": pc_time_perf_counter_s,
        "flip_time_psychopy_s": flip_time_psychopy_s,
        "frame_index": int(frame_index),
        "value": value,
    })


def phase_seconds(phase):
    if phase == "baseline":
        return BASELINE_SECONDS
    if phase == "cue":
        return CUE_SECONDS
    if phase == "imagery":
        return IMAGERY_SECONDS
    return COOLDOWN_SECONDS


def draw_phase(renderer, phase, direction, trial_index, total_trials, seconds_left):
    if phase == "baseline":
        renderer.draw_baseline(trial_index, total_trials, seconds_left)
    elif phase == "cue":
        renderer.draw_cue(direction, trial_index, total_trials, seconds_left)
    elif phase == "imagery":
        renderer.draw_imagery(direction, trial_index, total_trials, seconds_left)
    else:
        renderer.draw_cooldown(trial_index, total_trials, seconds_left)


def state_label_for(direction, phase):
    if phase == "baseline":
        return f"baseline_before_{direction}"
    if phase == "cue":
        return f"cue_{direction}"
    if phase == "imagery":
        return f"imagery_{direction}"
    return f"cooldown_after_{direction}"


def present_phase(win, renderer, events, recorder, refresh_rate_hz, row, phase, frame_counter, total_trials):
    direction = row["direction_label"]
    trial_index = int(row["trial_index"])
    seconds = phase_seconds(phase)
    state_label = state_label_for(direction, phase)
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(state_label)
    if phase == "cue":
        play_beep(760, 120)
    if phase == "imagery":
        play_beep(1100, 140)
    started = False
    for local_frame in range(total_frames):
        seconds_left = seconds - local_frame / refresh_rate_hz
        draw_phase(renderer, phase, direction, trial_index, total_trials, seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, phase, direction, trial_index, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, phase, direction, trial_index, perf_time, flip_time, frame_counter, direction)
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, phase, direction, trial_index, time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, phase, direction, trial_index, time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_trial(win, renderer, events, recorder, refresh_rate_hz, row, frame_counter, total_trials):
    for phase in ["baseline", "cue", "imagery", "cooldown"]:
        frame_counter, stopped = present_phase(win, renderer, events, recorder, refresh_rate_hz, row, phase, frame_counter, total_trials)
        if stopped:
            return frame_counter, True
    return frame_counter, False


def make_metadata(recorder, refresh_rate_hz, schedule, completed_trials, stopped_by_escape):
    metadata = recorder.metadata()
    metadata.update({
        "test": "box_motion_cognition",
        "protocol_version": PROTOCOL_VERSION,
        "display_refresh_rate_hz": refresh_rate_hz,
        "samples_per_direction": SAMPLES_PER_DIRECTION,
        "baseline_seconds": BASELINE_SECONDS,
        "cue_seconds": CUE_SECONDS,
        "imagery_seconds": IMAGERY_SECONDS,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "planned_trial_count": len(schedule),
        "completed_trial_count": int(completed_trials),
        "stopped_by_escape": bool(stopped_by_escape),
        "direction_labels": DIRECTION_LABELS,
        "schedule": schedule,
        "software_timing_note": "Training uses imagery windows only, baseline-normalized by the baseline phase before each trial.",
    })
    return metadata


def save_session_progress(session_dir, recorder, events, metadata, make_summary=False):
    eeg_df = recorder.dataframe()
    eeg_df.to_csv(session_dir / "eeg_samples.csv", index=False)
    pd.DataFrame(events).to_csv(session_dir / "events.csv", index=False)
    annotations = recorder.get_annotations()
    pd.DataFrame(annotations).to_csv(session_dir / "brainaccess_annotations.csv", index=False)
    save_metadata(session_dir, metadata)
    if len(eeg_df) > 0 and len(events) > 0:
        save_labeled_box_motion_samples(session_dir)
    if make_summary and len(eeg_df) > 0 and len(events) > 0:
        plot_recording_summary(session_dir, SUMMARY_NORMALIZATION)
    return eeg_df


def main():
    session_dir = make_session_output_dir()
    schedule = make_schedule()
    events = []
    frame_counter = 0
    completed_trials = 0
    stopped_by_escape = False
    win = make_window(BACKGROUND_COLOR)
    refresh_rate_hz = estimate_refresh_rate(win)
    renderer = BoxMotionRenderer(win)
    try:
        total_minutes = len(schedule) * (BASELINE_SECONDS + CUE_SECONDS + IMAGERY_SECONDS + COOLDOWN_SECONDS) / 60.0
        wait_with_text(
            win,
            f"BOX MOTION IMAGERY\n\n{len(schedule)} trials, about {total_minutes:.1f} minutes.\nThink only about moving the box in the cued direction.\nDo not move your eyes, jaw, hands, feet, or shoulders.\n\nPress ESC to stop and save partial data.",
            8.0,
        )
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            countdown(win, 3)
            for row in schedule:
                frame_counter, stopped_by_escape = present_trial(win, renderer, events, recorder, refresh_rate_hz, row, frame_counter, len(schedule))
                if not stopped_by_escape:
                    completed_trials = int(row["trial_index"])
                metadata = make_metadata(recorder, refresh_rate_hz, schedule, completed_trials, stopped_by_escape)
                save_session_progress(session_dir, recorder, events, metadata, make_summary=False)
                print(f"Saved trial {row['trial_index']}/{len(schedule)}: {session_dir}")
                if stopped_by_escape:
                    break
            recorder.stop()
            metadata = make_metadata(recorder, refresh_rate_hz, schedule, completed_trials, stopped_by_escape)
            save_session_progress(session_dir, recorder, events, metadata, make_summary=True)
        with open(session_dir / "manifest.json", "w", encoding="utf-8") as file:
            json.dump({"session_dir": str(session_dir), "protocol_version": PROTOCOL_VERSION, "schedule": schedule}, file, indent=2)
        wait_with_text(win, f"SAVED\n\n{session_dir}", 3.0, escape_allowed=False)
    finally:
        win.close()


if __name__ == "__main__":
    main()
