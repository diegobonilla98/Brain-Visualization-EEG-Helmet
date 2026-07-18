import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import event, visual

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))

from analysis_common import plot_recording_summary
from brainaccess_stream import BrainAccessStream, save_metadata
from stimulus_common import countdown, estimate_refresh_rate, frames_for_seconds, make_window, perf_time_after_flip, play_beep, wait_with_text
from motor_cognition_common import CONDITIONS, PROTOCOL_VERSION, TARGET_COLUMNS, condition_targets, resolve_project_path, save_labeled_motor_samples, target_mask


DEVICE_NAME = "BA MAXI 034"
OUTPUT_ROOT = "sessions"
STUDY_PREFIX = "basic_motor_cognition"
GAIN_NAME = "X8"
SAMPLES_PER_CLASS = 4
BASELINE_SECONDS = 5.0
CUE_SECONDS = 2.0
IMAGERY_SECONDS = 10.0
COOLDOWN_SECONDS = 3.0
SUMMARY_NORMALIZATION = "per_channel"
RANDOM_SEED = 20260620
ANIMATION_CYCLE_SECONDS = 2.8
ANIMATION_FRAME_COUNT = 16
ANIMATION_FRAME_DIR = Path(__file__).resolve().parent / "assets" / "motor_animation_frames"
STICKMAN_X = 0.42
STICKMAN_Y = -0.05
STICKMAN_SIZE = 0.78
TEXT_X = -0.40
TEXT_Y = 0.02


class StimulusRenderer:
    def __init__(self, win):
        self.win = win
        self.text = visual.TextStim(win, text="", height=0.044, color="white", pos=(TEXT_X, TEXT_Y), wrapWidth=0.82, alignText="center")
        self.animation_label = visual.TextStim(win, text="", height=0.034, color="#d9deea", pos=(STICKMAN_X, -0.74), wrapWidth=0.82, alignText="center")
        self.frames = {}
        for mask in range(16):
            for frame_index in range(ANIMATION_FRAME_COUNT):
                path = ANIMATION_FRAME_DIR / f"mask_{mask:02d}_frame_{frame_index:02d}.png"
                if not path.exists():
                    raise FileNotFoundError(f"Missing animation frame {path}. Run basic_motor_cognition/make_motor_animation_frames.py")
                self.frames[(mask, frame_index)] = visual.ImageStim(win, image=str(path), pos=(STICKMAN_X, STICKMAN_Y), size=(STICKMAN_SIZE, STICKMAN_SIZE), interpolate=True)

    def animation_frame_index(self, local_seconds):
        phase = (local_seconds % ANIMATION_CYCLE_SECONDS) / ANIMATION_CYCLE_SECONDS
        return int(phase * ANIMATION_FRAME_COUNT) % ANIMATION_FRAME_COUNT

    def draw(self, text, targets, phase, local_seconds):
        self.text.text = text
        self.text.draw()
        mask = 0 if phase in ["baseline", "cooldown"] else target_mask(targets)
        frame_index = self.animation_frame_index(local_seconds) if mask > 0 else 0
        self.frames[(mask, frame_index)].draw()
        self.animation_label.text = "Feel the motion. Do not move muscles." if mask > 0 else "Relax. No movement thought."
        self.animation_label.draw()


def make_study_output_dir():
    root = resolve_project_path(OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"session_{stamp}_{STUDY_PREFIX}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def make_trial_schedule():
    rng = np.random.default_rng(RANDOM_SEED + int(time.time()))
    schedule = []
    sample_index = 1
    for repetition_index in range(SAMPLES_PER_CLASS):
        order = list(CONDITIONS)
        rng.shuffle(order)
        for condition in order:
            row = dict(condition)
            row["repetition_index"] = repetition_index + 1
            row["sample_index"] = sample_index
            schedule.append(row)
            sample_index += 1
    return schedule


def add_event(events, event_type, state_label, state_phase, condition, pc_time_perf_counter_s, flip_time_psychopy_s, frame_index, value, trial_index, session_index):
    row = {
        "event_type": event_type,
        "state_label": state_label,
        "state_phase": state_phase,
        "condition_name": condition["condition_name"],
        "class_label": condition["class_label"],
        "active_intent": int(condition["active_intent"]),
        "pc_time_perf_counter_s": pc_time_perf_counter_s,
        "flip_time_psychopy_s": flip_time_psychopy_s,
        "frame_index": frame_index,
        "value": value,
        "trial_index": trial_index,
        "session_index": session_index,
    }
    targets = condition_targets(condition)
    if state_phase in ["baseline", "cooldown"]:
        targets = {target: 0 for target in TARGET_COLUMNS}
        row["class_label"] = "rest"
        row["active_intent"] = 0
    for target in TARGET_COLUMNS:
        row[target] = int(targets.get(target, 0))
    events.append(row)


def state_label_for(condition, phase):
    if phase == "baseline":
        return f"baseline_before_{condition['condition_name']}"
    if phase == "cue":
        return f"cue_{condition['condition_name']}"
    if phase == "imagery":
        return f"imagery_{condition['condition_name']}"
    return f"cooldown_after_{condition['condition_name']}"


def phase_seconds(phase):
    if phase == "baseline":
        return BASELINE_SECONDS
    if phase == "cue":
        return CUE_SECONDS
    if phase == "imagery":
        return IMAGERY_SECONDS
    return COOLDOWN_SECONDS


def remaining_time_text(trial_index, total_trials, phase_seconds_left, remaining_trials):
    trial_seconds = BASELINE_SECONDS + CUE_SECONDS + IMAGERY_SECONDS + COOLDOWN_SECONDS
    remaining_seconds = phase_seconds_left + remaining_trials * trial_seconds
    minutes = int(remaining_seconds // 60)
    seconds = int(remaining_seconds % 60)
    return f"Sample {trial_index}/{total_trials}\nLeft in session: {minutes:02d}:{seconds:02d}"


def text_for_phase(condition, phase, seconds_left, trial_index, total_trials):
    remaining_trials = max(0, total_trials - trial_index)
    progress = remaining_time_text(trial_index, total_trials, seconds_left, remaining_trials)
    if phase == "baseline":
        return f"{progress}\n\nBASELINE\n\nRelax and look at the figure.\nNo movement thought.\n\nPhase left: {int(seconds_left) + 1}s"
    if phase == "cue":
        return f"{progress}\n\nGET READY\n\n{condition['screen_text']}\n\n{condition['instruction']}\n\nDo not move.\nPhase left: {int(seconds_left) + 1}s"
    if phase == "imagery":
        if condition["class_label"] == "rest":
            return f"{progress}\n\nREST\n\nStay relaxed. Do not imagine movement.\n\nPhase left: {int(seconds_left) + 1}s"
        return f"{progress}\n\n{condition['screen_text']}\n\nImagine the movement from inside your body.\nFeel the intention and rhythm.\nNo muscle contraction.\n\nPhase left: {int(seconds_left) + 1}s"
    return f"{progress}\n\nCOOLDOWN\n\nRelease the image. Return to neutral rest.\n\nPhase left: {int(seconds_left) + 1}s"


def session_metadata(recorder, refresh_rate_hz, session_index, schedule, stopped_by_escape, completed_trials):
    metadata = recorder.metadata()
    metadata.update({
        "test": "basic_motor_cognition",
        "protocol_version": PROTOCOL_VERSION,
        "session_index": session_index,
        "session_count": 1,
        "display_refresh_rate_hz": refresh_rate_hz,
        "baseline_seconds": BASELINE_SECONDS,
        "cue_seconds": CUE_SECONDS,
        "imagery_seconds": IMAGERY_SECONDS,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "samples_per_class": SAMPLES_PER_CLASS,
        "planned_sample_count": len(schedule),
        "completed_sample_count": completed_trials,
        "stopped_by_escape": stopped_by_escape,
        "target_columns": TARGET_COLUMNS,
        "class_labels": [condition["class_label"] for condition in CONDITIONS],
        "trial_schedule": schedule,
        "software_timing_note": "Stimulus state is logged after PsychoPy win.flip and aligned to EEG callback receive timing using time.perf_counter.",
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
        save_labeled_motor_samples(session_dir)
    if make_summary and len(eeg_df) > 0 and len(events) > 0:
        plot_recording_summary(session_dir, SUMMARY_NORMALIZATION)
    return eeg_df


def present_state(win, renderer, events, recorder, refresh_rate_hz, condition, phase, frame_counter, trial_index, session_index, total_trials, beep_frequency=None):
    seconds = phase_seconds(phase)
    state_label = state_label_for(condition, phase)
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(state_label)
    if beep_frequency is not None:
        play_beep(beep_frequency, 140)
    started = False
    stopped = False
    targets = condition_targets(condition)
    if phase in ["baseline", "cooldown"]:
        targets = {target: 0 for target in TARGET_COLUMNS}
    for local_frame in range(total_frames):
        seconds_left = seconds - (local_frame / refresh_rate_hz)
        local_seconds = local_frame / refresh_rate_hz
        renderer.draw(text_for_phase(condition, phase, seconds_left, trial_index, total_trials), targets, phase, local_seconds)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, phase, condition, perf_time, flip_time, frame_counter, None, trial_index, session_index)
            started = True
        add_event(events, "frame", state_label, phase, condition, perf_time, flip_time, frame_counter, condition["screen_text"], trial_index, session_index)
        frame_counter += 1
        if "escape" in event.getKeys():
            stopped = True
            break
    add_event(events, "state_end", state_label, phase, condition, time.perf_counter(), None, frame_counter, "stopped_by_escape" if stopped else None, trial_index, session_index)
    return frame_counter, stopped


def present_trial(win, renderer, events, recorder, refresh_rate_hz, condition, frame_counter, trial_index, session_index, total_trials):
    frame_counter, stopped = present_state(win, renderer, events, recorder, refresh_rate_hz, condition, "baseline", frame_counter, trial_index, session_index, total_trials)
    if stopped:
        return frame_counter, True
    frame_counter, stopped = present_state(win, renderer, events, recorder, refresh_rate_hz, condition, "cue", frame_counter, trial_index, session_index, total_trials, beep_frequency=750)
    if stopped:
        return frame_counter, True
    frame_counter, stopped = present_state(win, renderer, events, recorder, refresh_rate_hz, condition, "imagery", frame_counter, trial_index, session_index, total_trials, beep_frequency=1100)
    if stopped:
        return frame_counter, True
    frame_counter, stopped = present_state(win, renderer, events, recorder, refresh_rate_hz, condition, "cooldown", frame_counter, trial_index, session_index, total_trials)
    return frame_counter, stopped


def run_session(win, renderer, study_dir, refresh_rate_hz):
    session_index = 1
    session_dir = study_dir
    schedule = make_trial_schedule()
    total_trials = len(schedule)
    events = []
    frame_counter = 0
    stopped_by_escape = False
    completed_trials = 0
    eeg_df = pd.DataFrame()
    total_minutes = total_trials * (BASELINE_SECONDS + CUE_SECONDS + IMAGERY_SECONDS + COOLDOWN_SECONDS) / 60.0
    wait_with_text(
        win,
        f"ONE MOTOR IMAGERY SESSION\n\n{total_trials} samples, about {total_minutes:.1f} minutes.\nEach sample has its own baseline rest before imagery.\n\nPress ESC to stop and save partial data.",
        6.0,
    )
    with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
        recorder.start()
        countdown(win, 3)
        for trial_index, condition in enumerate(schedule, start=1):
            frame_counter, stopped_by_escape = present_trial(win, renderer, events, recorder, refresh_rate_hz, condition, frame_counter, trial_index, session_index, total_trials)
            if not stopped_by_escape:
                completed_trials = trial_index
            metadata = session_metadata(recorder, refresh_rate_hz, session_index, schedule, stopped_by_escape, completed_trials)
            eeg_df = save_session_progress(session_dir, recorder, events, metadata, make_summary=False)
            print(f"Saved sample {trial_index}/{total_trials}: {session_dir}")
            if stopped_by_escape:
                break
        recorder.stop()
        metadata = session_metadata(recorder, refresh_rate_hz, session_index, schedule, stopped_by_escape, completed_trials)
        eeg_df = save_session_progress(session_dir, recorder, events, metadata, make_summary=True)
    return {
        "session_index": session_index,
        "session_dir": str(session_dir),
        "planned_sample_count": total_trials,
        "completed_sample_count": completed_trials,
        "stopped_by_escape": stopped_by_escape,
        "rows": int(len(eeg_df)),
        "duration_s": float(len(eeg_df) / 250.0) if len(eeg_df) > 0 else 0.0,
    }


def save_manifest(study_dir, session_summaries):
    manifest = {
        "study_dir": str(study_dir),
        "protocol_version": PROTOCOL_VERSION,
        "session_count": 1,
        "samples_per_class": SAMPLES_PER_CLASS,
        "sessions": session_summaries,
        "target_columns": TARGET_COLUMNS,
        "conditions": CONDITIONS,
    }
    with open(study_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def main():
    study_dir = make_study_output_dir()
    session_summaries = []
    win = make_window("black")
    refresh_rate_hz = estimate_refresh_rate(win)
    renderer = StimulusRenderer(win)
    stopped_by_escape = False
    try:
        wait_with_text(
            win,
            "BASIC MOTOR COGNITION V3\n\nThis version trains on baseline-normalized motor imagery.\nKeep muscles relaxed. Use the animated motion only as a mental rhythm.\n\nPress ESC during a session to stop and save partial data.",
            7.0,
        )
        summary = run_session(win, renderer, study_dir, refresh_rate_hz)
        session_summaries.append(summary)
        save_manifest(study_dir, session_summaries)
        if summary["stopped_by_escape"]:
            stopped_by_escape = True
        if stopped_by_escape:
            wait_with_text(win, f"STOPPED AND SAVED\n\nSaved to:\n{study_dir}", 3.0, escape_allowed=False)
        else:
            wait_with_text(win, f"DONE\n\nSaved to:\n{study_dir}", 4.0, escape_allowed=False)
    finally:
        save_manifest(study_dir, session_summaries)
        win.close()


if __name__ == "__main__":
    main()
