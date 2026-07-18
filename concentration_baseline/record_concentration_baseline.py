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
from brainaccess_stream import BrainAccessStream, save_metadata
from concentration_common import CLASS_LABELS, PROTOCOL_VERSION, resolve_project_path, save_labeled_concentration_samples
from stimulus_common import countdown, estimate_refresh_rate, frames_for_seconds, make_window, perf_time_after_flip, play_beep, wait_with_text


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
OUTPUT_ROOT = "sessions"
SESSION_PREFIX = "session"
BLOCKS_PER_CLASS = 6
INSTRUCTION_SECONDS = 5.0
TASK_SECONDS = 28.0
REST_BETWEEN_BLOCKS_SECONDS = 3.0
RANDOM_SEED = 20260621
SUMMARY_NORMALIZATION = "per_channel"
BACKGROUND_COLOR = "#111111"
DOT_COLOR = "#f2f2f2"


COMMANDS = {
    "concentrated": "CONCENTRATION BLOCK\n\nLook at the center dot.\nSilently count your breaths from 1 to 10, then restart.\nIf attention drifts, return to the dot and current count.\nKeep jaw, face, shoulders, hands, and feet relaxed.",
    "not_concentrated": "NON-CONCENTRATION BLOCK\n\nLook at the same center dot with relaxed attention.\nDo not count, solve, rehearse, or control breathing.\nLet thoughts drift without following a task.\nKeep jaw, face, shoulders, hands, and feet relaxed.",
}


class ConcentrationRenderer:
    def __init__(self, win):
        self.win = win
        self.command = visual.TextStim(win, text="", height=0.052, color="white", pos=(0.0, 0.0), wrapWidth=1.55, alignText="center")
        self.dot = visual.Circle(win, radius=0.018, fillColor=DOT_COLOR, lineColor=DOT_COLOR, pos=(0.0, 0.0))
        self.timer = visual.TextStim(win, text="", height=0.034, color="#cfcfcf", pos=(0.0, -0.38), wrapWidth=1.3, alignText="center")

    def draw_command(self, text, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.command.text = f"{text}\n\nStarting in {int(seconds_left) + 1}s"
        self.command.draw()

    def draw_task(self, block_index, total_blocks, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.dot.draw()
        self.timer.text = f"Block {block_index}/{total_blocks}    {int(seconds_left) + 1}s    Press ESC to stop"
        self.timer.draw()

    def draw_rest(self, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.command.text = f"RESET\n\nBlink if needed. Relax jaw and shoulders.\n\nNext block in {int(seconds_left) + 1}s"
        self.command.draw()


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
    for class_label in CLASS_LABELS:
        labels.extend([class_label] * BLOCKS_PER_CLASS)
    rng.shuffle(labels)
    rows = []
    for index, class_label in enumerate(labels, start=1):
        rows.append({
            "block_index": index,
            "class_label": class_label,
            "state_label": class_label,
            "command": COMMANDS[class_label],
        })
    return rows


def add_event(events, event_type, state_label, state_phase, class_label, block_index, pc_time_perf_counter_s, flip_time_psychopy_s, frame_index, value):
    events.append({
        "event_type": event_type,
        "state_label": state_label,
        "state_phase": state_phase,
        "class_label": class_label,
        "block_index": int(block_index),
        "pc_time_perf_counter_s": pc_time_perf_counter_s,
        "flip_time_psychopy_s": flip_time_psychopy_s,
        "frame_index": int(frame_index),
        "value": value,
    })


def present_instruction(win, renderer, events, recorder, refresh_rate_hz, row, frame_counter):
    state_label = f"instruction_{row['class_label']}_{row['block_index']:02d}"
    total_frames = frames_for_seconds(refresh_rate_hz, INSTRUCTION_SECONDS)
    recorder.annotate(state_label)
    play_beep(780, 160)
    started = False
    for local_frame in range(total_frames):
        seconds_left = INSTRUCTION_SECONDS - local_frame / refresh_rate_hz
        renderer.draw_command(row["command"], seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "instruction", row["class_label"], row["block_index"], perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "instruction", row["class_label"], row["block_index"], perf_time, flip_time, frame_counter, "instruction")
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, "instruction", row["class_label"], row["block_index"], time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, "instruction", row["class_label"], row["block_index"], time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_task(win, renderer, events, recorder, refresh_rate_hz, row, frame_counter, total_blocks):
    state_label = row["class_label"]
    total_frames = frames_for_seconds(refresh_rate_hz, TASK_SECONDS)
    recorder.annotate(state_label)
    play_beep(1100, 130)
    started = False
    for local_frame in range(total_frames):
        seconds_left = TASK_SECONDS - local_frame / refresh_rate_hz
        renderer.draw_task(row["block_index"], total_blocks, seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "task", row["class_label"], row["block_index"], perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "task", row["class_label"], row["block_index"], perf_time, flip_time, frame_counter, "dot")
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, "task", row["class_label"], row["block_index"], time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, "task", row["class_label"], row["block_index"], time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_rest(win, renderer, events, recorder, refresh_rate_hz, block_index, frame_counter):
    state_label = f"rest_after_{block_index:02d}"
    total_frames = frames_for_seconds(refresh_rate_hz, REST_BETWEEN_BLOCKS_SECONDS)
    recorder.annotate(state_label)
    started = False
    for local_frame in range(total_frames):
        seconds_left = REST_BETWEEN_BLOCKS_SECONDS - local_frame / refresh_rate_hz
        renderer.draw_rest(seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "rest", "unknown", block_index, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "rest", "unknown", block_index, perf_time, flip_time, frame_counter, "rest")
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, "rest", "unknown", block_index, time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, "rest", "unknown", block_index, time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def make_metadata(recorder, refresh_rate_hz, schedule, completed_blocks, stopped_by_escape):
    metadata = recorder.metadata()
    metadata.update({
        "test": "concentration_baseline",
        "protocol_version": PROTOCOL_VERSION,
        "display_refresh_rate_hz": refresh_rate_hz,
        "blocks_per_class": BLOCKS_PER_CLASS,
        "instruction_seconds": INSTRUCTION_SECONDS,
        "task_seconds": TASK_SECONDS,
        "rest_between_blocks_seconds": REST_BETWEEN_BLOCKS_SECONDS,
        "planned_block_count": len(schedule),
        "completed_block_count": int(completed_blocks),
        "stopped_by_escape": bool(stopped_by_escape),
        "class_labels": CLASS_LABELS,
        "schedule": schedule,
        "software_timing_note": "State starts are logged immediately after PsychoPy win.flip using time.perf_counter. Training uses only state_phase task windows.",
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
        save_labeled_concentration_samples(session_dir)
    if make_summary and len(eeg_df) > 0 and len(events) > 0:
        plot_recording_summary(session_dir, SUMMARY_NORMALIZATION)
    return eeg_df


def main():
    session_dir = make_session_output_dir()
    schedule = make_schedule()
    events = []
    frame_counter = 0
    completed_blocks = 0
    stopped_by_escape = False
    win = make_window(BACKGROUND_COLOR)
    refresh_rate_hz = estimate_refresh_rate(win)
    renderer = ConcentrationRenderer(win)
    try:
        total_minutes = len(schedule) * (INSTRUCTION_SECONDS + TASK_SECONDS + REST_BETWEEN_BLOCKS_SECONDS) / 60.0
        wait_with_text(
            win,
            f"CONCENTRATION BASELINE\n\n{len(schedule)} blocks, about {total_minutes:.1f} minutes.\nThe labeled screen is always the same dot.\nFollow the command given before each block.\n\nPress ESC to stop and save partial data.",
            8.0,
        )
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            countdown(win, 3)
            for row in schedule:
                frame_counter, stopped_by_escape = present_instruction(win, renderer, events, recorder, refresh_rate_hz, row, frame_counter)
                if not stopped_by_escape:
                    frame_counter, stopped_by_escape = present_task(win, renderer, events, recorder, refresh_rate_hz, row, frame_counter, len(schedule))
                if not stopped_by_escape:
                    completed_blocks = int(row["block_index"])
                    frame_counter, stopped_by_escape = present_rest(win, renderer, events, recorder, refresh_rate_hz, int(row["block_index"]), frame_counter)
                metadata = make_metadata(recorder, refresh_rate_hz, schedule, completed_blocks, stopped_by_escape)
                save_session_progress(session_dir, recorder, events, metadata, make_summary=False)
                print(f"Saved block {row['block_index']}/{len(schedule)}: {session_dir}")
                if stopped_by_escape:
                    break
            recorder.stop()
            metadata = make_metadata(recorder, refresh_rate_hz, schedule, completed_blocks, stopped_by_escape)
            save_session_progress(session_dir, recorder, events, metadata, make_summary=True)
        with open(session_dir / "manifest.json", "w", encoding="utf-8") as file:
            json.dump({"session_dir": str(session_dir), "protocol_version": PROTOCOL_VERSION, "schedule": schedule}, file, indent=2)
        wait_with_text(win, f"SAVED\n\n{session_dir}", 3.0, escape_allowed=False)
    finally:
        win.close()


if __name__ == "__main__":
    main()
