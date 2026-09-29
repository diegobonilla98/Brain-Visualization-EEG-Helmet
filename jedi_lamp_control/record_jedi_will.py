import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from psychopy import event, visual


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eeg_quality_suite"))
sys.path.insert(0, str(PROJECT_ROOT / "universal_brain_recorder"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_common import plot_recording_summary
from brainaccess_stream import BrainAccessStream, save_metadata
from jedi_common import CLASS_LABELS, PROTOCOL_VERSION, save_labeled_samples
from session_common import write_session_files
from stimulus_common import estimate_refresh_rate, frames_for_seconds, make_window, perf_time_after_flip, play_beep, wait_with_text
from traceability_common import SCHEMA_VERSION, file_provenance, rebuild_catalog, utc_now, write_json as write_traceability_json


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
OUTPUT_ROOT = PROJECT_ROOT / "sessions"
SESSION_PREFIX = "jedi_lamp_will"
CALIBRATION_SECONDS = 30.0
BLOCKS_PER_CLASS = 8
INSTRUCTION_SECONDS = 4.0
TASK_SECONDS = 10.0
REST_SECONDS = 4.0
RANDOM_SEED = 20260719
BACKGROUND_COLOR = "#080b14"
TASK_TAGS = ["attention", "executive_control", "imagery", "motor", "motor_cortex", "premotor", "eyes_open", "active", "interactive"]
NOTHING_VARIANTS = [
    ("free_thought", "Think about anything you want. Let thoughts move naturally."),
    ("mental_arithmetic", "Do casual mental arithmetic or count in an irregular way."),
    ("inner_speech", "Silently talk to yourself about any topic."),
    ("overt_speech", "Speak naturally for this block. Any harmless topic is fine."),
    ("small_movements", "Make ordinary small hand or posture movements. Do not use the Jedi pose."),
    ("visual_exploration", "Look around the screen and room naturally. Do not perform the trigger ritual."),
    ("memory", "Recall a recent event, song, place, or conversation."),
    ("mixed_activity", "Do anything ordinary except the exact Jedi trigger ritual."),
]
WILL_COMMAND = "Extend your chosen hand into the same Jedi pose every time. Fix your gaze on the center. Imagine the lamp switching and silently command NOW. Hold the pose and intention until the block ends."


class Renderer:
    def __init__(self, window):
        self.window = window
        self.title = visual.TextStim(window, text="", height=0.075, color="white", pos=(0.0, 0.18), wrapWidth=1.55, alignText="center")
        self.detail = visual.TextStim(window, text="", height=0.043, color="#cbd5e1", pos=(0.0, -0.05), wrapWidth=1.55, alignText="center")
        self.timer = visual.TextStim(window, text="", height=0.038, color="#8fa3c7", pos=(0.0, -0.38), wrapWidth=1.5, alignText="center")
        self.dot = visual.Circle(window, radius=0.020, fillColor="#eef2ff", lineColor="#eef2ff")

    def instruction(self, class_label, command, seconds_left):
        self.window.color = BACKGROUND_COLOR
        self.title.text = "THE WILL" if class_label == "the_will" else "NOTHING"
        self.title.color = "#60a5fa" if class_label == "the_will" else "#cbd5e1"
        self.detail.text = command
        self.timer.text = f"Labeled block begins in {int(seconds_left) + 1}s"
        self.title.draw()
        self.detail.draw()
        self.timer.draw()

    def task(self, block_index, total_blocks, seconds_left):
        self.window.color = BACKGROUND_COLOR
        self.dot.draw()
        self.timer.text = f"Block {block_index}/{total_blocks}    {int(seconds_left) + 1}s    ESC stops and saves"
        self.timer.draw()

    def rest(self, seconds_left):
        self.window.color = BACKGROUND_COLOR
        self.title.text = "RESET"
        self.title.color = "#a5b4fc"
        self.detail.text = "Return your hand to a neutral position. Blink, swallow, and relax if needed."
        self.timer.text = f"Next instruction in {int(seconds_left) + 1}s"
        self.title.draw()
        self.detail.draw()
        self.timer.draw()

    def calibration(self, seconds_left):
        self.window.color = "#03050a"
        self.title.text = "EYES CLOSED · RELAXED"
        self.title.color = "white"
        self.detail.text = "Remain completely still. Relax your jaw, face, shoulders, hands, and feet. Breathe naturally."
        self.timer.text = f"Calibration    {int(np.ceil(seconds_left)):02d}s"
        self.title.draw()
        self.detail.draw()
        self.timer.draw()


def make_session_dir():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"session_{time.strftime('%Y%m%d_%H%M%S')}_{SESSION_PREFIX}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_schedule():
    rng = np.random.default_rng(RANDOM_SEED + int(time.time()))
    nothing_rows = []
    for index in range(BLOCKS_PER_CLASS):
        variant, command = NOTHING_VARIANTS[index % len(NOTHING_VARIANTS)]
        nothing_rows.append({"class_label": "nothing", "negative_variant": variant, "command": command})
    will_rows = [{"class_label": "the_will", "negative_variant": "", "command": WILL_COMMAND} for _ in range(BLOCKS_PER_CLASS)]
    rows = nothing_rows + will_rows
    for _ in range(1000):
        rng.shuffle(rows)
        labels = [row["class_label"] for row in rows]
        if all(not (labels[index] == labels[index + 1] == labels[index + 2]) for index in range(len(labels) - 2)):
            break
    return [{"block_index": index + 1, **row} for index, row in enumerate(rows)]


def add_event(events, event_type, state_label, state_phase, class_label, block_index, negative_variant, timestamp, value=""):
    events.append({
        "event_index": len(events),
        "event_type": event_type,
        "state_label": state_label,
        "state_phase": state_phase,
        "class_label": class_label,
        "block_index": int(block_index),
        "negative_variant": negative_variant,
        "pc_time_perf_counter_s": float(timestamp),
        "value": value,
    })


def run_calibration(window, renderer, recorder, refresh_rate, events):
    started_at = None
    stopped = False
    recorder.annotate("calibration_start:eyes_closed_relaxed")
    while True:
        seconds_left = CALIBRATION_SECONDS if started_at is None else max(0.0, CALIBRATION_SECONDS - (time.perf_counter() - started_at))
        renderer.calibration(seconds_left)
        _, timestamp = perf_time_after_flip(window)
        if started_at is None:
            started_at = timestamp
            add_event(events, "state_start", "eyes_closed_relaxed_calibration", "calibration", "unknown", -1, "", timestamp)
        if "escape" in event.getKeys():
            stopped = True
            break
        if time.perf_counter() - started_at >= CALIBRATION_SECONDS:
            break
    ended_at = time.perf_counter()
    add_event(events, "state_end", "eyes_closed_relaxed_calibration", "calibration", "unknown", -1, "", ended_at, "aborted" if stopped else "completed")
    recorder.annotate("calibration_end:aborted" if stopped else "calibration_end:completed")
    return {"label": "eyes_closed_relaxed_calibration", "planned_duration_s": CALIBRATION_SECONDS, "duration_s": ended_at - started_at, "start_pc_time_perf_counter_s": started_at, "end_pc_time_perf_counter_s": ended_at, "completed": not stopped, "eyes_closed": True, "static_posture": True, "tags": ["baseline", "eyes_closed", "passive"]}, stopped


def run_timed_phase(window, renderer, recorder, refresh_rate, events, row, phase, total_blocks):
    seconds = INSTRUCTION_SECONDS if phase == "instruction" else TASK_SECONDS if phase == "task" else REST_SECONDS
    state_label = f"instruction_{row['block_index']:02d}" if phase == "instruction" else row["class_label"] if phase == "task" else f"rest_{row['block_index']:02d}"
    class_label = row["class_label"] if phase in {"instruction", "task"} else "unknown"
    recorder.annotate(f"{phase}:{state_label}")
    started = False
    for local_frame in range(frames_for_seconds(refresh_rate, seconds)):
        seconds_left = seconds - local_frame / refresh_rate
        if phase == "instruction":
            renderer.instruction(row["class_label"], row["command"], seconds_left)
        elif phase == "task":
            renderer.task(row["block_index"], total_blocks, seconds_left)
        else:
            renderer.rest(seconds_left)
        _, timestamp = perf_time_after_flip(window)
        if not started:
            add_event(events, "state_start", state_label, phase, class_label, row["block_index"], row["negative_variant"], timestamp)
            started = True
            if phase == "task":
                play_beep(1000, 120)
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, phase, class_label, row["block_index"], row["negative_variant"], time.perf_counter(), "stopped_by_escape")
            return True
    add_event(events, "state_end", state_label, phase, class_label, row["block_index"], row["negative_variant"], time.perf_counter())
    return False


def save_progress(session_dir, recorder, events, metadata, summary=False):
    eeg = recorder.dataframe()
    eeg.to_csv(session_dir / "eeg_samples.csv", index=False)
    pd.DataFrame(events).to_csv(session_dir / "events.csv", index=False)
    pd.DataFrame(recorder.get_annotations()).to_csv(session_dir / "brainaccess_annotations.csv", index=False)
    save_metadata(session_dir, metadata)
    if len(eeg) and len(events):
        save_labeled_samples(session_dir)
    if summary and len(eeg):
        plot_recording_summary(session_dir, "per_channel")


def make_metadata(recorder, schedule, refresh_rate, completed_blocks, stopped, calibration, session_started_utc, traceability_id):
    metadata = recorder.metadata()
    metadata.update({
        "test": "jedi_lamp_will",
        "protocol_version": PROTOCOL_VERSION,
        "traceability_id": traceability_id,
        "subject_id": "subject_001",
        "session_started_at_utc": session_started_utc,
        "class_labels": CLASS_LABELS,
        "display_refresh_rate_hz": refresh_rate,
        "calibration": calibration,
        "blocks_per_class": BLOCKS_PER_CLASS,
        "instruction_seconds": INSTRUCTION_SECONDS,
        "task_seconds": TASK_SECONDS,
        "rest_seconds": REST_SECONDS,
        "completed_block_count": completed_blocks,
        "planned_block_count": len(schedule),
        "stopped_by_escape": stopped,
        "will_command": WILL_COMMAND,
        "nothing_variants": [{"name": name, "command": command} for name, command in NOTHING_VARIANTS],
        "schedule": schedule,
        "validation_group": "session_id",
        "timing_note": "State starts are timestamped immediately after PsychoPy display flips. Training excludes instructions, rests, calibration, and task onset transitions.",
    })
    return metadata


def task_segments_from_events(events, schedule):
    output = []
    for row in schedule:
        starts = [event_row for event_row in events if event_row["event_type"] == "state_start" and event_row["state_phase"] == "task" and event_row["block_index"] == row["block_index"]]
        ends = [event_row for event_row in events if event_row["event_type"] == "state_end" and event_row["state_phase"] == "task" and event_row["block_index"] == row["block_index"]]
        if not starts:
            continue
        start = starts[0]
        end = ends[-1] if ends else None
        output.append({
            "segment_index": len(output),
            "task_name": row["class_label"],
            "task_slug": row["class_label"],
            "description": row["command"],
            "tags": TASK_TAGS,
            "block_index": row["block_index"],
            "negative_variant": row["negative_variant"],
            "start_pc_time_perf_counter_s": start["pc_time_perf_counter_s"],
            "end_pc_time_perf_counter_s": end["pc_time_perf_counter_s"] if end else None,
            "duration_s": end["pc_time_perf_counter_s"] - start["pc_time_perf_counter_s"] if end else None,
        })
    return output


def main():
    session_dir = make_session_dir()
    schedule = make_schedule()
    events = []
    completed_blocks = 0
    stopped = False
    calibration = None
    traceability_id = uuid.uuid4().hex
    session_started_utc = utc_now()
    window = make_window(BACKGROUND_COLOR)
    refresh_rate = estimate_refresh_rate(window)
    renderer = Renderer(window)
    try:
        duration_minutes = (CALIBRATION_SECONDS + len(schedule) * (INSTRUCTION_SECONDS + TASK_SECONDS + REST_SECONDS)) / 60.0
        wait_with_text(window, f"JEDI LAMP TRAINING\n\nTwo classes: NOTHING and THE WILL.\n\nTHE WILL must use exactly the same pose, gaze, imagery, and silent command every time.\nNOTHING deliberately contains many other thoughts, speech, and ordinary movements.\n\n{len(schedule)} blocks · about {duration_minutes:.1f} minutes\n\nPress ESC at any time to stop and save partial data.", 10.0)
        with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME) as recorder:
            recorder.start()
            calibration, stopped = run_calibration(window, renderer, recorder, refresh_rate, events)
            if not stopped:
                play_beep(1200, 180)
            for row in schedule:
                if stopped:
                    break
                stopped = run_timed_phase(window, renderer, recorder, refresh_rate, events, row, "instruction", len(schedule))
                if not stopped:
                    stopped = run_timed_phase(window, renderer, recorder, refresh_rate, events, row, "task", len(schedule))
                if not stopped:
                    completed_blocks += 1
                    stopped = run_timed_phase(window, renderer, recorder, refresh_rate, events, row, "rest", len(schedule))
                metadata = make_metadata(recorder, schedule, refresh_rate, completed_blocks, stopped, calibration, session_started_utc, traceability_id)
                save_progress(session_dir, recorder, events, metadata)
            recorder.stop()
            metadata = make_metadata(recorder, schedule, refresh_rate, completed_blocks, stopped, calibration, session_started_utc, traceability_id)
            save_progress(session_dir, recorder, events, metadata, summary=True)
        write_session_files(session_dir, "Jedi lamp will", "Two-state intentional lamp-toggle experiment with diverse negative behavior and a consistent will ritual.", TASK_TAGS, "jedi_lamp_experiment", PROTOCOL_VERSION, extra={"calibration": calibration, "class_labels": CLASS_LABELS})
        with open(session_dir / "manifest.json", "w", encoding="utf-8") as file:
            json.dump({"session_dir": str(session_dir), "protocol_version": PROTOCOL_VERSION, "completed_blocks": completed_blocks, "calibration": calibration}, file, indent=2)
        segments = task_segments_from_events(events, schedule)
        traceability = {
            "schema_version": SCHEMA_VERSION,
            "traceability_id": traceability_id,
            "session_id": session_dir.name,
            "recording_path": str(session_dir.resolve().relative_to(PROJECT_ROOT)),
            "source_type": "jedi_lamp_experiment",
            "traceability_status": "protocol_annotated",
            "created_at_utc": utc_now(),
            "recording_started_at_utc": session_started_utc,
            "recording_ended_at_utc": utc_now(),
            "subject_id": "subject_001",
            "protocol_version": PROTOCOL_VERSION,
            "primary_task": {"name": "jedi_lamp_will", "description": "Two-state intentional relay-toggle experiment.", "tags": TASK_TAGS},
            "calibration": calibration,
            "task_segments": segments,
            "available_state_labels": sorted(set(str(row["state_label"]) for row in events)),
            "device": {
                "requested": metadata.get("device_name_requested"),
                "connected": metadata.get("device_name_connected"),
                "model": metadata.get("device_model"),
                "serial_number": metadata.get("serial_number"),
                "sample_frequency_hz": metadata.get("sample_frequency_hz"),
                "channel_count": metadata.get("eeg_channel_count"),
                "gain_name": metadata.get("gain_name"),
                "bias_channel_name": metadata.get("bias_channel_name"),
            },
            "files": [
                file_provenance(session_dir / "eeg_samples.csv", "raw_eeg"),
                file_provenance(session_dir / "eeg_samples_labeled.csv", "derived_labeled_eeg"),
                file_provenance(session_dir / "events.csv", "task_events"),
                file_provenance(session_dir / "brainaccess_annotations.csv", "sdk_annotations"),
                file_provenance(session_dir / "metadata.json", "recording_metadata"),
                file_provenance(session_dir / "session.json", "canonical_session_metadata"),
                file_provenance(session_dir / "SESSION.md", "natural_language_session_info"),
                file_provenance(session_dir / "manifest.json", "experiment_manifest"),
                file_provenance(session_dir / "summary.png", "recording_summary_plot"),
            ],
            "parent_sources": [],
            "notes": "Calibration and class intervals were generated by the timed PsychoPy protocol.",
            "inference_evidence": [],
        }
        write_traceability_json(session_dir / "traceability.json", traceability)
        rebuild_catalog()
        wait_with_text(window, f"SAVED\n\n{session_dir}", 3.0, escape_allowed=False)
    finally:
        window.close()
        print("Session:", session_dir)


if __name__ == "__main__":
    main()
