import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from psychopy import event, visual

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_common import plot_recording_summary
from brainaccess_stream import BrainAccessStream, save_metadata
from stimulus_common import countdown, estimate_refresh_rate, frames_for_seconds, make_window, perf_time_after_flip, play_beep, wait_with_text
from image_emotion_common import (
    DATA_ROOT,
    DEFAULT_EMOSET_ROOT,
    EMOTION_LABELS,
    EMOTION_TO_INDEX,
    balanced_emotion_schedule,
    load_emoset_manifest,
    save_labeled_image_samples,
    write_json,
)


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
USE_BIAS = True
BIAS_CHANNEL_NAME = "Iz"
DATASET_ROOT = DEFAULT_EMOSET_ROOT
DATASET_SPLIT_NAME = "train"
OUTPUT_ROOT = DATA_ROOT
SESSION_PREFIX = "session"
PROTOCOL_VERSION = "image_emotion_passive_viewing_v1"
IMAGES_PER_SESSION = 30
EMOTIONS_TO_USE = EMOTION_LABELS
AVOID_PREVIOUSLY_COLLECTED_IMAGES = True
RANDOM_SEED = 20260620
FIXATION_SECONDS_MIN = 1.2
FIXATION_SECONDS_MAX = 1.8
IMAGE_SECONDS = 4.5
WASHOUT_SECONDS = 2.0
BREAK_EVERY_IMAGES = 10
BREAK_SECONDS = 8.0
SUMMARY_NORMALIZATION = "per_channel"
BACKGROUND_COLOR = "#171717"
FIXATION_COLOR = "#e8e8e8"
WASHOUT_COLOR = "#242424"
IMAGE_FIT_MODE = "contain"
SHOW_FIXATION_DOT_ON_IMAGE = True
FIXATION_DOT_SIZE = 0.012
PRELOAD_NEXT_IMAGE_DURING_WASHOUT = False


def make_session_output_dir():
    root = Path(OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"session_{stamp}_{SESSION_PREFIX}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def image_size_for_window(win, image_path):
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    image_aspect = float(width_px) / float(height_px)
    screen_aspect = float(win.size[0]) / float(win.size[1])
    if IMAGE_FIT_MODE == "cover":
        if image_aspect >= screen_aspect:
            return image_aspect, 1.0
        return screen_aspect, screen_aspect / image_aspect
    if image_aspect >= screen_aspect:
        return screen_aspect, screen_aspect / image_aspect
    return image_aspect, 1.0


class EmotionStimulusRenderer:
    def __init__(self, win):
        self.win = win
        self.fixation = visual.TextStim(win, text="+", height=0.12, color=FIXATION_COLOR, pos=(0.0, 0.0))
        self.small_fixation = visual.Circle(win, radius=FIXATION_DOT_SIZE, fillColor="white", lineColor="white", pos=(0.0, 0.0))
        self.progress = visual.TextStim(win, text="", height=0.035, color="#d6d6d6", pos=(0.0, -0.43), wrapWidth=1.7, alignText="center")
        self.washout = visual.TextStim(win, text="", height=0.055, color="#e0e0e0", pos=(0.0, 0.0), wrapWidth=1.5, alignText="center")

    def make_image(self, image_path):
        size = image_size_for_window(self.win, image_path)
        return visual.ImageStim(self.win, image=str(image_path), size=size, interpolate=True)

    def draw_fixation(self, trial_index, total_trials, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.fixation.draw()
        self.progress.text = f"Image {trial_index}/{total_trials}  |  steady gaze  |  {int(seconds_left) + 1}s"
        self.progress.draw()

    def draw_image(self, image_stim):
        self.win.color = BACKGROUND_COLOR
        image_stim.draw()
        if SHOW_FIXATION_DOT_ON_IMAGE:
            self.small_fixation.draw()

    def draw_washout(self, trial_index, total_trials, seconds_left):
        self.win.color = WASHOUT_COLOR
        self.washout.text = f"Blink now if needed\n\nImage {trial_index}/{total_trials}\nNext image in {int(seconds_left) + 1}s"
        self.washout.draw()

    def draw_break(self, completed, total_trials, seconds_left):
        self.win.color = BACKGROUND_COLOR
        self.washout.text = f"Short rest\n\nCompleted {completed}/{total_trials}\nRelax jaw and shoulders\n\nContinuing in {int(seconds_left) + 1}s"
        self.washout.draw()


def add_event(events, event_type, state_label, state_phase, emotion_label, trial_row, pc_time_perf_counter_s, flip_time_psychopy_s, frame_index, value):
    row = {
        "event_type": event_type,
        "state_label": state_label,
        "state_phase": state_phase,
        "emotion_label": emotion_label,
        "emotion_index": int(EMOTION_TO_INDEX.get(str(emotion_label), -1)),
        "trial_index": int(trial_row["trial_index"]),
        "image_id": str(trial_row["image_id"]),
        "image_path": str(trial_row["image_path"]),
        "annotation_path": str(trial_row["annotation_path"]),
        "pc_time_perf_counter_s": pc_time_perf_counter_s,
        "flip_time_psychopy_s": flip_time_psychopy_s,
        "frame_index": int(frame_index),
        "value": value,
    }
    for column in ["brightness", "colorfulness", "facial_expression"]:
        row[column] = trial_row[column] if column in trial_row.index else None
    events.append(row)


def blank_trial_row():
    return pd.Series({
        "trial_index": -1,
        "image_id": "none",
        "image_path": "",
        "annotation_path": "",
        "brightness": None,
        "colorfulness": None,
        "facial_expression": None,
    })


def present_fixation(win, renderer, events, recorder, refresh_rate_hz, seconds, trial_row, frame_counter, total_trials):
    trial_index = int(trial_row["trial_index"])
    emotion_label = str(trial_row["emotion"])
    state_label = f"fixation_{trial_index:03d}"
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(state_label)
    started = False
    for local_frame in range(total_frames):
        seconds_left = seconds - local_frame / refresh_rate_hz
        renderer.draw_fixation(trial_index, total_trials, seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "fixation", emotion_label, trial_row, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "fixation", emotion_label, trial_row, perf_time, flip_time, frame_counter, "fixation")
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, "fixation", emotion_label, trial_row, time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, "fixation", emotion_label, trial_row, time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_image(win, renderer, events, recorder, refresh_rate_hz, seconds, trial_row, frame_counter, image_stim):
    trial_index = int(trial_row["trial_index"])
    emotion_label = str(trial_row["emotion"])
    state_label = f"image_{emotion_label}_{trial_index:03d}"
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(state_label)
    started = False
    for local_frame in range(total_frames):
        renderer.draw_image(image_stim)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "image", emotion_label, trial_row, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "image", emotion_label, trial_row, perf_time, flip_time, frame_counter, str(trial_row["image_id"]))
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, "image", emotion_label, trial_row, time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, "image", emotion_label, trial_row, time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_washout(win, renderer, events, recorder, refresh_rate_hz, seconds, trial_row, frame_counter, total_trials):
    trial_index = int(trial_row["trial_index"])
    emotion_label = str(trial_row["emotion"])
    state_label = f"washout_{trial_index:03d}"
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(state_label)
    started = False
    for local_frame in range(total_frames):
        seconds_left = seconds - local_frame / refresh_rate_hz
        renderer.draw_washout(trial_index, total_trials, seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "washout", emotion_label, trial_row, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "washout", emotion_label, trial_row, perf_time, flip_time, frame_counter, "washout")
        frame_counter += 1
        if "escape" in event.getKeys():
            add_event(events, "state_end", state_label, "washout", emotion_label, trial_row, time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
    add_event(events, "state_end", state_label, "washout", emotion_label, trial_row, time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_break(win, renderer, events, recorder, refresh_rate_hz, seconds, completed, total_trials, frame_counter):
    trial_row = blank_trial_row()
    state_label = f"break_after_{completed:03d}"
    total_frames = frames_for_seconds(refresh_rate_hz, seconds)
    recorder.annotate(state_label)
    started = False
    for local_frame in range(total_frames):
        seconds_left = seconds - local_frame / refresh_rate_hz
        renderer.draw_break(completed, total_trials, seconds_left)
        flip_time, perf_time = perf_time_after_flip(win)
        if not started:
            add_event(events, "state_start", state_label, "break", "break", trial_row, perf_time, flip_time, frame_counter, None)
            started = True
        add_event(events, "frame", state_label, "break", "break", trial_row, perf_time, flip_time, frame_counter, "break")
        frame_counter += 1
        keys = event.getKeys()
        if "escape" in keys:
            add_event(events, "state_end", state_label, "break", "break", trial_row, time.perf_counter(), None, frame_counter, "stopped_by_escape")
            return frame_counter, True
        if "space" in keys:
            break
    add_event(events, "state_end", state_label, "break", "break", trial_row, time.perf_counter(), None, frame_counter, None)
    return frame_counter, False


def present_trial(win, renderer, events, recorder, refresh_rate_hz, trial_row, frame_counter, total_trials, rng):
    image_stim = renderer.make_image(Path(trial_row["image_path"]))
    fixation_seconds = float(rng.uniform(FIXATION_SECONDS_MIN, FIXATION_SECONDS_MAX))
    frame_counter, stopped = present_fixation(win, renderer, events, recorder, refresh_rate_hz, fixation_seconds, trial_row, frame_counter, total_trials)
    if stopped:
        return frame_counter, True
    play_beep(900, 90)
    frame_counter, stopped = present_image(win, renderer, events, recorder, refresh_rate_hz, IMAGE_SECONDS, trial_row, frame_counter, image_stim)
    if stopped:
        return frame_counter, True
    frame_counter, stopped = present_washout(win, renderer, events, recorder, refresh_rate_hz, WASHOUT_SECONDS, trial_row, frame_counter, total_trials)
    return frame_counter, stopped


def make_session_metadata(recorder, refresh_rate_hz, schedule, stopped_by_escape, completed_trials):
    metadata = recorder.metadata()
    metadata.update({
        "test": "image_emotion_cognition",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(DATASET_ROOT),
        "dataset_split_name": DATASET_SPLIT_NAME,
        "display_refresh_rate_hz": refresh_rate_hz,
        "images_per_session": IMAGES_PER_SESSION,
        "planned_image_count": int(len(schedule)),
        "completed_image_count": int(completed_trials),
        "stopped_by_escape": bool(stopped_by_escape),
        "emotion_labels": EMOTION_LABELS,
        "emotions_to_use": list(EMOTIONS_TO_USE),
        "fixation_seconds_min": FIXATION_SECONDS_MIN,
        "fixation_seconds_max": FIXATION_SECONDS_MAX,
        "image_seconds": IMAGE_SECONDS,
        "washout_seconds": WASHOUT_SECONDS,
        "break_every_images": BREAK_EVERY_IMAGES,
        "break_seconds": BREAK_SECONDS,
        "image_fit_mode": IMAGE_FIT_MODE,
        "show_fixation_dot_on_image": SHOW_FIXATION_DOT_ON_IMAGE,
        "neurological_design": "Passive affective picture viewing with jittered fixation baseline, fixed stimulus exposure, post-stimulus washout for blinks, balanced emotion sampling, no on-image emotion words, and event timestamps logged immediately after display flips using time.perf_counter.",
        "training_guardrail": "Training scripts use EEG samples only. Image paths, annotations, brightness, colorfulness, and facial-expression metadata are saved for audit/grouping but not used as model features.",
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
        save_labeled_image_samples(session_dir)
    if make_summary and len(eeg_df) > 0 and len(events) > 0:
        plot_recording_summary(session_dir, SUMMARY_NORMALIZATION)
    return eeg_df


def save_session_manifest(session_dir, schedule, session_summary):
    manifest = {
        "session_dir": str(session_dir),
        "session_summary": session_summary,
        "emotion_labels": EMOTION_LABELS,
        "schedule": schedule.to_dict(orient="records"),
    }
    write_json(session_dir / "manifest.json", manifest)
    return manifest


def run_session(win, renderer, session_dir, schedule, refresh_rate_hz):
    rng = np.random.default_rng(RANDOM_SEED + int(time.time()))
    events = []
    frame_counter = 0
    stopped_by_escape = False
    completed_trials = 0
    eeg_df = pd.DataFrame()
    total_trials = len(schedule)
    with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME, use_bias=USE_BIAS, bias_channel_name=BIAS_CHANNEL_NAME) as recorder:
        recorder.start()
        countdown(win, 3)
        for row_index, trial_row in schedule.iterrows():
            if completed_trials > 0 and BREAK_EVERY_IMAGES > 0 and completed_trials % BREAK_EVERY_IMAGES == 0:
                frame_counter, stopped_by_escape = present_break(win, renderer, events, recorder, refresh_rate_hz, BREAK_SECONDS, completed_trials, total_trials, frame_counter)
                if stopped_by_escape:
                    break
            frame_counter, stopped_by_escape = present_trial(win, renderer, events, recorder, refresh_rate_hz, trial_row, frame_counter, total_trials, rng)
            if not stopped_by_escape:
                completed_trials = int(trial_row["trial_index"])
            metadata = make_session_metadata(recorder, refresh_rate_hz, schedule, stopped_by_escape, completed_trials)
            eeg_df = save_session_progress(session_dir, recorder, events, metadata, make_summary=False)
            print(f"Saved image {int(trial_row['trial_index'])}/{total_trials}: {session_dir}")
            if stopped_by_escape:
                break
        recorder.stop()
        metadata = make_session_metadata(recorder, refresh_rate_hz, schedule, stopped_by_escape, completed_trials)
        eeg_df = save_session_progress(session_dir, recorder, events, metadata, make_summary=True)
    return {
        "planned_image_count": int(total_trials),
        "completed_image_count": int(completed_trials),
        "stopped_by_escape": bool(stopped_by_escape),
        "rows": int(len(eeg_df)),
        "duration_s": float(len(eeg_df) / 250.0) if len(eeg_df) > 0 else 0.0,
    }


def main():
    manifest = load_emoset_manifest(DATASET_ROOT, DATASET_SPLIT_NAME)
    schedule = balanced_emotion_schedule(
        manifest,
        IMAGES_PER_SESSION,
        EMOTIONS_TO_USE,
        RANDOM_SEED,
        data_root=OUTPUT_ROOT,
        avoid_previous=AVOID_PREVIOUSLY_COLLECTED_IMAGES,
    )
    session_dir = make_session_output_dir()
    schedule.to_csv(session_dir / "session_schedule.csv", index=False)
    win = make_window(BACKGROUND_COLOR)
    refresh_rate_hz = estimate_refresh_rate(win)
    renderer = EmotionStimulusRenderer(win)
    session_summary = {}
    try:
        wait_with_text(
            win,
            "IMAGE EMOTION COGNITION\n\nLook naturally at each image and let the emotional response happen.\nDo not name the emotion, analyze the picture, move your face, clench your jaw, or press keys.\nKeep gaze near the center dot. Blink only during the blank screen after each image.\n\nThe run saves after every image. Press ESC to stop and keep partial data.",
            10.0,
        )
        session_summary = run_session(win, renderer, session_dir, schedule, refresh_rate_hz)
        save_session_manifest(session_dir, schedule, session_summary)
        if session_summary["stopped_by_escape"]:
            wait_with_text(win, f"STOPPED AND SAVED\n\nSaved to:\n{session_dir}", 3.0, escape_allowed=False)
        else:
            wait_with_text(win, f"DONE\n\nSaved to:\n{session_dir}", 4.0, escape_allowed=False)
    finally:
        save_session_manifest(session_dir, schedule, session_summary)
        win.close()


if __name__ == "__main__":
    main()
