import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_ROOT = PROJECT_ROOT / "sessions"
SUBJECT_ID = "subject_001"
SESSION_SCHEMA_VERSION = "brainz_session_v2"
TAG_VOCABULARY = {
    "Sensory modality": ["visual", "auditory", "somatosensory", "proprioceptive", "multisensory"],
    "Action": ["motor", "fine_motor", "gross_motor", "walking", "speech", "oculomotor", "facial_motor"],
    "Cognition": ["imagery", "motor_imagery", "attention", "working_memory", "problem_solving", "executive_control", "visuospatial", "language", "reading", "emotion", "creativity", "decision_making"],
    "State": ["rest", "eyes_open", "eyes_closed", "active", "passive", "high_effort", "unknown_context"],
    "Protocol": ["baseline", "naturalistic", "interactive", "flicker", "ssvep", "blink", "jaw_emg", "movement_artifact"],
    "Expected system": ["prefrontal", "frontal", "motor_cortex", "premotor", "somatosensory_cortex", "parietal", "temporal", "auditory_cortex", "visual_cortex", "occipital", "language_network", "limbic", "default_mode_network"],
}
DEFAULT_TASK_TAGS = {
    "scrolling_reels": ["visual", "auditory", "multisensory", "fine_motor", "oculomotor", "attention", "interactive", "naturalistic", "visual_cortex", "auditory_cortex", "parietal"],
    "watching_tv_show": ["visual", "auditory", "multisensory", "passive", "attention", "naturalistic", "visual_cortex", "auditory_cortex", "temporal"],
    "listening_to_classical_music": ["auditory", "passive", "attention", "emotion", "naturalistic", "auditory_cortex", "temporal", "limbic"],
    "solving_rubiks_cubes": ["visual", "proprioceptive", "fine_motor", "active", "working_memory", "problem_solving", "executive_control", "visuospatial", "interactive", "prefrontal", "motor_cortex", "parietal", "visual_cortex"],
    "solving_hard_rubicks_cubes": ["visual", "proprioceptive", "fine_motor", "active", "high_effort", "working_memory", "problem_solving", "executive_control", "visuospatial", "interactive", "prefrontal", "motor_cortex", "parietal", "visual_cortex"],
    "solving_hard_rubiks_cubes": ["visual", "proprioceptive", "fine_motor", "active", "high_effort", "working_memory", "problem_solving", "executive_control", "visuospatial", "interactive", "prefrontal", "motor_cortex", "parietal", "visual_cortex"],
    "reading_an_article": ["visual", "oculomotor", "active", "attention", "working_memory", "language", "reading", "visual_cortex", "language_network", "temporal"],
    "reading_aloud": ["visual", "auditory", "speech", "facial_motor", "oculomotor", "active", "attention", "language", "reading", "motor_cortex", "auditory_cortex", "language_network"],
    "physical_pain": ["somatosensory", "passive", "attention", "somatosensory_cortex", "limbic"],
    "complete_resting": ["rest", "eyes_closed", "passive", "baseline", "default_mode_network"],
    "walking": ["motor", "gross_motor", "walking", "proprioceptive", "active", "movement_artifact", "motor_cortex", "premotor", "somatosensory_cortex"],
    "chess_puzzles": ["visual", "oculomotor", "active", "high_effort", "attention", "working_memory", "problem_solving", "executive_control", "visuospatial", "decision_making", "prefrontal", "parietal", "visual_cortex"],
    "playing_guitar": ["auditory", "proprioceptive", "fine_motor", "motor", "active", "creativity", "multisensory", "movement_artifact", "motor_cortex", "premotor", "auditory_cortex"],
    "basic_motor_cognition": ["visual", "imagery", "motor_imagery", "attention", "active", "baseline", "motor_cortex", "premotor", "parietal"],
    "image_emotion_cognition": ["visual", "passive", "attention", "emotion", "baseline", "visual_cortex", "occipital", "limbic"],
    "eyes_open_closed_and_blinks": ["visual", "eyes_open", "eyes_closed", "blink", "oculomotor", "baseline", "visual_cortex", "occipital"],
    "jaw_clench": ["motor", "facial_motor", "jaw_emg", "baseline", "movement_artifact", "motor_cortex"],
    "ssvep_flicker": ["visual", "passive", "attention", "flicker", "ssvep", "baseline", "visual_cortex", "occipital"],
    "standalone_brainaccess_recording": ["unknown_context"],
}


def slugify(value):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text[:80] or "untitled"


def all_tags():
    return [tag for tags in TAG_VOCABULARY.values() for tag in tags]


def tags_for_task(task_name, existing=None):
    selected = [str(value) for value in (existing or []) if str(value) in all_tags()]
    inferred = DEFAULT_TASK_TAGS.get(slugify(task_name), [])
    return sorted(set(selected + inferred), key=all_tags().index)


def make_session_dir(title, stamp=None):
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_ROOT / f"session_{timestamp}_{slugify(title)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def event_summary(events_path):
    path = Path(events_path)
    if not path.exists():
        return {"present": False, "event_count": 0, "columns": [], "event_types": [], "state_labels": []}
    frame = pd.read_csv(path, low_memory=False)
    def values(column):
        if column not in frame:
            return []
        return sorted(value for value in frame[column].dropna().astype(str).unique().tolist() if value not in {"", "nan"})
    labels = []
    for column in ["state_label", "emotion_label", "event_subtype", "phase"]:
        labels.extend(values(column))
    return {
        "present": True,
        "event_count": int(len(frame)),
        "columns": frame.columns.tolist(),
        "event_types": values("event_type"),
        "state_labels": sorted(set(labels)),
    }


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def write_session_files(session_dir, title, description, tags, source_type, protocol_version, legacy_path="", extra=None):
    path = Path(session_dir)
    metadata = {}
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    events = event_summary(path / "events.csv")
    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": path.name,
        "subject_id": SUBJECT_ID,
        "title": str(title).strip(),
        "description": str(description).strip(),
        "tags": [tag for tag in all_tags() if tag in tags] if tags else tags_for_task(title),
        "tags_meaning": "Protocol and expected-system descriptors, not claims that a brain region was directly measured as active.",
        "source_type": source_type,
        "protocol_version": protocol_version,
        "recorded_at_utc": metadata.get("session_started_at_utc", metadata.get("started_at_utc", metadata.get("created_at_utc"))),
        "legacy_path": legacy_path,
        "events": events,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        session.update(extra)
    write_json(path / "session.json", session)
    tag_text = ", ".join(session["tags"]) if session["tags"] else "none"
    event_text = ", ".join(events["event_types"]) if events["event_types"] else "none"
    state_text = ", ".join(events["state_labels"]) if events["state_labels"] else "none"
    description_text = session["description"] or "No description was provided."
    text = (
        f"# {session['title']}\n\n"
        f"{description_text}\n\n"
        f"Subject: {SUBJECT_ID}\n\n"
        f"Session: {path.name}\n\n"
        f"Protocol: {protocol_version or 'not specified'}\n\n"
        f"Tags: {tag_text}\n\n"
        f"Events: {events['event_count']} rows; event types: {event_text}.\n\n"
        f"Event/state labels: {state_text}.\n\n"
        "Tags describe the task and brain systems expected to be relevant. They are not direct localization or clinical findings.\n"
    )
    (path / "SESSION.md").write_text(text, encoding="utf-8")
    return session
