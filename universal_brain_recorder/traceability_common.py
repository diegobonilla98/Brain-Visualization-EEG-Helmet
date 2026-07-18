import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from session_common import SESSIONS_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SESSIONS_ROOT
SIDECAR_ROOT = MODULE_ROOT / "legacy_sidecars"
CATALOG_CSV = MODULE_ROOT / "recording_catalog.csv"
CATALOG_JSON = MODULE_ROOT / "recording_catalog.json"
SCHEMA_VERSION = "brainz_traceability_v1"
TASK_DESCRIPTIONS = {
    "basic_motor_cognition": "Motor imagery experiment with rest, hand, feet, and both-hands task states.",
    "motor_imagery_erd_v3": "Motor imagery ERD protocol with per-trial baseline, cue, imagery, and cooldown phases.",
    "image_emotion_cognition": "Passive affective image-viewing EEG experiment.",
    "image_emotion_passive_viewing_v1": "Passive EmoSet image viewing with fixation baseline and washout phases.",
    "eyes_open_closed_and_blinks": "Eyes-open, eyes-closed, and deliberate blink quality experiment.",
    "jaw_clench": "Relaxed-jaw and deliberate jaw-clench artifact experiment.",
    "ssvep_flicker": "Steady-state visual evoked potential flicker experiment with rest periods.",
    "derived_baseline_dataset": "Derived baseline EEG exports assembled from existing source recordings.",
    "standalone_brainaccess_recording": "Standalone BrainAccess MAXI EEG recording with limited historical task metadata.",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def slugify(value):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text[:80] or "unnamed_task"


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def sha256_file(path, block_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if len(block) == 0:
                break
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path, role, relative_to=PROJECT_ROOT):
    path = Path(path)
    try:
        relative = path.resolve().relative_to(Path(relative_to).resolve())
        stored_path = str(relative)
    except ValueError:
        stored_path = str(path.resolve())
    return {
        "path": stored_path,
        "role": role,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def event_state_labels(events_path):
    if not events_path.exists():
        return []
    header = pd.read_csv(events_path, nrows=0).columns
    if "state_label" not in header:
        return []
    values = pd.read_csv(events_path, usecols=["state_label"])["state_label"].dropna().astype(str)
    return sorted(value for value in values.unique().tolist() if value.lower() not in {"", "nan", "unlabeled"})


def inferred_event_segments(events_path, stream_start_perf_s=None):
    if not events_path.exists():
        return []
    header = pd.read_csv(events_path, nrows=0).columns.tolist()
    required = {"event_type", "state_label", "pc_time_perf_counter_s"}
    if not required.issubset(header):
        return []
    frame = pd.read_csv(events_path, usecols=list(required))
    frame = frame[frame["event_type"].isin(["state_start", "state_end"])].sort_values("pc_time_perf_counter_s")
    segments = []
    current = None
    for row in frame.itertuples(index=False):
        event_type = str(row.event_type)
        label = str(row.state_label)
        event_time = float(row.pc_time_perf_counter_s)
        relative_time = event_time - float(stream_start_perf_s) if stream_start_perf_s is not None else None
        if event_type == "state_start":
            if current is not None:
                current["end_pc_time_perf_counter_s"] = event_time
                current["end_t_from_stream_start_s"] = relative_time
                current["duration_s"] = event_time - current["start_pc_time_perf_counter_s"]
                segments.append(current)
            current = {
                "segment_index": len(segments),
                "task_name": label,
                "task_slug": slugify(label),
                "description": "Segment reconstructed from legacy state events.",
                "tags": [],
                "start_pc_time_perf_counter_s": event_time,
                "start_t_from_stream_start_s": relative_time,
                "start_utc_time": None,
                "end_pc_time_perf_counter_s": None,
                "end_t_from_stream_start_s": None,
                "end_utc_time": None,
                "duration_s": None,
                "traceability_origin": "inferred_from_existing_events",
            }
        elif current is not None and label == current["task_name"]:
            current["end_pc_time_perf_counter_s"] = event_time
            current["end_t_from_stream_start_s"] = relative_time
            current["duration_s"] = event_time - current["start_pc_time_perf_counter_s"]
            segments.append(current)
            current = None
    if current is not None:
        segments.append(current)
    return segments


def inferred_task(metadata, fallback):
    candidates = [
        metadata.get("test"),
        metadata.get("protocol_version"),
        fallback,
    ]
    for candidate in candidates:
        if candidate is not None and len(str(candidate).strip()) > 0:
            name = str(candidate)
            return name, TASK_DESCRIPTIONS.get(name, f"Legacy EEG session inferred from metadata identifier '{name}'.")
    return "unknown_legacy_task", "Legacy EEG recording with insufficient metadata to reconstruct the task."


def standard_recording_traceability(session_dir):
    session_dir = Path(session_dir)
    metadata_path = session_dir / "metadata.json"
    events_path = session_dir / "events.csv"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    task_name, task_description = inferred_task(metadata, session_dir.name)
    files = []
    roles = {
        "eeg_samples.csv": "raw_eeg",
        "events.csv": "task_events",
        "metadata.json": "recording_metadata",
        "brainaccess_annotations.csv": "sdk_annotations",
        "session_schedule.csv": "stimulus_schedule",
        "manifest.json": "session_manifest",
    }
    for filename, role in roles.items():
        path = session_dir / filename
        if path.exists():
            files.append(file_provenance(path, role))
    labels = event_state_labels(events_path)
    segments = inferred_event_segments(events_path, metadata.get("stream_start_perf_s"))
    evidence = [str(path.name) for path in [metadata_path, events_path] if path.exists()]
    identifier = uuid.uuid5(uuid.NAMESPACE_URL, str(session_dir.resolve())).hex
    return {
        "schema_version": SCHEMA_VERSION,
        "traceability_id": identifier,
        "session_id": session_dir.name,
        "recording_path": str(session_dir.resolve().relative_to(PROJECT_ROOT)),
        "source_type": "existing_standard_recording",
        "traceability_status": "inferred_from_existing_metadata",
        "created_at_utc": utc_now(),
        "participant_alias": metadata.get("participant_alias", ""),
        "consent_confirmed": None,
        "protocol_version": metadata.get("protocol_version", metadata.get("test", "")),
        "primary_task": {
            "name": task_name,
            "description": task_description,
            "tags": labels,
        },
        "task_segments": segments,
        "available_state_labels": labels,
        "device": {
            "requested": metadata.get("device_name_requested"),
            "connected": metadata.get("device_name_connected"),
            "model": metadata.get("device_model"),
            "serial_number": metadata.get("serial_number"),
            "sample_frequency_hz": metadata.get("sample_frequency_hz"),
            "channel_count": metadata.get("eeg_channel_count"),
        },
        "files": files,
        "parent_sources": [],
        "notes": "Task traceability was reconstructed after recording. Raw data and original metadata were not modified.",
        "inference_evidence": evidence,
    }


def derived_dataset_traceability(session_dir):
    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    manifest = read_json(manifest_path)
    files = [file_provenance(manifest_path, "derivation_manifest")]
    for path in sorted(session_dir.glob("*.csv")):
        files.append(file_provenance(path, "derived_eeg"))
    sources = manifest.get("sources", [])
    if isinstance(sources, dict):
        sources = [{"name": key, "path": value} for key, value in sources.items()]
    identifier = uuid.uuid5(uuid.NAMESPACE_URL, str(session_dir.resolve())).hex
    return {
        "schema_version": SCHEMA_VERSION,
        "traceability_id": identifier,
        "session_id": session_dir.name,
        "recording_path": str(session_dir.resolve().relative_to(PROJECT_ROOT)),
        "source_type": "derived_baseline_dataset",
        "traceability_status": "inferred_from_existing_manifest",
        "created_at_utc": utc_now(),
        "participant_alias": "",
        "consent_confirmed": None,
        "protocol_version": "derived_baseline_dataset",
        "primary_task": {
            "name": "derived_baseline_dataset",
            "description": TASK_DESCRIPTIONS["derived_baseline_dataset"],
            "tags": ["baseline", "derived"],
        },
        "task_segments": [],
        "available_state_labels": [],
        "device": {},
        "files": files,
        "parent_sources": sources,
        "notes": "This directory contains derived exports. Source recordings remain the authoritative raw EEG.",
        "inference_evidence": ["manifest.json"],
    }


def standalone_traceability(csv_path):
    csv_path = Path(csv_path)
    npz_path = csv_path.with_suffix(".npz")
    files = [file_provenance(csv_path, "raw_eeg")]
    if npz_path.exists():
        files.append(file_provenance(npz_path, "raw_eeg_binary"))
    identifier = uuid.uuid5(uuid.NAMESPACE_URL, str(csv_path.resolve())).hex
    return {
        "schema_version": SCHEMA_VERSION,
        "traceability_id": identifier,
        "session_id": csv_path.stem,
        "recording_path": str(csv_path.resolve().relative_to(PROJECT_ROOT)),
        "source_type": "standalone_recording",
        "traceability_status": "inferred_from_filename",
        "created_at_utc": utc_now(),
        "participant_alias": "",
        "consent_confirmed": None,
        "protocol_version": "",
        "primary_task": {
            "name": "standalone_brainaccess_recording",
            "description": TASK_DESCRIPTIONS["standalone_brainaccess_recording"],
            "tags": ["unannotated", "legacy"],
        },
        "task_segments": [],
        "available_state_labels": [],
        "device": {
            "sample_frequency_hz": 250,
            "channel_count": 32,
        },
        "files": files,
        "parent_sources": [],
        "notes": "No session events or task metadata were found. Task identity remains unknown.",
        "inference_evidence": [csv_path.name],
    }


def traceability_files():
    paths = list(PROJECT_ROOT.rglob("traceability.json"))
    if SIDECAR_ROOT.exists():
        paths.extend(SIDECAR_ROOT.glob("*.traceability.json"))
    return sorted(set(path.resolve() for path in paths), key=str)


def rebuild_catalog():
    rows = []
    records = []
    for path in traceability_files():
        traceability = read_json(path)
        records.append(traceability)
        task = traceability.get("primary_task", {})
        device = traceability.get("device", {})
        rows.append({
            "traceability_id": traceability.get("traceability_id"),
            "session_id": traceability.get("session_id"),
            "recording_path": traceability.get("recording_path"),
            "source_type": traceability.get("source_type"),
            "traceability_status": traceability.get("traceability_status"),
            "task_name": task.get("name"),
            "task_description": task.get("description"),
            "tags": "|".join(str(value) for value in task.get("tags", [])),
            "participant_alias": traceability.get("participant_alias"),
            "protocol_version": traceability.get("protocol_version"),
            "sample_frequency_hz": device.get("sample_frequency_hz"),
            "channel_count": device.get("channel_count"),
            "file_count": len(traceability.get("files", [])),
            "sidecar_path": str(path.relative_to(PROJECT_ROOT)),
        })
    MODULE_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["source_type", "session_id"]).to_csv(CATALOG_CSV, index=False)
    write_json(CATALOG_JSON, {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "recording_count": len(records),
        "recordings": records,
    })
    return CATALOG_CSV, CATALOG_JSON


def session_output_dir(task_name):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = DATA_ROOT / f"session_{stamp}_{slugify(task_name)}"
    path.mkdir(parents=True, exist_ok=False)
    return path
