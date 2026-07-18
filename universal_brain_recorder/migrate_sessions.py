import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from session_common import DEFAULT_TASK_TAGS, PROJECT_ROOT, SESSIONS_ROOT, slugify, tags_for_task, write_session_files
from traceability_common import SCHEMA_VERSION, file_provenance, read_json, rebuild_catalog, standard_recording_traceability, utc_now, write_json


RUN_MIGRATION = True
VERIFY_HASHES = True
SOURCE_ROOTS = [
    PROJECT_ROOT / "universal_brain_recorder" / "data",
    PROJECT_ROOT / "basic_motor_cognition" / "data",
    PROJECT_ROOT / "concentration_baseline" / "data",
    PROJECT_ROOT / "box_motion_cognition" / "data",
    PROJECT_ROOT / "image_emotion_cognition" / "data",
    PROJECT_ROOT / "recordings",
]
TASK_TITLES = {
    "basic_motor_cognition": "Basic motor cognition",
    "image_emotion_cognition": "Image emotion cognition",
    "eyes_open_closed_and_blinks": "Eyes open, eyes closed, and blinks",
    "jaw_clench": "Jaw clench",
    "ssvep_flicker": "SSVEP flicker",
    "standalone_brainaccess_recording": "Standalone BrainAccess recording",
}


def sha256_files(folder):
    records = {}
    for path in sorted(Path(folder).rglob("*"), key=str):
        if path.is_file():
            records[str(path.relative_to(folder))] = file_provenance(path, "migration_source", relative_to=folder)["sha256"]
    return records


def authoritative_session_dirs():
    sessions = set()
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for eeg_path in root.rglob("eeg_samples.csv"):
            if "inference" in eeg_path.parent.name.lower() or "baseline" in eeg_path.parent.name.lower():
                continue
            sessions.add(eeg_path.parent.resolve())
    return sorted(sessions, key=str)


def timestamp_for(path):
    for candidate in [path.name] + [parent.name for parent in path.parents]:
        match = re.search(r"(20\d{6}_\d{6})", candidate)
        if match:
            return match.group(1)
    modified = datetime.fromtimestamp((path / "eeg_samples.csv").stat().st_mtime)
    return modified.strftime("%Y%m%d_%H%M%S")


def task_details(source):
    traceability_path = source / "traceability.json"
    traceability = read_json(traceability_path) if traceability_path.exists() else standard_recording_traceability(source)
    task = traceability.get("primary_task", {})
    name = str(task.get("name") or source.name)
    title = TASK_TITLES.get(slugify(name), name.strip().capitalize())
    description = str(task.get("description") or "").strip()
    metadata_path = source / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        name = str(metadata.get("primary_task_name") or name)
        title = TASK_TITLES.get(slugify(name), name.strip().capitalize())
        description = str(metadata.get("primary_task_description") or description).strip()
    return traceability, name, title, description


def destination_for(source, task_name):
    name = f"session_{timestamp_for(source)}_{slugify(task_name)}"
    return SESSIONS_ROOT / name


def refresh_traceability(destination, original_relative, traceability, task_name, title, description, tags):
    traceability["schema_version"] = SCHEMA_VERSION
    traceability["session_id"] = destination.name
    traceability["recording_path"] = str(destination.relative_to(PROJECT_ROOT))
    traceability["subject_id"] = "subject_001"
    traceability["legacy_recording_path"] = original_relative
    traceability["primary_task"] = {"name": task_name, "title": title, "description": description, "tags": tags}
    traceability["traceability_status"] = "user_annotated" if traceability.get("traceability_status") == "user_annotated" else "protocol_inferred_and_canonicalized"
    for segment in traceability.get("task_segments", []):
        segment["tags"] = tags_for_task(segment.get("task_name", task_name), segment.get("tags", []))
    roles = {
        "eeg_samples.csv": "raw_eeg",
        "events.csv": "task_events",
        "metadata.json": "recording_metadata",
        "brainaccess_annotations.csv": "sdk_annotations",
        "session_schedule.csv": "stimulus_schedule",
        "manifest.json": "session_manifest",
        "recording_summary.json": "recording_summary",
        "summary.png": "recording_summary_image",
        "session.json": "canonical_session_metadata",
        "SESSION.md": "natural_language_session_info",
    }
    traceability["files"] = [file_provenance(destination / filename, role) for filename, role in roles.items() if (destination / filename).exists()]
    write_json(destination / "traceability.json", traceability)


def migrate_directory(source):
    original_relative = str(source.relative_to(PROJECT_ROOT))
    traceability, task_name, title, description = task_details(source)
    destination = destination_for(source, task_name)
    before = sha256_files(source) if VERIFY_HASHES else {}
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination}")
    shutil.move(str(source), str(destination))
    after = sha256_files(destination) if VERIFY_HASHES else {}
    if VERIFY_HASHES and before != after:
        raise RuntimeError(f"Hash verification failed for {original_relative}")
    tags = tags_for_task(task_name, traceability.get("primary_task", {}).get("tags", []))
    protocol = traceability.get("protocol_version", "")
    write_session_files(destination, title, description, tags, traceability.get("source_type", "existing_standard_recording"), protocol, original_relative)
    refresh_traceability(destination, original_relative, traceability, task_name, title, description, tags)
    return {
        "legacy_path": original_relative,
        "session_path": str(destination.relative_to(PROJECT_ROOT)),
        "session_id": destination.name,
        "title": title,
        "tags": tags,
        "verified_file_count": len(before),
        "hashes_verified": VERIFY_HASHES,
    }


def migrate_standalone():
    csv_path = PROJECT_ROOT / "brainaccess_maxi_32ch.csv"
    if not csv_path.exists():
        return None
    stamp = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    destination = SESSIONS_ROOT / f"session_{stamp}_standalone_brainaccess_recording"
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for source in [csv_path, csv_path.with_suffix(".npz")]:
        if source.exists():
            target_name = "eeg_samples.csv" if source.suffix.lower() == ".csv" else source.name
            before = file_provenance(source, "migration_source")["sha256"]
            shutil.move(str(source), str(destination / target_name))
            after = file_provenance(destination / target_name, "migration_destination")["sha256"]
            if before != after:
                raise RuntimeError(f"Hash verification failed for {source}")
            moved.append(target_name)
    title = TASK_TITLES["standalone_brainaccess_recording"]
    description = "Legacy standalone 32-channel BrainAccess recording. No event file or task description was available."
    tags = DEFAULT_TASK_TAGS["standalone_brainaccess_recording"]
    write_session_files(destination, title, description, tags, "standalone_recording", "", "brainaccess_maxi_32ch.csv")
    traceability = standard_recording_traceability(destination)
    traceability["traceability_id"] = uuid.uuid5(uuid.NAMESPACE_URL, str(destination.resolve())).hex
    traceability["source_type"] = "standalone_recording"
    refresh_traceability(destination, "brainaccess_maxi_32ch.csv", traceability, "standalone_brainaccess_recording", title, description, tags)
    return {
        "legacy_path": "brainaccess_maxi_32ch.csv",
        "session_path": str(destination.relative_to(PROJECT_ROOT)),
        "session_id": destination.name,
        "title": title,
        "tags": tags,
        "verified_file_count": len(moved),
        "hashes_verified": True,
    }


def main():
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    sources = authoritative_session_dirs()
    print("Raw sessions found:", len(sources))
    for source in sources:
        print(source.relative_to(PROJECT_ROOT), "->", destination_for(source, task_details(source)[1]).relative_to(PROJECT_ROOT))
    if not RUN_MIGRATION:
        print("RUN_MIGRATION is False; no files were moved.")
        return
    records = [migrate_directory(source) for source in sources]
    standalone = migrate_standalone()
    if standalone is not None:
        records.append(standalone)
    manifest = {
        "schema_version": "brainz_session_migration_v1",
        "migrated_at_utc": utc_now(),
        "session_count": len(records),
        "sessions": records,
    }
    write_json(SESSIONS_ROOT / "migration_manifest.json", manifest)
    rebuild_catalog()
    print("Migrated sessions:", len(records))
    print("Canonical root:", SESSIONS_ROOT)


if __name__ == "__main__":
    main()
