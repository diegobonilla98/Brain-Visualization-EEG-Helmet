import json
from pathlib import Path

import pandas as pd

from session_common import SESSIONS_ROOT, all_tags, tags_for_task, write_json, write_session_files
from traceability_common import file_provenance, rebuild_catalog


def main():
    rows = []
    for session_dir in sorted(SESSIONS_ROOT.glob("session_*")):
        session_path = session_dir / "session.json"
        traceability_path = session_dir / "traceability.json"
        if not session_path.exists() or not traceability_path.exists():
            continue
        session = json.loads(session_path.read_text(encoding="utf-8"))
        traceability = json.loads(traceability_path.read_text(encoding="utf-8"))
        task_name = traceability.get("primary_task", {}).get("name") or session["title"]
        tags = tags_for_task(task_name)
        if not tags:
            tags = [tag for tag in all_tags() if tag in session.get("tags", [])]
        write_session_files(
            session_dir,
            session["title"],
            session.get("description", ""),
            tags,
            session.get("source_type", traceability.get("source_type", "")),
            session.get("protocol_version", traceability.get("protocol_version", "")),
            session.get("legacy_path", ""),
        )
        traceability["subject_id"] = "subject_001"
        traceability.setdefault("primary_task", {})["tags"] = tags
        for segment in traceability.get("task_segments", []):
            segment["tags"] = tags
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["task_tags"] = tags
            write_json(metadata_path, metadata)
        refreshed_files = []
        for record in traceability.get("files", []):
            path = Path(record.get("path", ""))
            absolute = path if path.is_absolute() else SESSIONS_ROOT.parent / path
            if absolute.exists():
                refreshed_files.append(file_provenance(absolute, record.get("role", "session_artifact")))
        traceability["files"] = refreshed_files
        write_json(traceability_path, traceability)
        rows.append({
            "session_id": session_dir.name,
            "title": session["title"],
            "description": session.get("description", ""),
            "tag_count": len(tags),
            "tags": "|".join(tags),
        })
    pd.DataFrame(rows).to_csv(SESSIONS_ROOT / "session_tag_catalog.csv", index=False)
    rebuild_catalog()
    print("Tagged sessions:", len(rows))
    print("Tag vocabulary size:", len(all_tags()))
    print("Catalog:", SESSIONS_ROOT / "session_tag_catalog.csv")


if __name__ == "__main__":
    main()
