import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traceability_common import CATALOG_CSV, PROJECT_ROOT, SCHEMA_VERSION, rebuild_catalog, read_json, slugify, utc_now, write_json


OUTPUT_ROOT = Path(__file__).resolve().parent / "labeled_dataset"
RAW_OUTPUT_ROOT = OUTPUT_ROOT / "raw_sessions"
DERIVED_OUTPUT_ROOT = OUTPUT_ROOT / "derived_datasets"
CHUNK_ROWS = 50000
OVERWRITE = True
INCLUDE_DERIVED_BASELINES = True
INCLUDE_STANDALONE_RECORDING = True
CANONICAL_LABEL_VERSION = "brainz_canonical_tasks_v2"
TAXONOMY = {
    "scrolling_reels": {
        "task_domain": "passive_social_media",
        "engagement_type": "passive_interactive",
        "stimulus_modality": "visual_auditory",
        "movement_expectation": "minimal_hand_movement",
    },
    "watching_tv_show": {
        "task_domain": "passive_video",
        "engagement_type": "passive",
        "stimulus_modality": "visual_auditory",
        "movement_expectation": "still",
    },
    "listening_to_classical_music": {
        "task_domain": "passive_audio",
        "engagement_type": "passive",
        "stimulus_modality": "auditory",
        "movement_expectation": "still",
    },
    "solving_rubiks_cubes": {
        "task_domain": "visuospatial_problem_solving",
        "engagement_type": "active_cognitive_motor",
        "stimulus_modality": "visual_proprioceptive",
        "movement_expectation": "hands",
    },
    "solving_hard_rubiks_cubes": {
        "task_domain": "visuospatial_problem_solving",
        "engagement_type": "active_cognitive_motor_high_difficulty",
        "stimulus_modality": "visual_proprioceptive",
        "movement_expectation": "hands",
    },
    "reading_an_article": {
        "task_domain": "reading",
        "engagement_type": "active_cognitive",
        "stimulus_modality": "visual",
        "movement_expectation": "minimal",
    },
    "physical_pain": {
        "task_domain": "somatosensory_stimulus",
        "engagement_type": "passive_nociceptive",
        "stimulus_modality": "somatosensory",
        "movement_expectation": "still",
    },
    "complete_resting": {
        "task_domain": "rest",
        "engagement_type": "passive_rest",
        "stimulus_modality": "none_eyes_closed",
        "movement_expectation": "still",
    },
    "walking": {
        "task_domain": "locomotion",
        "engagement_type": "active_motor",
        "stimulus_modality": "proprioceptive",
        "movement_expectation": "whole_body",
    },
    "reading_aloud": {
        "task_domain": "reading_and_speech",
        "engagement_type": "active_cognitive_motor",
        "stimulus_modality": "visual_auditory",
        "movement_expectation": "speech",
    },
    "chess_puzzles": {
        "task_domain": "visuospatial_problem_solving",
        "engagement_type": "active_cognitive_high_difficulty",
        "stimulus_modality": "visual",
        "movement_expectation": "minimal_hand_movement",
    },
    "playing_guitar": {
        "task_domain": "music_performance",
        "engagement_type": "active_cognitive_motor",
        "stimulus_modality": "auditory_proprioceptive",
        "movement_expectation": "hands_and_arms",
    },
    "basic_motor_cognition": {
        "task_domain": "motor_imagery",
        "engagement_type": "active_cognitive",
        "stimulus_modality": "visual_instruction",
        "movement_expectation": "imagined_only",
    },
    "image_emotion_cognition": {
        "task_domain": "passive_affective_visual",
        "engagement_type": "passive",
        "stimulus_modality": "visual",
        "movement_expectation": "still",
    },
    "eyes_open_closed_and_blinks": {
        "task_domain": "eeg_quality",
        "engagement_type": "controlled_state",
        "stimulus_modality": "visual_state",
        "movement_expectation": "blink_only_when_requested",
    },
    "jaw_clench": {
        "task_domain": "eeg_quality",
        "engagement_type": "controlled_artifact",
        "stimulus_modality": "somatomotor",
        "movement_expectation": "jaw_only_when_requested",
    },
    "ssvep_flicker": {
        "task_domain": "ssvep",
        "engagement_type": "passive_visual_attention",
        "stimulus_modality": "visual_flicker",
        "movement_expectation": "still",
    },
    "resting_baseline": {
        "task_domain": "rest",
        "engagement_type": "passive_rest",
        "stimulus_modality": "none_or_fixation",
        "movement_expectation": "still",
    },
    "unknown_legacy_task": {
        "task_domain": "unknown",
        "engagement_type": "unknown",
        "stimulus_modality": "unknown",
        "movement_expectation": "unknown",
    },
}


def canonical_task_name(value):
    text = slugify(value)
    aliases = {
        "scrolling_reels": "scrolling_reels",
        "watching_tv_show": "watching_tv_show",
        "listening_to_classical_music": "listening_to_classical_music",
        "solving_rubiks_cubes": "solving_rubiks_cubes",
        "solving_hard_rubicks_cubes": "solving_hard_rubiks_cubes",
        "solving_hard_rubiks_cubes": "solving_hard_rubiks_cubes",
        "reading_an_article": "reading_an_article",
        "physical_pain": "physical_pain",
        "complete_resting": "complete_resting",
        "walking": "walking",
        "reading_aloud": "reading_aloud",
        "chess_puzzles": "chess_puzzles",
        "playing_guitar": "playing_guitar",
        "basic_motor_cognition": "basic_motor_cognition",
        "motor_imagery_erd_v3": "basic_motor_cognition",
        "image_emotion_cognition": "image_emotion_cognition",
        "image_emotion_passive_viewing_v1": "image_emotion_cognition",
        "eyes_open_closed_and_blinks": "eyes_open_closed_and_blinks",
        "jaw_clench": "jaw_clench",
        "ssvep_flicker": "ssvep_flicker",
        "standalone_brainaccess_recording": "unknown_legacy_task",
    }
    if text == "standalone_brainaccess_recording":
        return "unknown_legacy_task"
    return aliases.get(text, text)


def inferred_taxonomy(tags):
    values = set(str(tag) for tag in tags)
    modalities = [name for name in ["visual", "auditory", "somatosensory", "proprioceptive"] if name in values]
    if "multisensory" in values or len(modalities) > 1:
        modality = "multisensory"
    elif modalities:
        modality = modalities[0]
    else:
        modality = "unknown"
    if "active" in values:
        engagement = "active"
    elif "passive" in values:
        engagement = "passive"
    elif "rest" in values:
        engagement = "rest"
    else:
        engagement = "unknown"
    movements = [name for name in ["gross_motor", "fine_motor", "facial_motor", "speech", "oculomotor", "motor"] if name in values]
    movement = "|".join(movements) if movements else "still_or_unknown"
    cognitive = [name for name in ["imagery", "motor_imagery", "attention", "working_memory", "problem_solving", "executive_control", "visuospatial", "language", "reading", "emotion", "creativity", "decision_making"] if name in values]
    return {
        "task_domain": "|".join(cognitive) if cognitive else "general_brain_state",
        "engagement_type": engagement,
        "stimulus_modality": modality,
        "movement_expectation": movement,
    }


def label_quality(traceability):
    status = str(traceability.get("traceability_status", ""))
    if status == "user_annotated":
        return "user_annotated", "high"
    if status.startswith("inferred_from_existing"):
        return "protocol_inferred", "medium"
    return "unknown", "unknown"


def source_eeg_path(traceability):
    recording_path = PROJECT_ROOT / traceability["recording_path"]
    if recording_path.is_file():
        return recording_path
    return recording_path / "eeg_samples.csv"


def time_values(frame, row_offset, sample_rate):
    if "sample_time_est_from_chunk_s" in frame.columns:
        return frame["sample_time_est_from_chunk_s"].to_numpy(dtype=float), "sample_time_est_from_chunk_s"
    if "sample_time_est_s" in frame.columns:
        return frame["sample_time_est_s"].to_numpy(dtype=float), "sample_time_est_s"
    if "t_from_stream_start_s" in frame.columns:
        return frame["t_from_stream_start_s"].to_numpy(dtype=float), "t_from_stream_start_s"
    return (np.arange(len(frame), dtype=float) + row_offset) / float(sample_rate), "row_index_estimate"


def valid_sample_mask(frame):
    if "valid_eeg_sample" in frame.columns:
        values = frame["valid_eeg_sample"]
        if values.dtype == object:
            return values.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
        return values.to_numpy(dtype=bool)
    if "streaming" in frame.columns:
        return frame["streaming"].to_numpy(dtype=float) > 0.5
    return np.ones(len(frame), dtype=bool)


def segment_arrays(traceability):
    segments = sorted(
        [segment for segment in traceability.get("task_segments", []) if segment.get("start_pc_time_perf_counter_s") is not None],
        key=lambda segment: float(segment["start_pc_time_perf_counter_s"]),
    )
    starts = np.asarray([float(segment["start_pc_time_perf_counter_s"]) for segment in segments], dtype=float)
    ends = np.asarray([
        float(segment["end_pc_time_perf_counter_s"]) if segment.get("end_pc_time_perf_counter_s") is not None else np.inf
        for segment in segments
    ], dtype=float)
    return segments, starts, ends


def canonical_columns(frame, traceability, row_offset):
    primary = traceability.get("primary_task", {})
    primary_name = canonical_task_name(primary.get("name", "unknown_legacy_task"))
    primary_tags = primary.get("tags", [])
    taxonomy = TAXONOMY.get(primary_name, inferred_taxonomy(primary_tags))
    provenance, confidence = label_quality(traceability)
    sample_rate = float(traceability.get("device", {}).get("sample_frequency_hz") or 250.0)
    times, timing_source = time_values(frame, row_offset, sample_rate)
    valid_samples = valid_sample_mask(frame)
    segments, starts, ends = segment_arrays(traceability)
    task_labels = np.full(len(frame), "unlabeled_transition", dtype=object)
    task_descriptions = np.full(len(frame), "", dtype=object)
    task_tags = np.full(len(frame), "", dtype=object)
    segment_indices = np.full(len(frame), -1, dtype=np.int32)
    if len(segments) > 0 and timing_source != "row_index_estimate":
        candidates = np.searchsorted(starts, times, side="right") - 1
        valid = candidates >= 0
        valid_indices = np.where(valid)[0]
        valid[valid_indices] = times[valid_indices] <= ends[candidates[valid_indices]]
        for segment_index, segment in enumerate(segments):
            mask = valid & (candidates == segment_index)
            task_labels[mask] = slugify(segment["task_name"])
            task_descriptions[mask] = str(segment.get("description", ""))
            task_tags[mask] = "|".join(str(value) for value in segment.get("tags", []))
            segment_indices[mask] = int(segment.get("segment_index", segment_index))
    elif len(segments) > 0:
        task_labels[:] = slugify(segments[0]["task_name"])
        task_descriptions[:] = str(segments[0].get("description", ""))
        task_tags[:] = "|".join(str(value) for value in segments[0].get("tags", []))
        segment_indices[:] = int(segments[0].get("segment_index", 0))
    elif primary_name != "unknown_legacy_task":
        task_labels[:] = primary_name
        task_descriptions[:] = str(primary.get("description", ""))
        task_tags[:] = "|".join(str(value) for value in primary.get("tags", []))
    task_labels[~valid_samples] = "invalid_stream_sample"
    task_descriptions[~valid_samples] = "The device streaming validity flag marks this row as invalid."
    task_tags[~valid_samples] = "invalid|stream"
    segment_indices[~valid_samples] = -1
    if "sample_index" not in frame.columns:
        frame.insert(0, "sample_index", np.arange(row_offset, row_offset + len(frame), dtype=np.int64))
    output = pd.DataFrame({
        "sample_index": frame["sample_index"].to_numpy(),
        "canonical_task_family": primary_name,
        "canonical_task_label": task_labels,
        "task_description": task_descriptions,
        "task_tags": task_tags,
        "task_domain": taxonomy["task_domain"],
        "engagement_type": taxonomy["engagement_type"],
        "stimulus_modality": taxonomy["stimulus_modality"],
        "movement_expectation": taxonomy["movement_expectation"],
        "segment_index": segment_indices,
        "is_valid_eeg_sample": valid_samples,
        "is_task_labeled": (task_labels != "unlabeled_transition") & (task_labels != "invalid_stream_sample"),
        "timing_alignment_source": timing_source,
        "label_provenance": provenance,
        "label_confidence": confidence,
        "traceability_id": traceability.get("traceability_id", ""),
        "session_id": traceability.get("session_id", ""),
        "participant_alias": traceability.get("participant_alias", ""),
        "protocol_version": traceability.get("protocol_version", ""),
        "source_recording_path": traceability.get("recording_path", ""),
        "canonical_label_version": CANONICAL_LABEL_VERSION,
    })
    return output


def write_raw_labeled_dataset(traceability):
    source_path = source_eeg_path(traceability)
    session_id = traceability["session_id"]
    output_dir = RAW_OUTPUT_ROOT / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "eeg_samples_labeled.parquet"
    labels_path = output_dir / "sample_labels.csv"
    manifest_path = output_dir / "dataset_manifest.json"
    if parquet_path.exists() and not OVERWRITE:
        return read_json(manifest_path)
    parquet_writer = None
    labels_header = True
    row_offset = 0
    label_counts = {}
    total_rows = 0
    for chunk in pd.read_csv(source_path, chunksize=CHUNK_ROWS):
        labels = canonical_columns(chunk, traceability, row_offset)
        labeled = pd.concat([chunk.reset_index(drop=True), labels.drop(columns=["sample_index"]).reset_index(drop=True)], axis=1)
        table = pa.Table.from_pandas(labeled, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
        parquet_writer.write_table(table)
        labels.to_csv(labels_path, mode="w" if labels_header else "a", header=labels_header, index=False)
        labels_header = False
        for label, count in labels["canonical_task_label"].value_counts().items():
            label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)
        row_offset += len(chunk)
        total_rows += len(chunk)
    if parquet_writer is not None:
        parquet_writer.close()
    device = traceability.get("device", {})
    primary = canonical_task_name(traceability.get("primary_task", {}).get("name", "unknown_legacy_task"))
    provenance, confidence = label_quality(traceability)
    invalid_count = int(label_counts.get("invalid_stream_sample", 0))
    transition_count = int(label_counts.get("unlabeled_transition", 0))
    valid_count = int(total_rows - invalid_count)
    labeled_count = int(total_rows - invalid_count - transition_count)
    valid_fraction = valid_count / max(1, total_rows)
    labeled_fraction = labeled_count / max(1, total_rows)
    duration_s = total_rows / max(1.0, float(device.get("sample_frequency_hz") or 250.0))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "canonical_label_version": CANONICAL_LABEL_VERSION,
        "generated_at_utc": utc_now(),
        "session_id": session_id,
        "traceability_id": traceability.get("traceability_id"),
        "source_type": traceability.get("source_type"),
        "source_eeg_path": str(source_path.resolve().relative_to(PROJECT_ROOT)),
        "labeled_parquet_path": str(parquet_path.resolve().relative_to(PROJECT_ROOT)),
        "sample_labels_path": str(labels_path.resolve().relative_to(PROJECT_ROOT)),
        "primary_task_family": primary,
        "label_provenance": provenance,
        "label_confidence": confidence,
        "row_count": total_rows,
        "valid_sample_count": valid_count,
        "valid_sample_fraction": valid_fraction,
        "labeled_sample_count": labeled_count,
        "labeled_sample_fraction": labeled_fraction,
        "label_counts": label_counts,
        "sample_frequency_hz": device.get("sample_frequency_hz") or 250,
        "channel_count": device.get("channel_count") or 32,
        "participant_alias": traceability.get("participant_alias", ""),
        "group_id": traceability.get("traceability_id"),
        "is_raw_authoritative_recording": True,
        "is_duplicate_or_derived": False,
        "requires_validity_filter": invalid_count > 0,
        "recommended_for_autoencoder": valid_fraction >= 0.90 and duration_s >= 30.0,
        "recommended_for_downstream": primary != "unknown_legacy_task" and labeled_fraction >= 0.20,
    }
    write_json(manifest_path, manifest)
    return manifest


def manifest_entries(manifest):
    if "exports" in manifest:
        return manifest["exports"]
    if "classes" in manifest:
        return [
            {
                "dataset": row.get("class_name"),
                "output_file": row.get("output_file"),
                "rows": row.get("rows"),
                "duration_s": row.get("duration_s"),
                "source_recording": row.get("source_file", ""),
            }
            for row in manifest["classes"]
        ]
    return []


def write_derived_labeled_datasets():
    outputs = []
    for manifest_path in sorted((PROJECT_ROOT / "recordings").glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        entries = manifest_entries(manifest)
        if len(entries) == 0:
            continue
        output_dir = DERIVED_OUTPUT_ROOT / manifest_path.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            filename = entry.get("output_file")
            if filename is None:
                continue
            source_path = manifest_path.parent / filename
            if not source_path.exists():
                continue
            task_label = slugify(entry.get("dataset", source_path.stem))
            parquet_path = output_dir / f"{source_path.stem}_labeled.parquet"
            frame = pd.read_csv(source_path)
            frame["canonical_task_family"] = "resting_baseline"
            frame["canonical_task_label"] = task_label
            frame["task_domain"] = TAXONOMY["resting_baseline"]["task_domain"]
            frame["engagement_type"] = TAXONOMY["resting_baseline"]["engagement_type"]
            frame["stimulus_modality"] = TAXONOMY["resting_baseline"]["stimulus_modality"]
            frame["movement_expectation"] = TAXONOMY["resting_baseline"]["movement_expectation"]
            frame["label_provenance"] = "derived_manifest"
            frame["label_confidence"] = "medium"
            frame["source_recording_path"] = str(entry.get("source_recording", ""))
            frame["canonical_label_version"] = CANONICAL_LABEL_VERSION
            frame.to_parquet(parquet_path, compression="zstd", index=False)
            outputs.append({
                "session_id": manifest_path.parent.name,
                "traceability_id": "",
                "source_type": "derived_baseline_dataset",
                "source_eeg_path": str(source_path.resolve().relative_to(PROJECT_ROOT)),
                "labeled_parquet_path": str(parquet_path.resolve().relative_to(PROJECT_ROOT)),
                "sample_labels_path": "",
                "primary_task_family": "resting_baseline",
                "label_provenance": "derived_manifest",
                "label_confidence": "medium",
                "row_count": int(len(frame)),
                "valid_sample_count": int(len(frame)),
                "valid_sample_fraction": 1.0,
                "labeled_sample_count": int(len(frame)),
                "labeled_sample_fraction": 1.0,
                "label_counts": {task_label: int(len(frame))},
                "sample_frequency_hz": entry.get("sample_frequency_hz", 250),
                "channel_count": len([column for column in frame.columns if column.endswith("_uV")]),
                "participant_alias": "",
                "group_id": f"derived:{manifest_path.parent.name}:{source_path.stem}",
                "is_raw_authoritative_recording": False,
                "is_duplicate_or_derived": True,
                "requires_validity_filter": False,
                "recommended_for_autoencoder": False,
                "recommended_for_downstream": False,
            })
    return outputs


def artifact_type(path):
    name = path.name.lower()
    if name == "sample_labels.csv":
        return "canonical_sample_labels"
    if name == "eeg_samples_labeled.parquet":
        return "canonical_labeled_eeg"
    if name == "eeg_samples.csv":
        return "raw_eeg"
    if "eeg_samples" in name and "labeled" in name:
        return "legacy_labeled_eeg"
    if name == "events.csv":
        return "task_events"
    if "annotation" in name:
        return "sdk_annotations"
    if "prediction" in name:
        return "predictions"
    if "training_windows" in name:
        return "training_windows"
    if "embedding" in name or "trajectory" in name:
        return "learned_representation"
    if path.suffix.lower() == ".npz":
        return "binary_array"
    if name == "metadata.json":
        return "metadata"
    if name == "manifest.json":
        return "manifest"
    return "recorded_or_derived_artifact"


def containing_traceability(path, traceabilities):
    resolved = path.resolve()
    candidates = []
    for traceability_path, traceability in traceabilities:
        recording_path = PROJECT_ROOT / traceability.get("recording_path", "")
        recording_path = recording_path.resolve()
        if recording_path.is_file() and resolved == recording_path:
            candidates.append((len(str(recording_path)), traceability))
        elif recording_path.is_file() and resolved.parent == recording_path.parent and resolved.stem == recording_path.stem:
            candidates.append((len(str(recording_path)), traceability))
        elif recording_path.is_dir() and (recording_path in resolved.parents or recording_path == resolved.parent):
            candidates.append((len(str(recording_path)), traceability))
    if len(candidates) == 0:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def build_artifact_manifest(dataset_manifests):
    traceability_paths = sorted(set(PROJECT_ROOT.rglob("traceability.json")).union(PROJECT_ROOT.rglob("*.traceability.json")))
    traceabilities = [(path, read_json(path)) for path in traceability_paths]
    lineage = {}
    for manifest in dataset_manifests:
        for key in ["source_eeg_path", "labeled_parquet_path", "sample_labels_path"]:
            value = str(manifest.get(key, "")).strip()
            if len(value) > 0:
                lineage[(PROJECT_ROOT / value).resolve()] = manifest
    rows = []
    excluded_parts = {"__pycache__", "cache"}
    for path in sorted(PROJECT_ROOT.rglob("*"), key=str):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".npz", ".parquet"}:
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        traceability = containing_traceability(path, traceabilities)
        dataset_manifest = lineage.get(path.resolve())
        if dataset_manifest is not None:
            primary_name = dataset_manifest["primary_task_family"]
            session_id = dataset_manifest["session_id"]
            traceability_id = dataset_manifest["traceability_id"]
            source_type = dataset_manifest["source_type"]
            traceability_status = dataset_manifest["label_provenance"]
            context_basis = "canonical_dataset_manifest"
        elif traceability is not None:
            primary_name = canonical_task_name(traceability.get("primary_task", {}).get("name", ""))
            session_id = traceability.get("session_id", "")
            traceability_id = traceability.get("traceability_id", "")
            source_type = traceability.get("source_type", "")
            traceability_status = traceability.get("traceability_status", "")
            context_basis = "session_traceability"
        else:
            primary_name = ""
            session_id = ""
            traceability_id = ""
            source_type = ""
            traceability_status = ""
            context_basis = "path_inference"
            path_text = str(path.relative_to(PROJECT_ROOT)).lower()
            context_map = [
                ("basic_motor_cognition", "basic_motor_cognition"),
                ("image_emotion_cognition", "image_emotion_cognition"),
                ("eyes_inference", "eyes_open_closed_and_blinks"),
                ("eeg_quality_suite", "eeg_quality"),
                ("brain_state_representation", "mixed_recordings"),
                ("brain_music", "brain_music_composition"),
                ("concentration_baseline", "concentration"),
                ("box_motion_cognition", "box_motion_imagery"),
                ("eeg_channel_health_report", "eeg_quality"),
                ("recording_catalog", "mixed_recordings"),
            ]
            for pattern, context in context_map:
                if pattern in path_text:
                    primary_name = context
                    break
            if OUTPUT_ROOT.resolve() in path.resolve().parents and len(primary_name) == 0:
                primary_name = "mixed_recordings"
                source_type = "unified_canonical_dataset"
                context_basis = "dataset_level_artifact"
        rows.append({
            "artifact_path": str(path.resolve().relative_to(PROJECT_ROOT)),
            "artifact_type": artifact_type(path),
            "size_bytes": int(path.stat().st_size),
            "session_id": session_id,
            "traceability_id": traceability_id,
            "canonical_task_family": primary_name,
            "traceability_status": traceability_status,
            "source_type": source_type,
            "has_direct_session_traceability": traceability is not None or dataset_manifest is not None,
            "context_basis": context_basis,
        })
    output_path = OUTPUT_ROOT / "recorded_artifact_manifest.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path, rows


def update_source_traceability(manifest):
    source_path = PROJECT_ROOT / manifest["source_eeg_path"]
    traceability_path = source_path.parent / "traceability.json" if source_path.name == "eeg_samples.csv" else None
    if traceability_path is None or not traceability_path.exists():
        return
    traceability = read_json(traceability_path)
    traceability["canonical_labels"] = {
        "version": CANONICAL_LABEL_VERSION,
        "generated_at_utc": utc_now(),
        "primary_task_family": manifest["primary_task_family"],
        "label_provenance": manifest["label_provenance"],
        "label_confidence": manifest["label_confidence"],
        "labeled_parquet_path": manifest["labeled_parquet_path"],
        "sample_labels_path": manifest["sample_labels_path"],
        "label_counts": manifest["label_counts"],
    }
    write_json(traceability_path, traceability)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(CATALOG_CSV)
    raw_rows = catalog[catalog["source_type"].isin(["live_universal_recording", "existing_standard_recording", "standalone_recording"])]
    manifests = []
    for index, row in enumerate(raw_rows.itertuples(index=False), start=1):
        sidecar_path = PROJECT_ROOT / row.sidecar_path
        traceability = read_json(sidecar_path)
        if traceability.get("source_type") == "standalone_recording" and not INCLUDE_STANDALONE_RECORDING:
            continue
        print(f"Labeling raw recording {index}/{len(raw_rows)}: {traceability['session_id']}")
        manifest = write_raw_labeled_dataset(traceability)
        manifests.append(manifest)
        update_source_traceability(manifest)
    if INCLUDE_DERIVED_BASELINES:
        manifests.extend(write_derived_labeled_datasets())
    manifest_frame = pd.DataFrame(manifests)
    manifest_frame["label_counts_json"] = manifest_frame["label_counts"].apply(json.dumps)
    manifest_frame.drop(columns=["label_counts"]).to_csv(OUTPUT_ROOT / "autoencoder_manifest.csv", index=False)
    count_rows = []
    for manifest in manifests:
        for label, count in manifest["label_counts"].items():
            count_rows.append({
                "session_id": manifest["session_id"],
                "group_id": manifest["group_id"],
                "primary_task_family": manifest["primary_task_family"],
                "canonical_task_label": label,
                "sample_count": count,
                "sample_frequency_hz": manifest["sample_frequency_hz"],
                "duration_s": count / max(1.0, float(manifest["sample_frequency_hz"])),
                "label_provenance": manifest["label_provenance"],
                "label_confidence": manifest["label_confidence"],
            })
    pd.DataFrame(count_rows).to_csv(OUTPUT_ROOT / "sample_label_counts.csv", index=False)
    artifact_path, artifact_rows = build_artifact_manifest(manifests)
    write_json(OUTPUT_ROOT / "dataset_index.json", {
        "schema_version": SCHEMA_VERSION,
        "canonical_label_version": CANONICAL_LABEL_VERSION,
        "generated_at_utc": utc_now(),
        "recording_count": len(manifests),
        "artifact_count": len(artifact_rows),
        "autoencoder_manifest": "autoencoder_manifest.csv",
        "sample_label_counts": "sample_label_counts.csv",
        "recorded_artifact_manifest": str(artifact_path.name),
        "taxonomy": TAXONOMY,
    })
    rebuild_catalog()
    print("Labeled recordings:", len(manifests))
    print("Autoencoder manifest:", OUTPUT_ROOT / "autoencoder_manifest.csv")
    print("Sample label counts:", OUTPUT_ROOT / "sample_label_counts.csv")
    print("Artifact manifest:", artifact_path)


if __name__ == "__main__":
    main()
