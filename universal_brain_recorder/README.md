# Universal Brain Recorder

This folder provides a task-annotated, indefinite-duration EEG recorder and a project-wide provenance catalog.

## Record a session

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
Set-Location "D:\Brainz"
& $python .\universal_brain_recorder\record_universal_task.py
```

The Tkinter recorder asks for:

- A session title such as `scrolling reels` or `listening to classical music`.
- A natural-language description.
- Tags selected from sensory, action, cognitive, state, protocol, and expected-system groups.
- Explicit consent confirmation.

Raw EEG is streamed incrementally to disk rather than retained until the end. This allows long recordings without memory usage growing with session duration.

The live panel can mark instantaneous events, begin and end labeled intervals, stop the stream, and show the event timeline and elapsed time. Event boundaries use `state_start` and `state_end` so the existing Brainz labeling utilities can assign labels to EEG samples later.

## Session contents

Every authoritative recording is saved directly under `sessions` with:

- `eeg_samples.csv`: untouched 32-channel raw streamed EEG and timing.
- `events.csv`: task boundaries, notes, and markers.
- `brainaccess_annotations.csv`: BrainAccess SDK annotations.
- `metadata.json`: device, operator annotations, protocol, and timing.
- `recording_summary.json`: duration, samples, chunks, and valid fraction.
- `traceability.json`: subject ID, consent status, task segments, provenance, and SHA-256 hashes.
- `session.json`: canonical title, description, controlled tags, event summary, subject ID, and legacy path when migrated.
- `SESSION.md`: natural-language session report.

All recordings use the same `subject_001` identity. Expected-system tags describe the protocol and are not claims of directly localized brain activation. No command-line arguments are required.

## Existing recordings

Run:

```powershell
& $python .\universal_brain_recorder\backfill_existing_traceability.py
```

This repairs canonical traceability for standard sessions when needed. Legacy task identity is explicitly marked as inferred rather than user-authored.

The backfill also:

- Links derived baseline datasets to their source manifests.
- Computes SHA-256 hashes for authoritative source files.
- Rebuilds `recording_catalog.csv` and `recording_catalog.json`.

The catalog is automatically rebuilt after every new universal recording.

The one-time `migrate_sessions.py` utility consolidates legacy raw sessions into `sessions`, records their previous paths, verifies every moved file by SHA-256, and writes `migration_manifest.json`.

## Canonical labels for model training

Run:

```powershell
& $python .\universal_brain_recorder\label_all_recorded_data.py
```

This reads every raw EEG session, its traceability sidecar, and its event timing without modifying the source files. Outputs are written under `universal_brain_recorder/labeled_dataset`.

For each authoritative raw session it creates:

- A compressed `eeg_samples_labeled.parquet` containing the original data and canonical labels.
- A lightweight `sample_labels.csv` that can be joined to the raw CSV by `sample_index`.
- A `dataset_manifest.json` containing group identity, validity, label counts, and training recommendations.

Canonical fields separate:

- Broad task family.
- Exact within-session state.
- Task domain, engagement type, stimulus modality, and expected movement.
- User-authored, protocol-inferred, derived, or unknown provenance.
- Label confidence and session grouping.
- Valid EEG samples, unlabeled transitions, and invalid stream rows.

Project-wide outputs include:

- `autoencoder_manifest.csv`: authoritative training sources and session-level group IDs.
- `sample_label_counts.csv`: class balance and duration.
- `recorded_artifact_manifest.csv`: task/provenance context for recorded and derived CSV, NPZ, and Parquet artifacts.
- `dataset_index.json`: canonical taxonomy and dataset version.

Derived baseline exports are labeled but marked as duplicate/derived and are not recommended as default autoencoder inputs. Sessions with poor stream validity are also flagged rather than silently included.

EEG recordings remain sensitive biometric and health-adjacent data. Use aliases rather than real names and do not upload sessions without explicit permission.
