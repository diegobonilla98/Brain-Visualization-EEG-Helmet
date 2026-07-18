# AGENTS.md

This file is for future AI agents taking control of `E:\Brainz`.

The project records and analyzes EEG from a BrainAccess MAXI 32-channel helmet, currently `BA MAXI 034`, at 250 Hz. It includes low-level recording utilities, quality tests, motor imagery experiments, passive image-emotion experiments, and some visualization assets.

This is an experimental research/code project, not a medical device, diagnostic tool, or clinical decision system. Treat all recordings and predictions as experimental signals unless a qualified clinician and validated protocol are involved.

## Required Runtime

Use the conda Python environment unless the user explicitly says otherwise:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
```

Do not add command-line argument parsing unless requested. This project is intentionally constant-driven: change top-level variables in scripts so the files can be run directly from an IDE.

Keep imports at the top of files. Do not wrap imports in try/except. Do not add decorative comment headers. Do not add comments unless the user asks or the code would otherwise be hard to maintain.

## Project Map

- `BrainAccessBoard/BrainAccessBoard_v2.6.1.exe`: BrainAccess desktop board software kept with the project.
- `eeg_quality_suite/`: hardware streaming, basic recording protocols, stream diagnostics, and simple predictors.
- `basic_motor_cognition/`: motor imagery recording, training, and live inference.
- `concentration_baseline/`: simple concentrated vs not-concentrated recording, training, and live inference.
- `box_motion_cognition/`: simple up/down/left/right box-motion imagery recording, training, and live inference.
- `image_emotion_cognition/`: EmoSet passive image viewing, feature extraction, and emotion model training.
- `sessions/`: all authoritative EEG sessions, regardless of experiment type.
- `recordings/`: legacy derived baselines and project-level quality reports only.
- `assets/brain.stl`: 3D brain model asset.
- `brain_visualize/viz3d.py`: 3D visualization.
- `info/`: reference images for headset/electrode placement.
- `first_crap/`: early exploratory scripts. Do not build new work here unless asked.

The folder is not currently a git repository. Be careful with existing data files and do not delete or rewrite recordings unless the user explicitly requests it.

## Hardware And SDK

The BrainAccess SDK entry point is `eeg_quality_suite/brainaccess_stream.py`.

Main class:

```python
BrainAccessStream(device_name="BA MAXI 034", gain_name="X8", use_bias=True, bias_channel_name="Iz")
```

The stream wrapper does the following:

- Calls `brainaccess.core.init()`.
- Scans devices with `core.scan()`.
- Connects through `EEGManager().connect(device_name)`.
- Reads device info, battery, feature count, and sample frequency.
- Assumes the 32-channel MAXI 10-20 labels:
  `Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO3, POz, PO4, O1, Oz, O2, Iz`.
- Enables `SAMPLE_NUMBER`, all requested EEG channels, and `STREAMING`.
- Sets gain via `GainMode.X8`.
- Sets bias with `set_channel_bias()` on `Iz` by default.
- Calls `load_config()`.
- Registers a chunk callback with `set_callback_chunk()`.
- Starts streaming with `start_stream()`.
- Converts callback chunks to a pandas DataFrame with estimated sample times.

Important output columns:

- `sample_index`
- `device_sample`
- `pc_time_received_s`
- `sample_time_est_s`
- `sample_time_est_from_chunk_s`
- `chunk_timing_error_s`
- `t_from_stream_start_s`
- one column per EEG channel, e.g. `C3_uV`
- `streaming`, used as the validity flag

Annotations are sent with `recorder.annotate(text)` and retrieved with `recorder.get_annotations()`. Most experiment scripts also write their own `events.csv` using `time.perf_counter()` timestamps immediately after display flips. The event CSV is the primary source for labeling samples.

## Data Layout

Every recording/session directory generally contains:

- `eeg_samples.csv`: raw streamed EEG sample table.
- `events.csv`: experiment state markers.
- `brainaccess_annotations.csv`: SDK annotations when available.
- `metadata.json`: device, protocol, timing, and session settings.
- `eeg_samples_labeled.csv`: derived table after assigning labels from events.
- `summary.png`: quick visual recording summary.
- model-specific metrics, predictions, or training-window CSVs when relevant.

Derived labeled files can be regenerated from `eeg_samples.csv` and `events.csv`. Prefer regenerating derived files over manually editing them.

## Common EEG Processing

Shared preprocessing utilities live in:

- `eeg_quality_suite/analysis_common.py`
- `basic_motor_cognition/motor_cognition_common.py`
- `concentration_baseline/concentration_common.py`
- `box_motion_cognition/box_motion_common.py`
- `image_emotion_cognition/image_emotion_common.py`

Common steps:

- Infer sampling rate from `metadata.json`, falling back to sample timestamps, then 250 Hz.
- Use `streaming` or `valid_eeg_sample` to mark valid samples.
- Set invalid EEG samples to `NaN`.
- Interpolate missing samples before filtering.
- Apply 50 Hz notch when sample rate allows.
- Apply bandpass filtering.
- Apply common average reference.
- Extract Welch bandpower features.
- Build overlapping windows and assign majority labels.

Be careful with leakage. Existing accuracies are mostly window-level metrics. Overlapping windows from the same trial/image are highly correlated, so validation must group related windows together. The image-emotion trainer already groups by session or image ID; preserve that behavior.

## Quality Suite

Run scripts from `E:\Brainz`:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\eeg_quality_suite\01_record_ssvep_flicker.py
& $python .\eeg_quality_suite\02_record_eyes_open_closed.py
& $python .\eeg_quality_suite\03_record_jaw_clench.py
& $python .\eeg_quality_suite\04_record_stream_stability.py
& $python .\eeg_quality_suite\05_record_psychopy_stream_stability.py
& $python .\eeg_quality_suite\10_predict_ssvep.py
& $python .\eeg_quality_suite\11_predict_eyes_open_closed_and_blinks.py
& $python .\eeg_quality_suite\12_predict_jaw_clench.py
& $python .\eeg_quality_suite\13_diagnose_stream_quality.py
```

Current known quality sessions are in `sessions/`:

- 250 Hz sample rate.
- 32 enabled EEG channels.
- 100% valid sample fraction in the stored quality recordings.
- Eyes open/closed/blink ML accuracy around 96%.
- Jaw clench ML accuracy around 96%.
- SSVEP ML accuracy around 90%.

Treat these as sanity checks, not final proof of generalization.

## Motor Cognition

Main files:

- `basic_motor_cognition/record_motor_cognition.py`
- `basic_motor_cognition/train_motor_cognition.py`
- `basic_motor_cognition/infer_motor_cognition.py`
- `basic_motor_cognition/motor_cognition_common.py`

Protocol version:

```python
PROTOCOL_VERSION = "motor_imagery_erd_v3"
```

Classes:

- `rest`
- `left_hand`
- `right_hand`
- `both_feet`
- `both_hands`

Recording constants:

- `DEVICE_NAME = "BA MAXI 034"`
- `GAIN_NAME = "X8"`
- `SAMPLES_PER_CLASS = 4`
- `BASELINE_SECONDS = 5.0`
- `CUE_SECONDS = 2.0`
- `IMAGERY_SECONDS = 10.0`
- `COOLDOWN_SECONDS = 3.0`

Training constants:

- `WINDOW_SECONDS = 3.0`
- `STEP_SECONDS = 0.25`
- `MIN_VALID_FRACTION = 0.95`
- `MIN_LABEL_FRACTION = 0.90`
- `TRANSITION_SKIP_SECONDS = 1.50`
- `BASELINE_SKIP_SECONDS = 1.00`
- `MIN_TRAINING_WINDOWS = 100`

Training logic:

- Load matching protocol sessions from `sessions/`.
- Keep only sessions matching `motor_imagery_erd_v3`.
- Label samples from state-start events.
- Use only imagery windows.
- Skip transition onset.
- Compute each trial's baseline features.
- Compute ERD-style baseline-subtracted motor bandpower features.
- Train a two-stage model:
  - active intent vs rest
  - active class among left hand, right hand, both feet, both hands
- Save model bundles to `basic_motor_cognition/models`.

Run:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\basic_motor_cognition\record_motor_cognition.py
& $python .\basic_motor_cognition\train_motor_cognition.py
& $python .\basic_motor_cognition\infer_motor_cognition.py
```

Live inference:

- Uses the latest model from `basic_motor_cognition/models` unless `MODEL_PATH` is set.
- Records a 20-second baseline.
- Uses rolling 3-second windows.
- Subtracts current-session baseline before predicting.

## Concentration Baseline

Main files:

- `concentration_baseline/record_concentration_baseline.py`
- `concentration_baseline/train_concentration_baseline.py`
- `concentration_baseline/infer_concentration_baseline.py`
- `concentration_baseline/concentration_common.py`

Protocol version:

```python
PROTOCOL_VERSION = "concentration_baseline_v1"
```

Classes:

- `concentrated`
- `not_concentrated`

Recording logic:

- Saves sessions under `sessions/`.
- Uses balanced blocks for concentrated and non-concentrated states.
- The instruction screen tells the user what mental state to enter.
- Labeled task windows use the same central-dot visual for both classes to reduce obvious visual confounds.
- Concentration command: gaze at the dot and silently count breaths from 1 to 10, restarting after 10 or after attention drifts.
- Non-concentration command: gaze at the same dot, do not count or solve, and let thoughts drift.
- Saves after every block so partial data survives Escape.

Training logic:

- Loads sessions matching `concentration_baseline_v1`.
- Uses only `state_phase == "task"` windows.
- Extracts EEG-only regional bandpower, contrast, and ratio-like features.
- Validates by session when multiple sessions exist, otherwise by block.
- Saves model bundles to `concentration_baseline/models`.

Run:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\concentration_baseline\record_concentration_baseline.py
& $python .\concentration_baseline\train_concentration_baseline.py
& $python .\concentration_baseline\infer_concentration_baseline.py
```

## Box Motion Cognition

Main files:

- `box_motion_cognition/record_box_motion.py`
- `box_motion_cognition/train_box_motion.py`
- `box_motion_cognition/infer_box_motion.py`
- `box_motion_cognition/box_motion_common.py`

Protocol version:

```python
PROTOCOL_VERSION = "box_motion_imagery_v1"
```

Classes:

- `up`
- `down`
- `left`
- `right`

Recording logic:

- Saves sessions under `sessions/`.
- Each trial has baseline, cue, imagery, and cooldown phases.
- The user imagines moving a screen box in the cued direction without actual movement.
- Training uses only imagery windows.
- Each imagery window is baseline-normalized against the baseline phase from the same trial.
- Saves after every trial so partial data survives Escape.

Training logic:

- Loads sessions matching `box_motion_imagery_v1`.
- Skips the first part of each imagery phase to avoid cue transition contamination.
- Extracts motor-area bandpower, asymmetry, and baseline-delta features.
- Validates by session when multiple sessions exist, otherwise by trial.
- Saves model bundles to `box_motion_cognition/models`.

Run:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\box_motion_cognition\record_box_motion.py
& $python .\box_motion_cognition\train_box_motion.py
& $python .\box_motion_cognition\infer_box_motion.py
```

## Image Emotion Cognition

Main files:

- `image_emotion_cognition/record_image_emotion_cognition.py`
- `image_emotion_cognition/train_image_emotion_cognition.py`
- `image_emotion_cognition/image_emotion_common.py`

Dataset:

```python
DEFAULT_EMOSET_ROOT = Path("F:/EmoSet-118K")
```

Emotion labels:

- `amusement`
- `awe`
- `contentment`
- `excitement`
- `anger`
- `disgust`
- `fear`
- `sadness`

Recording protocol:

- Passive affective image viewing.
- Jittered fixation baseline before each image.
- Fixed image exposure.
- Washout period after the image for blink/recovery.
- Balanced emotion sampling.
- No emotion text is shown on the image.
- Display flip timestamps are logged with `time.perf_counter()`.

Recording constants:

- `DEVICE_NAME = "BA MAXI 034"`
- `GAIN_NAME = "X8"`
- `PROTOCOL_VERSION = "image_emotion_passive_viewing_v1"`
- `IMAGES_PER_SESSION = 30` in code, though an existing session metadata shows 50 planned images.
- `FIXATION_SECONDS_MIN = 1.2`
- `FIXATION_SECONDS_MAX = 1.8`
- `IMAGE_SECONDS = 4.5`
- `WASHOUT_SECONDS = 2.0`
- `BREAK_EVERY_IMAGES = 10`
- `BREAK_SECONDS = 8.0`

Training constants:

- `WINDOW_SECONDS = 2.5`
- `STEP_SECONDS = 0.5`
- `IMAGE_ONSET_SKIP_SECONDS = 0.50`
- `MIN_VALID_FRACTION = 0.98`
- `MIN_LABEL_FRACTION = 0.95`
- `MIN_TRAINING_WINDOWS = 80`
- `MIN_CLASS_WINDOWS = 12`
- `TRAIN_FEATURE_ENSEMBLE = True`
- `TRAIN_DEEP_MODEL = True`
- `MIN_DEEP_WINDOWS = 120`

Training logic:

- Load matching protocol sessions from `sessions/`.
- Use image-phase EEG only.
- Use fixation phase as per-trial baseline where available.
- Build EEG-only engineered features:
  - regional bandpower
  - baseline deltas
  - per-channel bandpower
  - left-right asymmetries
  - frontal/posterior contrasts
- Build EEG tensors for a temporal convnet.
- Validate with grouped CV by session when possible, otherwise by image ID.
- Do not use image pixels, image paths, brightness, colorfulness, facial expression annotations, or metadata as model features.
- Save models and metrics to `image_emotion_cognition/models`.

Run:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\image_emotion_cognition\record_image_emotion_cognition.py
& $python .\image_emotion_cognition\train_image_emotion_cognition.py
```

Existing session:

- `sessions/session_20260620_230440_image_emotion_cognition`
- 50 planned images in metadata.
- 29 completed images.
- Stopped by Escape.
- 102681 EEG rows.
- 32 channels at 250 Hz.

## Baseline And EEG Experiment Discipline

Use a baseline on every session when it makes sense. EEG shifts with electrode contact, skin impedance, fatigue, posture, time of day, alertness, and recent movement. Do not assume yesterday's baseline is valid today.

Recommended baselines:

- General quality: 30 to 60 seconds eyes open, relaxed, looking at a fixation point.
- Alpha check: 30 seconds eyes open and 30 seconds eyes closed.
- Motor imagery: per-trial baseline before each imagery cue, plus a session-level relaxed baseline before inference.
- Emotion viewing: fixation baseline before every image.
- Live inference: collect a fresh resting baseline immediately before prediction.

Before recording:

- Confirm the user is seated, comfortable, and unlikely to move.
- Confirm the helmet is stable and symmetric on the head.
- Check electrode placement against the 10-20 layout used by the code.
- Make sure reference/bias setup is intentional. Current code uses bias on `Iz`.
- Check battery level.
- Check that all expected channels are enabled.
- Inspect whether channels are flat, saturated, extremely noisy, or dominated by line noise.
- If gel/saline/contact quality is relevant for the helmet setup, improve contact before recording.
- Remove or distance obvious electrical noise sources when possible.
- Keep phone, charger cables, and unnecessary USB devices away from the headset and leads.
- Use the same monitor, refresh-rate setup, and viewing distance when comparing visual-stimulus sessions.

During recording:

- Ask the user to keep the head still.
- Keep facial muscles relaxed.
- Relax the jaw and tongue.
- Do not clench teeth unless the protocol is specifically jaw clench.
- Avoid swallowing during stimulus windows when possible.
- Avoid blinking during short stimulus windows when possible; blink during rest/washout periods.
- Keep shoulders, neck, hands, and feet relaxed unless the protocol asks for overt movement.
- For motor imagery, imagine movement kinesthetically without actually moving.
- Avoid counting, subvocalizing, talking, or changing breathing pattern during trials.
- Keep gaze stable, especially for SSVEP and visual emotion sessions.
- Abort or pause if the user becomes uncomfortable, dizzy, anxious, photosensitive, or fatigued.

After recording:

- Check `valid_eeg_sample` or `streaming` validity.
- Review `summary.png`.
- Review channel health before trusting model metrics.
- Confirm events align with sample timestamps.
- Confirm class balance before training.
- Save raw recordings unchanged.
- Regenerate derived labels or metrics rather than editing raw files.

## Safety And Ethics

Do not present flashing visual stimuli to someone with known photosensitive epilepsy or seizure risk unless a qualified professional has approved the protocol. SSVEP flicker can be provocative.

Stop immediately if there is discomfort, headache, visual disturbance, nausea, panic, dizziness, unusual sensations, or any neurological concern.

Get explicit consent before recording. EEG data can be sensitive biometric/health-adjacent data. Do not upload raw EEG, metadata, images of the participant, or identifiable session data without explicit permission.

Do not interpret emotion, intention, disease state, cognitive state, or neurological health from these models as fact. Report predictions as model outputs with uncertainty and validation caveats.

## Model Evaluation Rules

When changing training code, preserve or improve these safeguards:

- Group overlapping windows by trial, image, or session during validation.
- Keep all windows from the same trial/image out of both train and validation simultaneously.
- Report balanced accuracy, macro F1, class counts, and confusion matrix.
- Track validation group level in metrics.
- Keep leakage audits for image-emotion models.
- Do not add image metadata as features unless the user explicitly asks for a non-EEG control experiment.
- Be skeptical of very high accuracy from a single session or correlated windows.

For more reliable claims, prefer:

- Multiple sessions on different days.
- Held-out sessions.
- More trials per class.
- Same-session baseline recalibration.
- Clear artifact rejection.
- Reporting per-class metrics, not only aggregate accuracy.

## Development Rules For Future Agents

- Read relevant files before editing.
- Prefer `rg` and `rg --files` for search.
- Keep changes scoped.
- Use the existing constant-driven style.
- Use `apply_patch` or direct IDE-compatible edits; do not rewrite unrelated files.
- Do not delete recordings, models, or manifests unless explicitly requested.
- When adding a new experiment, create a new protocol version string.
- When changing event semantics, update the corresponding label assignment function.
- When changing channels, update feature extraction and metadata expectations together.
- When changing timing, log both planned timing and actual display flip timing.
- When touching training code, run at least the relevant training script if data is available.
- When touching recording code, do not start a real hardware session unless the user is physically ready and explicitly asked for it.

## Quick Commands

Quality diagnostics:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\eeg_quality_suite\13_diagnose_stream_quality.py
```

Train motor model:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\basic_motor_cognition\train_motor_cognition.py
```

Train concentration model:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\concentration_baseline\train_concentration_baseline.py
```

Train box-motion model:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\box_motion_cognition\train_box_motion.py
```

Train image-emotion model:

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "E:\Brainz"
Set-Location $project
& $python .\image_emotion_cognition\train_image_emotion_cognition.py
```

Check project files:

```powershell
$project = "E:\Brainz"
rg --files $project
```
