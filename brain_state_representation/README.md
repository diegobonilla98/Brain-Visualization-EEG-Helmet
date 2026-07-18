# Brain-State Representation Models

This folder contains the preserved Visualization Model and the deployed Universal Model, hybrid self-supervised and label-assisted spatiotemporal EEG representation pipelines for the 32-channel BrainAccess MAXI at 250 Hz. Both produce a 160-dimensional public brain-state representation while separating recording-session and artifact information into private training-only factors.

The Universal Model adds multi-label general neurofunctional tag prototypes and tag-set relational geometry. Its selected training configuration strengthens multi-horizon future prediction, stable-segment acceleration, and multi-scale agreement. The current universal model and map are selected through `latest_universal_model_path.txt` and `latest_map_path.txt`; the original Visualization Model remains intact.

The factorized model retains the causal 2, 4, and 8-second channel-aware time-frequency encoder and adds:

- A public `z_state` representation used by visualization, music, and downstream tasks.
- Private `z_session` and `z_artifact` factors used only during training.
- Gradient-reversal session and artifact adversaries that discourage those nuisance signals in `z_state`.
- Cross-session task and state contrastive positives, with every training batch containing paired sessions for each repeated task.
- Masked-input invariance and future latent prediction at 0.5, 2, and 8 seconds.
- Orthogonality between the public state and private nuisance factors.
- Confidence-weighted canonical task/chunk supervision.
- Leakage-embargoed final-time validation blocks from every session.

The maintained trainer initializes only the backbone from the saved Visualization Model and deliberately reinitializes the public and private projections. It uses within-task session adversaries, aligns same-state distributions across sessions, subtracts learned session and artifact residuals before producing `z_state`, and guarantees useful cross-session and clean/artifact pairs in every training batch. The single terminal blink segment is training-only because it cannot be divided into independent train and validation segments.

Checkpoint selection excludes adversary cross-entropy because minimizing that value would favor a successful nuisance classifier rather than an invariant encoder. Selection starts at epoch 6 after the adversaries have learned useful gradients. Final metrics include conditional-session and artifact probes and a provisional invariance gate; the gate is a development safeguard, not a claim of unseen-session generalization.

The current dataset has only two sessions each for image-emotion and scrolling-reels. The trainer therefore uses both sessions for cross-session learning and validates on non-overlapping final-time blocks separated from training by the complete encoder window and prediction horizon. These metrics are held-out-time metrics, not held-out-session claims.

The v2 encoder uses the known 10-20 electrode geometry as a graph and processes causal 2, 4, and 8-second windows together. Every electrode has separate learnable calibration, depthwise temporal filters, an identity embedding, and a dynamic reliability estimate. A direct per-channel seven-band frequency branch is fused with temporal features before graph processing.

Training combines:

- Conservative VICReg anti-collapse regularization using weak EEG-plausible transformations.
- Contiguous time masking, small sensor noise, mild gain changes, and limited channel dropout.
- Direct per-channel delta, theta, alpha, beta, and low-gamma frequency inputs.
- Per-channel seven-band spectral reconstruction.
- Previous/next latent prediction.
- Transition-aware acceleration regularization that is disabled across task boundaries.
- Cross-window agreement between 2, 4, and 8-second representations.
- Auxiliary task-family and coarse intra-task classification.
- Supervised contrastive structure for task labels that have repeated independent sessions.
- Per-session channelwise robust calibration.

Task classification is restricted to task families with multiple sessions. Unique one-session tasks still control transition-aware smoothing but are not allowed to create a misleading task/session shortcut.

The visualization pipeline uses robust scaling, whitened PCA, UMAP as a nonlinear manifold teacher, and a smooth parametric neural projection for deterministic real-time coordinates. Fast state, slow context, and latent velocity are mapped together. A separate learned two-component projection supplies the circular hue coordinate.

## Scripts

- `train_brain_state_model.py` launches the Visualization Model trainer.
- `visualization_model.py` contains the maintained model architecture and objectives.
- `train_visualization_model.py` contains the maintained factorized cross-session training and evaluation workflow.
- `universal_model.py` adds tag prototypes and relational tag geometry to the factorized encoder.
- `train_universal_model.py` runs architecture selection or the full Universal Model training according to its top-level `RUN_MODE` constant.
- `UNIVERSAL_MODEL_REPORT.md` records the architecture comparison, final metrics, limitations, and deployed artifacts.
- `build_brain_state_map.py` encodes every selected recording, fits the 2D or 3D manifold, exports all 160-dimensional embeddings to CSV, saves the reducer, and displays a trajectory map.
- `replay_brain_state.py` replays one saved recording as the moving ball and trajectory tail without needing the helmet.
- `realtime_brain_state.py` connects to `BA MAXI 034`, collects a fresh 30-second eyes-closed relaxed baseline, and displays the fused multi-window trajectory at four updates per second beside the seven highest-scoring smoothed multi-label tag predictions.

## Run

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
$project = "D:\Brainz"
Set-Location $project
& $python .\brain_state_representation\train_brain_state_model.py
& $python .\brain_state_representation\train_universal_model.py
& $python .\brain_state_representation\build_brain_state_map.py
& $python .\brain_state_representation\replay_brain_state.py
& $python .\brain_state_representation\realtime_brain_state.py
```

Training uses the constants at the top of `train_visualization_model.py` or `train_universal_model.py`; it does not require command-line arguments. The default batch size is sized for the 6 GB RTX 4050 Laptop GPU. After training finishes, build a fresh map before launching the visualization or music projection.

Set `OUTPUT_DIMENSIONS = 2` in `build_brain_state_map.py` and rebuild the map for a native 2D projection. The live and replay scripts automatically use the saved map dimensionality.

Saved artifacts are written under `models`. The current embeddings CSV contains `recording`, `time_s`, `hue`, `state_1..3`, and `embedding_001..160`. Keep every window from one trial, image, or session in only one validation fold.

Live sessions are written under `live_sessions` with the plotted coordinates, full 160-dimensional state, all 37 smoothed tag scores, ranked top tags, multi-window weights and disagreement, and learned channel-reliability summaries.

The existing data appear to come from a small number of sessions and likely one participant. A genuinely universal encoder requires substantially more days, participants, tasks, rest states, artifact conditions, and held-out downstream evaluations. The coordinates and colors are learned model outputs, not direct measurements of emotion, intention, cognition, disease, or neurological health.
