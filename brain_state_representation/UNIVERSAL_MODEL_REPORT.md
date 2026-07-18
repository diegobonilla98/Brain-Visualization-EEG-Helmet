# Universal EEG brain-state encoder

## Selected model

The deployed model is `prototype_relational_spatiotemporal_cohesive`, a 160-dimensional encoder trained from 22 usable single-subject sessions and 37 general neurofunctional tags.

It combines:

- channel-specific calibration and reliability, rather than treating electrodes identically;
- explicit 10-20 electrode adjacency graph processing;
- time-domain and seven-band spectral inputs;
- 2, 4, and 8 second temporal scales with learned fusion;
- future-state prediction at 0.5, 2, and 8 seconds;
- stable-segment acceleration and cross-scale consistency losses;
- VICReg-style variance/covariance regularization and spectral reconstruction;
- multi-label tag prediction through learned tag prototypes;
- a relational objective that makes session embeddings reflect tag-set similarity;
- private session/artifact factors and conditional session adversaries.

## Architecture selection

All development candidates used the same initialization, training sessions, future-time validation split, and three completely unseen audit sessions.

| Candidate | Unseen tag micro F1 | Unseen tag macro F1 | Unseen Jaccard | Tag-geometry Spearman | Scale disagreement | Adjacent distance | Acceleration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Factorized baseline | — | 0.252 | — | 0.310 | — | 0.322 | 0.501 |
| Prototype relational | 0.615 | 0.443 | 0.454 | 0.343 | 0.251 | 0.317 | 0.491 |
| Prototype relational + stronger domain loss | — | 0.424 | — | 0.348 | — | 0.304 | 0.495 |
| Prototype relational + session-style augmentation | — | 0.424 | — | — | 0.228 | 0.303 | 0.481 |
| Selected cohesive model | 0.636 | 0.430 | 0.477 | 0.337 | 0.214 | 0.290 | 0.464 |

The cohesive model was selected because it tied the best overall selection score, improved unseen micro F1 and Jaccard, slightly reduced conditional session decoding, and materially improved temporal and multi-scale cohesion.

The unseen frozen downstream probe achieved micro F1 0.452, macro F1 0.222, and Jaccard 0.291. This is the stricter estimate for learning a new tag mapping from frozen embeddings with the current very small number of independent sessions.

## Final all-session training

Training used 4,878 windows and a 904-window embargoed future-time validation split. Validation reached its best EMA checkpoint at epoch 5 and then failed to improve for six epochs, stopping at epoch 11.

| Metric | Result |
|---|---:|
| Effective rank | 59.60 |
| Held-out-time task balanced accuracy | 0.987 |
| Held-out-time tag micro F1 | 0.751 |
| Held-out-time tag macro F1 | 0.713 |
| Held-out-time tag Jaccard | 0.664 |
| Session/tag geometry Spearman | 0.363 |
| Single-electrode dropout cosine | 0.986 |
| Multi-scale disagreement | 0.167 |
| Normalized adjacent distance, median | 0.235 |
| Normalized acceleration, median | 0.375 |
| Conditional session probe | 0.910 |

The final 3D map contains 25,366 temporally ordered states from all 22 sessions. Its median adjacent coordinate distance is 0.080 and median acceleration is 0.061. The parametric mapper has teacher RMSE 0.193 and its PCA stage retains 77.1% of dynamic-feature variance.

## Honest interpretation

The representation is meaningfully tag-aware, high-rank, robust to individual channel loss, and demonstrably smoother across time and window scales than the non-cohesive candidates.

Compared with the preserved Visualization Model, scale disagreement improved from 0.207 to 0.167, median adjacent distance from 0.296 to 0.235, and median acceleration from 0.469 to 0.375. Global session decoding decreased from 0.913 to 0.889 and conditional session decoding from 0.956 to 0.910. These are improvements, not evidence that session identity has been eliminated.

It is not yet a proven session-invariant universal brain encoder. Conditional session decoding remains 0.910 and the provisional invariance gate fails. Most tasks occur in only one session, so physiology, electrode contact, task, environment, and session identity are statistically entangled. Artifact prediction is likewise confounded by which experiments contain artifact events and must not be interpreted as clean artifact disentanglement.

Further architecture tuning on this dataset is more likely to trade task information for apparent invariance than solve the confound. The highest-value new data is repeated instances of the same tasks on different days, with a common relaxed baseline and standardized hardware setup in every session.

## Artifacts and use

- Model: `models/universal_brain_model.pt`
- Metrics: `models/universal_brain_model_metrics.json`
- Universal-model pointer: `models/latest_universal_model_path.txt`
- 3D map pointer: `models/latest_map_path.txt`
- Embeddings pointer: `models/latest_embeddings_path.txt`

The real-time visualizer and brain-music projector both recognize `universal_tag_semantic_model_v1`. They use the current map pointer, which is tied to this model. The live embedding visualizer reconstructs the selected prototype tag head, smooths its 37 simultaneous outputs over 1.5 seconds, displays the top seven predictions with scores, and stores every tag score in the trajectory CSV.

```powershell
& "C:\Users\diego\anaconda3\envs\python311\python.exe" .\brain_state_representation\realtime_brain_state.py
& "C:\Users\diego\anaconda3\envs\python311\python.exe" .\brain_music\brain_music_visualizer.py
```

Both live paths retain the 30-second relaxed, eyes-closed session calibration.
