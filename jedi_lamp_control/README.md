# Jedi Lamp Control

This experiment learns a deliberate two-state EEG trigger and toggles the Arduino relay only after sustained agreement.

The classes are:

- `nothing`: deliberately broad negative behavior including free thought, arithmetic, inner or overt speech, ordinary small movements, visual exploration, memory, and mixed activity.
- `the_will`: one repeatable ritual using the same hand pose, center gaze, lamp-switch imagery, and silent `NOW` command every time.

The physical ritual is intentionally fixed. Changing hands, pose, gaze, or mental command between recording and demonstration reduces reliability. Predictions are experimental model outputs, not measurements of intent or neurological state.

## 1. Record sessions

```powershell
& "C:\Users\diego\anaconda3\envs\python311\python.exe" .\jedi_lamp_control\record_jedi_will.py
```

Each session starts with a visible 30-second static eyes-closed calibration, then records eight blocks per class. Both labeled classes show the same neutral dot during the scored interval to reduce visual cue leakage. Instructions, calibration, rests, and the first 1.5 seconds of each task block are excluded from training. Partial sessions save after every block.

Record multiple complete sessions on different days. With at least two sessions, validation holds out entire sessions. One-session block validation is provisional and usually optimistic.

## 2. Train and select the model

```powershell
& "C:\Users\diego\anaconda3\envs\python311\python.exe" .\jedi_lamp_control\train_jedi_will.py
```

The trainer automatically discovers every `jedi_lamp_will_v1` session under `sessions`, extracts spatial, temporal, per-electrode, and spectral features, and compares:

- selected-feature L2 logistic regression;
- selected-feature elastic-net logistic regression;
- linear and RBF SVMs;
- Extra Trees and Random Forests;
- compact and broad soft-voting ensembles.

All overlapping windows remain grouped by session, or by block when only one session exists. Selection uses macro F1 and balanced accuracy with an explicit penalty for false `the_will` activations. The saved activation threshold is chosen from out-of-fold predictions with a target false-will rate no higher than 10% when possible.

## 3. Run the lamp demonstration

Upload `mind_control_hardware/rele_simple/rele_simple.ino` to the Arduino, then run:

```powershell
& "C:\Users\diego\anaconda3\envs\python311\python.exe" .\jedi_lamp_control\infer_jedi_lamp.py
```

Leave `SERIAL_PORT = ""` for automatic Arduino/USB-serial discovery or set a fixed port such as `COM3`. The script sends `Y` for relay on and `N` for relay off at 115200 baud.

Live inference performs a 20-second session-specific `nothing` calibration and raises the activation threshold above the observed false-trigger distribution. A toggle requires six of the latest eight predictions to agree, a smoothed probability above threshold, and an expired refractory period. After toggling, another event is impossible until eight of ten predictions agree on release and the signal returns below the release threshold. Every accepted will event toggles the current lamp state.

Press Escape to stop. The relay is commanded off before the serial connection closes.

## Electrical safety

The Arduino code drives an active-low relay on pin 7. Do not switch exposed mains wiring on a breadboard. Use an enclosed, correctly rated relay module, strain relief, insulation, fuse/protection appropriate to the load, and qualified help for mains-voltage wiring. Test first with a low-voltage lamp or LED supply. EEG equipment, the computer, and any body-contact equipment must remain electrically isolated from unsafe wiring.
