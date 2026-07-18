# Brain State Sonata

This folder turns the trained spatial-temporal EEG representation into deterministic real-time piano composition and a synchronized visual performance.

It uses the 37 chromatic piano samples from `C:\Users\diego\Downloads\mp3 Notes\mp3 Notes`, covering C3 through C6. The first launch decodes the MP3 files with FFmpeg and saves a reusable cache under `brain_music/cache`. Later launches load the complete instrument in roughly a tenth of a second.

## Run the recorded demo first

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
Set-Location "D:\Brainz"
& $python .\brain_music\replay_brain_music.py
```

This replays the newest recorded brain-state trajectory with live audio. Set `RECORDING_FILTER` to part of a recording path to choose a specific session. Set `AUDIO_ENABLED = False` for a silent visualization test.

## Run with the helmet

```powershell
$python = "C:\Users\diego\anaconda3\envs\python311\python.exe"
Set-Location "D:\Brainz"
& $python .\brain_music\realtime_brain_music.py
```

The live script connects to `BA MAXI 034`, performs the same causal filtering used by representation training, collects a fresh 30-second eyes-closed relaxed baseline, fuses causal 2, 4, and 8-second windows every 250 ms, projects the state into the learned manifold, and drives both music and visualization. Keep the face, jaw, tongue, neck, shoulders, hands, and feet relaxed and breathe naturally. Music remains muted during calibration. Open your eyes when the display reports that calibration is complete.

## Musical mapping

The system contains no random note selection.

- Manifold state I chooses the tonal center, with key changes permitted only at phrase boundaries.
- Manifold state II controls melodic register and the continuous target scale degree.
- Manifold state III selects Ionian, Dorian, Lydian, or Aeolian color at phrase boundaries.
- Latent trajectory speed controls tempo, note density, and intensity.
- Latent direction controls melodic contour with a maximum two-degree movement per event.
- Trajectory curvature introduces controlled seventh-chord tension and shorter articulation.
- Stable trajectories create longer, quieter, more legato phrases and a deeper ambience.
- The independent learned hue coordinate controls stereo placement, ambience, and every synchronized visual color.
- Harmony follows mode-specific progressions; bass, chord voicing, and melody are quantized to the active scale.

The composer runs on an eighth-note grid with four-beat harmony, phrase-level key/mode changes, three- or four-note voicings, bass movement, bounded melodic motion, and velocity/duration envelopes. The audio engine provides 32-voice polyphony, equal-power stereo panning, two feedback ambience delays, and soft limiting.

## Visualization

The dashboard combines:

- The learned manifold as a faint spatial context.
- A glowing hue-driven state orb and fading trajectory.
- Three pulse rings synchronized to the musical beat.
- Current key, mode, chord, tempo, and recent notes.
- Motion and curvature meters.
- A chromatic harmony wheel showing the active scale and chord tones.
- A three-octave piano keyboard showing active notes and chord voicing.

The display automatically supports either a saved 2D or 3D brain-state map.

## Outputs

Every run creates a timestamped directory under `brain_music/sessions`.

- `music_events.csv` records every note, role, velocity, duration, pan, key, mode, chord, tempo, and hue.
- `brain_states.csv` is also saved for live sessions with the complete 160-dimensional representation, multi-window weights and disagreement, and channel-reliability summaries at every update.
- `metadata.json` records the model, manifold, sample directory, timing, and event counts.

The music and visual state remain experimental representations. They are not measurements of emotion, intention, disease, or neurological health.
