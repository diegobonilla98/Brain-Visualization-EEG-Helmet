import json
import hashlib
import math
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
import torch
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "brain_state_representation"))

from brain_state_common import CHANNEL_NAMES, StateDynamics, latest_pointer, load_joblib, torch_device
from brain_state_model import build_encoder_from_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = MODULE_ROOT / "sessions"
CACHE_ROOT = MODULE_ROOT / "cache"
DEFAULT_SAMPLE_ROOT = Path(r"C:\Users\diego\Downloads\mp3 Notes\mp3 Notes")
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "brain_state_representation" / "models"
NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
MODES = {
    "IONIAN": [0, 2, 4, 5, 7, 9, 11],
    "DORIAN": [0, 2, 3, 5, 7, 9, 10],
    "LYDIAN": [0, 2, 4, 6, 7, 9, 11],
    "AEOLIAN": [0, 2, 3, 5, 7, 8, 10],
}
PROGRESSIONS = {
    "IONIAN": [0, 4, 5, 3, 0, 3, 4, 0],
    "DORIAN": [0, 3, 6, 0, 5, 3, 0, 0],
    "LYDIAN": [0, 1, 4, 0, 5, 1, 4, 0],
    "AEOLIAN": [0, 5, 2, 6, 0, 3, 6, 0],
}


@dataclass
class PianoVoice:
    midi: int
    sample: np.ndarray
    position: int
    duration_frames: int
    gain_left: float
    gain_right: float
    attack_frames: int
    release_frames: int


@dataclass
class MusicalEvent:
    timestamp: float
    elapsed_s: float
    step: int
    role: str
    midi: int
    note: str
    velocity: float
    duration_s: float
    pan: float
    key: str
    mode: str
    chord: str
    bpm: float
    hue: float


def midi_name(midi):
    octave = int(midi) // 12 - 1
    return f"{NOTE_NAMES[int(midi) % 12]}{octave}"


def sample_name_to_midi(path):
    match = re.fullmatch(r"([a-g])(-?)([3-6])", path.stem.lower())
    if match is None:
        raise ValueError(f"Cannot parse piano sample name: {path.name}")
    letter, sharp, octave_text = match.groups()
    semitone = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[letter]
    if sharp == "-":
        semitone += 1
    return (int(octave_text) + 1) * 12 + semitone


class PianoSampleLibrary:
    def __init__(self, root=DEFAULT_SAMPLE_ROOT, sample_rate=44100):
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        self.samples = {}
        if not self.root.exists():
            raise FileNotFoundError(f"Piano sample directory does not exist: {self.root}")
        paths = sorted(self.root.glob("*.mp3"))
        signature_text = "|".join(f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in paths)
        signature = hashlib.sha1(signature_text.encode("utf-8")).hexdigest()
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_ROOT / f"piano_samples_{self.sample_rate}.npz"
        cache_valid = False
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_valid = str(cached["signature"].item()) == signature
                if cache_valid:
                    midis = cached["midis"].astype(int)
                    lengths = cached["lengths"].astype(int)
                    audio = cached["audio"]
                    self.samples = {int(midi): audio[index, :lengths[index]].copy() for index, midi in enumerate(midis)}
        for path in paths if len(self.samples) == 0 else []:
            midi = sample_name_to_midi(path)
            segment = AudioSegment.from_file(path).set_frame_rate(self.sample_rate).set_channels(2).set_sample_width(2)
            array = np.asarray(segment.get_array_of_samples(), dtype=np.float32).reshape(-1, 2) / 32768.0
            array -= np.mean(array, axis=0, keepdims=True)
            peak = float(np.max(np.abs(array)))
            if peak > 1e-8:
                array *= 0.82 / peak
            self.samples[midi] = array.astype(np.float32)
        if not cache_valid:
            midis = np.asarray(sorted(self.samples), dtype=np.int16)
            lengths = np.asarray([len(self.samples[int(midi)]) for midi in midis], dtype=np.int32)
            audio = np.zeros((len(midis), int(np.max(lengths)), 2), dtype=np.float32)
            for index, midi in enumerate(midis):
                audio[index, :lengths[index]] = self.samples[int(midi)]
            np.savez_compressed(cache_path, signature=np.asarray(signature), midis=midis, lengths=lengths, audio=audio)
        expected = set(range(48, 85))
        missing = sorted(expected.difference(self.samples))
        if len(missing) > 0:
            raise RuntimeError(f"Missing piano samples: {[midi_name(midi) for midi in missing]}")

    def sample(self, midi):
        midi = int(np.clip(round(midi), min(self.samples), max(self.samples)))
        return self.samples[midi], midi


class PolyphonicPianoEngine:
    def __init__(self, library, block_size=512, master_gain=0.32, enabled=True):
        self.library = library
        self.sample_rate = library.sample_rate
        self.block_size = int(block_size)
        self.master_gain = float(master_gain)
        self.enabled = bool(enabled)
        self.voices = []
        self.lock = threading.Lock()
        self.stream = None
        self.wet = 0.16
        self.delay_a = np.zeros((int(round(self.sample_rate * 0.107)), 2), dtype=np.float32)
        self.delay_b = np.zeros((int(round(self.sample_rate * 0.163)), 2), dtype=np.float32)
        self.delay_a_position = 0
        self.delay_b_position = 0

    def start(self):
        if self.enabled:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=self.block_size,
                latency="low",
                callback=self._callback,
            )
            self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_ambience(self, value):
        self.wet = float(np.clip(value, 0.08, 0.30))

    def trigger(self, midi, velocity, duration_s, pan=0.0):
        sample, used_midi = self.library.sample(midi)
        duration_frames = int(np.clip(round(duration_s * self.sample_rate), int(0.10 * self.sample_rate), len(sample)))
        pan_value = float(np.clip(pan, -1.0, 1.0))
        angle = (pan_value + 1.0) * math.pi / 4.0
        gain = float(np.clip(velocity, 0.0, 1.0))
        voice = PianoVoice(
            midi=used_midi,
            sample=sample,
            position=0,
            duration_frames=duration_frames,
            gain_left=gain * math.cos(angle),
            gain_right=gain * math.sin(angle),
            attack_frames=max(1, int(round(0.008 * self.sample_rate))),
            release_frames=max(1, min(int(round(0.16 * self.sample_rate)), duration_frames // 2)),
        )
        with self.lock:
            if len(self.voices) >= 32:
                self.voices = self.voices[-31:]
            self.voices.append(voice)
        return used_midi

    def _feedback_delay(self, source, buffer, position, feedback):
        indices = (position + np.arange(len(source))) % len(buffer)
        delayed = buffer[indices].copy()
        buffer[indices] = source + delayed * float(feedback)
        return delayed, int((position + len(source)) % len(buffer))

    def _callback(self, outdata, frames, time_info, status):
        mixed = np.zeros((frames, 2), dtype=np.float32)
        remaining = []
        with self.lock:
            for voice in self.voices:
                count = min(frames, voice.duration_frames - voice.position, len(voice.sample) - voice.position)
                if count <= 0:
                    continue
                indices = np.arange(voice.position, voice.position + count)
                envelope = np.ones(count, dtype=np.float32)
                attack = indices < voice.attack_frames
                envelope[attack] = indices[attack] / max(1, voice.attack_frames)
                release_start = voice.duration_frames - voice.release_frames
                release = indices >= release_start
                envelope[release] *= np.maximum(0.0, (voice.duration_frames - indices[release]) / max(1, voice.release_frames))
                segment = voice.sample[voice.position:voice.position + count]
                mixed[:count, 0] += segment[:, 0] * envelope * voice.gain_left
                mixed[:count, 1] += segment[:, 1] * envelope * voice.gain_right
                voice.position += count
                if voice.position < voice.duration_frames and voice.position < len(voice.sample):
                    remaining.append(voice)
            self.voices = remaining
        delayed_a, self.delay_a_position = self._feedback_delay(mixed, self.delay_a, self.delay_a_position, 0.34)
        delayed_b, self.delay_b_position = self._feedback_delay(mixed, self.delay_b, self.delay_b_position, 0.27)
        output = mixed * (1.0 - self.wet * 0.45) + (delayed_a + delayed_b) * (self.wet * 0.50)
        outdata[:] = np.tanh(output * self.master_gain * 1.25)


class CoordinateFilter:
    def __init__(self, step_seconds, smoothing_seconds):
        self.alpha = 1.0 - math.exp(-float(step_seconds) / float(smoothing_seconds))
        self.value = None

    def update(self, value):
        value = np.asarray(value, dtype=float)
        if self.value is None:
            self.value = value.copy()
        else:
            self.value += self.alpha * (value - self.value)
        return self.value.copy()


class BrainStateProjector:
    def __init__(self, model_root=DEFAULT_MODEL_ROOT, model_path="", map_path="", device="auto", step_seconds=0.25):
        model_root = Path(model_root)
        self.map_path = Path(map_path) if len(str(map_path).strip()) > 0 else latest_pointer(model_root, "latest_map_path.txt", "brain_state_map_*.joblib")
        self.map_bundle = load_joblib(self.map_path)
        self.model_path = Path(model_path) if len(str(model_path).strip()) > 0 else Path(self.map_bundle["model_path"])
        self.model_bundle = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.device = torch_device(device)
        self.encoder = build_encoder_from_config(self.model_bundle["config"], self.model_bundle["adjacency"])
        self.encoder.load_state_dict(self.model_bundle["encoder_state"])
        self.encoder.to(self.device).eval()
        self.config = self.model_bundle["config"]
        self.window_samples = int(round(float(self.config["window_seconds"]) * float(self.config["sample_rate"])))
        self.dynamics = StateDynamics(step_seconds, float(self.map_bundle["fast_smooth_seconds"]), float(self.map_bundle["slow_smooth_seconds"]))
        self.coordinate_filter = CoordinateFilter(step_seconds, float(self.map_bundle["coordinate_smooth_seconds"]))
        self.last_scale_disagreement = None
        self.last_scale_weights = None
        self.last_channel_reliability = None

    def encode(self, filtered_window, center, scale):
        normalized = np.clip(
            (np.asarray(filtered_window, dtype=np.float32) - center[:, None]) / scale[:, None],
            -float(self.config["clip_value"]),
            float(self.config["clip_value"]),
        )
        with torch.inference_mode():
            tensor = torch.from_numpy(normalized[None]).to(self.device)
            if str(self.config.get("architecture", "v1")) in {"channel_aware_multiscale_time_frequency_v2", "visualization_model_v1", "universal_tag_semantic_model_v1"}:
                fused, scales, weights, _, reliability = self.encoder(tensor, return_scales=True)
                embedding = fused.cpu().numpy()[0]
                normalized_scales = torch.nn.functional.normalize(scales, dim=2)
                normalized_fused = torch.nn.functional.normalize(fused, dim=1)[:, None, :]
                self.last_scale_disagreement = float((1.0 - torch.sum(normalized_scales * normalized_fused, dim=2)).mean().cpu())
                self.last_scale_weights = weights.cpu().numpy()[0]
                self.last_channel_reliability = torch.stack(reliability, dim=1).mean(dim=1).cpu().numpy()[0, :, 0]
            else:
                embedding = self.encoder(tensor).cpu().numpy()[0]
                self.last_scale_disagreement = None
                self.last_scale_weights = None
                self.last_channel_reliability = None
        return self.project_embedding(embedding)

    def project_embedding(self, embedding):
        dynamic_feature = self.dynamics.update(embedding)
        scaled = self.map_bundle["scaler"].transform(dynamic_feature[None])
        reduced = self.map_bundle["pca"].transform(scaled)
        raw_coordinate = self.map_bundle["parametric_mapper"].predict(reduced)[0]
        coordinate = self.coordinate_filter.update(raw_coordinate)
        hue_components = self.map_bundle["hue_pca"].transform(dynamic_feature[None])[0]
        hue = float(np.mod(np.arctan2(hue_components[1], hue_components[0]) / (2.0 * np.pi) + 0.5, 1.0))
        return embedding, coordinate, hue


class BrainMusicComposer:
    def __init__(self, audio_engine, map_bounds, inference_step_seconds=0.25):
        self.audio = audio_engine
        self.bounds = np.asarray(map_bounds, dtype=float)
        self.inference_step_seconds = float(inference_step_seconds)
        self.coordinate = np.mean(self.bounds, axis=1)
        self.normalized = np.full(3, 0.5, dtype=float)
        self.previous_coordinate = None
        self.previous_velocity = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.speed = 0.0
        self.curvature = 0.0
        self.hue = 0.5
        self.embedding = None
        self.bpm = 76.0
        self.target_bpm = 76.0
        self.key_pc = 0
        self.pending_key_pc = 0
        self.mode_name = "IONIAN"
        self.pending_mode_name = "IONIAN"
        self.progression_index = -1
        self.chord_degree = 0
        self.chord_midis = [60, 64, 67]
        self.chord_name = "C"
        self.step_index = 0
        self.next_step_time = None
        self.started_at = None
        self.melody_position = 7.0
        self.events = []
        self.recent_events = deque(maxlen=64)

    def update_brain_state(self, coordinate, hue, embedding=None):
        coordinate = np.asarray(coordinate, dtype=float)
        span = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-6)
        normalized = np.clip((coordinate - self.bounds[:, 0]) / span, 0.0, 1.0)
        if self.previous_coordinate is not None:
            new_velocity = (coordinate - self.previous_coordinate) / self.inference_step_seconds / span
            self.velocity = 0.72 * self.velocity + 0.28 * new_velocity
            acceleration = (self.velocity - self.previous_velocity) / self.inference_step_seconds
            self.speed = 0.82 * self.speed + 0.18 * float(np.linalg.norm(self.velocity))
            self.curvature = 0.84 * self.curvature + 0.16 * float(np.linalg.norm(acceleration))
            self.previous_velocity = self.velocity.copy()
        self.previous_coordinate = coordinate.copy()
        self.coordinate = coordinate
        self.normalized = 0.75 * self.normalized + 0.25 * normalized
        self.hue = float(hue)
        self.embedding = embedding
        self.pending_key_pc = int(np.floor(self.normalized[0] * 11.999))
        mode_index = min(3, int(np.floor(self.normalized[2] * 4.0)))
        self.pending_mode_name = list(MODES)[mode_index]
        activity = float(np.clip(self.speed * 2.2, 0.0, 1.0))
        self.target_bpm = 62.0 + 54.0 * activity + 10.0 * self.normalized[1]
        self.audio.set_ambience(0.10 + 0.15 * (1.0 - activity) + 0.04 * abs(math.sin(self.hue * 2.0 * math.pi)))

    def _nearest_midi(self, pitch_class, preferred, low=48, high=84):
        candidates = [midi for midi in range(low, high + 1) if midi % 12 == pitch_class % 12]
        return min(candidates, key=lambda midi: abs(midi - preferred))

    def _scale_midi(self, scale_degree, preferred_octave):
        scale = MODES[self.mode_name]
        octave_shift, degree = divmod(int(scale_degree), len(scale))
        pitch_class = (self.key_pc + scale[degree]) % 12
        preferred = 12 * (preferred_octave + 1) + pitch_class + octave_shift * 12
        return self._nearest_midi(pitch_class, preferred, 48, 84)

    def _update_harmony(self):
        if self.step_index % 32 == 0:
            self.key_pc = self.pending_key_pc
            self.mode_name = self.pending_mode_name
        progression = PROGRESSIONS[self.mode_name]
        state_bias = -1 if self.normalized[0] < 0.28 else (1 if self.normalized[0] > 0.72 else 0)
        self.progression_index = (self.progression_index + 1 + state_bias) % len(progression)
        self.chord_degree = progression[self.progression_index]
        degrees = [self.chord_degree, self.chord_degree + 2, self.chord_degree + 4]
        if self.curvature > 0.55:
            degrees.append(self.chord_degree + 6)
        self.chord_midis = [self._scale_midi(degree, 4) for degree in degrees]
        root_name = NOTE_NAMES[self.chord_midis[0] % 12]
        quality_interval = (self.chord_midis[1] - self.chord_midis[0]) % 12
        quality = "m" if quality_interval == 3 else ""
        extension = "7" if len(degrees) == 4 else ""
        self.chord_name = f"{root_name}{quality}{extension}"

    def _record_and_play(self, now, role, midi, velocity, duration_s, pan):
        used_midi = self.audio.trigger(midi, velocity, duration_s, pan)
        event = MusicalEvent(
            timestamp=time.perf_counter(),
            elapsed_s=now - self.started_at,
            step=self.step_index,
            role=role,
            midi=used_midi,
            note=midi_name(used_midi),
            velocity=float(velocity),
            duration_s=float(duration_s),
            pan=float(pan),
            key=NOTE_NAMES[self.key_pc],
            mode=self.mode_name,
            chord=self.chord_name,
            bpm=float(self.bpm),
            hue=float(self.hue),
        )
        self.events.append(event)
        self.recent_events.append(event)

    def _compose_step(self, now):
        if self.step_index % 8 == 0:
            self._update_harmony()
        activity = float(np.clip(self.speed * 2.2, 0.0, 1.0))
        smoothness = 1.0 - float(np.clip(self.curvature * 0.65, 0.0, 1.0))
        velocity = 0.34 + 0.34 * activity + 0.12 * self.normalized[2]
        duration = 0.20 + 0.40 * smoothness
        hue_pan = math.sin(self.hue * 2.0 * math.pi) * 0.48
        if self.step_index % 8 == 0:
            bass = self._nearest_midi(self.chord_midis[0] % 12, 48 + self.key_pc, 48, 60)
            self._record_and_play(now, "bass", bass, velocity * 0.88, min(0.62, duration + 0.12), -0.30)
            for voice_index, midi in enumerate(self.chord_midis):
                pan = -0.42 + voice_index * (0.84 / max(1, len(self.chord_midis) - 1))
                self._record_and_play(now, "harmony", midi, velocity * 0.52, min(0.62, duration + 0.08), pan)
        elif self.step_index % 8 == 4:
            fifth = self._nearest_midi((self.chord_midis[0] + 7) % 12, 55 + self.key_pc, 48, 64)
            self._record_and_play(now, "bass", fifth, velocity * 0.58, min(0.56, duration), -0.22)
        density = 1 if activity > 0.58 else 2
        if self.step_index % density == 0:
            target_position = self.normalized[1] * 13.0
            directional_push = float(np.clip(self.velocity[1] * 2.0, -1.5, 1.5))
            self.melody_position += np.clip((target_position - self.melody_position) * 0.24 + directional_push, -2.0, 2.0)
            chord_tone = [0, 2, 4, 6][self.step_index % min(4, len(self.chord_midis))]
            degree = int(round(self.melody_position + chord_tone * 0.35))
            melody = self._scale_midi(degree, 4)
            if melody < 60:
                melody += 12
            self._record_and_play(now, "melody", melody, min(0.92, velocity), duration, hue_pan)

    def tick(self, now):
        if self.started_at is None:
            self.started_at = now
            self.next_step_time = now
        self.bpm += 0.05 * (self.target_bpm - self.bpm)
        step_seconds = 30.0 / max(40.0, self.bpm)
        scheduled = 0
        while now >= self.next_step_time and scheduled < 4:
            self._compose_step(self.next_step_time)
            self.step_index += 1
            self.next_step_time += step_seconds
            scheduled += 1

    def active_notes(self, now, horizon_s=0.70):
        if self.started_at is None:
            return []
        elapsed = now - self.started_at
        return [event.midi for event in self.recent_events if elapsed - event.elapsed_s <= horizon_s]

    def snapshot(self):
        recent_notes = "  ".join(event.note for event in list(self.recent_events)[-6:])
        return {
            "key": NOTE_NAMES[self.key_pc],
            "mode": self.mode_name,
            "chord": self.chord_name,
            "bpm": self.bpm,
            "speed": self.speed,
            "curvature": self.curvature,
            "hue": self.hue,
            "recent_notes": recent_notes,
            "scale_pitch_classes": [(self.key_pc + interval) % 12 for interval in MODES[self.mode_name]],
            "chord_pitch_classes": [midi % 12 for midi in self.chord_midis],
        }

    def save_events(self, output_dir, metadata):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [event.__dict__ for event in self.events]
        events_path = output_dir / "music_events.csv"
        metadata_path = output_dir / "metadata.json"
        pd.DataFrame(rows).to_csv(events_path, index=False)
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        return events_path, metadata_path


def make_session_dir(prefix):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = OUTPUT_ROOT / f"{prefix}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output
