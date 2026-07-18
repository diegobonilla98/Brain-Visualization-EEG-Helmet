import csv
import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eeg_quality_suite"))

from brainaccess_stream import BrainAccessStream
from session_common import PROJECT_ROOT, TAG_VOCABULARY, all_tags, make_session_dir, tags_for_task, write_session_files
from traceability_common import SCHEMA_VERSION, file_provenance, rebuild_catalog, slugify, utc_now, write_json


DEVICE_NAME = "BA MAXI 034"
GAIN_NAME = "X8"
USE_BIAS = True
BIAS_CHANNEL_NAME = "Iz"
WRITER_POLL_SECONDS = 0.10
PROTOCOL_VERSION = "universal_task_recording_v2"
WINDOW_TITLE = "Brainz Universal EEG Recorder"
OPENAI_MODEL = "gpt-5.6-terra"
PREVIEW_SAMPLE_COUNT = 2500


class TagRecommendation(BaseModel):
    tags: list[str]
    rationale: str


def openai_api_key():
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("OPENAI_API_TOKEN") or os.getenv("OPENAI_API_KEY")


def recommend_tags_with_openai(title, description):
    api_key = openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_TOKEN or OPENAI_API_KEY was not found in D:\\Brainz\\.env")
    vocabulary = {group: tags for group, tags in TAG_VOCABULARY.items()}
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        store=False,
        instructions=(
            "You are an EEG experimental-neuroscience tagging expert. Select only exact strings from the supplied controlled vocabulary. "
            "Tags must describe general sensory input, overt or imagined motor involvement, cognitive processes, behavioral state, protocol confounds, "
            "and broad functional systems plausibly engaged by the task. Never return the task name as a tag. Do not claim that scalp EEG precisely localizes activation. "
            "Choose a compact set of high-confidence tags and explain the selection in one sentence."
        ),
        input=f"Session title: {title}\nDescription: {description or 'No description provided.'}\nControlled vocabulary: {json.dumps(vocabulary)}",
        text_format=TagRecommendation,
    )
    recommendation = response.output_parsed
    if recommendation is None:
        raise RuntimeError("The OpenAI response did not contain a tag recommendation.")
    allowed = set(all_tags())
    selected = [tag for tag in all_tags() if tag in recommendation.tags and tag in allowed]
    if not selected:
        raise RuntimeError("The OpenAI response did not select any valid controlled tags.")
    return selected, recommendation.rationale.strip()


def create_recording_preview(eeg_path, preview_path):
    preview = pd.read_csv(eeg_path, nrows=PREVIEW_SAMPLE_COUNT, low_memory=False)
    if preview.empty or "sample_index" not in preview.columns:
        raise RuntimeError("The raw EEG file was created but failed its post-save integrity check.")
    preview.to_csv(preview_path, index=False)
    return {
        "raw_csv_size_bytes": int(Path(eeg_path).stat().st_size),
        "preview_rows": int(len(preview)),
        "column_count": int(len(preview.columns)),
        "first_sample_index": int(preview["sample_index"].iloc[0]),
        "raw_csv_verified_readable": True,
    }


class IncrementalEEGWriter:
    def __init__(self, recorder, output_path):
        self.recorder = recorder
        self.output_path = Path(output_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.sample_counter = 0
        self.chunk_counter = 0
        self.valid_sample_count = 0
        self.first_sample_time_est_s = None
        self.last_sample_time_est_s = None
        self.headers = [
            "sample_index",
            "sample_offset_in_chunk",
            "pc_time_received_s",
            "sample_time_est_s",
            "sample_time_est_from_chunk_s",
            "chunk_timing_error_s",
            "t_from_stream_start_s",
        ] + list(recorder.channel_labels)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def drain_chunks(self):
        with self.recorder.lock:
            chunks = list(self.recorder.chunks)
            self.recorder.chunks.clear()
        return chunks

    def has_chunks(self):
        with self.recorder.lock:
            return len(self.recorder.chunks) > 0

    def write_chunks(self, writer, chunks):
        sample_rate = float(self.recorder.sample_frequency)
        streaming_index = self.recorder.channel_indices.get("streaming")
        for receive_perf_s, array, chunk_size in chunks:
            count = min(int(chunk_size), array.shape[1])
            if count <= 0:
                continue
            self.chunk_counter += 1
            if self.first_sample_time_est_s is None:
                self.first_sample_time_est_s = receive_perf_s - ((count - 1) / sample_rate)
            rows = []
            for sample_offset in range(count):
                sample_time_from_chunk_s = receive_perf_s - ((count - 1 - sample_offset) / sample_rate)
                sample_time_est_s = self.first_sample_time_est_s + self.sample_counter / sample_rate
                row = [
                    self.sample_counter,
                    sample_offset,
                    receive_perf_s,
                    sample_time_est_s,
                    sample_time_from_chunk_s,
                    sample_time_from_chunk_s - sample_time_est_s,
                    sample_time_est_s - self.recorder.stream_start_perf_s,
                ]
                for label in self.recorder.channel_labels:
                    channel_index = self.recorder.channel_indices.get(label)
                    row.append(array[channel_index, sample_offset] if channel_index is not None and channel_index < array.shape[0] else "")
                rows.append(row)
                if streaming_index is None or float(array[streaming_index, sample_offset]) > 0.5:
                    self.valid_sample_count += 1
                self.last_sample_time_est_s = sample_time_est_s
                self.sample_counter += 1
            writer.writerows(rows)

    def run(self):
        with open(self.output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)
            while not self.stop_event.is_set() or self.has_chunks():
                chunks = self.drain_chunks()
                if chunks:
                    self.write_chunks(writer, chunks)
                    file.flush()
                if not chunks:
                    time.sleep(WRITER_POLL_SECONDS)

    def summary(self):
        duration = self.sample_counter / max(1.0, float(self.recorder.sample_frequency))
        return {
            "sample_count": int(self.sample_counter),
            "chunk_count": int(self.chunk_counter),
            "duration_s": float(duration),
            "valid_sample_count": int(self.valid_sample_count),
            "valid_sample_fraction": float(self.valid_sample_count / max(1, self.sample_counter)),
            "first_sample_time_est_s": self.first_sample_time_est_s,
            "last_sample_time_est_s": self.last_sample_time_est_s,
        }


def event_row(event_index, event_type, event_subtype, state_label, text, stream_start):
    now = time.perf_counter()
    return {
        "event_index": int(event_index),
        "event_type": event_type,
        "event_subtype": event_subtype,
        "state_label": state_label,
        "task_name": state_label,
        "text": text,
        "pc_time_perf_counter_s": now,
        "t_from_stream_start_s": now - stream_start,
        "utc_time": datetime.now(timezone.utc).isoformat(),
    }


def close_segment(segment, event):
    segment["end_pc_time_perf_counter_s"] = event["pc_time_perf_counter_s"]
    segment["end_t_from_stream_start_s"] = event["t_from_stream_start_s"]
    segment["end_utc_time"] = event["utc_time"]
    segment["duration_s"] = segment["end_t_from_stream_start_s"] - segment["start_t_from_stream_start_s"]


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1120x790")
        self.root.minsize(960, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.ui_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.writer = None
        self.recorder = None
        self.output_dir = None
        self.recording = False
        self.recording_ready = False
        self.started_perf = None
        self.selected_tags = {}
        self.tag_menu_buttons = {}
        self.configure_style()
        self.build_ui()
        self.root.after(100, self.poll_worker)

    def configure_style(self):
        self.root.configure(bg="#0b1020")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#0b1020")
        style.configure("Card.TFrame", background="#151c30")
        style.configure("TLabel", background="#0b1020", foreground="#e8eefc", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0b1020", foreground="#ffffff", font=("Segoe UI Semibold", 24))
        style.configure("Subtitle.TLabel", background="#0b1020", foreground="#91a4c8", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#151c30", foreground="#dce6fb", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#151c30", foreground="#7dd3fc", font=("Segoe UI Semibold", 12))
        style.configure("TEntry", fieldbackground="#0f172a", foreground="#ffffff", insertcolor="#ffffff", bordercolor="#334155")
        style.configure("TCheckbutton", background="#151c30", foreground="#dce6fb", font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", "#151c30")], foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background="#2563eb", foreground="#ffffff", padding=(18, 10), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#3b82f6"), ("disabled", "#334155")])
        style.configure("Stop.TButton", background="#dc2626", foreground="#ffffff", padding=(18, 10), font=("Segoe UI Semibold", 10))
        style.map("Stop.TButton", background=[("active", "#ef4444"), ("disabled", "#334155")])
        style.configure("Treeview", background="#0f172a", foreground="#e8eefc", fieldbackground="#0f172a", rowheight=27)
        style.configure("Treeview.Heading", background="#1e293b", foreground="#ffffff")

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Universal EEG Recorder", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="One session format · structured tags · live event timeline · subject_001", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 16))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        self.setup_card = ttk.Frame(body, style="Card.TFrame", padding=18)
        self.setup_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.live_card = ttk.Frame(body, style="Card.TFrame", padding=18)
        self.live_card.pack(side="right", fill="both", expand=True, padx=(10, 0))
        self.build_setup_card()
        self.build_live_card()
        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(16, 0))
        self.status_var = tk.StringVar(value="Ready to configure a session")
        self.elapsed_var = tk.StringVar(value="00:00")
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")
        ttk.Label(footer, textvariable=self.elapsed_var, style="Subtitle.TLabel").pack(side="right")

    def build_setup_card(self):
        ttk.Label(self.setup_card, text="Session details", style="Section.TLabel").pack(anchor="w")
        ttk.Label(self.setup_card, text="Title", style="Card.TLabel").pack(anchor="w", pady=(12, 4))
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(self.setup_card, textvariable=self.title_var)
        self.title_entry.pack(fill="x")
        ttk.Label(self.setup_card, text="Natural-language description", style="Card.TLabel").pack(anchor="w", pady=(12, 4))
        self.description_text = tk.Text(self.setup_card, height=4, bg="#0f172a", fg="#ffffff", insertbackground="#ffffff", relief="flat", wrap="word", padx=8, pady=8, font=("Segoe UI", 10))
        self.description_text.pack(fill="x")
        tag_header = ttk.Frame(self.setup_card, style="Card.TFrame")
        tag_header.pack(fill="x", pady=(14, 6))
        ttk.Label(tag_header, text="Functional and sensory tags", style="Section.TLabel").pack(side="left")
        self.ai_tag_button = ttk.Button(tag_header, text="AI neurology expert", command=self.request_ai_tags)
        self.ai_tag_button.pack(side="right")
        ttk.Button(tag_header, text="Local suggest", command=self.suggest_tags).pack(side="right", padx=(0, 6))
        tags_frame = ttk.Frame(self.setup_card, style="Card.TFrame")
        tags_frame.pack(fill="x", pady=(2, 8))
        for group_index, (group, tags) in enumerate(TAG_VOCABULARY.items()):
            menu_button = ttk.Menubutton(tags_frame, text=f"{group} (0)")
            menu = tk.Menu(menu_button, tearoff=False, bg="#0f172a", fg="#e8eefc", activebackground="#2563eb", activeforeground="#ffffff")
            menu_button.configure(menu=menu)
            menu_button.grid(row=group_index // 2, column=group_index % 2, sticky="ew", padx=(0, 8) if group_index % 2 == 0 else (8, 0), pady=5)
            self.tag_menu_buttons[group] = menu_button
            for tag in tags:
                variable = tk.BooleanVar(value=False)
                self.selected_tags[tag] = variable
                variable.trace_add("write", lambda *_args, category=group: self.update_tag_menu_label(category))
                menu.add_checkbutton(label=tag.replace("_", " "), variable=variable, onvalue=True, offvalue=False)
        tags_frame.columnconfigure(0, weight=1)
        tags_frame.columnconfigure(1, weight=1)
        self.selected_tags_var = tk.StringVar(value="No tags selected")
        ttk.Label(self.setup_card, textvariable=self.selected_tags_var, style="Card.TLabel", wraplength=470).pack(anchor="w", pady=(2, 6))
        self.ai_rationale_var = tk.StringVar(value="")
        ttk.Label(self.setup_card, textvariable=self.ai_rationale_var, style="Card.TLabel", wraplength=470).pack(anchor="w", pady=(0, 4))
        consent_frame = ttk.Frame(self.setup_card, style="Card.TFrame")
        consent_frame.pack(fill="x", pady=(14, 8))
        self.consent_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(consent_frame, text="Explicit consent to record this EEG session", variable=self.consent_var).pack(anchor="w")
        self.start_button = ttk.Button(self.setup_card, text="Start recording", style="Accent.TButton", command=self.start_recording)
        self.start_button.pack(fill="x", pady=(6, 0))

    def build_live_card(self):
        ttk.Label(self.live_card, text="Live event timeline", style="Section.TLabel").pack(anchor="w")
        self.live_indicator = ttk.Label(self.live_card, text="● NOT RECORDING", style="Card.TLabel")
        self.live_indicator.pack(anchor="w", pady=(8, 12))
        ttk.Label(self.live_card, text="Event label", style="Card.TLabel").pack(anchor="w")
        self.event_var = tk.StringVar()
        self.event_entry = ttk.Entry(self.live_card, textvariable=self.event_var, state="disabled")
        self.event_entry.pack(fill="x", pady=(4, 8))
        button_row = ttk.Frame(self.live_card, style="Card.TFrame")
        button_row.pack(fill="x")
        self.mark_button = ttk.Button(button_row, text="Mark instant", command=self.mark_event, state="disabled")
        self.mark_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.begin_button = ttk.Button(button_row, text="Begin interval", command=self.begin_event, state="disabled")
        self.begin_button.pack(side="left", fill="x", expand=True, padx=4)
        self.end_button = ttk.Button(button_row, text="End interval", command=self.end_event, state="disabled")
        self.end_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.active_event_var = tk.StringVar(value="Active interval: none")
        ttk.Label(self.live_card, textvariable=self.active_event_var, style="Card.TLabel").pack(anchor="w", pady=(10, 8))
        self.timeline = ttk.Treeview(self.live_card, columns=("time", "kind", "label"), show="headings", height=15)
        self.timeline.heading("time", text="Time")
        self.timeline.heading("kind", text="Type")
        self.timeline.heading("label", text="Label")
        self.timeline.column("time", width=70, anchor="center")
        self.timeline.column("kind", width=95, anchor="center")
        self.timeline.column("label", width=250)
        self.timeline.pack(fill="both", expand=True, pady=(4, 12))
        self.stop_button = ttk.Button(self.live_card, text="Stop and save session", style="Stop.TButton", command=self.stop_recording, state="disabled")
        self.stop_button.pack(fill="x")

    def selected_tag_values(self):
        return [tag for tag in all_tags() if self.selected_tags[tag].get()]

    def update_tag_menu_label(self, group):
        count = sum(self.selected_tags[tag].get() for tag in TAG_VOCABULARY[group])
        self.tag_menu_buttons[group].configure(text=f"{group} ({count})")
        selected = self.selected_tag_values()
        text = ", ".join(tag.replace("_", " ") for tag in selected)
        self.selected_tags_var.set(text if text else "No tags selected")

    def suggest_tags(self):
        suggestions = tags_for_task(self.title_var.get())
        for tag, variable in self.selected_tags.items():
            variable.set(tag in suggestions)
        self.ai_rationale_var.set("Local rule-based suggestion. Review the selected tags before recording.")

    def request_ai_tags(self):
        title = self.title_var.get().strip()
        description = self.description_text.get("1.0", "end").strip()
        if not title:
            messagebox.showwarning("Session title", "Enter the session title before requesting AI tags.")
            return
        self.ai_tag_button.configure(state="disabled", text="AI expert working…")
        self.status_var.set(f"Asking {OPENAI_MODEL} to select functional tags…")
        threading.Thread(target=self.ai_tag_worker, args=(title, description), daemon=True).start()

    def ai_tag_worker(self, title, description):
        try:
            tags, rationale = recommend_tags_with_openai(title, description)
            self.ui_queue.put(("ai_tags", tags, rationale))
        except Exception as error:
            self.ui_queue.put(("ai_error", str(error)))

    def set_setup_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.title_entry.configure(state=state)
        self.description_text.configure(state=state)
        self.start_button.configure(state=state)
        self.ai_tag_button.configure(state=state)
        for menu_button in self.tag_menu_buttons.values():
            menu_button.configure(state=state)

    def set_live_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.event_entry.configure(state=state)
        self.mark_button.configure(state=state)
        self.begin_button.configure(state=state)
        self.end_button.configure(state=state)
        self.stop_button.configure(state=state)

    def start_recording(self):
        title = self.title_var.get().strip()
        description = self.description_text.get("1.0", "end").strip()
        tags = self.selected_tag_values()
        if not title:
            messagebox.showerror("Missing title", "Enter a clear title for this session.")
            return
        if not tags:
            messagebox.showerror("Missing tags", "Select at least one task or expected-system tag.")
            return
        if not self.consent_var.get():
            messagebox.showerror("Consent required", "Confirm explicit consent before recording EEG.")
            return
        self.output_dir = make_session_dir(title)
        self.stop_event.clear()
        self.recording = True
        self.recording_ready = False
        self.set_setup_enabled(False)
        self.start_button.configure(state="disabled")
        self.status_var.set("Connecting to BrainAccess helmet…")
        self.live_indicator.configure(text="● CONNECTING", foreground="#fbbf24")
        payload = {"title": title, "description": description, "tags": tags}
        self.worker = threading.Thread(target=self.recording_worker, args=(payload,), daemon=True)
        self.worker.start()

    def recording_worker(self, payload):
        events = []
        segments = []
        active_interval = None
        session_started_utc = utc_now()
        traceability_id = uuid.uuid4().hex
        try:
            with BrainAccessStream(device_name=DEVICE_NAME, gain_name=GAIN_NAME, use_bias=USE_BIAS, bias_channel_name=BIAS_CHANNEL_NAME) as recorder:
                self.recorder = recorder
                recorder.start()
                self.writer = IncrementalEEGWriter(recorder, self.output_dir / "eeg_samples.csv")
                self.writer.start()
                start = event_row(0, "state_start", "task_start", payload["title"], payload["description"], recorder.stream_start_perf_s)
                events.append(start)
                segments.append({
                    "segment_index": 0,
                    "task_name": payload["title"],
                    "task_slug": slugify(payload["title"]),
                    "description": payload["description"],
                    "tags": payload["tags"],
                    "start_pc_time_perf_counter_s": start["pc_time_perf_counter_s"],
                    "start_t_from_stream_start_s": start["t_from_stream_start_s"],
                    "start_utc_time": start["utc_time"],
                    "end_pc_time_perf_counter_s": None,
                    "end_t_from_stream_start_s": None,
                    "end_utc_time": None,
                    "duration_s": None,
                })
                recorder.annotate(f"task_start:{slugify(payload['title'])}")
                self.ui_queue.put(("ready", start["t_from_stream_start_s"]))
                while not self.stop_event.is_set():
                    try:
                        action, label = self.command_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if action == "mark":
                        event = event_row(len(events), "marker", "user_marker", payload["title"], label, recorder.stream_start_perf_s)
                        events.append(event)
                        recorder.annotate(f"mark:{slugify(label)}")
                        self.ui_queue.put(("event", event))
                    elif action == "begin":
                        if active_interval is not None:
                            event = event_row(len(events), "state_end", "labeled_interval_end", active_interval, "interval replaced", recorder.stream_start_perf_s)
                            events.append(event)
                            self.ui_queue.put(("event", event))
                        active_interval = label
                        event = event_row(len(events), "state_start", "labeled_interval_start", label, "", recorder.stream_start_perf_s)
                        events.append(event)
                        recorder.annotate(f"event_start:{slugify(label)}")
                        self.ui_queue.put(("event", event))
                    elif action == "end" and active_interval is not None:
                        event = event_row(len(events), "state_end", "labeled_interval_end", active_interval, "", recorder.stream_start_perf_s)
                        events.append(event)
                        recorder.annotate(f"event_end:{slugify(active_interval)}")
                        active_interval = None
                        self.ui_queue.put(("event", event))
                if active_interval is not None:
                    event = event_row(len(events), "state_end", "labeled_interval_end", active_interval, "session stopped", recorder.stream_start_perf_s)
                    events.append(event)
                end = event_row(len(events), "state_end", "task_end", payload["title"], "session stopped by user", recorder.stream_start_perf_s)
                events.append(end)
                close_segment(segments[-1], end)
                recorder.annotate(f"task_end:{slugify(payload['title'])}")
                recorder.stop()
                self.writer.stop()
                annotations = recorder.get_annotations()
                recorder_metadata = recorder.metadata()
                writer_summary = self.writer.summary()
            session_ended_utc = utc_now()
            events_path = self.output_dir / "events.csv"
            annotations_path = self.output_dir / "brainaccess_annotations.csv"
            metadata_path = self.output_dir / "metadata.json"
            summary_path = self.output_dir / "recording_summary.json"
            preview_path = self.output_dir / "eeg_preview.csv"
            integrity_path = self.output_dir / "recording_integrity.json"
            pd.DataFrame(events).to_csv(events_path, index=False)
            pd.DataFrame(annotations).to_csv(annotations_path, index=False)
            metadata = {
                **recorder_metadata,
                "protocol_version": PROTOCOL_VERSION,
                "traceability_id": traceability_id,
                "subject_id": "subject_001",
                "consent_confirmed": True,
                "primary_task_name": payload["title"],
                "primary_task_description": payload["description"],
                "task_tags": payload["tags"],
                "session_started_at_utc": session_started_utc,
                "session_ended_at_utc": session_ended_utc,
                "event_count": len(events),
                "task_segment_count": len(segments),
                "incremental_disk_writer": True,
            }
            write_json(metadata_path, metadata)
            write_json(summary_path, writer_summary)
            integrity = create_recording_preview(self.output_dir / "eeg_samples.csv", preview_path)
            integrity["sample_count_reported_by_writer"] = writer_summary["sample_count"]
            write_json(integrity_path, integrity)
            write_session_files(self.output_dir, payload["title"], payload["description"], payload["tags"], "live_universal_recording", PROTOCOL_VERSION)
            traceability = {
                "schema_version": SCHEMA_VERSION,
                "traceability_id": traceability_id,
                "session_id": self.output_dir.name,
                "recording_path": str(self.output_dir.resolve().relative_to(Path(__file__).resolve().parent.parent)),
                "source_type": "live_universal_recording",
                "traceability_status": "user_annotated",
                "created_at_utc": utc_now(),
                "recording_started_at_utc": session_started_utc,
                "recording_ended_at_utc": session_ended_utc,
                "subject_id": "subject_001",
                "consent_confirmed": True,
                "protocol_version": PROTOCOL_VERSION,
                "primary_task": {"name": payload["title"], "description": payload["description"], "tags": payload["tags"]},
                "task_segments": segments,
                "available_state_labels": sorted(set(str(event["state_label"]) for event in events)),
                "device": {
                    "requested": recorder_metadata.get("device_name_requested"),
                    "connected": recorder_metadata.get("device_name_connected"),
                    "model": recorder_metadata.get("device_model"),
                    "serial_number": recorder_metadata.get("serial_number"),
                    "sample_frequency_hz": recorder_metadata.get("sample_frequency_hz"),
                    "channel_count": recorder_metadata.get("eeg_channel_count"),
                    "gain_name": recorder_metadata.get("gain_name"),
                    "bias_channel_name": recorder_metadata.get("bias_channel_name"),
                },
                "recording_summary": writer_summary,
                "files": [
                    file_provenance(self.output_dir / "eeg_samples.csv", "raw_eeg"),
                    file_provenance(events_path, "task_events"),
                    file_provenance(annotations_path, "sdk_annotations"),
                    file_provenance(metadata_path, "recording_metadata"),
                    file_provenance(summary_path, "recording_summary"),
                    file_provenance(preview_path, "eeg_preview"),
                    file_provenance(integrity_path, "recording_integrity"),
                    file_provenance(self.output_dir / "session.json", "canonical_session_metadata"),
                    file_provenance(self.output_dir / "SESSION.md", "natural_language_session_info"),
                ],
                "parent_sources": [],
                "notes": "Title, description, tags, and event labels were entered in the recording UI.",
                "inference_evidence": [],
            }
            write_json(self.output_dir / "traceability.json", traceability)
            rebuild_catalog()
            self.ui_queue.put(("saved", self.output_dir, writer_summary, integrity))
        except Exception as error:
            self.ui_queue.put(("error", str(error)))

    def require_event_label(self):
        label = self.event_var.get().strip()
        if not label:
            messagebox.showwarning("Event label", "Enter a short event label first.")
            return None
        return label

    def mark_event(self):
        label = self.require_event_label()
        if label:
            self.command_queue.put(("mark", label))
            self.event_var.set("")

    def begin_event(self):
        label = self.require_event_label()
        if label:
            self.command_queue.put(("begin", label))
            self.active_event_var.set(f"Active interval: {label}")
            self.event_var.set("")

    def end_event(self):
        self.command_queue.put(("end", ""))
        self.active_event_var.set("Active interval: none")

    def stop_recording(self):
        if not self.recording:
            return
        self.stop_button.configure(state="disabled")
        self.set_live_enabled(False)
        self.status_var.set("Stopping stream and saving session…")
        self.stop_event.set()

    def add_timeline_event(self, event):
        seconds = max(0.0, float(event["t_from_stream_start_s"]))
        timestamp = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        subtype = str(event["event_subtype"]).replace("labeled_interval_", "")
        label = event["text"] if event["event_type"] == "marker" else event["state_label"]
        self.timeline.insert("", "end", values=(timestamp, subtype, label))
        children = self.timeline.get_children()
        if children:
            self.timeline.see(children[-1])

    def poll_worker(self):
        while True:
            try:
                message = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "ready":
                self.recording_ready = True
                self.started_perf = time.perf_counter()
                self.set_live_enabled(True)
                self.live_indicator.configure(text="● RECORDING", foreground="#22c55e")
                self.status_var.set(f"Recording to {self.output_dir}")
            elif kind == "event":
                self.add_timeline_event(message[1])
            elif kind == "saved":
                self.recording = False
                self.recording_ready = False
                self.live_indicator.configure(text="● SAVED", foreground="#60a5fa")
                summary = message[2]
                self.status_var.set(f"Saved {summary['sample_count']:,} samples to {message[1]}")
                size_mb = message[3]["raw_csv_size_bytes"] / (1024 * 1024)
                messagebox.showinfo(
                    "Session saved and verified",
                    f"Saved {summary['sample_count']:,} samples ({size_mb:.1f} MB).\n\n"
                    f"Raw data: {message[1] / 'eeg_samples.csv'}\n"
                    f"Editor-friendly preview: {message[1] / 'eeg_preview.csv'}",
                )
            elif kind == "ai_tags":
                selected = set(message[1])
                for tag, variable in self.selected_tags.items():
                    variable.set(tag in selected)
                self.ai_rationale_var.set(f"{OPENAI_MODEL}: {message[2]}")
                self.ai_tag_button.configure(state="normal", text="AI neurology expert")
                self.status_var.set(f"AI selected {len(selected)} controlled functional tags. Review them before recording.")
            elif kind == "ai_error":
                self.ai_tag_button.configure(state="normal", text="AI neurology expert")
                self.status_var.set("AI tag suggestion failed")
                messagebox.showerror("AI tag suggestion", message[1])
            elif kind == "error":
                self.recording = False
                self.recording_ready = False
                self.live_indicator.configure(text="● ERROR", foreground="#ef4444")
                self.status_var.set("Recording failed")
                self.set_setup_enabled(True)
                messagebox.showerror("Recording error", message[1])
        if self.recording_ready and self.started_perf is not None:
            elapsed = time.perf_counter() - self.started_perf
            self.elapsed_var.set(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        self.root.after(100, self.poll_worker)

    def on_close(self):
        if self.recording:
            if not messagebox.askyesno("Stop recording?", "Stop and save the active EEG session before closing?"):
                return
            self.stop_recording()
            self.status_var.set("Saving before close…")
            self.root.after(200, self.close_when_saved)
            return
        self.root.destroy()

    def close_when_saved(self):
        if self.recording:
            self.root.after(200, self.close_when_saved)
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
