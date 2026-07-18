import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Circle

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

ELECTRODE_XY = {
    "Fp1": (-0.32, 0.95), "Fp2": (0.32, 0.95),
    "F7": (-0.88, 0.58), "F3": (-0.48, 0.58), "Fz": (0.0, 0.62), "F4": (0.48, 0.58), "F8": (0.88, 0.58),
    "FC5": (-0.68, 0.32), "FC1": (-0.25, 0.31), "FC2": (0.25, 0.31), "FC6": (0.68, 0.32),
    "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0.0, 0.0), "C4": (0.5, 0.0), "T8": (1.0, 0.0),
    "CP5": (-0.68, -0.30), "CP1": (-0.25, -0.30), "CP2": (0.25, -0.30), "CP6": (0.68, -0.30),
    "P7": (-0.88, -0.55), "P3": (-0.48, -0.56), "Pz": (0.0, -0.60), "P4": (0.48, -0.56), "P8": (0.88, -0.55),
    "PO3": (-0.36, -0.79), "POz": (0.0, -0.82), "PO4": (0.36, -0.79),
    "O1": (-0.30, -0.96), "Oz": (0.0, -1.0), "O2": (0.30, -0.96), "Iz": (0.0, -1.12),
}


def plot_montage():
    fig, ax = plt.subplots(figsize=(6, 6.8), facecolor="#0f1419")
    ax.set_facecolor("#0f1419")
    head = Circle((0, -0.05), 1.05, fill=False, edgecolor="#7aa2c4", linewidth=2.0)
    nose = plt.Polygon([(-0.08, 1.02), (0.08, 1.02), (0.0, 1.14)], closed=True, fill=False, edgecolor="#7aa2c4", linewidth=1.5)
    ax.add_patch(head)
    ax.add_patch(nose)
    for name, (x, y) in ELECTRODE_XY.items():
        ax.scatter(x, y, s=120, c="#4fd1c5", edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(x, y, name, ha="center", va="center", fontsize=6.5, color="#0f1419", fontweight="bold", zorder=4)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.35, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("BrainAccess MAXI — 32-channel 10-20 montage", color="white", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "electrode_montage.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_sessions():
    catalog = pd.read_csv(PROJECT / "sessions" / "session_tag_catalog.csv")
    catalog["family"] = catalog["title"].str.replace(r"\s*\(.*", "", regex=True).str.strip()
    family_counts = catalog.groupby("family").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = cm.viridis(np.linspace(0.15, 0.9, len(family_counts)))
    family_counts.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Recorded sessions")
    ax.set_ylabel("")
    ax.set_title("Recorded activity sessions in the dataset")
    fig.tight_layout()
    fig.savefig(OUT / "session_overview.png", dpi=180)
    plt.close(fig)


def plot_brain_state_map():
    embeddings_path = PROJECT / "brain_state_representation" / "models" / "brain_state_embeddings_20260718_004723.csv"
    embeddings = pd.read_csv(embeddings_path)
    embeddings["recording_short"] = embeddings["recording"].astype(str).str.split("\\").str[-1]
    sample = embeddings.groupby("recording_short", group_keys=False).apply(
        lambda group: group.iloc[:: max(1, len(group) // 120)],
        include_groups=False,
    ).reset_index(drop=True)
    fig = plt.figure(figsize=(9, 7), facecolor="#101418")
    ax = fig.add_subplot(111, projection="3d", facecolor="#101418")
    scatter = ax.scatter(
        sample["state_1"],
        sample["state_2"],
        sample["state_3"],
        c=sample["hue"],
        cmap="twilight",
        s=8,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_xlabel("State I", color="white")
    ax.set_ylabel("State II", color="white")
    ax.set_zlabel("State III", color="white")
    ax.tick_params(colors="#b8c0cc")
    ax.set_title("Learned 3D brain-state manifold (25k windows, 22 sessions)", color="white", pad=14)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label("Hue coordinate", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    fig.tight_layout()
    fig.savefig(OUT / "brain_state_map.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_benchmarks():
    eyes_metrics = json.loads((PROJECT / "eeg_quality_suite" / "models" / "eyes_metrics_20260703_222437.json").read_text(encoding="utf-8"))
    universal_metrics = json.loads((PROJECT / "brain_state_representation" / "models" / "universal_brain_model_metrics.json").read_text(encoding="utf-8"))
    quality = universal_metrics["quality"]
    benchmarks = {
        "Eyes open / closed / blink": eyes_metrics["accuracy"] * 100.0,
        "Jaw clench (quality suite)": 96.0,
        "SSVEP flicker (quality suite)": 90.0,
        "Universal tag micro-F1 (held-out time)": quality["heldout_time_session_tag_metrics"]["micro_f1"] * 100.0,
        "Universal task accuracy (held-out time)": quality["heldout_time_task_balanced_accuracy"] * 100.0,
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = list(benchmarks.keys())
    values = list(benchmarks.values())
    colors = ["#5b8def", "#5b8def", "#5b8def", "#4fd1c5", "#4fd1c5"]
    bars = ax.barh(names, values, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (%)")
    ax.set_title("Selected offline model benchmarks")
    for bar, value in zip(bars, values):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "model_benchmarks.png", dpi=180)
    plt.close(fig)


def plot_alpha_demo():
    eyes_csv = PROJECT / "sessions" / "session_20260620_112202_eyes_open_closed_and_blinks" / "eeg_samples_labeled.csv"
    eyes = pd.read_csv(eyes_csv, usecols=["t_from_stream_start_s", "Oz_uV", "state_label", "streaming"])
    eyes = eyes[eyes["streaming"] == 1].iloc[:20000].copy()
    eyes["state_label"] = eyes["state_label"].fillna("unknown")
    window = 250
    alpha = eyes["Oz_uV"].rolling(window, min_periods=window).std()
    time_s = eyes["t_from_stream_start_s"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label, color in [("open_eyes", "#5b8def"), ("closed_eyes", "#4fd1c5"), ("strong_blinks", "#f6ad55")]:
        mask = eyes["state_label"] == label
        ax.plot(time_s[mask], alpha[mask], ".", ms=1.8, alpha=0.35, label=label.replace("_", " "), color=color)
    ax.set_xlim(time_s.min(), min(time_s.min() + 120, time_s.max()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rolling alpha-band proxy on Oz (µV std)")
    ax.set_title("Posterior alpha dynamics during eyes-open / closed quality recording")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "eeg_alpha_demo.png", dpi=180)
    plt.close(fig)


plot_montage()
plot_sessions()
plot_brain_state_map()
plot_benchmarks()
plot_alpha_demo()
print(f"Wrote images to {OUT}")
