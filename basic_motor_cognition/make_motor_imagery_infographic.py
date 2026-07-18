from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
OUTPUT_PNG = OUTPUT_DIR / "motor_imagery_infographic.png"
OUTPUT_SVG = OUTPUT_DIR / "motor_imagery_infographic.svg"
FIGURE_SIZE = (14, 9)
BACKGROUND = "#f7f8fb"
INK = "#172033"
MUTED = "#5d6678"
BLUE = "#2f6fdb"
RED = "#d24b4b"
GREEN = "#2f9b70"
PURPLE = "#7b5cc7"


def draw_person(ax, x, y, scale, highlight):
    head = Circle((x, y + 1.45 * scale), 0.18 * scale, fill=False, linewidth=2.4, color=INK)
    ax.add_patch(head)
    ax.plot([x, x], [y + 1.25 * scale, y + 0.55 * scale], color=INK, linewidth=2.4)
    ax.plot([x - 0.55 * scale, x + 0.55 * scale], [y + 1.03 * scale, y + 1.03 * scale], color=INK, linewidth=2.4)
    ax.plot([x, x - 0.42 * scale], [y + 0.55 * scale, y], color=INK, linewidth=2.4)
    ax.plot([x, x + 0.42 * scale], [y + 0.55 * scale, y], color=INK, linewidth=2.4)
    limb_points = {
        "left_arm": ((x - 0.55 * scale, y + 1.03 * scale), (x - 0.92 * scale, y + 0.72 * scale), BLUE),
        "right_arm": ((x + 0.55 * scale, y + 1.03 * scale), (x + 0.92 * scale, y + 0.72 * scale), RED),
        "left_leg": ((x - 0.42 * scale, y), (x - 0.68 * scale, y - 0.48 * scale), GREEN),
        "right_leg": ((x + 0.42 * scale, y), (x + 0.68 * scale, y - 0.48 * scale), PURPLE),
    }
    for limb in highlight:
        start, end, color = limb_points[limb]
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18 * scale, linewidth=4.0, color=color)
        ax.add_patch(arrow)


def add_panel(ax, x, y, title, body, highlight):
    ax.text(x, y + 1.9, title, ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
    draw_person(ax, x, y + 0.25, 0.9, highlight)
    ax.text(x, y - 0.65, body, ha="center", va="top", fontsize=10.5, color=MUTED, linespacing=1.25)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(7, 8.45, "Motor Imagery: What To Think During Recording", ha="center", va="center", fontsize=24, fontweight="bold", color=INK)
    ax.text(7, 8.02, "Imagine the feeling and intention of movement. Do not tense, twitch, press, clench, or move.", ha="center", va="center", fontsize=13, color=MUTED)
    panels = [
        (2.0, 5.8, "Left Arm", "Feel the left arm\nstarting to lift or reach.", ["left_arm"]),
        (5.3, 5.8, "Right Arm", "Feel the right arm\nstarting to lift or reach.", ["right_arm"]),
        (8.7, 5.8, "Left Leg", "Feel the left leg\nstarting to step or kick.", ["left_leg"]),
        (12.0, 5.8, "Right Leg", "Feel the right leg\nstarting to step or kick.", ["right_leg"]),
        (2.0, 2.4, "Both Hands", "Imagine both hands\nopening or squeezing air.", ["left_arm", "right_arm"]),
        (5.3, 2.4, "Both Legs", "Imagine both legs\nstarting to walk.", ["left_leg", "right_leg"]),
        (8.7, 2.4, "Left Side", "Imagine left arm and\nleft leg together.", ["left_arm", "left_leg"]),
        (12.0, 2.4, "All Four", "Imagine all limbs moving\nwithout muscle activity.", ["left_arm", "right_arm", "left_leg", "right_leg"]),
    ]
    for panel in panels:
        add_panel(ax, *panel)
    ax.text(7, 0.45, "Best cue: kinesthetic imagination. Think about the body sensation, not a visual movie of yourself moving.", ha="center", va="center", fontsize=12.5, color=INK)
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180, facecolor=BACKGROUND)
    fig.savefig(OUTPUT_SVG, facecolor=BACKGROUND)
    plt.close(fig)
    print(OUTPUT_PNG)
    print(OUTPUT_SVG)


if __name__ == "__main__":
    main()
