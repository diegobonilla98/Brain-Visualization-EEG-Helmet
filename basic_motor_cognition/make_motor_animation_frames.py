from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


OUTPUT_DIR = Path(__file__).resolve().parent / "assets" / "motor_animation_frames"
FRAME_COUNT = 16
FIGURE_SIZE = (4.2, 4.2)
DPI = 150
INK = "#f2f5ff"
NEUTRAL = "#8d96aa"
COLORS = {
    "left_arm": "#3d8cff",
    "right_arm": "#ff5a5f",
    "left_leg": "#2fc47c",
    "right_leg": "#a174ff",
}
LIMBS = ["left_arm", "right_arm", "left_leg", "right_leg"]


def point(x, y):
    return x, y


def limb_positions(limb, amount):
    side = -1.0 if limb in ["left_arm", "left_leg"] else 1.0
    if limb.endswith("arm"):
        shoulder = point(0.09 * side, 0.30)
        elbow = point(side * (0.17 + 0.04 * amount), 0.12 - 0.06 * amount)
        hand = point(side * (0.26 + 0.09 * amount), -0.02 - 0.03 * amount)
        neutral_hand = point(side * 0.21, -0.13)
        return shoulder, elbow, hand, neutral_hand
    hip = point(0.055 * side, -0.17)
    knee = point(side * (0.08 + 0.05 * amount), -0.36 + 0.10 * amount)
    foot = point(side * (0.10 + 0.11 * amount), -0.58 + 0.04 * amount)
    neutral_foot = point(side * 0.10, -0.60)
    return hip, knee, foot, neutral_foot


def draw_line(ax, start, end, color, width):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=width, solid_capstyle="round")


def draw_frame(mask, frame_index):
    active_limbs = [limb for index, limb in enumerate(LIMBS) if (mask >> index) & 1]
    cycle_phase = frame_index / float(FRAME_COUNT)
    amount = 0.5 - 0.5 * np.cos(2.0 * np.pi * cycle_phase)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.82, 0.62)
    ax.axis("off")
    ax.add_patch(Circle((0.0, 0.47), 0.055, fill=False, linewidth=3.0, color=INK))
    draw_line(ax, point(0.0, 0.40), point(0.0, -0.17), INK, 4.0)
    draw_line(ax, point(-0.09, 0.30), point(0.09, 0.30), INK, 4.0)
    draw_line(ax, point(-0.055, -0.17), point(0.055, -0.17), INK, 4.0)
    for limb in LIMBS:
        used_amount = amount if limb in active_limbs else 0.0
        root, joint, end, neutral_end = limb_positions(limb, used_amount)
        color = COLORS[limb] if limb in active_limbs else NEUTRAL
        width = 5.2 if limb in active_limbs else 3.2
        draw_line(ax, root, joint, color, width)
        draw_line(ax, joint, end, color, width)
        ax.add_patch(Circle(joint, 0.014, color=color))
        ax.add_patch(Circle(end, 0.017 if limb in active_limbs else 0.011, color=color))
        if limb in active_limbs:
            arrow = FancyArrowPatch(neutral_end, end, arrowstyle="-|>", mutation_scale=16, linewidth=2.2, color=color, alpha=0.65)
            ax.add_patch(arrow)
    output = OUTPUT_DIR / f"mask_{mask:02d}_frame_{frame_index:02d}.png"
    fig.savefig(output, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for mask in range(16):
        for frame_index in range(FRAME_COUNT):
            draw_frame(mask, frame_index)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
