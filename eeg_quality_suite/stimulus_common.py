import time
import threading
import tkinter as tk

import numpy as np
from psychopy import visual, event, core, monitors


MONITOR_NAME = "brainz_runtime_monitor"
MONITOR_WIDTH_CM = 53.0
MONITOR_DISTANCE_CM = 60.0


def get_primary_screen_size():
    root = tk.Tk()
    root.withdraw()
    width = int(root.winfo_screenwidth())
    height = int(root.winfo_screenheight())
    root.destroy()
    return width, height


def make_window(background="black", fullscr=True):
    screen_size = get_primary_screen_size()
    monitor = monitors.Monitor(MONITOR_NAME, width=MONITOR_WIDTH_CM, distance=MONITOR_DISTANCE_CM)
    monitor.setSizePix(screen_size)
    win = visual.Window(
        size=screen_size,
        fullscr=fullscr,
        monitor=monitor,
        color=background,
        units="height",
        allowGUI=False,
        waitBlanking=True,
        checkTiming=False,
    )
    win.mouseVisible = False
    return win


def estimate_refresh_rate(win, fallback_hz=60.0):
    frame_times = []
    for frame_index in range(90):
        flip_time = win.flip()
        if frame_index >= 10:
            frame_times.append(float(flip_time))
    intervals = np.diff(np.asarray(frame_times, dtype=float))
    intervals = intervals[(intervals > 0.001) & (intervals < 0.1)]
    if len(intervals) > 0:
        return float(1.0 / np.median(intervals))
    period = getattr(win, "monitorFramePeriod", None)
    if period is not None and period > 0:
        rate = 1.0 / float(period)
    else:
        rate = fallback_hz
    return float(rate)


def draw_center_text(win, text, height=0.08, color="white", pos=(0, 0), wrap_width=1.6):
    stim = visual.TextStim(win, text=text, height=height, color=color, pos=pos, wrapWidth=wrap_width)
    stim.draw()


def wait_with_text(win, text, seconds, color="white", escape_allowed=True):
    clock = core.Clock()
    while clock.getTime() < seconds:
        draw_center_text(win, text, color=color)
        win.flip()
        if escape_allowed and "escape" in event.getKeys():
            raise KeyboardInterrupt


def countdown(win, seconds=3, prefix="Starting in"):
    for remaining in range(seconds, 0, -1):
        wait_with_text(win, f"{prefix}\n{remaining}", 1.0)


def frames_for_seconds(refresh_rate_hz, seconds):
    return int(round(float(refresh_rate_hz) * float(seconds)))


def flicker_pattern(refresh_rate_hz, target_hz):
    frames_per_cycle = max(2, int(round(refresh_rate_hz / float(target_hz))))
    actual_hz = refresh_rate_hz / float(frames_per_cycle)
    bright_frames = max(1, frames_per_cycle // 2)
    dark_frames = max(1, frames_per_cycle - bright_frames)
    return frames_per_cycle, bright_frames, dark_frames, actual_hz


def play_beep(frequency=1000, duration_ms=180):
    def worker():
        if __import__("platform").system().lower().startswith("win"):
            import winsound
            winsound.Beep(int(frequency), int(duration_ms))
        else:
            print("\a", end="", flush=True)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def perf_time_after_flip(win):
    flip_time = win.flip()
    perf_time = time.perf_counter()
    return flip_time, perf_time
