"""Splice the zoom insert into the original video and encode (H.264 CRF 16, yuv420p, 30 fps, original silent
audio padded). Usage: compose.py A|B  (A = zoom only, B = zoom + profile-line illustration).

Layout: original frames 0-449 | zoom insert (crossfades in from the original close-up) | re-rendered tail for
original frames 450-799 (same static close-up, smoother edge-point / phase effects at the original times)."""
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_lib as rl  # noqa: E402

FF = rl.FFMPEG
VDIR = rl.VIDEO_DIR
FRAMES = os.path.join(rl.WORK, "frames_base")  # from render_insert.py
SRC = os.path.join(VDIR, "transitionToScanSequence_smooth.mp4")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "A"
NAME = {"A": "transitionToScanSequence_smooth_zoom.mp4",
        "B": "transitionToScanSequence_smooth_zoom_profileLine.mp4"}[VERSION]
W, H = 1920, 1080
FB = W * H * 3
SPLICE = 450   # insert between original frames 449 and 450 (camera static on the close-up 412-511)
XF = 8         # crossfade frames from the original close-up into the insert

import tail_effects as te  # noqa: E402

if VERSION == "B":
    import profile_overlay as po  # noqa: E402
NOTCH = 14     # insert frames at each end (camera ~at the close-up) where the left-edge sliver is faded out

bases = sorted(f for f in os.listdir(FRAMES) if f.startswith("base_"))
N = len(bases)


def insert_frame(k: int) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(os.path.join(FRAMES, bases[k])), cv2.COLOR_BGR2RGB)
    if VERSION == "B":
        img = po.composite(img, po.overlay(k))
    return img


def smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


dec = subprocess.Popen([FF, "-v", "error", "-i", SRC, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE)
enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", "30", "-i", "-", "-i", SRC, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-af", "apad", "-shortest", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                        os.path.join(VDIR, NAME)], stdin=subprocess.PIPE)


def read() -> "np.ndarray | None":
    buf = dec.stdout.read(FB)
    return None if len(buf) < FB else np.frombuffer(buf, np.uint8).reshape(H, W, 3)


t0 = time.perf_counter()
n_out = 0
prev = None
for i in range(SPLICE):
    prev = read()
    enc.stdin.write(prev.tobytes())
    n_out += 1
dec.kill()    # the original's frames 450-799 are replaced by the re-rendered tail
last = None
for k in range(N):
    base = insert_frame(k)
    if k < NOTCH:                                # edge sliver: hidden while static, fades as the camera moves off
        base = te.fill_notch(base, 1.0 if k < XF else 1 - smooth((k - XF + 1) / (NOTCH - XF)))
    if k >= N - NOTCH:                           # ... and fades back out before the static tail
        base = te.fill_notch(base, smooth((k - (N - NOTCH) + 1) / (NOTCH - XF)))
    img = base.astype(np.float32)
    if k < XF:                                   # dissolve in from the original close-up
        w = smooth((k + 1) / (XF + 1))
        img = (1 - w) * prev + w * img
    last = np.clip(np.round(img), 0, 255).astype(np.uint8)
    enc.stdin.write(last.tobytes())
    n_out += 1
for f in range(SPLICE, 800):                     # the tail: same close-up, re-rendered effects
    tf = te.frame(f)
    if f == SPLICE:
        print(f"  insert end -> tail start: mean abs diff {np.abs(tf.astype(int) - last).mean():.3f}")
    enc.stdin.write(tf.tobytes())
    n_out += 1
enc.stdin.close()
enc.wait()
print(f"version {VERSION}: wrote {n_out} frames ({n_out / 30:.2f} s) -> {NAME} in {time.perf_counter() - t0:.0f} s; "
      f"insert {N} frames at {SPLICE / 30:.2f} s")
