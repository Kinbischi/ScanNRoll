"""Render the base frames of the zoom insert (shared by versions A and B): 2x supersampled, downsampled to
1920x1080, written as lossless PNGs to work/frames_base + the per-frame poses (work/path_poses.npy)."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_path as mp  # noqa: E402
import render_lib as rl  # noqa: E402
from insert_renderer import H, W, InsertScene  # noqa: E402

FRAMES = os.path.join(rl.WORK, "frames_base")
os.makedirs(FRAMES, exist_ok=True)
poses, params = mp.frame_poses()
np.save(os.path.join(rl.WORK, "path_poses.npy"), poses)
SS = 2
scene = InsertScene(mp.z0 - 150000, mp.z1 + 300000, supersample=SS)
print(f"scene: {scene.n_points:,} points; {len(poses)} frames at {W * SS}x{H * SS} -> {W}x{H}")
t0 = time.perf_counter()
for k, x in enumerate(poses):
    big = scene.render(x)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(FRAMES, f"base_{k:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if k % 40 == 0:
        print(f"  frame {k:3d}  {time.perf_counter() - t0:6.1f} s")
print(f"done in {time.perf_counter() - t0:.0f} s")
