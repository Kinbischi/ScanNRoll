"""Version-B overlay: a magenta (#F20782, the laser colour in the footage) line through one scan profile's points,
drawn left->right, then stepped to neighbouring profiles with a fading trail, faded out before the zoom-out.
Drawn at 2x with anti-aliasing, then area-downsampled to 1920x1080 RGBA."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_path as mp  # noqa: E402
import path_lib as pth  # noqa: E402
import render_lib as rl  # noqa: E402
from insert_renderer import H, W, ZREF  # noqa: E402

MAGENTA = (242, 7, 130)   # RGB
SS = 2
poses = np.load(os.path.join(rl.WORK, "path_poses.npy"))  # written by render_insert.py
_, params = mp.frame_poses()
SZ, SY = poses[0][7], poses[0][8]

# timeline (insert frame indices, 30 fps)
DRAW0, DRAW1 = 70, 92                      # 2.33-3.07 s: draw-on left -> right (camera looking along the belt)
STEP0, STEP_EVERY, N_STEPS = 96, 4, 11     # then 11 more profiles toward the back, one every 4 frames (3.2-4.53 s)
STRIDE = 4                                 # every 4th profile (1.8 mm): adjacent ones (0.45 mm) smear into one line
TRAIL_DECAY, TRAIL_FLOOR = 0.8, 0.4        # older lines dim to 40 % and stay, so the whole 12-profile stack shows
FADE0, FADE1 = 142, 154                    # 4.73-5.13 s: fade out before the zoom-out (5.4 s)

# the profile at the view centre at ~2.7 s
z_disp = params[82][2]
z_data = ZREF + (z_disp - ZREF) / SZ
i0 = int(np.argmin(np.abs(rl.zpath - z_data)))
print(f"highlighted profile {i0} (z {rl.zpath[i0]:.0f}), then every {STRIDE}th up to {i0 + STRIDE * N_STEPS}; "
      f"step {np.mean(np.diff(rl.zpath[i0:i0 + STRIDE * N_STEPS + 1:STRIDE])) / 100:.2f} mm")


def profile_world(i: int) -> np.ndarray:
    """The drawn points of profile i (point_step 2, as in the viewer), sorted across the belt, displayed coords."""
    p = rl.processed[i]
    xs, hs = p.x[::2], p.z[::2]
    o = np.argsort(xs)
    P = np.column_stack((xs[o], hs[o] * SY, np.full(o.size, ZREF + (rl.zpath[i] - ZREF) * SZ)))
    return P


def project(P: np.ndarray, x: np.ndarray) -> np.ndarray:
    R = pth.Rotation.from_rotvec(x[:3]).as_matrix()
    pc = (P - x[3:6]) @ R.T
    f = (H / 2) / np.tan(np.radians(x[6]) / 2)
    return np.column_stack((f * pc[:, 0] / pc[:, 2] + W / 2, f * pc[:, 1] / pc[:, 2] + H / 2)), pc[:, 2]


def partial(uv: np.ndarray, frac: float) -> np.ndarray:
    """Leading part of a polyline up to `frac` of its screen arc length."""
    seg = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = frac * cum[-1]
    j = int(np.searchsorted(cum, target, side="right"))
    if j >= len(uv):
        return uv
    t = (target - cum[j - 1]) / max(seg[j - 1], 1e-9)
    return np.vstack([uv[:j], uv[j - 1] + t * (uv[j] - uv[j - 1])])


def draw_line(canvas: np.ndarray, uv: np.ndarray, alpha: float, width: float, halo: float) -> None:
    """Draw (white halo + magenta core) onto a premultiplied RGBA float canvas at SS scale ("over" blend)."""
    if len(uv) < 2 or alpha <= 0:
        return
    pts = np.round(uv * SS * 16).astype(np.int32).reshape(-1, 1, 2)  # 4 fractional bits
    for colour, w in (((255, 255, 255), width + 2 * halo), (MAGENTA, width)):
        layer = np.zeros(canvas.shape[:2], np.uint8)
        cv2.polylines(layer, [pts], False, 255, int(round(w * SS)), cv2.LINE_AA, shift=4)
        a = layer.astype(np.float32) / 255.0 * alpha
        for c in range(3):
            canvas[..., c] = canvas[..., c] * (1 - a) + colour[c] * a
        canvas[..., 3] = canvas[..., 3] * (1 - a) + a


def smooth(t: float) -> float:
    return float(pth.smootherstep(np.array(t)))


def overlay(k: int) -> "np.ndarray | None":
    """RGBA overlay (1920x1080, float 0-255 RGB + 0-1 alpha) for insert frame k, or None."""
    if k < DRAW0 or k >= FADE1:
        return None
    x = poses[k]
    fade = 1.0 - smooth((k - FADE0) / (FADE1 - FADE0)) if k >= FADE0 else 1.0
    canvas = np.zeros((H * SS, W * SS, 4), np.float32)
    if k < STEP0:
        uv, z = project(profile_world(i0), x)
        uv = uv[z > 0]
        if len(uv) and uv[0, 0] > uv[-1, 0]:  # reveal from the screen-left end, whatever the camera orientation
            uv = uv[::-1]
        frac = smooth((k - DRAW0) / (DRAW1 - DRAW0)) if k < DRAW1 else 1.0
        draw_line(canvas, partial(uv, frac), fade, 3.0, 1.5)
    else:
        cur = min(N_STEPS, (k - STEP0) // STEP_EVERY + 1)   # how many steps taken
        for age in range(cur, -1, -1):                     # oldest first, current on top
            uv, z = project(profile_world(i0 + STRIDE * (cur - age)), x)
            a = (1.0 if age == 0 else max(TRAIL_FLOOR, TRAIL_DECAY ** age)) * fade
            if a > 0.06:
                draw_line(canvas, uv[z > 0], a, 3.0 if age == 0 else 2.0, 1.5 if age == 0 else 1.0)
    small = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    return small


def composite(rgb: np.ndarray, ov: "np.ndarray | None") -> np.ndarray:
    if ov is None:
        return rgb
    a = ov[..., 3:4]
    out = rgb.astype(np.float32) * (1 - a) + ov[..., :3]  # canvas is premultiplied (built over transparent black)
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    for k in (80, 90, 104, 120, 138, 148):
        base = cv2.cvtColor(cv2.imread(os.path.join(rl.WORK, "frames_base", f"base_{k:03d}.png")), cv2.COLOR_BGR2RGB)
        ov = overlay(k)
        img = composite(base, ov)
        cv2.imwrite(os.path.join(rl.WORK, f"B_{k:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("preview frames written")
