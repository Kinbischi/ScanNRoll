"""Re-rendered tail of the video (original frames 450-799, static close-up) with smoother effects, at the original
event times: blue edge points (orig 512, gone at 565) appear as crisp dots sweeping head -> tail and fade out;
the body colour (637) and the rupture tip (678) sweep along each segment with a soft edge instead of popping.

prepare() renders the colour states + depth once (2x supersampled) and caches them; frame(f) composites frame f."""
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.path.dirname(HERE), "work")   # = render_lib.WORK (not imported here: it loads the data)
CACHE = os.path.join(WORK, "tail_cache.npz")
sys.path.insert(0, HERE)
W, H, FPS, SS = 1920, 1080, 30, 2

# effect timeline, in ORIGINAL frame numbers (kept identical so the edit / music sync stays valid)
MARK_IN, MARK_SWEEP, MARK_RAMP = 512, 0.75, 0.18   # edge points sweep in over 0.75 s, each dot ramps 0.18 s
MARK_OUT, MARK_FADE = 565, 0.40                    # ... and fade out over 0.4 s
BODY_IN, BODY_SWEEP = 637, 0.90                    # body colour sweep
RUP_IN, RUP_SWEEP = 678, 0.50                      # rupture colour sweep
SOFT = 0.05                                         # sweep edge softness (fraction of the phase length)
HIDDEN_ALPHA = 0.5                                  # far-edge points behind the bead: drawn semi-transparent
MARK_EVERY = 3                                      # edge point of every 3rd profile -> a clean dotted outline
MARK_RGB, MARK_R, MARK_RIM = (30, 91, 255), 4.0, 1.5  # vivid blue dot, radius / white rim (px at 1080p)


def _ease(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def notch_mask(img: np.ndarray) -> np.ndarray:
    """White pixels at the left frame edge (x < 100, y > 640) connected to the border: the sliver of the previous
    segment's floor gap that the reconstructed camera shows but the original close-up does not (float mask 0-1)."""
    white = (img.min(axis=2) > 200).astype(np.uint8)
    box = np.zeros_like(white)
    box[640:, :100] = white[640:, :100]
    n, lab = cv2.connectedComponents(box, connectivity=8)
    keep = np.isin(lab, np.unique(lab[:, 0][lab[:, 0] > 0]))
    m = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


FLOOR_RGB = np.array([139, 69, 19], np.float32)  # saddlebrown, as rendered


def fill_notch(img: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Paint the edge sliver (notch_mask) with the floor colour, blended by `weight`."""
    m = notch_mask(img)[..., None] * weight
    if not m.any():
        return img
    return np.clip(img.astype(np.float32) * (1 - m) + FLOOR_RGB * m, 0, 255).astype(np.uint8)


def prepare() -> None:
    import render_lib as rl
    from insert_renderer import ZREF, InsertScene

    x = np.load(os.path.join(WORK, "path_poses.npy"))[-1]         # the close-up pose (= last insert frame)
    sz, sy = x[7], x[8]
    sc = InsertScene(x[5] - 150000, x[5] + 300000, phases=True, supersample=SS)
    for a in ("body", "rup"):
        sc.acts[a].prop.color = "green"
    plain = sc.render(x)
    depth = -sc.pl.get_image_depth()                              # camera depth per 2x pixel (NaN = background)
    sc.acts["body"].prop.color = "royalblue"
    body = sc.render(x)
    sc.acts["rup"].prop.color = "crimson"
    allc = sc.render(x)
    down = [cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in (plain, body, allc)]
    down = [fill_notch(im) for im in down]

    # per-pixel data z (along the belt): back-project the 2x depth, then average finite values per 2x2 block
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    f2 = (H * SS / 2) / np.tan(np.radians(x[6]) / 2)
    jj, ii = np.meshgrid(np.arange(W * SS) + 0.5, np.arange(H * SS) + 0.5)
    pc = np.stack([(jj - W * SS / 2) / f2 * depth, (ii - H * SS / 2) / f2 * depth, depth], axis=-1)
    zdisp = x[5] + pc @ R[:, 2]                                   # world z = C_z + (R^T pc)_z
    zdata = ZREF + (zdisp - ZREF) / sz
    blocks = zdata.reshape(H, SS, W, SS)
    with np.errstate(invalid="ignore"):
        zmap = np.nanmean(blocks, axis=(1, 3))

    # edge points (widthOuterIdx) of every MARK_EVERY-th segment profile, visible from this camera
    seg = rl.segment_table()
    marks = []
    for k, (z0, z1) in enumerate(seg[:, :2]):
        if z1 < x[5] - 50000 or z0 > x[5] + 250000:
            continue
        a, b = np.searchsorted(rl.zpath, [z0, z1 + 1])
        for i in range(a, b, MARK_EVERY):
            p = rl.processed[i]
            idx = p.widthOuterIdx
            if idx is None or len(idx) != 2:
                continue
            for j in (int(idx[0]), int(idx[1])):
                P = np.array([[p.x[j], p.z[j] * sy, ZREF + (rl.zpath[i] - ZREF) * sz]])
                q = (P - x[3:6]) @ R.T
                d = q[0, 2]
                u = (H / 2) / np.tan(np.radians(x[6]) / 2) * q[0, 0] / d + W / 2
                v = (H / 2) / np.tan(np.radians(x[6]) / 2) * q[0, 1] / d + H / 2
                if not (-10 < u < W + 10 and -10 < v < H + 10):
                    continue
                c, r = int(u * SS), int(v * SS)
                win = depth[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] if 0 <= r < H * SS and 0 <= c < W * SS \
                    else np.array([np.nan])
                near = np.nanmin(win) if np.isfinite(win).any() else np.inf
                visible = float(d <= near + 120.0)                # not hidden behind the bead (1.2 mm tolerance)
                marks.append((u, v, d, (rl.zpath[i] - z0) / max(z1 - z0, 1.0), visible))
    marks = np.array(marks)
    np.savez_compressed(CACHE, plain=down[0], body=down[1], allc=down[2], zmap=zmap, seg=seg,
                        marks=marks[np.argsort(-marks[:, 2])])     # far first -> near dots drawn on top
    print(f"tail cache: {len(marks)} edge points ({int(marks[:, 4].sum())} visible, rest behind the bead); "
          f"z-map finite {np.isfinite(zmap).mean():.2f}")


_cache = {}


def _load():
    if not _cache:
        c = np.load(CACHE)
        _cache.update({k: c[k] for k in c.files})
        plain, body, allc = (_cache[k].astype(np.float32) for k in ("plain", "body", "allc"))
        _cache["d_body"], _cache["d_rup"], _cache["plain_f"] = body - plain, allc - body, plain
        z, seg = _cache["zmap"], _cache["seg"]
        k = np.clip(np.searchsorted(seg[:, 0], np.nan_to_num(z, nan=-1.0), side="right") - 1, 0, len(seg) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            _cache["s_body"] = (z - seg[k, 3]) / (seg[k, 4] - seg[k, 3])
            _cache["s_rup"] = (z - seg[k, 5]) / (seg[k, 6] - seg[k, 5])
    return _cache


def _sweep_weight(s: np.ndarray, t: float, dur: float) -> np.ndarray:
    """Soft-edged reveal along s (0 = start of the phase, 1 = end) whose front travels over `dur` seconds."""
    if t <= 0:
        return np.zeros_like(s)
    front = -SOFT + (1 + 2 * SOFT) * float(_ease(np.array(t / dur)))
    w = _ease((front - s) / SOFT + 0.5)
    return np.nan_to_num(w, nan=0.0)


def _dots(t_in: float, t_out: float) -> "np.ndarray | None":
    """Premultiplied RGBA overlay of the edge points at time t_in after MARK_IN (t_out after MARK_OUT, or <0)."""
    c = _load()
    m = c["marks"]
    a = _ease((t_in - m[:, 3] * MARK_SWEEP) / MARK_RAMP)
    if t_out > 0:
        a = a * (1 - _ease(np.array(t_out / MARK_FADE)))
    if not (a > 0.01).any():
        return None
    canvas = np.zeros((H * SS, W * SS, 4), np.float32)
    for (u, v, _, _, vis), grow in zip(m, a):
        ai = grow * (1.0 if vis else HIDDEN_ALPHA)
        if ai <= 0.01:
            continue
        r = (MARK_R * (0.55 + 0.45 * grow) * (1.0 if vis else 0.85)) * SS
        ctr = (int(round(u * SS * 16)), int(round(v * SS * 16)))
        for colour, rad in (((255, 255, 255), r + MARK_RIM * SS * (1.0 if vis else 0.6)), (MARK_RGB, r)):
            layer = np.zeros((H * SS, W * SS), np.uint8)
            x0, y0 = int(u * SS - rad - 3), int(v * SS - rad - 3)
            x1, y1 = int(u * SS + rad + 4), int(v * SS + rad + 4)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W * SS), min(y1, H * SS)
            if x0 >= x1 or y0 >= y1:
                continue
            cv2.circle(layer, ctr, int(round(rad * 16)), 255, -1, cv2.LINE_AA, shift=4)
            al = layer[y0:y1, x0:x1].astype(np.float32) / 255.0 * ai
            sub = canvas[y0:y1, x0:x1]
            for ch in range(3):
                sub[..., ch] = sub[..., ch] * (1 - al) + colour[ch] * al
            sub[..., 3] = sub[..., 3] * (1 - al) + al
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)


def frame(f: int) -> np.ndarray:
    """Tail frame for ORIGINAL frame number f (450..799), RGB uint8."""
    c = _load()
    t_body, t_rup = (f - BODY_IN) / FPS, (f - RUP_IN) / FPS
    img = c["plain_f"] + _sweep_weight(c["s_body"], t_body, BODY_SWEEP)[..., None] * c["d_body"] \
        + _sweep_weight(c["s_rup"], t_rup, RUP_SWEEP)[..., None] * c["d_rup"]
    if f >= MARK_IN:
        ov = _dots((f - MARK_IN) / FPS, (f - MARK_OUT) / FPS if f >= MARK_OUT else -1.0)
        if ov is not None:
            img = img * (1 - ov[..., 3:4]) + ov[..., :3]
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    prepare()
    for f in (511, 520, 530, 545, 568, 572, 640, 650, 665, 682, 690, 799):
        cv2.imwrite(os.path.join(WORK, f"tailnew_{f}.png"), cv2.cvtColor(frame(f), cv2.COLOR_RGB2BGR))
    print("preview frames written")
