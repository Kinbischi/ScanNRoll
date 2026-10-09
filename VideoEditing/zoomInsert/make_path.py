"""Camera path for the zoom insert (shared by versions A and B): orbit keyframes -> per-frame pose vectors.

Inputs (fitted once, see ../README.md): pose_scaled_zscale.npy = the original close-up camera (9-vector: rotvec,
centre, vertical view angle, s_z, s_y); orbit_close.npy = the same as orbit parameters + a target on segment 49."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import path_lib as pth  # noqa: E402
import render_lib as rl  # noqa: E402

FPS = 30
SEGMENT = 49   # the segment the close-up shows (identified from its body/rupture phase pattern)
x_close = np.load(os.path.join(HERE, "pose_scaled_zscale.npy"))
VA, SZ, SY = x_close[6], x_close[7], x_close[8]
orb = np.load(os.path.join(HERE, "orbit_close.npy"))
p0, T_mid = orb[:7], orb[7:]
ZREF = 1727000.0
seg = rl.segment_table()[SEGMENT]
z0, z1, b0, b1 = seg[0], seg[1], seg[3], seg[4]


def disp_z(z: float) -> float:
    return ZREF + (z - ZREF) * SZ


# keyframe targets on the bead (displayed coords): x = bead centre, y = ~55 % of the bead height (scaled)
T1 = np.array([T_mid[0], T_mid[1], disp_z(b0 + 0.15 * (b1 - b0))])
T2 = np.array([T_mid[0], T_mid[1], disp_z(b0 + 0.30 * (b1 - b0))])
T3 = np.array([T_mid[0], T_mid[1], disp_z(b0 + 0.45 * (b1 - b0))])
D0 = np.exp(p0[5])
KEYS = np.array([
    p0,                                                                        # K0 close-up (exact)
    [*T1, np.radians(14.0), np.radians(27.0), np.log(0.26 * D0), 0.0],        # K1 zoomed in, looking along the bead
    [*T2, np.radians(24.0), np.radians(25.0), np.log(0.25 * D0), 0.0],        # K2 slow drift along the belt
    [*T3, np.radians(55.0), np.radians(21.0), np.log(0.24 * D0), 0.0],        # K3 orbit to the side (comb view)
    p0,                                                                        # K4 back to the close-up
])
TIMES = np.array([0.0, 2.2, 4.1, 5.4, 7.6])
N = int(round(TIMES[-1] * FPS)) + 1  # include both end poses (first = last = close-up)


def frame_poses() -> np.ndarray:
    t = np.arange(N) / FPS
    params = pth.hermite_path(TIMES, KEYS, t)
    return np.array([pth.pose_from_params(q, VA, SZ, SY) for q in params]), params


if __name__ == "__main__":
    poses, params = frame_poses()
    assert np.allclose(poses[0], x_close, atol=1e-6) and np.allclose(poses[-1], x_close, atol=1e-6)
    sp = np.linalg.norm(np.diff(poses[:, 3:6], axis=0), axis=1)
    print(f"{N} frames; camera speed (units/frame) min {sp.min():.1f} max {sp.max():.1f}; "
          f"never stops inside: min speed over frames 5..N-6 = {sp[5:-5].min():.1f}")
    print(f"distance range {np.exp(params[:, 5]).min():.0f}..{np.exp(params[:, 5]).max():.0f}; "
          f"elevation {np.degrees(params[:, 4]).min():.1f}..{np.degrees(params[:, 4]).max():.1f} deg")
    np.save(os.path.join(rl.WORK, "path_poses.npy"), poses)
