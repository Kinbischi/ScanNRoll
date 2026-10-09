"""Orbit camera parametrisation for the zoom insert: (target T[3], azimuth, elevation, log-distance, roll).

Forward f = (cos el sin az, -sin el, cos el cos az) (az from +z toward +x, el = downward look angle); camera
C = T - dist * f; roll rotates the camera up about f, measured from the world-up-aligned up vector.
Pose vectors (insert_renderer): rotvec(3), C(3), view angle, s_z, s_y."""
import numpy as np
from scipy.spatial.transform import Rotation

WORLD_UP = np.array([0.0, 1.0, 0.0])


def _frame(az: float, el: float, roll: float) -> np.ndarray:
    f = np.array([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)])
    r0 = np.cross(f, WORLD_UP)
    r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, f)
    up = np.cos(roll) * u0 + np.sin(roll) * r0
    right = np.cross(f, up)
    return np.vstack([right, -up, f])  # rows: camera x (right), y (down), z (forward) in world


def pose_from_params(p: np.ndarray, view_angle: float, s_z: float, s_y: float) -> np.ndarray:
    """p = (Tx, Ty, Tz, az, el, logdist, roll) -> 9-vector pose."""
    T, az, el, ld, roll = p[:3], p[3], p[4], p[5], p[6]
    R = _frame(az, el, roll)
    C = T - np.exp(ld) * R[2]
    return np.concatenate([Rotation.from_matrix(R).as_rotvec(), C, [view_angle, s_z, s_y]])


def params_from_pose(x: np.ndarray, T_hint: np.ndarray) -> np.ndarray:
    """Inverse: target = the point on the view ray closest to T_hint."""
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    C, f = x[3:6], R[2]
    dist = float((T_hint - C) @ f)
    T = C + dist * f
    el = float(np.arcsin(-f[1]))
    az = float(np.arctan2(f[0], f[2]))
    r0 = np.cross(f, WORLD_UP)
    r0 /= np.linalg.norm(r0)
    u0 = np.cross(r0, f)
    up = -R[1]
    roll = float(np.arctan2(up @ r0, up @ u0))
    return np.array([*T, az, el, np.log(dist), roll])


def smootherstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def hermite_path(times: np.ndarray, keys: np.ndarray, t: np.ndarray) -> np.ndarray:
    """C1 cubic Hermite through keyframes (rows of `keys` at `times`), Catmull-Rom tangents inside, zero
    velocity at both ends -> starts and ends at rest, never stops in between."""
    n = len(times)
    tang = np.zeros_like(keys)
    for i in range(1, n - 1):
        tang[i] = (keys[i + 1] - keys[i - 1]) / (times[i + 1] - times[i - 1])
    out = np.empty((len(t), keys.shape[1]))
    for j, tt in enumerate(t):
        i = int(np.clip(np.searchsorted(times, tt, side="right") - 1, 0, n - 2))
        h = times[i + 1] - times[i]
        s = (tt - times[i]) / h
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        out[j] = h00 * keys[i] + h10 * h * tang[i] + h01 * keys[i + 1] + h11 * h * tang[i + 1]
    return out
