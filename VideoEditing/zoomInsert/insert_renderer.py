"""Renderer for the zoom insert: the cell-1 scene (linear path; floor + filament, optional phase layers) drawn
directly in video space (1920x1080, square pixels) with the camera + data scaling fitted to the existing close-up.

Pose vector x (9): rotvec(3), camera centre C(3) [world], vertical view angle (deg), s_z, s_y — the data is
scaled about (0, 0, ZREF) by (1, s_y, s_z), which is how the existing video's geometry was matched (it appears
~23 % taller and ~1.8 % longer along the belt than the processed data at true scale)."""
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_lib as rl  # noqa: E402

ZREF = 1727000.0           # world z the data scaling is anchored at (segment-49 middle)
W, H = 1920, 1080


class InsertScene:
    def __init__(self, z_lo: float, z_hi: float, point_size: float = 5.3, phases: bool = False,
                 profile_step: int = 1, point_step: int = 2, supersample: int = 1):
        self.ss = supersample  # render at ss x the video size (point size scaled too); callers downsample
        point_size = point_size * supersample
        self.pl = pv.Plotter(off_screen=True, window_size=[W * supersample, H * supersample])
        self.pl.set_background("white")
        self.acts = {}
        kw = dict(profile_step=profile_step, point_step=point_step)
        floor, _ = rl.layer_points(z_lo, z_hi, category="floor", **kw)
        layers = [("floor", floor, "saddlebrown")]
        if phases:
            fil = np.vstack([rl.layer_points(z_lo, z_hi, category="profile", section=s, **kw)[0]
                             for s in ("unclassified", "head")])
            layers += [("fil", fil, "green"),
                       ("body", rl.layer_points(z_lo, z_hi, category="profile", section="body", **kw)[0], "royalblue"),
                       ("rup", rl.layer_points(z_lo, z_hi, category="profile", section="rupture", **kw)[0], "crimson")]
        else:
            layers += [("fil", rl.layer_points(z_lo, z_hi, category="profile", **kw)[0], "green")]
        for name, pts, col in layers:
            self.acts[name] = self.pl.add_points(pv.PolyData(pts), color=col, point_size=point_size,
                                                 render_points_as_spheres=False)
        self.n_points = sum(a.mapper.dataset.n_points for a in self.acts.values())

    def set_pose(self, x: np.ndarray) -> np.ndarray:
        """Apply pose/scaling vector x; returns the rotation matrix (rows = camera axes, OpenCV convention)."""
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        for a in self.acts.values():
            a.SetOrigin(0.0, 0.0, ZREF)
            a.SetScale(1.0, float(x[8]), float(x[7]))
        cam = self.pl.camera
        cam.position = tuple(x[3:6])
        cam.focal_point = tuple(x[3:6] + 50000.0 * R[2])
        cam.up = tuple(-R[1])
        cam.view_angle = float(x[6])
        self.pl.renderer.ResetCameraClippingRange()
        return R

    def render(self, x: np.ndarray) -> np.ndarray:
        self.set_pose(x)
        self.pl.render()
        return np.asarray(self.pl.screenshot(return_img=True))[..., :3].copy()

    def world_to_scene(self, P: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Data-world points -> displayed (scaled) points, for overlays."""
        Q = np.array(P, float).copy()
        Q[..., 1] *= x[8]
        Q[..., 2] = ZREF + (Q[..., 2] - ZREF) * x[7]
        return Q

    def project(self, Pdisp: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Displayed points -> pixel coords (u, v) + camera depth, matching VTK's camera (square pixels)."""
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        pc = (Pdisp - x[3:6]) @ R.T
        f = (H / 2) / np.tan(np.radians(x[6]) / 2)
        u = f * pc[:, 0] / pc[:, 2] + W / 2
        v = f * pc[:, 1] / pc[:, 2] + H / 2
        return np.column_stack((u, v)), pc[:, 2]
