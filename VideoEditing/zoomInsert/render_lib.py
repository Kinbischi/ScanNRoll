"""Shared data + paths for the zoom-insert video side project (see ../README.md).

Loads the processed cache once (via dataAnalysisSetup.load), lays the profiles out on the LINEAR print path (as the
first dataAnalysis cell does with PATH_LAYOUT = "linear"), and extracts the cell-1 layers exactly like the viewer
(get_profile_points_for_plot + the same first-point-per-voxel downsampling). Off the analysis pipeline."""
import os
import shutil
import sys

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.dirname(HERE)                 # VideoEditing/ (source + output videos, git-ignored)
REPO = os.path.dirname(VIDEO_DIR)
WORK = os.path.join(VIDEO_DIR, "work")            # generated frames + caches (git-ignored)
os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, REPO)
pv.OFF_SCREEN = True

import profile3Dplotting as p3d  # noqa: E402
from dataAnalysisSetup import load  # noqa: E402
from profileProcessingAlgorithms import _contiguous_runs, _segment_mask, profile_advance_distances  # noqa: E402

VOXEL = 10   # same as dataAnalysisSetup.VOXEL_SIZE (profile units, 0.01 mm)
_ANACONDA_FFMPEG = r"C:\Users\zimme\anaconda3\pkgs\ffmpeg-4.2.2-he774522_0\Library\bin\ffmpeg.exe"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or _ANACONDA_FFMPEG  # needs libx264 + aac

_raw, processed = load()
zpath = np.cumsum(profile_advance_distances(processed))  # linear layout: profile i at world (x, height, zpath[i])


def layer_points(z_lo: float, z_hi: float, profile_step: int = 1, point_step: int = 2,
                 category: str = "profile", section: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World points of one cell-1 layer for the profiles with zpath in [z_lo, z_hi], plus each point's profile index.
    The voxel grid is absolute, so a whole-profile subset keeps exactly the points the viewer draws."""
    a, b = np.searchsorted(zpath, [z_lo, z_hi])
    per = p3d.get_profile_points_for_plot(processed[a:b], 1, point_step, category=category, section=section)
    out, owner = [], []
    for j, arr in enumerate(per):
        i = a + j
        if arr.shape[0] and i % profile_step == 0:  # profile_step counted on the global index, like the viewer
            out.append(arr + np.array([0.0, 0.0, zpath[i]]))
            owner.append(np.full(arr.shape[0], i))
    if not out:
        return np.empty((0, 3)), np.empty(0, int)
    pts, own = np.vstack(out), np.concatenate(owner)
    key = np.floor(pts / VOXEL).astype(np.int64)
    key -= key.min(axis=0)
    dims = key.max(axis=0) + 1
    _, idx = np.unique(np.ravel_multi_index((key[:, 0], key[:, 1], key[:, 2]), dims), return_index=True)
    idx = np.sort(idx)
    return pts[idx], own[idx]


def segment_table() -> np.ndarray:
    """One row per cleaned segment run: z0, z1, segmentShapeStatus (-1 = none), then the first/last z of the body,
    rupture and head (overshoot-peak) phases (NaN if the phase is absent)."""
    rows = []
    for r in _contiguous_runs(_segment_mask(processed)):
        sec = np.array([np.nan if processed[i].segmentSection is None else processed[i].segmentSection for i in r])
        status = processed[r[0]].segmentShapeStatus
        spans = []
        for code in (1.0, 2.0, 3.0):  # body, rupture, head (segmentShape._SECTION_*)
            hit = np.flatnonzero(sec == code)
            spans += [zpath[r[hit[0]]], zpath[r[hit[-1]]]] if hit.size else [np.nan, np.nan]
        rows.append([zpath[r[0]], zpath[r[-1]], -1.0 if status is None else float(status), *spans])
    return np.array(rows)
