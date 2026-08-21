"""Shared setup for the `dataAnalysis` plot workbench: imports, display config, tunable constants, and a
cached data loader — so every `dataAnalysis` cell can be run first in a fresh kernel.

Each cell starts with `from dataAnalysisSetup import load, ...` + `raw, processed = load()`. Importing this
module (once per kernel) sets the PyVista + matplotlib (Qt) display config; `load()` reads the processed
cache and reconstructs raw ("before") once and caches it, so repeated cell runs are instant. This is NOT a
processing module — it only wires up the workbench (no `process_profiles` here); the cache is built by
`profileProcessing.py`.
"""
import pyvista as pv

import profile3Dplotting
from datasetConfig import PROCESSED_FILE
from featureComparison import compare_features
from featurePlcTrends import DEFAULT_FEATURES, plot_feature_plc_trends
from plcData import ALL_PLC_COLUMNS
from profileLoading import load_profiles, read_file_attrs
from profileProcessingAlgorithms import unlevel_profiles

pv.global_theme.notebook = False  # show() pops the native interactive window (not a static inline image)

# Interactive matplotlib backend so the 2D plot widgets (checkboxes / radios in compare_features and
# plot_feature_plc_trends) work — otherwise the 2D cells render a static image. `IPython.get_ipython()`
# returns the running kernel (or None outside IPython), so this configures the backend even from a module.
try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "qt")
except (ImportError, AttributeError):
    pass

PROFILE_STEP = 3   # draw every Nth profile  (3D subsampling; points ~ total / (PROFILE_STEP * POINT_STEP))
POINT_STEP = 2     # draw every Nth point
VOXEL_SIZE = 10    # keep one point per cube of this edge (profile units, 0.01 mm; None = off) — see plottingClass

# Per-segment shape features for the 2D stepwise plot (thinning / rupture / startup; one value per segment)
# — compared per speed level. The body steadiness values + segmentRuptures are heat-map-only (see
# HEATMAP_FEATURES), so they are excluded here; the stepwise plot focuses on the thinning rates + geometry.
SEGMENT_SHAPE_FEATURES = ("segmentBodyAreaThinning", "segmentBodyWidthThinning", "segmentBodyHeightThinning",
                          "segmentCriticalArea", "segmentRuptureLength", "segmentHeadOvershoot")

# y-features for the stepwise feature-vs-PLC cell (+ the derived pipePressureDifference, per level).
# Both width measures (flank/outer) and both heights (p95/smooth) are included, as in the heat map / other 2D plots.
PLC_FEATURES = ("widthFlank", "widthOuter", "heightP95", "heightSmooth", "areaSimpson", "areaShoelace",
                "segmentVolume", "segmentLength", "defectLength", "pipePressureDifference",
                *SEGMENT_SHAPE_FEATURES)

# Heat-map features: geometry + PLC colour the filament points; defectLength, the 0/1 flags, the segmentSection
# phase code and segmentShapeStatus colour all points. bodyThinning + segFlags expand to sub-panels; only
# segmentShapeStatus is categorical.
HEATMAP_FEATURES = ("widthFlank", "widthOuter", "heightP95", "heightSmooth", "areaSimpson", "areaShoelace",
                    "segmentVolume", "segmentLength", "defectLength", "segmentHeadOvershoot",
                    "segmentBodyAreaThinning", "segmentBodyWidthThinning", "segmentBodyHeightThinning",
                    "segmentBodyAreaSteadiness", "segmentBodyWidthSteadiness", "segmentBodyHeightSteadiness",
                    "segmentCriticalArea", "segmentRuptureLength",   # segmentRuptures -> read off segmentShapeStatus
                    "isNotFlat", "isSegment", "isContinuousFilament",
                    "segmentShapeStatus", "segmentSection")

_cache: dict = {}  # in-process cache of the loaded data (so repeated cell runs don't reload)


def load(force: bool = False) -> "tuple[list, list]":
    """Load the processed cache + reconstruct raw ("before"), caching the result. Returns (raw, processed).

    Cached in-process, so running any cell that calls `load()` is instant after the first load. Pass
    `force=True` to re-read the cache from disk (e.g. after reprocessing) without restarting the kernel.
    """
    if force or "processed" not in _cache:
        processed = load_profiles(PROCESSED_FILE)                # levelled columnar cache
        attrs = read_file_attrs(PROCESSED_FILE)
        raw = unlevel_profiles(processed, float(attrs["level_angle"]), float(attrs["level_offset"]))  # "before"
        print(f"{len(raw)} raw (reconstructed) / {len(processed)} processed profiles")
        _cache["raw"], _cache["processed"] = raw, processed
    return _cache["raw"], _cache["processed"]


# Public workbench surface: what the dataAnalysis cells import from here (data loader + the plot entry points,
# re-exported modules/constants, and the tunable config). Declared so re-exports don't read as unused imports.
__all__ = ["load", "profile3Dplotting", "compare_features", "plot_feature_plc_trends",
           "ALL_PLC_COLUMNS", "DEFAULT_FEATURES", "PROFILE_STEP", "POINT_STEP", "VOXEL_SIZE",
           "SEGMENT_SHAPE_FEATURES", "PLC_FEATURES", "HEATMAP_FEATURES"]
