# %% Imports and configuration
# Cell-based (# %%) plot workbench. Run cell-by-cell in VS Code's Interactive Window /
# Jupyter (Shift+Enter), or as a plain script (`python dataAnalysis.py`) — the # %%
# markers are ordinary comments. Processing lives in profileProcessing.py; this file only
# loads and visualises. Run profileProcessing.py first if the processed cache is stale.
import pyvista as pv

import profile3Dplotting
from datasetConfig import PROCESSED_FILE
from featureComparison import compare_features
from plcData import PLC_COLUMNS
from profileLoading import load_profiles, read_file_attrs
from profileProcessingAlgorithms import unlevel_profiles

# In VS Code's Interactive Window / Jupyter, PyVista auto-detects the kernel and renders
# a static (non-interactive) image (its 'trame' inline backend isn't installed). Forcing
# notebook=False makes show() pop the native interactive desktop window instead (it blocks
# the cell until you close the window). Harmless when run as a plain script. Install
# trame + trame-vtk and use pv.set_jupyter_backend("trame") if you want interactive INLINE.
pv.global_theme.notebook = False

# PROCESSED_FILE comes from datasetConfig (switch datasets there).

# 3D subsampling for the dense clouds: draw every PROFILE_STEP-th profile and every
# POINT_STEP-th point (points drawn ~ total / (PROFILE_STEP * POINT_STEP)).
PROFILE_STEP = 3
POINT_STEP = 2

# Voxel downsampling (see plottingClass): overlay a 3-D grid of cubes and keep ONE point per cube,
# thinning dense areas to cut lag/overdraw/memory. Composes with PROFILE_STEP / POINT_STEP above.
# VOXEL_SIZE = cube edge in profile units (0.01 mm), so 100 = a 1 mm cube; larger = fewer points;
# None = off. NB the cubes bin height too, so it is not a uniform on-screen spacing: steep bead
# flanks keep points stacked ~VOXEL_SIZE apart in height (expected, not a bug).
VOXEL_SIZE = 10

# processed : the levelled columnar cache written by profileProcessing.py (points / width / PLC / ...)
# raw       : unprocessed x/z (not levelled), the "before" for the overlay cell. Reconstructed for
#   free by inverting the stored uniform leveling transform (level_angle / level_offset), so no second
#   (slow) load of the raw file is needed — every columnar cache carries the transform.
processed = load_profiles(PROCESSED_FILE)
_attrs = read_file_attrs(PROCESSED_FILE)
raw = unlevel_profiles(processed, float(_attrs["level_angle"]), float(_attrs["level_offset"]))
print(f"{len(raw)} raw (reconstructed) / {len(processed)} processed profiles")


# %% 3D - overlay raw (grey), processed (green), floor baselines (red), z=0 reference (yellow)
# raw is reconstructed from processed (same profiles, 1:1), so the two clouds overlay directly.
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)  # path from processed (carries rollerbandSpeed); raw is aligned
pl.plot(raw, "profile", "grey", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "baseline", "red")  # fitted floor lines from stored m/b (reprocess cache to populate)
pl.plot(processed, "zeroBaseline", "yellow")  # z=0 reference (uniform leveling target) for the floor/bead split
pl.show()


# %% 3D - points coloured by category: floor (brown) vs profile/bead (green)
# floorMask comes from the cache (categorize_floor_points runs in the pipeline).
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot(processed, "profile", "saddlebrown", category="floor", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 3D - bead points with both width methods marked (needs a reprocessed cache)
# widthPoints (flank-foot method) in red, beadWidthPoints (outer bead points) in blue — both
# enlarged and sphere-rendered so the two chosen points stand out. The marker indices come from
# width_from_smoothed_slope (peaks) and width_from_bead_edges (beadWidthIdx).
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "widthPoints", "red", size=15, spheres=True)       # slope-peak method
pl.plot(processed, "beadWidthPoints", "blue", size=15, spheres=True)  # outer-bead-point method
pl.show()


# %% 3D - bead heat-map with a live feature selector (needs a reprocessed cache for area / PLC)
# Colour the bead points by a per-profile feature; click a button in the left-edge panel to switch
# the active feature in real time. Geometry features are grouped (width / height / area) and share a
# colour range in mm / mm^2; each PLC machine-log channel is its own group in its native units. Flat
# profiles have no bead points so they drop out, and NaN values show grey. Interactive window only
# (the buttons need the native VTK window).
GEOMETRY_FEATURES = ("width", "beadWidth", "beadHeight", "beadHeightSmooth", "area", "shoelaceArea")
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot_feature_heatmap(processed, features=GEOMETRY_FEATURES + PLC_COLUMNS,
                        initial="beadWidth", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 2D - compare per-profile features over time (normalized overlay, toggle curves with checkboxes)
# Overlay several geometry / PLC features on one axis, each robustly normalized to 0-1 (1-99 pct, so
# outliers don't flatten the curve). Each legend entry shows scale[lo-hi] (what maps to 0-1, with
# unit) and true[min-max] (real extremes -> a clipped outlier shows up as true max >> scale hi); its
# text is bold while the curve is shown. Left panel: the "show" column toggles each curve; the
# "smooth" column moving-average-smooths selected curves, with the slider setting the window (1 = raw).
# Needs a GUI backend (run as a script or %matplotlib qt); VS Code's inline backend renders a static
# image. profile_step decimates for responsiveness.
compare_features(processed,
                 features=("printHeadTorque", "width", "beadHeight", "area",
                           "viscoPump1_VMAflow", "mortarPumpFlow", "pressurePipeStart", "rollerbandSpeed"),
                 initial=("printHeadTorque", "width", "viscoPump1_VMAflow"),
                 time_unit="min", profile_step=5)


# %% [TUNING - safe to delete] dial the floor/bead split without reprocessing
# Re-runs the seed+grow categorization on the LOADED cache with the knobs below and re-plots floor
# (brown) vs bead (green), to fix floor points near the bead being mislabelled as bead. Non-destructive:
# it works on shallow copies, so the cache's floorMask (used by the cells above) is untouched, and it
# does NOT change the processing pipeline. Once the split looks right, copy the values into the pipeline
# (mapping at the bottom) and reprocess once. This whole cell -- imports included -- is self-contained;
# delete it to remove every trace.
import copy as _copy
from profileProcessingAlgorithms import categorize_floor_points, grow_profile_points

# Knobs. Primary levers for "floor points next to the bead get labelled bead": raise GROW_THRESHOLD
# and/or lower FILL_GAP -- both monotonically trim the bleed. The cache was built with GROW_THRESHOLD=20,
# FILL_GAP=5, GROW_USE_BASELINE=False, so the print below shows the before/after bead-point count.
GROW_THRESHOLD    = 150     # z (~0.01 mm) to grow the bead down to (cache 20; HIGHER = less edge bleed)
FILL_GAP          = 3      # bridge interior floor gaps up to this many points (cache 5; LOWER = less bleed)
SEED_THRESHOLD    = 300    # z to seed a confident bead point (HIGHER = stricter/fewer seeds)
MIN_SEED_LEN      = 3      # a bead core must span >= this many points (rejects lone spikes)
SEED_USE_BASELINE = True   # seed above each profile's own floor fit (as the pipeline already does)
GROW_USE_BASELINE = False  # grow height reference: False = uniform median z (as the cache); True = each
                           #   profile's OWN floor fit (consistent with the seed). Try True if WHOLE floor
                           #   regions on tilted/offset profiles are mislabelled -- but note it re-balances
                           #   both ways (can ADD bead where the cache under-grew), so watch the plot.
TUNE_SLICE        = slice(None)#0, 8000)  # focus a range so re-plotting stays snappy (slice(None) = all)

_tuned = [_copy.copy(p) for p in processed[TUNE_SLICE]]  # shallow: own floorMask, shares x/z/m/b
categorize_floor_points(_tuned, threshold=SEED_THRESHOLD, use_profile_baseline=SEED_USE_BASELINE)
grow_profile_points(_tuned, low_threshold=GROW_THRESHOLD, min_seed_length=MIN_SEED_LEN,
                    max_gap=FILL_GAP, use_profile_baseline=GROW_USE_BASELINE)
_bead = sum(int((~p.floorMask).sum()) for p in _tuned if p.floorMask is not None)
_orig = sum(int((~p.floorMask).sum()) for p in processed[TUNE_SLICE] if p.floorMask is not None)
print(f"bead points over {len(_tuned)} profiles:  tuned {_bead:,}  vs  cache {_orig:,}")

_pl = profile3Dplotting.plottingClass(_tuned, voxel_size=VOXEL_SIZE)
_pl.plot(_tuned, "profile", "saddlebrown", category="floor", profile_step=PROFILE_STEP, point_step=POINT_STEP)
_pl.plot(_tuned, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
_pl.show()

# To apply the settled values, edit the pipeline then run `python profileProcessing.py`:
#   GROW_USE_BASELINE -> GROW_USE_PROFILE_BASELINE   (profileProcessing.py)
#   SEED_USE_BASELINE -> SEED_USE_PROFILE_BASELINE   (profileProcessing.py)
#   SEED_THRESHOLD    -> FLOOR_POINT_THRESHOLD       (profileProcessingAlgorithms.py)
#   GROW_THRESHOLD    -> PROFILE_GROW_THRESHOLD      (profileProcessingAlgorithms.py)
#   MIN_SEED_LEN      -> MIN_SEED_LENGTH             (profileProcessingAlgorithms.py)
#   FILL_GAP          -> PROFILE_FILL_GAP            (profileProcessingAlgorithms.py)

# %%
