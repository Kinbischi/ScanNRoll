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
PROFILE_STEP = 1
POINT_STEP = 1

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
