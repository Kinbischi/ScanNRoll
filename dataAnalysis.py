# %% Imports and configuration
# Cell-based (# %%) plot workbench. Run cell-by-cell in VS Code's Interactive Window /
# Jupyter (Shift+Enter), or as a plain script (`python dataAnalysis.py`) — the # %%
# markers are ordinary comments. Processing lives in profileProcessing.py; this file only
# loads and visualises. Run profileProcessing.py first if the processed cache is stale.
import pyvista as pv

import profile3Dplotting
from profileLoading import load_profiles

# In VS Code's Interactive Window / Jupyter, PyVista auto-detects the kernel and renders
# a static (non-interactive) image (its 'trame' inline backend isn't installed). Forcing
# notebook=False makes show() pop the native interactive desktop window instead (it blocks
# the cell until you close the window). Harmless when run as a plain script. Install
# trame + trame-vtk and use pv.set_jupyter_backend("trame") if you want interactive INLINE.
pv.global_theme.notebook = False

RAW_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor.h5"
PROCESSED_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor_processed.h5"

# Raw index range [START:END) for the "before" view (0-based, END exclusive; None -> end).
# Keep this matching the range profileProcessing.py wrote to the cache so raw and
# processed line up (needed for the overlay cell).
START_PROFILE = 58000
END_PROFILE = None

# 3D subsampling for the dense clouds: draw every PROFILE_STEP-th profile and every
# POINT_STEP-th point (points drawn ~ total / (PROFILE_STEP * POINT_STEP)).
PROFILE_STEP = 1
POINT_STEP = 1

# raw       : unprocessed x/z straight from the sensor HDF5 (not levelled)
# processed : the levelled cache written by profileProcessing.py (peaks / width / isFlat)
raw = load_profiles(RAW_FILE, START_PROFILE, END_PROFILE)
processed = load_profiles(PROCESSED_FILE)
print(f"{len(raw)} raw / {len(processed)} processed profiles")


# %% 3D - overlay raw (grey), processed (green), floor baselines (red), z=0 reference (yellow)
# Requires raw and processed to be the SAME profiles: keep START_PROFILE/END_PROFILE
# matching the range profileProcessing.py wrote to the cache.
assert len(raw) == len(processed), (
    f"raw ({len(raw)}) and processed ({len(processed)}) differ; align START_PROFILE/"
    "END_PROFILE with the range profileProcessing.py used."
)
pl = profile3Dplotting.plottingClass(len(raw))
pl.plot(raw, "profile", "grey", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "baseline", "red")  # fitted floor lines from stored m/b (reprocess cache to populate)
pl.plot(processed, "zeroBaseline", "yellow")  # z=0 reference (uniform leveling target) for the floor/bead split
pl.show()


# %% 3D - points coloured by category: floor (brown) vs profile/bead (green)
# floorMask comes from the cache (categorize_floor_points runs in the pipeline).
pl = profile3Dplotting.plottingClass(len(processed))
pl.plot(processed, "profile", "saddlebrown", category="floor", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 3D - profile (bead) points only, floor removed
pl = profile3Dplotting.plottingClass(len(processed))
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 3D - bead points with both width methods marked (needs a reprocessed cache)
# widthPoints (slope-peak method) in red, beadWidthPoints (outer bead points) in blue — both
# enlarged and sphere-rendered so the two chosen points stand out. The marker indices come from
# find_smooth_slope + width_from_smoothed_slope (peaks) and width_from_bead_edges (beadWidthIdx).
pl = profile3Dplotting.plottingClass(len(processed))
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "widthPoints", "red", size=15, spheres=True)       # slope-peak method
pl.plot(processed, "beadWidthPoints", "blue", size=15, spheres=True)  # outer-bead-point method
pl.show()


# %% 3D - bead heat-map with a live feature selector (needs a reprocessed cache for area/shoelaceArea)
# Colour the bead points by a per-profile feature; click a button in the left-edge panel to switch
# the active feature in real time. The colour bar rescales per feature (area ~1e6 vs width ~1e3),
# flat profiles have no bead points so they drop out, and NaN values show grey. Interactive
# window only (the buttons need the native VTK window).
pl = profile3Dplotting.plottingClass(len(processed))
pl.plot_feature_heatmap(processed, features=("beadWidth", "width", "area", "shoelaceArea"),
                        initial="beadWidth", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()

