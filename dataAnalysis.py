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


# %% Load raw ("before") and processed ("after") — both from files, no processing here
# raw       : unprocessed x/z straight from the sensor HDF5 (not levelled)
# processed : the levelled cache written by profileProcessing.py (peaks / width / isFlat)
raw = load_profiles(RAW_FILE, START_PROFILE, END_PROFILE)
processed = load_profiles(PROCESSED_FILE)
print(f"{len(raw)} raw / {len(processed)} processed profiles")


# %% 3D - overlay raw (grey) and processed (green) in one scene
# Requires raw and processed to be the SAME profiles: keep START_PROFILE/END_PROFILE
# matching the range profileProcessing.py wrote to the cache.
assert len(raw) == len(processed), (
    f"raw ({len(raw)}) and processed ({len(processed)}) differ; align START_PROFILE/"
    "END_PROFILE with the range profileProcessing.py used."
)
pl = profile3Dplotting.plottingClass(len(raw))
pl.plot(raw, "profile", "grey", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()

