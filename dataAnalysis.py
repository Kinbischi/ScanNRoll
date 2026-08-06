# %% Imports & config — run once (processing lives in profileProcessing.py; run it first if the cache is stale)
import pyvista as pv

import profile3Dplotting
from datasetConfig import PROCESSED_FILE
from featureComparison import compare_features
from plcData import PLC_COLUMNS
from profileLoading import load_profiles, read_file_attrs
from profileProcessingAlgorithms import unlevel_profiles

pv.global_theme.notebook = False  # show() pops the native interactive window (not a static inline image)

PROFILE_STEP = 3   # draw every Nth profile  (3D subsampling; points ~ total / (PROFILE_STEP * POINT_STEP))
POINT_STEP = 2     # draw every Nth point
VOXEL_SIZE = 10    # keep one point per cube of this edge (profile units, 0.01 mm; None = off) — see plottingClass

processed = load_profiles(PROCESSED_FILE)               # levelled columnar cache
_attrs = read_file_attrs(PROCESSED_FILE)
raw = unlevel_profiles(processed, float(_attrs["level_angle"]), float(_attrs["level_offset"]))  # "before" overlay
print(f"{len(raw)} raw (reconstructed) / {len(processed)} processed profiles")


# %% 3D overlay — raw (grey) vs processed (green) + floor baselines (red), z=0 ref (yellow)
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot(raw, "profile", "grey", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "baseline", "red")
pl.plot(processed, "zeroBaseline", "yellow")
pl.show()


# %% 3D category — floor (brown) vs filament (green)
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot(processed, "profile", "saddlebrown", category="floor", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 3D width markers — slope-peak (red) vs filament-edge (blue)   (needs a reprocessed cache)
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot(processed, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.plot(processed, "widthPoints", "red", size=15, spheres=True)
pl.plot(processed, "filamentWidthPoints", "blue", size=15, spheres=True)
pl.show()


# %% 3D heat-map — colour filament by a feature; live feature + scale-mode buttons   (needs a reprocessed cache)
GEOMETRY_FEATURES = ("width", "filamentWidth", "filamentHeight", "filamentHeightSmooth", "area", "shoelaceArea",
                     "segmentVolume", "sliceVolume", "segmentLength")
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot_feature_heatmap(processed, features=GEOMETRY_FEATURES + PLC_COLUMNS,
                        initial="filamentWidth", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 3D defect length — colour pure-floor gaps by their length (floor points)   (needs a reprocessed cache)
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot_feature_heatmap(processed, features=("defectLength",), initial="defectLength",
                        category="floor", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 2D feature-vs-time — normalized overlay; toggle/smooth curves   (needs a GUI backend)
compare_features(processed,
                 features=("printHeadTorque", "width", "filamentHeight", "area",
                           "viscoPump1_VMAflow", "mortarPumpFlow", "pressurePipeStart", "rollerbandSpeed"),
                 initial=("printHeadTorque", "width", "viscoPump1_VMAflow"),
                 time_unit="min", profile_step=5)


# %% [TUNING — safe to delete] dial the floor/filament split live (shallow copies; pipeline untouched)
import copy as _copy
from profileProcessingAlgorithms import categorize_floor_points, grow_profile_points

# raise GROW_THRESHOLD / lower FILL_GAP to trim filament bleed into the floor (cache built with 20 / 5)
GROW_THRESHOLD    = 150     # z (0.01 mm) to grow the filament down to (HIGHER = less edge bleed)
FILL_GAP          = 3       # bridge interior floor gaps up to this many points (LOWER = less bleed)
SEED_THRESHOLD    = 300     # z to seed a confident filament point (HIGHER = fewer seeds)
MIN_SEED_LEN      = 3       # a filament core must span >= this many points
SEED_USE_BASELINE = True    # seed above each profile's own floor fit (as the pipeline does)
GROW_USE_BASELINE = False   # grow ref: False = uniform median z (cache); True = each profile's own floor fit
TUNE_SLICE        = slice(None)  # focus a range for a snappier re-plot, e.g. slice(0, 8000)

_tuned = [_copy.copy(p) for p in processed[TUNE_SLICE]]  # shallow: own floorMask, shares x/z/m/b
categorize_floor_points(_tuned, threshold=SEED_THRESHOLD, use_profile_baseline=SEED_USE_BASELINE)
grow_profile_points(_tuned, low_threshold=GROW_THRESHOLD, min_seed_length=MIN_SEED_LEN,
                    max_gap=FILL_GAP, use_profile_baseline=GROW_USE_BASELINE)
_filament = sum(int((~p.floorMask).sum()) for p in _tuned if p.floorMask is not None)
_orig = sum(int((~p.floorMask).sum()) for p in processed[TUNE_SLICE] if p.floorMask is not None)
print(f"filament points over {len(_tuned)} profiles:  tuned {_filament:,}  vs  cache {_orig:,}")

_pl = profile3Dplotting.plottingClass(_tuned, voxel_size=VOXEL_SIZE)
_pl.plot(_tuned, "profile", "saddlebrown", category="floor", profile_step=PROFILE_STEP, point_step=POINT_STEP)
_pl.plot(_tuned, "profile", "green", category="profile", profile_step=PROFILE_STEP, point_step=POINT_STEP)
_pl.show()

# Apply settled values in the pipeline, then reprocess: SEED/GROW_USE_BASELINE -> *_USE_PROFILE_BASELINE
# (profileProcessing.py); SEED_THRESHOLD/GROW_THRESHOLD/MIN_SEED_LEN/FILL_GAP -> FLOOR_POINT_THRESHOLD/
# PROFILE_GROW_THRESHOLD/MIN_SEED_LENGTH/PROFILE_FILL_GAP (profileProcessingAlgorithms.py).

# %%
