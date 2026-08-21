# %% 3D preprocessing (togglable) — raw/processed clouds, floor baselines + z=0, floor/filament, width markers; a left-edge checkbox toggles each layer
# Setup lives in dataAnalysisSetup.py, so any cell can be clicked and run first in a fresh kernel; load() caches.
from dataAnalysisSetup import PROFILE_STEP, POINT_STEP, VOXEL_SIZE, load, profile3Dplotting
raw, processed = load()
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.add_layer_toggles([  # (label, actor(s), label colour, shown at start) — floor + filament category view on by default
    ("raw",         pl.plot(raw, "profile", "grey", profile_step=PROFILE_STEP, point_step=POINT_STEP),       "grey",        False),
    ("processed",   pl.plot(processed, "profile", "green", profile_step=PROFILE_STEP, point_step=POINT_STEP), "green",       False),
    ("baseline",    pl.plot(processed, "baseline", "red"),                                                    "red",         False),
    ("z=0 ref",     pl.plot(processed, "zeroBaseline", "yellow"),                                             "yellow",      False),
    ("floor",       pl.plot(processed, "profile", "saddlebrown", category="floor",
                            profile_step=PROFILE_STEP, point_step=POINT_STEP),                                "saddlebrown", True),
    ("filament",    pl.plot(processed, "profile", "green", category="profile",
                            profile_step=PROFILE_STEP, point_step=POINT_STEP),                                "green",       True),
    ("width flank", pl.plot(processed, "widthFlankPoints", "red", size=15, spheres=True),                     "red",         False),
    ("width outer", pl.plot(processed, "widthOuterPoints", "blue", size=15, spheres=True),                    "blue",        False),
])
pl.show()


# %% 3D heat-map — colour the cloud by any feature (Geometry / Segment / PLC selector); bodyThinning + segFlags expand to sub-panels, segmentShapeStatus is categorical
from dataAnalysisSetup import ALL_PLC_COLUMNS, HEATMAP_FEATURES, PROFILE_STEP, POINT_STEP, VOXEL_SIZE, load, profile3Dplotting
raw, processed = load()
pl = profile3Dplotting.plottingClass(processed, voxel_size=VOXEL_SIZE)
pl.plot_feature_heatmap(processed, features=HEATMAP_FEATURES + ALL_PLC_COLUMNS,
                        initial="widthOuter", profile_step=PROFILE_STEP, point_step=POINT_STEP)
pl.show()


# %% 2D feature-vs-time — all features + PLC vars, category-grouped; one curve -> its real units   (needs a GUI backend)
from dataAnalysisSetup import ALL_PLC_COLUMNS, DEFAULT_FEATURES, compare_features, load
raw, processed = load()
compare_features(processed, features=DEFAULT_FEATURES + ALL_PLC_COLUMNS,
                 initial=("printHeadTorque",), initial_smooth=("printHeadTorque",), smooth_window_init=25,
                 time_unit="min", profile_step=5)


# %% 2D feature-vs-PLC (stepwise) — box/violin per level; toggle features, pick channel, median/mean   (needs a GUI backend)
from dataAnalysisSetup import PLC_FEATURES, load, plot_feature_plc_trends
raw, processed = load()
plot_feature_plc_trends(processed, kind="stepwise", features=PLC_FEATURES,
                        initial_feature="widthFlank", initial_channel="rollerbandSpeed", profile_step=5)


# %% 2D feature-vs-PLC (continuous) — hexbin density + trend; ANY subject on either axis   (needs a GUI backend)
# Shared pool of all features + every PLC channel, grouped by category: x = pick one, y = toggle several.
from dataAnalysisSetup import ALL_PLC_COLUMNS, DEFAULT_FEATURES, load, plot_feature_plc_trends
raw, processed = load()
plot_feature_plc_trends(processed, kind="continuous", features=DEFAULT_FEATURES, channels=ALL_PLC_COLUMNS,
                        initial_feature="widthFlank", initial_channel="pressurePrintHead", profile_step=5)


# %% [TUNING — safe to delete] dial the floor/filament split live (shallow copies; pipeline untouched)
import copy as _copy

from dataAnalysisSetup import PROFILE_STEP, POINT_STEP, VOXEL_SIZE, load, profile3Dplotting
from profileProcessingAlgorithms import categorize_floor_points, grow_profile_points

raw, processed = load()

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
