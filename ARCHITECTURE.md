# Architecture

This document describes how the code is structured today — the **as-is** state,
not an aspirational design. It is the reference for understanding the pipeline
before changing anything. For the conventions to follow when changing it, see
[CLAUDE.md](CLAUDE.md); for the backlog of improvements, see [TODO.md](TODO.md).

---

## 1. Overview

The system has two halves that meet at an HDF5 file:

```
 Baumer OX200 ──UDP──►  rawProfileUdpCapturing.py  ──►  HDf5data/*.h5  (raw profile store)
   sensor              (acquisition, standalone)                  │ read
                                                                  ▼
                        profileProcessing.py  ──────►  *_processed.h5  (small processed cache)
                        (process once: level →                    │ read
                         categorise floor/filament → width)           ▼
                                                            dataAnalysis.py
                                                            (plot workbench: cache → 3D)
```

- **Acquisition** (`rawProfileUdpCapturing.py`) is standalone: it shares no imports with the
  analysis code and owns its own `ProfileDataRaw` dataclass tuned to the wire format.
- **Analysis** (everything else) reads the HDF5 store into a different
  `profileData` dataclass and runs the processing / visualisation pipeline.

> The two dataclasses serve different layers and are now named distinctly —
> `ProfileDataRaw` (wire/storage) vs `profileData` (analysis). They share no
> fields and must not be merged.

---

## 2. Components & responsibilities

| Module | Responsibility | Status |
| ------ | -------------- | ------ |
| `rawProfileUdpCapturing.py` | Receive sensor UDP packets, parse the binary protocol, pair Z-profile + measurement blocks, write to HDF5. Owns `MeasurementData` and a wire-format `ProfileDataRaw`. | Active (standalone) |
| `profilePointsClass.py` | Defines the analysis `profileData` dataclass **only** — the pure data model, no processing logic and no project imports. | Active |
| `profileProcessingAlgorithms.py` | The processing functions that operate on lists of `profileData`: rotate, level, smooth, width detection, flatness flag, floor/filament categorisation, area, filament-segment volume, baseline fit, moving average (+ `FLATNESS_RMS_THRESHOLD`), and the along-track spacing helper `profile_advance_distances` (shared with the 3D layout). Imports only `profilePointsClass`. | Active |
| `profileLoading.py` | `load_profiles()` / `save_profiles()` — read/write `profileData` to HDF5. `save_profiles` writes the processed cache as a **columnar table** (padded `[N,L]` points, `[N,2]` index pairs, `[N]` scalar columns, names) for fast bulk loading; `load_profiles` reads that, and reads raw acquisition files (`x`/`z` + `arrival_time`, rest ignored), but **rejects** an old per-group *processed* cache with a "reprocess" error. `read_file_attrs()` returns file-level attributes (e.g. `raw_start`/`raw_end`). | Active |
| `plcData.py` | Load the machine PLC log (YT-Scope CSV) and join it to the profiles by timestamp. Owns the staging `PlcLog` dataclass, `load_plc_csv` (FILETIME→Unix, clock-offset corrected), and `join_plc_to_profiles` (nearest-sample; drops profiles outside the mutual overlap). Imports only `profilePointsClass`. | Active |
| `profile3Dplotting.py` | `plottingClass` — PyVista 3D rendering; builds the serpentine print path (per-profile along-track advance from `rollerbandSpeed` × dt, via `profile_advance_distances`, imported from `profileProcessingAlgorithms`) & tilt angles and places each profile along it. Points are batched into one actor per `plot()` call. Owns `FEATURE_DISPLAY` (feature → unit factor + label), the heat-map's display map. | Active |
| `featureComparison.py` | `compare_features()` — matplotlib 2D overlay comparing several per-profile features over time, each robustly normalised to 0–1 (percentile-clipped so outliers don't flatten it), with a `CheckButtons` panel to toggle curves. Imports `profilePointsClass` + `FEATURE_DISPLAY`. | Active |
| `featurePlcTrends.py` | `plot_feature_plc_trends()` — matplotlib 2D feature-vs-PLC-channel correlation plot, one channel family per call via `kind`: `"stepwise"` (discrete channels → box/violin per level) or `"continuous"` (analogue channels → hexbin density + trend); Spearman r; several features → normalised 0–1 trend lines. Live feature toggles, a channel radio, a median/mean statistic radio, and (stepwise) a box/violin shape radio; constant channels force-shown; idle excluded; fixed x-axis. Segment/defect features are aggregated **per run** in the stepwise view (one point per segment/defect) with `n=` counts, a >3-runs box rule, and >0.5 m / cross-level exclusions. Reuses `featureComparison`'s value/scale helpers, `FEATURE_DISPLAY`, and `profileProcessingAlgorithms._contiguous_runs`. | Active |
| `profileRegistration.py` | Align overlapping profiles in x (ICP / `minimize`), detect left/right/centre profiles, join them into a combined profile. | Legacy (dormant) |
| `datasetConfig.py` | The active experiment's file paths (`RAW_FILE`, `PLC_FILE`, derived `PROCESSED_FILE`) in one place, imported by both entry points so they can't drift. Switch datasets by moving the "ACTIVE" pair; others kept commented. | Active |
| `profileProcessing.py` | **Entry point (process).** Hosts the `process_profiles()` pipeline (composes the algorithm functions in order) and the run script: load raw HDF5 → process → join the PLC log by timestamp (trims to the overlap) → write the processed-HDF5 cache. Run once per dataset / when processing params change. | Active |
| `dataAnalysis.py` | **Entry point (plot workbench).** Cell-based (`# %%`) file: load the processed cache ("after") and reconstruct raw ("before") by inverting the stored leveling transform, then plot flexibly in 3D (PyVista, native window) — raw, processed, and an overlay — plus the 2D feature-vs-time comparison and feature-vs-PLC correlation views. No processing on this path (uses the cache, not `process_profiles`). | Active |
| `LidarProfileAnalysis_oldRegistration.py` | Previous entry point built around the registration path. | Legacy |

---

## 3. Module dependency graph

Legacy modules still use `from <module> import *`; newer/edited code uses explicit imports.

```
 profilePointsClass          base layer — the profileData model only, no project imports
   ▲    ▲    ▲    ▲
   │    │    │    └── plcData                PLC CSV load + timestamp join (imports profilePointsClass)
   │    │    └─────── profile3Dplotting      PyVista plotting (imports profilePointsClass + plcData + profileProcessingAlgorithms)
   │    │                   ▲
   │    │                   └── featureComparison   matplotlib 2D feature overlay (imports FEATURE_DISPLAY)
   │    │                             ▲
   │    │                             └── featurePlcTrends   matplotlib 2D feature-vs-PLC plot
   │    │                                 (imports featureComparison helpers + FEATURE_DISPLAY + PLC_COLUMNS)
   │    └──────────── profileProcessingAlgorithms  processing fns + FLATNESS_RMS_THRESHOLD
   │                        ▲
   └── profileLoading ──────┘               HDF5 I/O (also imports FLATNESS_RMS_THRESHOLD)

 Entry points compose the above (both also import datasetConfig for the file paths):
   profileProcessing → profileLoading + profileProcessingAlgorithms + plcData                        (process)
   dataAnalysis      → profileLoading + profile3Dplotting + plcData + featureComparison + featurePlcTrends   (plot)

 profileRegistration       LEGACY / dormant — imports profilePointsClass (wildcard), off active path
 rawProfileUdpCapturing    standalone — imports only stdlib + numpy + h5py
```

Edges: `profileProcessingAlgorithms`, `profileLoading`, `profile3Dplotting`, and `plcData` each
import `profilePointsClass`; `profileLoading` also imports `profileProcessingAlgorithms`
(`FLATNESS_RMS_THRESHOLD`); `profile3Dplotting` imports `plcData` (`PLC_COLUMNS`) and
`profileProcessingAlgorithms` (`profile_advance_distances`); and
`featureComparison` imports `profilePointsClass` + `profile3Dplotting` (`FEATURE_DISPLAY`); and
`featurePlcTrends` imports `featureComparison` (the `_feature_values`/`_scale_range`/`_normalise`
helpers) + `profile3Dplotting` (`FEATURE_DISPLAY`) + `plcData` (`PLC_COLUMNS`) + `profilePointsClass`
(and `scipy.stats.spearmanr`). The entry points compose these: `profileProcessing` imports
`profileLoading` + `profileProcessingAlgorithms` + `plcData`; `dataAnalysis` imports `profileLoading` +
`profile3Dplotting` + `plcData` + `featureComparison` + `featurePlcTrends`. No cycles.

- `profilePointsClass` is the foundation; everything depends on it.
- The active analysis modules now use **explicit** imports; only the dormant
  `profileRegistration.py` still uses `from x import *`.

---

## 4. Important execution flows

### 4.1 Active analysis pipeline — split into a process step and a plot step

Processing is decoupled from plotting via a small **processed-HDF5 cache** so the
visualisation can be re-run cheaply (the raw 810 MB load + actor build dominated; see
§7). Run the process step once, then iterate on the plot step.

**Process** (`profileProcessing.py`, run once / when params change):
```
load_profiles(RAW_FILE)                        # profileLoading → list[profileData] (+ arrivalTime)
process_profiles(profiles)                     # geometry pipeline (see below)
join_plc_to_profiles(profiles, load_plc_csv(PLC_FILE))  # plcData: attach PLC by time, trim to overlap
save_profiles(profiles, OUT, kind=..., raw_start=…, raw_end=…)  # profileLoading → processed *.h5
```

**Plot** (`dataAnalysis.py`, run freely):
```
load_profiles(PROCESSED_FILE)            # profileLoading  → list[profileData]
plottingClass(profiles, voxel_size=…)    # profile3Dplotting: precompute path + tilt from the profiles
plotter.plot(profiles, "profile", green) # batched: one actor for all profiles
plotter.plot(profiles, "widthPoints", …) # mark width peaks
plotter.show()                           # interactive PyVista window
```

`process_profiles()` runs these steps in order:

1. `rotate_and_shift_uniform` — level **all** profiles by one **median** rotation + shift
   (derived from each profile's floor fit), preserving the real height differences between
   profiles. Also stores each profile's own floor fit in `m`/`b`.
2. `categorize_floor_points` — per-point `floorMask` (floor vs filament) by height, with a symmetric
   **positional prior** (points far from the scan centre need more height to count as filament).
3. `grow_profile_points` — hysteresis: grow the filament from confident seeds into their connected
   lower flanks, then fill small interior gaps.
4. `flag_flat_profiles` — mark a profile flat when it has **no filament points** (floor only). The
   old line-fit-residual method is kept behind `use_line_fit=True`.
5. `width_from_smoothed_slope` → `widthFlank` (+ `widthFlankIdx`) — filament width between the feet of
   the two **outermost** filament flanks: smooths z and |dz/dx| internally, finds the outer flank peaks,
   then walks each flank down to its foot near the floor (so the markers sit at the filament base).
6. `width_from_filament_edges` → `widthOuter` (+ `widthOuterIdx`) — filament width the other way: the
   x-span between the outer filament points.
7. `measure_filament_height` — robust filament height above the `z = 0` median floor, two ways: the 95th
   percentile of the filament points' z → `heightP95`, and the max of a median-smoothed profile →
   `heightSmooth` (both ignore outlier spikes).
8. `measure_filament_area` — cross-sectional filament area over the shared `z = 0` median floor, two
   ways (Simpson integration → `areaSimpson`, shoelace polygon → `areaShoelace`) as a mutual cross-check.

After the geometry pipeline, the entry point runs one more step **outside** `process_profiles`
(it needs the CSV path, and it changes the profile *set*, not just per-profile fields):
`join_plc_to_profiles` (`plcData`) attaches the machine PLC log to each profile by timestamp —
profiles carry Unix-epoch `arrivalTime` (from the raw sensor `arrival_time`), the PLC log carries
Windows FILETIME converted to Unix and corrected for the PLC clock offset (`PLC_CLOCK_OFFSET_S`),
and each profile takes the **nearest-in-time** PLC sample. Profiles outside the two streams'
mutual time overlap are dropped (a contiguous head/tail trim); the surviving raw index span is
stored on the cache as `raw_start`/`raw_end` so the plot workbench can load the matching raw slice.
Then `clean_flat_runs` (`profileProcessingAlgorithms`) de-noises the segment/defect run structure —
bridging gaps shorter than `SEGMENT_MERGE_GAP_MM` (5 mm) and dropping segments shorter than
`MIN_SEGMENT_LENGTH_MM` (10 mm), so a stray flat profile no longer splits a segment and sub-mm blips are
removed (Exp1: 182 → 120 segments). It writes the cleaned result to a **dedicated `isSegment` flag**
(starting from `~isFlat`), leaving the raw `isFlat` (= "no filament points", a `floorMask` summary)
untouched — the two concepts stay separate. It must run here because it, too, needs the physical
distances. Finally `measure_filament_volume` and `measure_run_lengths` (`profileProcessingAlgorithms`)
run — also outside `process_profiles`, because they need the physical inter-profile distances
(`profile_advance_distances`, from `rollerbandSpeed`, populated only by the join) — measuring the cleaned
runs (segments = runs of `isSegment` via `_segment_mask`).
`measure_filament_volume` integrates `areaShoelace` along the print path over each **filament segment**
(a run of non-flat profiles between flat ones), setting `segmentVolume` (the segment total, broadcast
onto its profiles) and `sliceVolume` (each profile's own `area × gap` slab). `measure_run_lengths` sums
the advance over each run and broadcasts the total: `segmentLength` (filament-segment length) and
`defectLength` (length of a pure-floor / no-filament gap).

The plot workbench (`dataAnalysis.py`) draws floor vs filament in two colours (`category=`), flat
profiles highlighted (`flat_colour=`), the floor baselines and a `z = 0` reference
(`"baseline"` / `"zeroBaseline"`), and both width methods' points (`"widthFlankPoints"` /
`"widthOuterPoints"`). It can also colour the cloud by a per-profile feature with a live selector
panel (`plot_feature_heatmap`; the feature buttons are grouped under Geometry / Segment / PLC headers).
Because features live on different point sets, the heat-map **prebuilds one cloud per point set**
(`_feature_pointset`): filament geometry + PLC channels colour the **filament**
points, while `defectLength` and the `isSegment` / `isNotFlat` flags (which sit on floor/gap profiles)
colour **all** points. Switching within a point set only repoints the mapper (instant); crossing point
sets swaps which cloud is drawn (filament-only ↔ all) while keeping the camera, and a single shared
colour bar is re-tied to the active cloud. Selectable features include the geometry measures (width /
height / area, grouped and shown in mm / mm²), the two filament-segment volumes (`segmentVolume` /
`sliceVolume`, in cm³), the run lengths (`segmentLength` / `defectLength` in mm), the 0/1 segment flags,
and each joined PLC channel — set by `FEATURE_DISPLAY` in `profile3Dplotting.py`. A bottom-right radio
group switches the colour scale (linear / log / clip / rank) live.

For comparing features against each other (rather than one at a time in space),
`featureComparison.compare_features` (matplotlib) overlays several as time series on one axis,
each robustly normalised to 0–1 (percentile-clipped so a spike doesn't flatten the curve); the
legend shows both the scale range that maps to 0–1 and the true min–max (revealing clipped
outliers). Left-panel checkboxes toggle each curve ("show") and moving-average-smooth selected
curves ("smooth", with a window slider); the legend keeps every feature's colour and bolds the
shown ones. New `dataAnalysis.py` cell.

For checking whether a feature **correlates with a machine input**, `featurePlcTrends.plot_feature_plc_trends`
(matplotlib) puts a PLC channel on x and per-profile feature(s) on y. It is built for one channel family
at a time via the `kind` argument, so `dataAnalysis.py` opens it as **two cells**: `kind="stepwise"` for
the discrete/setpoint channels (`STEPWISE_CHANNELS`: `rollerbandSpeed`, `printHeadMixxingSpeed`, the two
viscotec pumps) aggregated **per level**, and `kind="continuous"` for the analogue signals
(`CONTINUOUS_CHANNELS`: pressures, torque) drawn as a density cloud (`mortarPumpFlow`/`rollerbandHeight`
are omitted). One feature shown → the rich single-feature view in real units (a **box or violin** per
level — a box/violin shape radio — or a hexbin density + central-per-quantile-bin trend + spread band)
with a Spearman r; two or more → each collapses to one normalised-0–1 central±band trend line on a shared
axis, with each feature's Spearman r in the legend (several boxes/densities can't overlay legibly). A
median/mean radio switches the central statistic everywhere (median+IQR ↔ mean+std) and, in the single
stepwise view, moves the central line on the box/violin. Live `CheckButtons` toggle the features and a
`RadioButtons` list picks the x-channel; **constant channels are force-shown** (the viscotec pumps read 0
in the current cache; their Spearman shows `n/a`). Selectable features are the per-profile geometry
measures plus the broadcast segment/defect aggregates (`segmentVolume`/`segmentLength`/`defectLength`).
In the stepwise view those **run features** are aggregated **per run** — one datapoint per segment/defect
(detected with `profileProcessingAlgorithms._contiguous_runs` on the full profile list, since decimation
would break run contiguity), not per profile — and each level shows an **`n=` count**, draws its
box/violin only where there are **> 3** runs, and **excludes** runs longer than **0.5 m** or that span
more than one level of the active channel (so it generalises to the viscotec channels in future datasets).
The stepwise **x-axis is fixed** to the channel's full level set. Idle profiles (`rollerbandSpeed == 0`)
are excluded so trends reflect actual printing. It reuses `featureComparison`'s value/scale helpers and
`FEATURE_DISPLAY`. Two `dataAnalysis.py` cells (need a GUI backend, like `compare_features`).

Key detail: `width_from_smoothed_slope` cascades box filters (`moving_average`) over `z` (locally —
no stored arrays), takes the gradient and smooths it again, runs `scipy.signal.find_peaks` on that
smoothed slope, keeps the **outermost** peaks (intermediate peaks ignored; `NaN` when fewer than
two), then walks each flank down to its foot. The per-profile alternative (`rotate_pointcloud` +
`translate_floor_to_zero`) is kept but dormant.

### 4.2 Acquisition pipeline (`rawProfileUdpCapturing.py`)

```
run_udp_listener()
  └─ recvfrom() loop:
       parse_header()                    # block_id, frame_type, frame_index
       ├─ measurement packet  → parse_udp_packet_measure() → MeasurementData
       ├─ Z-profile start     → buffer fragment
       └─ Z-profile continuation → reassemble → parse_udp_packet_zProfile()
       pair Z-profile with its MeasurementData (by block_id ±1)
       HDF5ProfileWriter.write_profile()  # group "profile_NNNNNN" per profile
```

Profiles and measurements arrive as separate UDP blocks and are paired by adjacent
`block_id`. Unpaired blocks are buffered in in-memory dicts until their partner
arrives.

### 4.3 Legacy registration pipeline (`LidarProfileAnalysis_oldRegistration.py`)

Per group: rotate / level → `registerAndShiftProfiles()` → `generateJoinedProfile()` →
area calculations. **Dormant** — depends on a `.y` attribute the current `profileData` no
longer has, and the CSV loaders it used (`loadProfiles`/`groupProfiles`) were removed in the
cleanup.

---

## 5. Coordinate-system convention

- Profiles are stored as `x` (across-track position) and `z` (height).
- In the 3D plot, height is currently placed into the **y** slot of the PyVista
  point array (`column_stack((x, z, 0))`), and the print path runs in the x–z
  plane. The code's own comments flag this as confusing.
- **Authoritative rule going forward:** height lives in `z`. Anything reading `.y`
  off a `profileData` is legacy and incorrect (see debt below).

---

## 6. Data storage format (HDF5)

One group per profile, named `profile_NNNNNN`, written by `HDF5ProfileWriter`:

- Datasets: `x`, `z` (the profile points), plus `measurement_*` arrays when present.
- Attributes: `block_id`, `frame_type`, `frame_index`, `source_ip`, timing
  (`timestamp_sec`, `timestamp_usec`), `encoderValue`, `quality`,
  `measurement_rate_hz`, `profile_length`, `arrival_time`, etc.

`load_profiles()` takes `x`/`z` (plus any known `profileData` fields present) and maps two raw
timing attributes onto fields: `arrival_time` → `arrivalTime` (the absolute clock for the PLC join)
and `timestamp_sec` + `timestamp_usec` → `sensorTime` (the low-jitter sensor clock used for the
physical plot spacing). The rest of the rich sensor metadata is written but not yet consumed.

### Processed cache (`save_profiles` / `load_profiles`)

A second, much smaller HDF5 holds the *processed* result so plotting can skip the raw
load and the processing pipeline. It is a **columnar "table"** — every per-profile field is one
array keyed by profile index (profile *i* = row *i*), so the whole cache loads in a handful of bulk
reads instead of ~N tiny per-group reads (measured **~142 s → ~5 s** on the 90.7k Exp1 cache):

- **`/points`** (per-point arrays, padded): `x`, `z` (rotated + levelled coordinates) and `floorMask`
  (per-point floor/filament split) as `[N, L]` matrices (L = max point count), with `lengths` `[N]`
  giving each profile's valid point count; plus `widthFlankIdx` (the two flank-foot indices) and
  `widthOuterIdx` (the two outer filament-point indices) as `[N, 2]` int (`-1` = absent / None).
- **`/scalars`** — every per-profile *scalar* field as a length-N `float64` dataset (NaN = unset):
  `m`/`b` (floor fit), `widthFlank` (smoothed-slope flank-foot method, NaN when < 2 flanks), `widthOuter`
  (filament-edge method), `heightP95` / `heightSmooth` (robust filament heights, percentile vs
  median-smoothed; NaN when flat), `areaSimpson` / `areaShoelace` (filament cross-section, integration vs
  shoelace; NaN when flat), `segmentVolume` / `sliceVolume` (filament-segment total vs per-profile slab
  volume; unset for flat profiles), `segmentLength` (filament-run length; unset on flat) / `defectLength`
  (pure-floor-run length; unset on non-flat), `isFlat` (raw "no filament points"), `isSegment` (cleaned
  "part of a real filament segment", from `clean_flat_runs`), `flatness` (line-fit RMS residual; unset for
  the default floor-based flat method), `arrivalTime` (absolute Unix capture time), `sensorTime` (sensor clock
  seconds, for inter-profile dt), and the 10 joined PLC channels (`mortarPumpFlow`, `pressure*`,
  `printHead*`, `rollerband*`, `viscoPump*`).
- **`/names`** — `[N]` strings.
- **File attributes**: `layout = "columnar"`, `n_profiles`, `kind = "processed"`, `source_file`
  (provenance), `raw_start`/`raw_end` (the raw index span the cache covers, after the PLC-overlap
  trim), and `level_angle`/`level_offset` — the uniform leveling transform. The plot workbench
  reconstructs the pre-leveling ("before") x/z by inverting it (`unlevel_profiles`) instead of
  reloading the raw file; `raw_start`/`raw_end` remain as provenance and the fallback for caches
  predating the stored transform.

`load_profiles` routes by file attribute: `layout="columnar"` → this table; a raw acquisition file
(`profile_NNNNNN` groups, no `layout`) → takes `x`/`z` + `arrival_time`/`timestamp_*`; an **old
per-group processed cache** (`kind="processed"` but no `layout`) is **rejected** with a "reprocess to
the columnar layout" error (the legacy processed-cache reader was removed — reprocess old Exp2/Exp3
caches before use). Padded storage suits the sensor's near-fixed profile width (negligible waste, gzip
squashes the padding); if profile lengths ever varied widely, `/points` would switch to concatenation
+ an offsets index. The default (floor-based) flat method caches `isFlat` directly and leaves
`flatness` unset; `load_profiles` only re-derives
`isFlat = flatness < FLATNESS_RMS_THRESHOLD` for the optional line-fit method
(`flag_flat_profiles(use_line_fit=True)`), which stores an RMS `flatness` — so that cutoff can be
retuned without reprocessing.

Example sizes: a 2000-profile slice is ~30 MB processed vs ~810 MB raw, and loads in
~1 s vs ~19 s. Switching a cache to the columnar layout requires one reprocess
(`python profileProcessing.py`); until then the old cache loads via the fallback path.

---

## 7. Known technical debt & risks

Severity is relative to *current* behaviour. Full remediation backlog in
[TODO.md](TODO.md).

| # | Issue | Risk |
| - | ----- | ---- |
| 1 | **`.y` vs `.z` mismatch.** `profileRegistration.py` reads `.y`, but `profileData` only defines `x`/`z`. | High *(latent)* — does not affect the active path, but the registration pipeline will `AttributeError` the moment it is re-enabled. |
| 2 | ~~**Name collision** between the wire-format `ProfileData` (`rawProfileUdpCapturing.py`) and analysis `profileData` (`profilePointsClass.py`).~~ **Resolved:** the wire-format class is now `ProfileDataRaw`. | — |
| 3 | **Wildcard imports** (`from x import *`) — now confined to the dormant `profileRegistration.py`; the active analysis modules use explicit imports. | Low — limited to off-path legacy code. |
| 4 | **Magic constants** — many are now named in `profileProcessingAlgorithms.py` (`MIN_PROFILE_POINTS`, `FLOOR_POINT_THRESHOLD`, the positional-prior, slope-peak and smoothing-window constants). Still un-named: path geometry `2000/5000/80000` (`profile3Dplotting.py`), the UDP address/port and parser byte offsets (`rawProfileUdpCapturing.py`). Values are unit-dependent (1 unit ≈ 0.01 mm). | Medium — a shared `config` module is still wanted for the rest. |
| 5 | **Dead code & unused imports** (active pipeline cleared): the CSV loaders, `find_border_points`, the `borderPoints` field, and unused imports were removed. Remaining is out-of-scope legacy (`profileRegistration.py`); the `profilePointsClass.py` max-height block is intentionally kept for later (the area block it sat with is now implemented as `measure_filament_area`). | Low — clutter, risk of "fixing" code that never runs. |
| 6 | **Missing type hints & docstrings** on most module-level functions. | Low/Medium — slows comprehension; no static-analysis safety net. |
| 7 | **No error handling at boundaries**: HDF5 load assumes well-formed files; UDP parser uses a broad `except Exception` and unbounded buffering dicts. | Medium — silent data loss / memory growth on malformed or out-of-order packets. |
| 8 | **Hardcoded paths & IP** in the entry scripts. | Low — non-portable; blocks reuse on another machine. |
| 9 | **No tests, lint, or type-check config.** | Medium — refactors are unguarded as the codebase grows. |
| 10 | ~~**`load_profiles` reads *all* profiles, then the caller slices.**~~ **Resolved:** `load_profiles(fileName, start, end)` reads only the requested range, and `profileProcessing.py` validates the range against a fast `count_profiles()` peek. | — |

### Suggested direction (not yet done)

Incremental, behaviour-preserving steps, in priority order, are tracked in
[TODO.md](TODO.md). The highest-leverage early moves are: a small `config`/constants
module for the magic numbers, replacing wildcard imports in new/edited files, and a
`pytest` suite around the pure functions (`moving_average`,
`get_baseline_from_profileBorder`, width detection) so later cleanups are safe.
