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
                         categorise floor/bead → width)           ▼
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
| `profileProcessingAlgorithms.py` | The processing functions that operate on lists of `profileData`: rotate, level, smooth, width detection, flatness flag, floor/bead categorisation, baseline fit, moving average (+ `FLATNESS_RMS_THRESHOLD`). Imports only `profilePointsClass`. | Active |
| `profileLoading.py` | `load_profiles()` / `save_profiles()` — one generic pair that reads/writes `profileData` to HDF5, storing whichever fields are set (arrays → datasets, scalars → attributes). `load_profiles` also reads raw acquisition files (takes `x`/`z`, ignores sensor metadata). | Active |
| `profile3Dplotting.py` | `plottingClass` — PyVista 3D rendering; computes the print path & per-profile tilt angles and places each profile along it. Points are batched into one actor per `plot()` call. | Active |
| `profileRegistration.py` | Align overlapping profiles in x (ICP / `minimize`), detect left/right/centre profiles, join them into a combined profile. | Legacy (dormant) |
| `profileProcessing.py` | **Entry point (process).** Hosts the `process_profiles()` pipeline (composes the algorithm functions in order) and the run script: load raw HDF5 → process → write the processed-HDF5 cache. Run once per dataset / when processing params change. | Active |
| `dataAnalysis.py` | **Entry point (plot workbench).** Cell-based (`# %%`) file: load raw ("before") and the processed cache ("after") from files, and plot flexibly in 3D (PyVista, native window) — raw, processed, and an overlay. No processing on this path (uses the cache, not `process_profiles`). | Active |
| `LidarProfileAnalysis_oldRegistration.py` | Previous entry point built around the registration path. | Legacy |

---

## 3. Module dependency graph

Legacy modules still use `from <module> import *`; newer/edited code uses explicit imports.

```
 profilePointsClass          base layer — the profileData model only, no project imports
   ▲    ▲    ▲
   │    │    └── profile3Dplotting           PyVista plotting (imports profilePointsClass only)
   │    └─────── profileProcessingAlgorithms processing fns + FLATNESS_RMS_THRESHOLD
   │                   ▲
   └── profileLoading ─┘                     HDF5 I/O (also imports FLATNESS_RMS_THRESHOLD)

 Entry points compose the above:
   profileProcessing → profileLoading + profileProcessingAlgorithms   (process; hosts process_profiles)
   dataAnalysis      → profileLoading + profile3Dplotting             (plot workbench)

 profileRegistration       LEGACY / dormant — imports profilePointsClass (wildcard), off active path
 rawProfileUdpCapturing    standalone — imports only stdlib + numpy + h5py
```

Edges: `profileProcessingAlgorithms`, `profileLoading`, and `profile3Dplotting` each import
`profilePointsClass`; `profileLoading` also imports `profileProcessingAlgorithms`
(`FLATNESS_RMS_THRESHOLD`). The entry points compose these: `profileProcessing` imports
`profileLoading` + `profileProcessingAlgorithms`; `dataAnalysis` imports `profileLoading` +
`profile3Dplotting`. `profile3Dplotting` depends on `profilePointsClass` only. No cycles.

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
load_profiles(RAW_FILE)                  # profileLoading  → list[profileData]
process_profiles(profiles)               # profilePointsClass pipeline (see below)
save_profiles(profiles, OUT, kind=...)   # profileLoading  → processed *.h5
```

**Plot** (`dataAnalysis.py`, run freely):
```
load_profiles(PROCESSED_FILE)            # profileLoading  → list[profileData]
plottingClass(len(profiles))             # profile3Dplotting: precompute path + tilt
plotter.plot(profiles, "profile", green) # batched: one actor for all profiles
plotter.plot(profiles, "widthPoints", …) # mark width peaks
plotter.show()                           # interactive PyVista window
```

`process_profiles()` runs these steps in order:

1. `rotate_and_shift_uniform` — level **all** profiles by one **median** rotation + shift
   (derived from each profile's floor fit), preserving the real height differences between
   profiles. Also stores each profile's own floor fit in `m`/`b`.
2. `categorize_floor_points` — per-point `floorMask` (floor vs bead) by height, with a symmetric
   **positional prior** (points far from the scan centre need more height to count as bead).
3. `grow_profile_points` — hysteresis: grow the bead from confident seeds into their connected
   lower flanks, then fill small interior gaps.
4. `flag_flat_profiles` — mark a profile flat when it has **no bead points** (floor only). The
   old line-fit-residual method is kept behind `use_line_fit=True`.
5. `find_smooth_slope` → `width_from_smoothed_slope` — bead width from the two **outermost**
   smoothed-slope peaks (the flanks).
6. `width_from_bead_edges` — bead width the other way: the x-span between the outer bead points.
7. `measure_bead_area` — cross-sectional bead area over the shared `z = 0` median floor, two
   ways (Simpson integration → `area`, shoelace polygon → `shoelaceArea`) as a mutual cross-check.

The plot workbench (`dataAnalysis.py`) draws floor vs bead in two colours (`category=`), flat
profiles highlighted (`flat_colour=`), the floor baselines and a `z = 0` reference
(`"baseline"` / `"zeroBaseline"`), and both width methods' points (`"widthPoints"` /
`"beadWidthPoints"`). It can also colour the bead by a per-profile feature with a live
selector panel (`plot_feature_heatmap`): all features are attached to the cloud as separate
scalar arrays, so clicking a feature button only repoints the mapper and rescales the colour bar.

Key detail: `find_smooth_slope` cascades box filters (`moving_average`) over `z`, takes the
gradient, and smooths it again; `width_from_smoothed_slope` runs `scipy.signal.find_peaks` on
that smoothed slope and spans the **outermost** peaks — intermediate peaks are ignored (`NaN`
only when fewer than two peaks). The per-profile alternative (`rotate_pointcloud` +
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

`load_profiles()` takes `x`/`z` (plus any known `profileData` fields present) and
ignores the rich sensor metadata, which is written but not yet consumed by the
analysis side.

### Processed cache (`save_profiles` / `load_profiles`)

A second, much smaller HDF5 holds the *processed* result so plotting can skip the raw
load and the processing pipeline. Same `profile_NNNNNN` group layout:

- Datasets: `x`, `z` (rotated + levelled coordinates), `peaks` (the two outermost slope-peak
  indices), `beadWidthIdx` (the two outer bead-point indices).
- Attributes: `name`, `width` (slope-peak method, NaN when < 2 peaks), `beadWidth` (bead-edge
  method), `area` / `shoelaceArea` (bead cross-section, integration vs shoelace; NaN when flat),
  `isFlat` (bool), `flatness` (line-fit RMS residual; None for the default floor-based flat
  method).
- File attributes: `kind = "processed"`, `source_file` (the raw path) for provenance.

The default (floor-based) flat method caches `isFlat` directly and leaves `flatness` unset;
`load_profiles` only re-derives `isFlat = flatness < FLATNESS_RMS_THRESHOLD` for the optional
line-fit method (`flag_flat_profiles(use_line_fit=True)`), which stores an RMS `flatness` — so
that cutoff can be retuned without reprocessing.

Smoothed arrays (`ySmooth`, `ySlopeSmooth`) are intermediates and are **not** stored;
a future "plot smoothed profile" option would need them added. Example sizes: a
2000-profile slice is ~30 MB processed vs ~810 MB raw, and loads in ~1 s vs ~19 s.

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
| 5 | **Dead code & unused imports** (active pipeline cleared): the CSV loaders, `find_border_points`, the `borderPoints` field, and unused imports were removed. Remaining is out-of-scope legacy (`profileRegistration.py`); the `profilePointsClass.py` max-height block is intentionally kept for later (the area block it sat with is now implemented as `measure_bead_area`). | Low — clutter, risk of "fixing" code that never runs. |
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
