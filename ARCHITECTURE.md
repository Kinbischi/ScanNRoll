# Architecture

This document describes how the code is structured today — the **as-is** state,
not an aspirational design. It is the reference for understanding the pipeline
before changing anything. For the conventions to follow when changing it, see
[CLAUDE.md](CLAUDE.md); for the backlog of improvements, see [TODO.md](TODO.md).

---

## 1. Overview

The system has two halves that meet at an HDF5 file:

```
 ┌─────────────┐     UDP      ┌──────────────────┐    HDF5    ┌────────────────────────┐
 │ Baumer OX200│ ───────────► │  rawProfileUdpCapturing.py │ ─────────► │  HDf5data/*.h5         │
 │   sensor    │   packets    │  (acquisition)   │   write    │  (profile store)       │
 └─────────────┘              └──────────────────┘            └───────────┬────────────┘
                                                                          │ read
                                                                          ▼
                                                       ┌──────────────────────────────────┐
                                                       │     dataAnalysis.py       │
                                                       │  load → process → measure → plot  │
                                                       └──────────────────────────────────┘
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
| `profileProcessingAlgorithms.py` | The processing functions that operate on lists of `profileData`: rotate, level, smooth, width detection, flatness flag, border detection, baseline fit, moving average (+ `FLATNESS_RMS_THRESHOLD`). Imports only `profilePointsClass`. | Active |
| `profileLoading.py` | `load_profiles()` / `save_profiles()` — one generic pair that reads/writes `profileData` to HDF5, storing whichever fields are set (arrays → datasets, scalars → attributes). `load_profiles` also reads raw acquisition files (takes `x`/`z`, ignores sensor metadata). Also holds legacy CSV loaders/plotters (`loadProfiles`, `plotProfiles`, `groupProfiles`). | Mixed (loaders active, CSV legacy) |
| `profile3Dplotting.py` | `plottingClass` — PyVista 3D rendering; computes the print path & per-profile tilt angles and places each profile along it. Points are batched into one actor per `plot()` call. | Active |
| `profileRegistration.py` | Align overlapping profiles in x (ICP / `minimize`), detect left/right/centre profiles, join them into a combined profile. | Legacy (dormant) |
| `profileProcessing.py` | **Entry point (process).** Hosts the `process_profiles()` pipeline (composes the algorithm functions in order) and the run script: load raw HDF5 → process → write the processed-HDF5 cache. Run once per dataset / when processing params change. | Active |
| `dataAnalysis.py` | **Entry point (plot).** Load the processed cache → plot in 3D. No processing. | Active |
| `LidarProfileAnalysis_oldRegistration.py` | Previous entry point built around the registration path. | Legacy |

---

## 3. Module dependency graph

Legacy modules still use `from <module> import *`; newer/edited code uses explicit imports.

```
 profilePointsClass            base layer: profileData only, no project imports
        ▲
 profileProcessingAlgorithms   processing functions + FLATNESS_RMS_THRESHOLD
        ▲                 ▲
 profileLoading        profile3Dplotting
        ▲   ▲              ▲
        │   └──────┐       │
 profileProcessing   dataAnalysis
 (process entry:     (plot entry)
  hosts process_profiles)

 profileRegistration       LEGACY / dormant — imports profilePointsClass, off the active path
 rawProfileUdpCapturing    standalone — imports only stdlib + numpy + h5py
```

Edges: `profileProcessing` imports `profileLoading` + `profileProcessingAlgorithms`;
`dataAnalysis` imports `profileLoading` + `profile3Dplotting`; `profileLoading` and
`profile3Dplotting` each import `profileProcessingAlgorithms` (for `FLATNESS_RMS_THRESHOLD`
and `get_baseline_from_profileBorder` respectively). No cycles.

- `profilePointsClass` is the foundation; everything depends on it.
- No circular imports exist today, but wildcard imports make the dependency
  surface implicit and fragile (any new top-level name leaks everywhere).

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

`process_profiles()` runs the steps in order: `rotate_pointcloud` (flatten via
baseline angle) → `translate_floor_to_zero` (subtract baseline offset) →
`find_smooth_slope` → `width_from_smoothed_slope` → `flag_flat_profiles` (sets
`isFlat`/`flatness` from the whole-profile line-fit residual). The plot step draws
flat profiles red via `plot(..., flat_colour='red')`.

Key processing detail: `find_smooth_slope` applies a **cascade of moving averages**
to `z`, takes the gradient, then smooths the gradient again; `width_from_smoothed_slope`
runs `scipy.signal.find_peaks` on that smoothed slope and reports the x-distance
between exactly two peaks (otherwise `width = NaN`).

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

`loadProfiles()` (CSV) → `groupProfiles()` → per group: rotate / level / find borders →
`registerAndShiftProfiles()` → `generateJoinedProfile()` → area calculations.
**Dormant** and depends on a `.y` attribute the current `profileData` no longer has.

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

- Datasets: `x`, `z` (rotated + levelled coordinates), `peaks` (slope-peak indices).
- Attributes: `name`, `width` (NaN when not exactly two peaks), `isFlat` (bool),
  `flatness` (line-fit RMS residual), `profileNumber`.
- File attributes: `kind = "processed"`, `source_file` (the raw path) for provenance.

`flatness` is the threshold-independent metric; `load_profiles` re-derives
`isFlat = flatness < FLATNESS_RMS_THRESHOLD` on load, so the flat/curved cutoff can be
retuned without reprocessing the cache.

Smoothed arrays (`ySmooth`, `ySlopeSmooth`) are intermediates and are **not** stored;
a future "plot smoothed profile" option would need them added. Example sizes: a
2000-profile slice is ~30 MB processed vs ~810 MB raw, and loads in ~1 s vs ~19 s.

---

## 7. Known technical debt & risks

Severity is relative to *current* behaviour. Full remediation backlog in
[TODO.md](TODO.md).

| # | Issue | Risk |
| - | ----- | ---- |
| 1 | **`.y` vs `.z` mismatch.** `profileRegistration.py`, `profileLoading.py` (`plotProfiles`/`loadProfiles`) and the commented block in `dataAnalysis.py` read `.y`, but `profileData` only defines `x`/`z`. | High *(latent)* — does not affect the active path, but the registration pipeline will `AttributeError` the moment it is re-enabled. |
| 2 | ~~**Name collision** between the wire-format `ProfileData` (`rawProfileUdpCapturing.py`) and analysis `profileData` (`profilePointsClass.py`).~~ **Resolved:** the wire-format class is now `ProfileDataRaw`. | — |
| 3 | **Wildcard imports** (`from x import *`) across all analysis modules. | Medium — hidden coupling, namespace leakage, hard to trace symbol origins. |
| 4 | **Magic constants** scattered and undocumented: smoothing windows `15/9/5/5/65/55/15/5`, peak `height=0.15, distance=50`, border threshold `20`, baseline `borderPoints=30` & error `50`, path geometry `2000/5000/80000`, UDP address/port, parser byte offsets. | Medium — tuning is opaque; values are unit-dependent (≈ 0.01 mm units). |
| 5 | **Dead code & unused imports**: large commented blocks in three modules; unused `copy`, `NearestNeighbors`, `pyvista` (in the class module); unused `loadProfiles`/`plotProfiles`. | Low — clutter, risk of "fixing" code that never runs. |
| 6 | **Missing type hints & docstrings** on most module-level functions. | Low/Medium — slows comprehension; no static-analysis safety net. |
| 7 | **No error handling at boundaries**: HDF5 load assumes well-formed files; UDP parser uses a broad `except Exception` and unbounded buffering dicts. | Medium — silent data loss / memory growth on malformed or out-of-order packets. |
| 8 | **Hardcoded paths & IP** in entry scripts and `loadProfiles()`. | Low — non-portable; blocks reuse on another machine. |
| 9 | **No tests, lint, or type-check config.** | Medium — refactors are unguarded as the codebase grows. |
| 10 | ~~**`load_profiles` reads *all* profiles, then the caller slices.**~~ **Resolved:** `load_profiles(fileName, start, end)` reads only the requested range, and `profileProcessing.py` validates the range against a fast `count_profiles()` peek. | — |

### Suggested direction (not yet done)

Incremental, behaviour-preserving steps, in priority order, are tracked in
[TODO.md](TODO.md). The highest-leverage early moves are: a small `config`/constants
module for the magic numbers, replacing wildcard imports in new/edited files, and a
`pytest` suite around the pure functions (`moving_average`,
`get_baseline_from_profileBorder`, width detection) so later cleanups are safe.
