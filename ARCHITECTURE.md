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
 │ Baumer OX200│ ───────────► │  udpCapturing.py │ ─────────► │  HDf5data/*.h5         │
 │   sensor    │   packets    │  (acquisition)   │   write    │  (profile store)       │
 └─────────────┘              └──────────────────┘            └───────────┬────────────┘
                                                                          │ read
                                                                          ▼
                                                       ┌──────────────────────────────────┐
                                                       │     LidarProfileAnalysis.py       │
                                                       │  load → process → measure → plot  │
                                                       └──────────────────────────────────┘
```

- **Acquisition** (`udpCapturing.py`) is standalone: it shares no imports with the
  analysis code and owns its own `ProfileData` dataclass tuned to the wire format.
- **Analysis** (everything else) reads the HDF5 store into a different
  `profileData` dataclass and runs the processing / visualisation pipeline.

> ⚠️ The two dataclasses have nearly identical names (`ProfileData` vs
> `profileData`) but completely different fields. See *Known technical debt*.

---

## 2. Components & responsibilities

| Module | Responsibility | Status |
| ------ | -------------- | ------ |
| `udpCapturing.py` | Receive sensor UDP packets, parse the binary protocol, pair Z-profile + measurement blocks, write to HDF5. Owns `MeasurementData` and a wire-format `ProfileData`. | Active (standalone) |
| `profilePointsClass.py` | Defines the analysis `profileData` dataclass **and** the free functions that operate on lists of it: rotate, level, smooth, width detection, border detection, baseline fit, moving average. The processing core. | Active |
| `profileLoading.py` | `load_hdf5_profiles()` reads the HDF5 store into `profileData` objects. Also holds legacy CSV loaders/plotters (`loadProfiles`, `plotProfiles`, `groupProfiles`). | Mixed (loader active, rest legacy) |
| `plottingProfiles3D.py` | `plottingClass` — PyVista 3D rendering; computes the print path & per-profile tilt angles and places each profile along it. | Active |
| `profileRegistration.py` | Align overlapping profiles in x (ICP / `minimize`), detect left/right/centre profiles, join them into a combined profile. | Legacy (dormant) |
| `LidarProfileAnalysis.py` | **Entry point.** Wires loading → processing → plotting for the current workflow. | Active |
| `LidarProfileAnalysis_oldRegistration.py` | Previous entry point built around the registration path. | Legacy |

---

## 3. Module dependency graph

All cross-module imports currently use `from <module> import *`.

```
        profilePointsClass        (no internal deps — the base layer)
            ▲     ▲     ▲
            │     │     └──────────────┐
            │     │                    │
 profileRegistration        plottingProfiles3D
            ▲                          ▲
            │                          │
        profileLoading                 │
            ▲                          │
            └───────────┬──────────────┘
                        │
              LidarProfileAnalysis        (entry point)

 udpCapturing  ── standalone, imports only stdlib + numpy + h5py
```

- `profilePointsClass` is the foundation; everything depends on it.
- No circular imports exist today, but wildcard imports make the dependency
  surface implicit and fragile (any new top-level name leaks everywhere).

---

## 4. Important execution flows

### 4.1 Active analysis pipeline (`LidarProfileAnalysis.py`)

```
load_hdf5_profiles(HDF5_FILE)            # profileLoading  → list[profileData]
plottingClass(len(profiles))             # plottingProfiles3D: precompute path + tilt
plotter.plot(profiles, "profile", blue)  # show raw profiles
rotate_pointcloud(profiles)              # profilePointsClass: flatten via baseline angle
translate_floor_to_zero(profiles)        # profilePointsClass: subtract baseline offset
find_smooth_slope(profiles)              # profilePointsClass: cascade moving-averages → slope
width_from_smoothed_slope(profiles)      # profilePointsClass: find_peaks → width
plotter.plot(profiles, "profile", green) # show processed profiles
plotter.plot(profiles, "widthPoints", …) # mark width peaks
plotter.show()                           # interactive PyVista window
```

Key processing detail: `find_smooth_slope` applies a **cascade of moving averages**
to `z`, takes the gradient, then smooths the gradient again; `width_from_smoothed_slope`
runs `scipy.signal.find_peaks` on that smoothed slope and reports the x-distance
between exactly two peaks (otherwise `width = NaN`).

### 4.2 Acquisition pipeline (`udpCapturing.py`)

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

`load_hdf5_profiles()` reads only `x` and `z` today; the rich metadata is written
but not yet consumed by the analysis side.

---

## 7. Known technical debt & risks

Severity is relative to *current* behaviour. Full remediation backlog in
[TODO.md](TODO.md).

| # | Issue | Risk |
| - | ----- | ---- |
| 1 | **`.y` vs `.z` mismatch.** `profileRegistration.py`, `profileLoading.py` (`plotProfiles`/`loadProfiles`) and the commented block in `LidarProfileAnalysis.py` read `.y`, but `profileData` only defines `x`/`z`. | High *(latent)* — does not affect the active path, but the registration pipeline will `AttributeError` the moment it is re-enabled. |
| 2 | **Name collision** between the wire-format `ProfileData` (`udpCapturing.py`) and analysis `profileData` (`profilePointsClass.py`). | Medium — easy to confuse when editing; obstructs any future sharing. |
| 3 | **Wildcard imports** (`from x import *`) across all analysis modules. | Medium — hidden coupling, namespace leakage, hard to trace symbol origins. |
| 4 | **Magic constants** scattered and undocumented: smoothing windows `15/9/5/5/65/55/15/5`, peak `height=0.15, distance=50`, border threshold `20`, baseline `borderPoints=30` & error `50`, path geometry `2000/5000/80000`, UDP address/port, parser byte offsets. | Medium — tuning is opaque; values are unit-dependent (≈ 0.01 mm units). |
| 5 | **Dead code & unused imports**: large commented blocks in three modules; unused `copy`, `NearestNeighbors`, `pyvista` (in the class module); unused `loadProfiles`/`plotProfiles`. | Low — clutter, risk of "fixing" code that never runs. |
| 6 | **Missing type hints & docstrings** on most module-level functions. | Low/Medium — slows comprehension; no static-analysis safety net. |
| 7 | **No error handling at boundaries**: HDF5 load assumes well-formed files; UDP parser uses a broad `except Exception` and unbounded buffering dicts. | Medium — silent data loss / memory growth on malformed or out-of-order packets. |
| 8 | **Hardcoded paths & IP** in entry scripts and `loadProfiles()`. | Low — non-portable; blocks reuse on another machine. |
| 9 | **No tests, lint, or type-check config.** | Medium — refactors are unguarded as the codebase grows. |

### Suggested direction (not yet done)

Incremental, behaviour-preserving steps, in priority order, are tracked in
[TODO.md](TODO.md). The highest-leverage early moves are: a small `config`/constants
module for the magic numbers, replacing wildcard imports in new/edited files, and a
`pytest` suite around the pure functions (`moving_average`,
`get_baseline_from_profileBorder`, width detection) so later cleanups are safe.
