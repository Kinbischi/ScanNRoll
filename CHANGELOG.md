# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not yet use formal version numbers; changes accumulate under
`[Unreleased]` until a versioning scheme is adopted.

## [Unreleased]

### Removed
- **Dead-code cleanup (behaviour-preserving).** Deleted the unused pre-HDF5 CSV loaders
  (`plotProfiles`, `loadProfiles`, `groupProfiles`) and `find_border_points`, the orphaned
  `profileData.borderPoints` field, and now-unused imports (`matplotlib`, `pathlib`, `os`,
  `time` in `profileLoading.py`; `h5py` in `profile3Dplotting.py`). Also fixed stale comments
  (the flatness-threshold and `FLOOR_POINT_THRESHOLD` unit notes; removed resolved `show()`
  notes), added a `MIN_PROFILE_POINTS` constant for the repeated `< 10` guard, made the two
  wildcard imports explicit, and split the categorisation toggle into `SEED_USE_PROFILE_BASELINE`
  / `GROW_USE_PROFILE_BASELINE`. No numeric or plot-output change. The
  `integrate_area`/`shoelace_area`/`find_max_height` block in `profileData` was kept for later.
- **Removed the write-only `profileData.profileNumber`** (with its buggy 1-digit `__post_init__`
  regex and `import re`) and the unused `ySlope` field. `profileNumber`'s only reader was the
  deleted `groupProfiles`; `ySlope` (raw `|dz/dx|`) was computed but never read. Old caches still
  load (both are ignored as foreign keys).

### Added
- **Interactive bead feature heat-map.** New `plottingClass.plot_feature_heatmap()` in
  `profile3Dplotting.py` colours the bead cloud by a per-profile scalar (`beadWidth`, `width`,
  `area`, `shoelaceArea`) and adds a left-edge button panel to switch the active feature live.
  Every feature is attached to the cloud as its own point-data array, so a click only repoints
  the mapper and rescales the colour bar (per-feature `clim`) — no recompute. Bead points only
  (`~floorMask`); flat profiles drop out and NaN values render grey. Interactive window only;
  new `dataAnalysis.py` cell.
- **Bead cross-sectional area (two methods).** New `measure_bead_area()` in
  `profileProcessingAlgorithms.py` measures the bead's cross-section over the shared `z = 0`
  median floor (from `rotate_and_shift_uniform`, not each profile's own fit), across the bead
  span from `floorMask`. Two independent numerical schemes cross-check each other: Simpson
  integration → `profileData.area`, shoelace polygon → `profileData.shoelaceArea` (they agree to
  ~0.02% on typical beads). NaN for flat profiles. Runs in `process_profiles` and is cached via
  the generic I/O. Reprocess to populate the new fields.
- **Bead-edge width method + dual-method width visualisation.** New `width_from_bead_edges()`
  measures bead width directly as the x-span between the outer (min-x / max-x) bead points using
  `floorMask`, stored in `profileData.beadWidth` / `beadWidthIdx`. Both width methods now run in
  `process_profiles`: the slope-peak method (`find_smooth_slope` + `width_from_smoothed_slope` →
  `peaks`/`width`) and the new bead-edge method. Plotting gained a `"beadWidthPoints"` subject and
  a `spheres` argument (`plottingClass.plot` can render enlarged sphere markers), and a new
  `dataAnalysis.py` cell shows both methods' chosen points (slope-peak red, bead-edge blue).
  Reprocess to populate the new fields.
- **Floor vs profile point categorization.** `categorize_floor_points()` in
  `profileProcessingAlgorithms.py` sets a per-point boolean `floorMask` on each profile
  (`True` = floor, else bead point) by absolute-height threshold (`FLOOR_POINT_THRESHOLD`
  = 20). Non-destructive and vectorised (superseding the older `find_border_points`). Runs
  in `process_profiles` and is cached (auto-persisted by the
  generic I/O). Plotting gained a `category="floor"|"profile"` argument
  (`plottingClass.plot` / `get_profile_points_for_plot`) to draw only points of a category;
  `dataAnalysis.py` has cells for a floor/bead two-colour view and a bead-only view.
- **Positional prior for floor/bead categorization.** New `position_height_penalty()` adds a
  per-point height penalty that is 0 within a central plateau (`BEAD_CENTER_HALFWIDTH` = 0.4 of
  the half-width) and ramps linearly to `EDGE_HEIGHT_PENALTY` (400 z-units) at the profile
  edges. It is added to both the seed threshold (`categorize_floor_points`) and the grow
  threshold (`grow_profile_points`). Because the printed bead sits in the middle of the scan,
  this makes points near the edges need more height to be classed as bead, suppressing raised
  edge floor (notably the right side, which sits ~1.3 mm up) that was being mislabelled bead.
  On a 1000-profile slice, profiles with a false right-edge bead dropped 115 → 67 while the
  central bead was untouched (129068 centre points unchanged, ~3.7% fewer bead points overall).
  The hysteresis / gap-fill logic is unchanged — only the thresholds became position-aware.
  Reprocess required.
- **Per-profile floor-fit categorization option.** `categorize_floor_points` and
  `grow_profile_points` gained `use_profile_baseline` (default `False` = the existing uniform
  median-levelled height). When `True`, floor/bead thresholds are applied to each point's height
  above its OWN floor fit, `z - (m*x + b)` (via `_categorization_height`), so a profile's floor
  sits at 0 regardless of its deviation from the dataset median. The stored `x`/`z` stay
  median-levelled (relative heights kept for visualisation); only the categorisation basis
  changes. `profileProcessing.py` exposes a `USE_PROFILE_BASELINE_CATEGORIZATION` toggle. On the
  example data it gives nearly identical results to the median basis — the uniform leveling
  already lands each profile within ~5 units of its own floor (residual `|b|` median 4.7) — so
  the median method is kept for comparison. Reprocess to apply.

### Changed
- **Slope-peak width now uses the outermost peaks (+ refactor).** `width_from_smoothed_slope`
  measures the width between the two OUTERMOST slope peaks (furthest-left/right flanks) instead of
  bailing to `NaN` whenever there weren't exactly two peaks — profiles with intermediate peaks now
  get a width (coverage 123 → 257 of 300 on a test slice; the two-peak results are unchanged).
  `peaks` now holds just those two width-defining points so the markers plot. Also refactored:
  magic numbers lifted to named constants (`HEIGHT_SMOOTH_WINDOWS`, `SLOPE_SMOOTH_WINDOWS`,
  `SLOPE_PEAK_MIN_HEIGHT` / `SLOPE_PEAK_MIN_DISTANCE`), the smoothing cascade looped, the dead
  unused `ySlope` computation dropped, and type hints/docstrings added.
- **Flat detection is now floor-based.** `flag_flat_profiles` flags a profile flat when the
  categorisation found no bead points (every point is floor, via `floorMask`), replacing the
  straight-line RMS-residual test as the default; the old method is kept behind
  `use_line_fit=True` (with a `max_bead_points` tolerance, default 0). In floor-based mode
  `flatness` is left None, so `load_profiles` preserves the cached `isFlat` (no load change). Now
  run in the `process_profiles` pipeline after the grow step — reprocess to populate `isFlat`.
- **Floor baseline fit cached, reused for plotting.** `rotate_and_shift_uniform` now stores
  each profile's floor fit in the (previously unused) `profileData.m`/`.b` fields (the
  intercept is carried through the final shift). `line_points_from_floorSides`
  (`profile3Dplotting.py`) reads those instead of refitting via
  `get_baseline_from_profileBorder`, so the baseline layer no longer re-runs the fit on
  every plot; the drawn line is numerically identical (~1e-11). `m`/`b` are auto-saved in
  the processed cache — reprocess to populate them; a profile with no stored fit falls back
  to a flat line at z = 0.
- **Robust floor baseline fit.** `get_baseline_from_profileBorder` now fits the floor with
  an iterative lower-envelope (clip points above the fit down, refit, repeat) instead of
  ordinary least squares. This ignores the curled-up paper edge (whose raised border points
  biased the old fit high and over-shifted every profile down) and any bead intruding into
  the border, so the `LSerror>50` two-sided fallback was removed. Cuts the median shift by
  ~8 units (~0.08 mm); reprocess required.
- **`dataAnalysis.py` is now a cell-based (`# %%`) plot workbench.** Runs cell-by-cell in
  VS Code's Interactive Window / Jupyter (or as a plain script) with a native PyVista
  window. Cells: load raw ("before") and the processed cache ("after") — both from files,
  no processing on this path; 3D raw, 3D processed, and a raw/processed overlay (guarded
  by an assert that the two line up). Dropped the old broken commented legacy block.
  `plottingClass` and the loaders are reused unchanged.
- **Split `profilePointsClass` into data model + algorithms.** `profilePointsClass.py`
  now holds only the `profileData` dataclass (a pure, dependency-free data model). The
  processing functions (rotate/level/smooth/width/flatness/borders + `moving_average`,
  `get_baseline_from_profileBorder`, `FLATNESS_RMS_THRESHOLD`) moved to a new
  `profileProcessingAlgorithms.py`, which imports only `profilePointsClass`. The
  `process_profiles` pipeline moved into the `profileProcessing.py` entry point. Import
  sites updated (`profileLoading`, `profile3Dplotting`, `profileProcessing`);
  `import scipy.signal` added so `find_peaks` no longer relies on side-effect imports.
- **Process a profile range.** `profileProcessing.py` now takes `START_PROFILE` /
  `END_PROFILE` (index range, END exclusive) instead of `MAX_PROFILES`, and validates
  the range against a fast `count_profiles()` peek first — an out-of-range start or
  empty range fails in <1 s (before the slow load), and an over-large end is clamped
  with a warning. `load_profiles(fileName, start, end)` reads only the requested slice,
  so processing a small part is ~1 s instead of ~19 s.
- **Unified profile I/O.** Replaced `load_hdf5_profiles`, `load_processed_profiles`
  and `save_processed_profiles` with one generic pair in `profileLoading.py`:
  `save_profiles()` writes whichever `profileData` fields are set (arrays → datasets,
  scalars → attributes; bulky intermediates skipped) and `load_profiles()` restores
  whichever are present, ignoring foreign keys — so it reads both processed caches and
  raw acquisition files (taking only `x`/`z` from the latter). Call sites in
  `profileProcessing.py` and `dataAnalysis.py` updated.
- **Separated processing from plotting** to make re-plotting fast. Processing now
  runs once via the new `profileProcessing.py`, which writes a small processed-HDF5
  cache; `dataAnalysis.py` is now a plot-only entry point that loads that
  cache. On a 2000-profile slice the re-plot loop dropped from ~34 s to ~1.7 s
  (cache is ~30 MB vs ~810 MB raw; load ~1 s vs ~19 s; plot ~0.65 s vs ~14.7 s).
- **Batched 3D plotting**: `plottingClass.add_3d_points_to_plot` now merges all
  profiles into a single PyVista actor per `plot()` call instead of one actor per
  profile. Rendered output is unchanged. `add_lines_to_plot` (the `"baseline"` layer)
  was likewise batched into one `line_segments_from_points` mesh — it had kept the
  per-line `add_mesh` anti-pattern, making the baseline layer ~103 s for 7151 profiles;
  now ~0.5 s.
- **Optional plot subsampling**: `plottingClass.plot` / `get_profile_points_for_plot`
  gained `profile_step` / `point_step` (default 1 = unchanged) to decimate the dense
  "profile" cloud so interaction stays smooth on large datasets. The full example
  dataset is ~37M points; `dataAnalysis.py` defaults to `(3, 2)` ≈ 6M.
- Renamed the wire-format dataclass `ProfileData` → `ProfileDataRaw` in
  `rawProfileUdpCapturing.py` to remove the name collision with the analysis `profileData`
  (`profilePointsClass.py`). All references within the module were updated;
  behaviour and the HDF5 schema are unchanged.

### Fixed
- `find_smooth_slope` / `width_from_smoothed_slope` used `return` inside their
  per-profile loop, so the **whole** pass aborted at the first profile with `< 10`
  points. Changed to `continue` so only that profile is skipped. *Behaviour change:*
  profiles after a too-short one are now processed/measured (previously left
  untouched). Width results on a representative 2000-profile slice were unchanged
  (no early short profile there); the round-trip through the cache is identical.

### Added
- **Flatness detection.** New `flag_flat_profiles()` in `profilePointsClass.py` sets
  `isFlat` (bool) and `flatness` (the RMS residual of a straight-line fit to the whole
  profile) on each `profileData`; a flat substrate fits a line (small residual) while a
  bead deviates (large residual). `FLATNESS_RMS_THRESHOLD = 250` sits in the valley of
  the bimodal distribution (flat mode < ~200, beaded mode > ~500); this catches small
  beads ramping up just after a flat run, which an earlier value of 400 mislabelled as
  flat. Added to the `process_profiles` pipeline and persisted in the cache. On the
  example dataset 15% of profiles are flat. `load_profiles` re-derives `isFlat` from
  the cached (threshold-independent) `flatness`, so the threshold can be tuned without
  reprocessing. The border `LSerror` was evaluated and rejected — it does not separate
  flat from beaded.
- **Flat profiles drawn red.** `plottingClass.plot` gained `flat_colour`: when set,
  `isFlat` profiles use it and the rest use `colour`. `dataAnalysis.py` passes
  `flat_colour='red'`.
- Processing pipeline + processed cache:
  - `process_profiles()` in `profilePointsClass.py` — runs the pipeline
    (rotate → level → smooth → width → flatness) in order and returns the list.
  - Processed HDF5 cache written via `save_profiles()` (`profile_NNNNNN` groups with
    `x`/`z`/`peaks` datasets and `name`/`width`/`isFlat`/`flatness`/`profileNumber`
    attributes; file attrs `kind`, `source_file`).
  - `profileProcessing.py` — process entry point (raw → process → cache).
- Project documentation and scaffolding (no runtime code changed):
  - `README.md` — project overview, quickstart, data flow, documentation map.
  - `ARCHITECTURE.md` — components, module dependency graph, execution flows,
    coordinate convention, HDF5 schema, and a known-technical-debt register.
  - `CLAUDE.md` — coding standards and conventions for future AI/human sessions.
  - `TODO.md` — prioritised backlog of suggested (not-yet-applied) improvements.
  - `CHANGELOG.md` — this file.
  - `requirements.txt` — runtime dependencies derived from the source imports.
  - `.gitignore` — Python, editor, and project data/output exclusions.

### Notes
- The documentation/scaffolding files above were added without touching runtime code;
  remaining identified issues (e.g. the `.y`/`.z` mismatch in dormant registration
  code, wildcard imports, magic constants) are catalogued in `TODO.md`.
