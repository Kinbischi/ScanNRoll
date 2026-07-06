# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not yet use formal version numbers; changes accumulate under
`[Unreleased]` until a versioning scheme is adopted.

## [Unreleased]

### Changed
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
  profile. Rendered output is unchanged.
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
