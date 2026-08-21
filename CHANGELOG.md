# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project does not yet use formal version numbers; changes accumulate under
`[Unreleased]` until a versioning scheme is adopted.

## [Unreleased]

### Added
- **Togglable 3D "preprocessing" cell (replaces the three static 3D cells).** One interactive `dataAnalysis`
  cell with a **left-edge show/hide checkbox per layer** — raw (grey) & processed (green) clouds, the floor
  baselines (red) + z=0 ref (yellow), the floor (brown) / filament (green) point categories, and the two
  width-marker sets (flank = red, outer = blue) — each independently togglable (opens on the floor/filament
  category view; the other layers start hidden). It subsumes the old **overlay**, **category**, and **width
  markers** cells, which were removed. Backed by a small `plottingClass.add_layer_toggles(layers)` helper;
  `plot()` and the `add_3d_points_to_plot` / `add_lines_to_plot` helpers now **return their actor** so callers
  can toggle it (existing calls ignore the return, unchanged). Verified headless (all eight actors returned,
  initial visibility + toggle callbacks, screenshots).
- **`dataAnalysisSetup.py` — any `dataAnalysis` cell can be run first.** New workbench-setup module holding the
  imports, display config (PyVista + the matplotlib Qt backend, set from the module via
  `IPython.get_ipython()`), the tunable constants (`PROFILE_STEP` / feature lists), and a **cached `load()`**
  (reads the processed cache + reconstructs raw once, in-process; `load(force=True)` re-reads after a
  reprocess). Each `dataAnalysis` cell now starts with `from dataAnalysisSetup import …` + `raw, processed =
  load()`, so **clicking any cell first in a fresh kernel works** (no more `NameError` from the setup cell not
  having run) and is instant after the first load. Replaces the earlier in-file `ensure_loaded()` guard, which
  couldn't help a truly cold kernel (its own definition lived in the un-run first cell).

### Changed
- **Heat-map body-taper parent button relabelled `bodyThinning` → `Thinning`** (the selector row is already
  labelled `body`, so the button no longer repeats it). Feature keys unchanged.

### Performance
- **Much faster processed-cache load (columnar "table" layout).** `save_profiles` now writes the
  processed cache as a columnar table — every per-profile field is one array keyed by profile index,
  instead of a `profile_NNNNNN` group each: `/points/x`, `/points/z`, `/points/floorMask` as padded
  `[N, L]` matrices (+ `/points/lengths`), `/points/peaks` & `/points/beadWidthIdx` as `[N, 2]` int
  (`-1` = absent), `/scalars/<field>` as `[N]` float64 (NaN = unset), and `/names`. Loading collapses
  from ~450k tiny per-object reads to a handful of bulk reads: **measured 20k-slice load 18.3 s →
  1.09 s (~17×), projected ~142 s → ~5 s** for the full 90.7k Exp1 cache. `load_profiles` reads this
  layout (via the `layout` file attr) and still reads the legacy per-group-attribute layout and raw
  acquisition files transparently. Padded storage fits the sensor's near-fixed profile width
  (negligible waste). Verified **byte-for-byte lossless**: all 33 fields × 23k real profiles
  identical (round-trip of a cache slice + a freshly-processed raw slice), plus edge cases (shortest
  profile, absent `peaks`/`beadWidthIdx` → None, flat profiles). Reprocess
  (`python profileProcessing.py`) to convert a cache; existing caches keep loading via the fallback.
- **Faster voxel downsampling.** `plottingClass._maybe_voxel` flattens the 3-D voxel index to one
  int64 key (bijective `ravel_multi_index`) so uniqueness is a 1-D sort instead of
  `np.unique(axis=0)`'s 3-column lexsort — **~5× faster** (6.0 s → 1.2 s on a 5.5 M-point cloud) with
  byte-identical output (same first point kept per voxel; exact row-wise fallback if the flattened
  index would overflow int64). Cuts the overlay cell's two voxel passes from ~13 s to ~2.5 s.

### Changed
- **2D plots trimmed + fully grouped; heat-map sub-panels moved off-centre.** The body **steadiness** values
  and `segmentRuptures` are now **heat-map-only** — removed from every 2D plot (feature-vs-time and both
  feature-vs-PLC views) so those stay focused on the thinning rates + rupture geometry. The **stepwise**
  feature panel is now **category-grouped** (Geometry / Segment / PLC headers via `_build_grouped_panel`),
  matching the continuous view and `featureComparison` — the last flat feature list is gone. In the 3D heat
  map, the expander member sub-panel and the `segmentShapeStatus` grey-out panel moved from the bottom-centre
  (where they stacked over the print-path cloud) to the **upper-right** (`_side_panel_x` / `_side_panel_top_y`,
  top-anchored), clear of the cloud, the colour bar, and the scale selector; the **camera-orientation gizmo**
  moved from its default upper-right to the **upper-left** (`AnchorToUpperLeft`) so it no longer clashes with
  those panels. The stepwise plot now also lists **both widths (flank/outer) and both heights (p95/smooth)**
  like the other views, and the featurePlcTrends grouped-panel **checkboxes are larger** (box size 34 → 64).
  Verified headless (heat-map screenshots of both panels top-right; grouped stepwise + continuous figures).
- **Body thinning split into area / width / height (+ steadiness); heat-map parent buttons.** The single
  body-taper feature is now a **family of six**, each measured over the same body plateau on a different
  signal: `segmentBody{Area,Width,Height}Thinning` (%/mm) + `segmentBody{Area,Width,Height}Steadiness`
  (Spearman, -1..1). This **renames** the old fields (`segmentBodyThinning` → `segmentBodyAreaThinning`,
  `segmentBodyThinningStability` → `segmentBodyAreaSteadiness`) and **adds** the width (widthOuter) and height
  (heightP95) versions — a **breaking cache-field change, so reprocess** (`python profileProcessing.py`) to
  populate them. In the 3D heat map these six sit behind a **`bodyThinning` parent button**: selecting it
  reveals the members as a **plain-gradient radio sub-panel** at the bottom-centre (`EXPANDER_GROUPS` /
  `_add_member_selector`), each rendered like a normal gradient feature (own colour scale, no lag) rather
  than the categorical grey-out panel. Labels renamed: rupture `start` → `critArea`; `steady` → `steadiness`.
- **`segFlags` reverted to three plain-gradient buttons behind a parent.** The previous combined categorical
  `segFlags` feature is removed; the flags `isNotFlat` / `isSegment` / `isContinuousFilament` are ordinary
  gradient features again, grouped behind a **`segFlags` parent button** (same expander mechanism as
  `bodyThinning`) — only `segmentShapeStatus` keeps the multicolour grey-out checkbox panel. No reprocess for
  this part (the flag fields are unchanged). The Segment panel leads with the **run aggregates** row and ends
  with a **debug** row `segFlags` / `sortout` / `phase`; the 2D plots still list every member individually
  under Segment (`_selector_tokens` collapses to parents only in the 3D selector). Verified headless (cloud
  build, parent↔member↔categorical panel transitions, radio sync, sub-panel clears the colour bar).
- **`segmentRuptures` and `sliceVolume` stay out of the heat map.** `segmentRuptures` is not a heat-map button
  (read its gate off `segmentShapeStatus` code 6, "no rupture") but remains a field + a stepwise
  feature-vs-PLC series (rupture rate per belt speed). `sliceVolume` (the per-profile slab) is shown in **no
  plot** — a backend-only field feeding `segmentVolume`.
- **`segmentShapeStatus` gets a distinct-colour scheme + an interactive checkbox panel in the heat map.** The
  `CATEGORICAL_FEATURES` registry (`profile3Dplotting.py`) maps `segmentShapeStatus` (0-6 sort-out reasons) to
  `{code: label}`; when it is the active feature the cloud is coloured by a **discrete distinct-colour LUT**
  (one solid palette colour per code, off-category points grey) and the gradient colour bar is replaced by a
  bottom-centre **checkbox panel**: one colour-matched toggle per category (a reusable widget pool,
  `_add_category_selector`), each with a shadowed label. **Unchecking a category greys those segments out** (a
  companion NaN-mask scalar), so you can isolate e.g. just the tiny-body segments. The scale-mode group
  (linear/log/clip/rank) is ignored for it. Every other feature — including `segmentSection` (phase) and the
  0/1 flags (`isSegment` / `isNotFlat` / `isContinuousFilament` / `segmentRuptures`) — keeps the plain viridis
  gradient + colour bar (e.g. phase head/peak = yellow). Fixes the misleading "7 shades of viridis" rendering
  of the multi-code sort-out feature, where the gradient falsely implied an ordering. Verified headless
  (gradient↔categorical switches, panel build, grey-out toggle, stray-widget hiding, screenshots).
- **Unified, visible sort-out of weird segments from the shape analysis (`segmentShapeStatus`).** Every
  `isSegment` run now carries a per-segment status code — 0 kept, or the reason it was sorted out of the
  thinning analysis: 1 too short (< 50 mm), 2 continuous filament (> 600 mm), 3 degenerate (no body),
  **4 tiny body** (plateau < `MIN_BODY_LENGTH_MM` = 15 mm — catches a degenerate 5 mm-body segment that
  otherwise reported a wild **+5.9 %/mm** taper), **5 high width change** (outer width swings >
  `WIDTH_CHANGE_MAX_FRAC` = 40 % of the body width across the body — a turbulent/spreading bead whose
  area-based taper misleads, e.g. the 2×-wide-but-flat pancaking segment), 6 didn't rupture. A sorted-out
  segment's 5 thinning/shape features + `segmentSection` become `None` (so it drops from the stepwise boxes),
  but `segmentRuptures` is **kept** on every valid-body segment so the rupture *rate* isn't lost. The status
  is a heat-map debug view (grouped in a new Segment "debug" row alongside `segmentSection`); the sort-out is
  **shape-analysis only** — `segmentLength` / `segmentVolume` / `defectLength` still cover every run. On Exp1:
  107 kept, sorted out 4 too-short / 2 continuous / 1 tiny-body / 2 high-width / 4 didn't-rupture; kept-taper
  range tightens from [−0.43, **+5.89**] to [−0.43, +0.37] %/mm. **Reprocess to apply.**
- **Body starts earlier for segments without a head overshoot (`_settle_index`).** The settle search used to
  pick the "overshoot peak" as the `argmax` over the whole first half of the segment and start the body
  *after* it, so a mid-segment spike or a broad dome peak shoved `body_start` late (e.g. to 0.50·L). It now
  only skips past a *prominent leading* overshoot — a bulge more than the ±band above the body level within
  the first `HEAD_OVERSHOOT_FRAC` (0.25) of the segment; otherwise the body starts at the first sustained
  band entry. No/little-overshoot segments' `body_start` drops from a median 0.39·L to 0.12·L (worst case
  0.50→0.36); genuine-overshoot segments are untouched (median |taper change| 0.000, Spearman old↔new 0.82).
  Domed segments (rise to a broad mid-peak, then decline) now span the whole dome, so ~7 of them flip from a
  small negative taper to ~0/positive — the honest reading (a dome has no net thinning; see
  `segmentBodyThinningStability`). **Reprocess to apply.**
- **Segment-shape analysis now gated to discrete segments, with the body held back from the rupture.**
  Three tuning changes to `measure_segment_shape` (`segmentShape.py`), driven by a read-only study of the
  120 Exp1 segments:
  - **Length gating in mm** replaces the old `MIN_SEGMENT_PROFILES = 40` profile-count cutoff. A segment is
    shape-analysed only if its arc-length is in [`SEGMENT_SHAPE_MIN_LENGTH_MM` = 50 mm,
    `MAX_SEGMENT_LENGTH_MM` = 600 mm]. The profile-count cutoff was speed-dependent and let 11 mm segments
    through at low speed (a −3.4 %/mm taper outlier); the mm floor removes the noisy short segments (taper
    range tightens from [−3.38, +0.78] to [−0.43, +0.46] %/mm). The 600 mm ceiling excludes the two longest
    runs (822 mm, 4242 mm), which also **fixes a latent `MemoryError`**: `scipy.stats.theilslopes` is O(n²)
    in memory and would try to allocate 3.24 GiB on the 4242 mm (20 838-point) body during reprocessing.
  - **`classify_continuous_filaments`** (`profileProcessingAlgorithms.py`) labels those over-long runs as a
    **continuous filament** in a new cached `isContinuousFilament` flag — a distinct object from a discrete
    segment (uninterrupted print, no single startup/rupture), excluded from the shape analysis and available
    for future continuous-filament features. `isSegment` is unchanged, so `segmentLength` / `segmentVolume` /
    `defectLength` still cover every run. Viewable in the heat map (grouped in the Segment "flags" row).
  - **Body-border pushback**: `body_end` is now held `SEGMENT_BODY_RUPTURE_MARGIN_MM` (3 mm) before the
    rupture cliff top (or the segment end on a non-rupturing segment), so the pre-rupture roll-off stays out
    of the taper fit (previously the body could touch the cliff top or run to the last profile). Cleans 43 of
    114 segments (the cliff-huggers) with no starved bodies.

  **Reprocess (`python profileProcessing.py`)** to populate `isContinuousFilament` and the re-gated segment
  fields; the segment-shape values change slightly and short/continuous runs now read NaN there.
- **Renamed the method-ambiguous geometry fields to `<measure><Method>`** so a name says how it was
  computed: `width`→`widthFlank` (smoothed-slope flank feet) + `peaks`→`widthFlankIdx`;
  `filamentWidth`→`widthOuter` (outer filament points) + `filamentWidthIdx`→`widthOuterIdx`;
  `filamentHeight`→`heightP95`, `filamentHeightSmooth`→`heightSmooth`; `area`→`areaSimpson` (Simpson
  integration), `shoelaceArea`→`areaShoelace`. Also renames the 3D width-marker subjects
  (`"widthPoints"`→`"widthFlankPoints"`, `"filamentWidthPoints"`→`"widthOuterPoints"`) and updates
  `FEATURE_DISPLAY`, the feature lists, and `_INDEX_PAIR_FIELDS`; the `FEATURE_DISPLAY` colour-group
  names (`width`/`height`/`area`) are kept and now also serve as the selector's measure labels. The
  heat-map feature buttons now show `measure  [method] [method]` (e.g. `area  [simpson] [shoelace]`),
  which also fixes the paired-button label overlap. **Reprocess (`python profileProcessing.py`)** — the
  cache scalar/index datasets are keyed by field name, so pre-rename caches load these fields as unset.
- **Renamed the deposited-material term `bead` → `filament`** throughout the active code and docs, to
  match the concrete-3D-printing domain. This renames the four `profileData` fields
  `beadWidth`/`beadHeight`/`beadHeightSmooth`/`beadWidthIdx` → `filamentWidth`/`filamentHeight`/
  `filamentHeightSmooth`/`filamentWidthIdx`, the functions `width_from_bead_edges` →
  `width_from_filament_edges`, `measure_bead_height` → `measure_filament_height`, `measure_bead_area`
  → `measure_filament_area`, the plot subject `"beadWidthPoints"` → `"filamentWidthPoints"`, the
  heat-map feature keys, and the `BEAD_CENTER_HALFWIDTH` constant → `FILAMENT_CENTER_HALFWIDTH`
  (value unchanged). Pure rename — no numeric or plot-output change. Because the renamed fields are
  stored under their new names in the columnar cache (`/scalars/filament*`, `/points/filamentWidthIdx`),
  **reprocess (`python profileProcessing.py`) — pre-rename caches no longer load** (they raise on the
  missing `filamentWidthIdx` dataset). The earlier changelog entries below keep the old `bead*` names
  as the record of what those features were called when introduced.
- **Physical profile spacing in the 3D layout.** `plottingClass` now places each profile along the
  print path by its real along-track advance — `rollerbandSpeed` (m/s) × the inter-profile dt — via
  the new `profile_advance_distances`, instead of a uniform 20 mm gap. dt comes from the sensor
  clock (new `profileData.sensorTime` = `timestamp_sec` + `timestamp_usec`, loaded from the raw
  file), falling back to `arrivalTime`, then to a uniform gap when neither speed nor time is present
  (e.g. raw-only profiles). `plottingClass.__init__` now takes the profiles list (not a count) so it
  can read those fields, and `add_3d_points_to_plot` / `add_lines_to_plot` share the one physical
  path instead of each rebuilding a uniform one. Reprocess to populate `sensorTime` in the cache.
- **`width_from_smoothed_slope` now reports flank feet and is self-contained.** It folds the former
  `find_smooth_slope` smoothing in as locals (so `find_smooth_slope` is gone), and after finding the
  two outermost flank peaks it walks each flank down to its **foot** near the floor
  (`FLANK_FOOT_HEIGHT`), so the width markers sit at the bead base instead of mid-flank. `width`
  values shift outward (closer to `beadWidth`); the `peaks` field now holds the two flank-foot
  indices. Reprocess to refresh.

### Removed
- **Retired the legacy per-group processed-cache reader.** Now that the processed cache is the
  columnar table, `load_profiles` reads only that (`layout="columnar"`) and raw acquisition files
  (`profile_NNNNNN` groups → `x`/`z` + `arrival_time`/`timestamp_*`, via the slimmed `_load_raw`). An
  old per-group *processed* cache (`kind="processed"`, no `layout`) is now rejected with a clear
  "reprocess to the columnar layout" error instead of loaded partially — reprocess old Exp2/Exp3
  caches before use. Also dropped `dataAnalysis`'s raw-reload fallback (every columnar cache carries
  the leveling transform, so the overlay's raw is always reconstructed) and the now-unused `RAW_FILE`
  import there.
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
- **Dropped the stored `profileData.ySmooth` / `ySlopeSmooth` fields.** The smoothing they held is
  now computed locally inside `width_from_smoothed_slope` (see Changed); nothing else read them and
  they were never persisted (`_TRANSIENT_FIELDS` is now empty). Old caches still load.

### Added
- **Per-segment shape features (thinning / startup / rupture).** New `measure_segment_shape`
  (in the new **`segmentShape.py`** module, run in `profileProcessing.main` after `clean_flat_runs` /
  `measure_run_lengths`) characterises each filament segment's cross-section (`areaShoelace`, smoothed)
  along the print path and **broadcasts** six per-segment values onto every profile of the segment (like
  `segmentVolume`): `segmentBodyThinning` (%/mm body thinning, Theil–Sen over the body plateau),
  `segmentBodyThinningStability` (Spearman −1..1), `segmentCriticalArea` (cross-section at the rupture start,
  mm²), `segmentRuptureLength` (mm terminal cliff), `segmentHeadOvershoot` (% startup bulge), and
  `segmentRuptures` (0/1 gate — the rupture/critical fields are NaN on segments that ended thick). The
  **body** is the plateau where the smoothed area stays within a ±band of `bodyLevel` — from where the
  ramp settles in (found via the early-overshoot-peak → sustained-in-band scan) to where it leaves the band
  before the cliff — and the taper is fit there; validated on the 120 real segments to be robust to its own
  parameters (Spearman 0.99 to the band width, 0.998 to the smoothing), while faithfully representing the
  actual plateau. The rupture start is the derivative cliff top; a single-point "sharpness" was rejected as
  smoothing-fragile. The features slot into every plot via the **Segment**
  selector group and `FEATURE_DISPLAY`; in the feature-vs-PLC **stepwise** view they are aggregated
  **per segment** (added to `SEGMENT_FEATURES`), so they can be compared per `rollerbandSpeed` level
  (e.g. does the cross-section at rupture, or the rupture rate, depend on belt speed?). The feature-vs-PLC
  run table now tolerates per-feature NaN so a non-rupturing segment still contributes its taper. A
  per-profile **`segmentSection`** flag (1 body / 2 rupture / 3 the overshoot-peak band; the start ramp and
  the body↔rupture shoulder are NaN, since no feature uses them as a whole) is also set — an **all-points
  heat-map feature** that shows *exactly* the regions the features are computed over, for debugging. The
  heat-map feature selector was regrouped so the many features
  fit: segment features are laid out by **phase** (startup / body / rupture / runs / flags) via the new
  `FEATURE_UI` map (decoupled from the colour groups, so each keeps its own clim), the PLC channels are
  **packed two per row** with short labels, and each row's button pitch is sized to its labels. New
  `profileData` fields (auto-persist via the columnar
  cache). **Reprocess** to populate them.
- **`pipePressureDifference` derived PLC channel** (`pressurePipeStart - pressurePipeEnd`). A computed
  `profileData` `@property` (like `isNotFlat`), so it needs **no cache slot and no reprocess** — it reads
  the already-cached pipe pressures. Registered as a PLC channel via the new `plcData.ALL_PLC_COLUMNS`
  (raw `PLC_COLUMNS` + `DERIVED_PLC_COLUMNS`), which now feeds `FEATURE_DISPLAY`, the selector's `PLC`
  group, and the `dataAnalysis.py` plot cells — so it appears in the **3D heat-map**, **feature-vs-time**,
  and **both feature-vs-PLC** views (plus the stepwise cell's y-features). `PLC_COLUMNS` alone still drives
  CSV parsing (the derived name is not a log column). `profilePointsClass.py` + `plcData.py` +
  `profile3Dplotting.py` + `dataAnalysis.py`. Also hardened `featurePlcTrends._binned_trend` against an
  **all-NaN x-subject** (no finite pairs → `np.quantile` used to raise) — now returns an empty trend, so
  putting an unset subject on the continuous x-axis no longer crashes.
- **Feature-vs-PLC continuous view: any subject on either axis.** The `kind="continuous"` cell now
  offers a **shared pool of all features + every PLC channel on both axes** — a single-select x-picker and
  a multi-select y-panel — so you can plot feature-vs-channel, channel-vs-channel, or feature-vs-feature
  (e.g. torque vs pressure). Both selectors are **category-grouped** (Geometry / Segment / PLC / Other);
  the x-panel enforces single-select across its groups (`_on_xpick`, re-entrancy-guarded). `_render` now
  sources x from `self._phys[xkey]` for continuous (was the channel-only `self._chan`). Discrete channels
  are included — they render as vertical stripes in the hexbin (the stepwise cell remains the richer view
  for discrete x). The stepwise cell is unchanged. Grouping is factored into a shared
  `profile3Dplotting.group_by_category(keys)` reused by `featureComparison` and `featurePlcTrends`.
  `featurePlcTrends.py` + `dataAnalysis.py` (continuous cell now passes `DEFAULT_FEATURES` + `PLC_COLUMNS`).
- **Feature-vs-time comparison plot: grouped, colour-matched selector + optional initial smoothing.**
  `compare_features` now lays its left panel out **by category** (Geometry / Segment / PLC / Other,
  sharing `profile3Dplotting.SELECTOR_CATEGORIES` with the 3D heat-map) under a bold header per group,
  and **tints each "show" checkbox with its curve's colour** so the panel maps visually to the plot.
  Curves are drawn thicker (`lw` 1.0 → 2.0) and the whole panel is inset from the left edge, fixing the
  clipped checkboxes. New `initial_smooth` + `smooth_window_init` args open chosen curves **pre-smoothed**
  (the `dataAnalysis.py` cell now opens showing only the smoothed `printHeadTorque`); `PLC_FEATURES` moved
  to the config cell so either feature-vs-PLC cell runs standalone. `_SELECTOR_CATEGORIES` promoted to
  public `SELECTOR_CATEGORIES`. `featureComparison.py` + `dataAnalysis.py`; no cache change.
  - **Follow-ups:** the cell now offers **all features + every PLC channel** (`DEFAULT_FEATURES` +
    `PLC_COLUMNS`, grouped); when **exactly one curve is shown it is drawn in its real units** on a
    self-scaled y-axis (labelled with the feature's unit) instead of the normalised 0-1 overlay
    (`_apply_display`, driven by every show/smooth/window change); the checkbox tick-boxes are bigger
    (`s` 60 → 90); and the "smooth"/"show" column headers are spaced apart so they no longer collide.
- **Unified 3D heat-map with automatic point-set switching.** `plot_feature_heatmap` now **prebuilds one
  point cloud per point set** (`_build_one_cloud`) and swaps the visible one when the active feature
  changes (`_feature_pointset` / `_ALL_POINT_FEATURES`): filament geometry + PLC channels colour the
  **filament** points, while `defectLength` and the `isSegment` / `isNotFlat` flags (which live on
  floor/gap profiles) colour **all** points. Switching within a point set is instant (repoint the
  mapper); crossing point sets swaps which cloud is drawn (filament-only ↔ all) while keeping the camera,
  with a single shared colour bar re-tied to the active cloud (`add_scalar_bar(mapper=…)`). The three
  separate heat-map cells (main / defect / flags) collapse into **one** `dataAnalysis.py` cell listing
  every feature; the per-call `category` argument is gone. The **feature buttons are grouped** in a
  single left column under shadowed category headers — Geometry / Segment / PLC (/ Other), via
  `SELECTOR_CATEGORIES` — and **paired measures** (the two features sharing a FEATURE_DISPLAY colour
  group, e.g. `area` + `shoelaceArea`) sit **side by side on one row** to save height; the default 3D
  window is enlarged so the panel fits. `profile3Dplotting.py` only; no cache change.
- **Cleaned segment structure in a dedicated `isSegment` flag (was overloaded onto `isFlat`).** New
  per-profile `profileData.isSegment` (cached) holds the morphologically-cleaned "part of a real filament
  segment" result, so `isFlat` stays a faithful raw floorMask summary (`isFlat == floorMask.all()`) and
  the two concepts are separate. `clean_flat_runs` now writes `isSegment` (starting from `~isFlat`),
  leaving `isFlat` untouched; `measure_run_lengths`, `measure_filament_volume`, and `featurePlcTrends`'
  run table key off the new `_segment_mask` (the cleaned `isSegment`). The 3D flat/gap highlight
  (`plot()`'s `flat_colour` / `get_profile_points_for_plot(want_flat)`)
  now marks gaps by `~isSegment`. **View the flags in the heat map**: `isSegment` and `isNotFlat` (a
  derived property = inverse of `isFlat`, so both read 1 = filament and share the colour) are 0/1
  heat-map features (share a 0-1 colour group), shown over **all** points in the unified heat-map (see
  above) so bridged floor-only gaps are visible and the cleanup is inspectable. **Reprocess to populate
  `isSegment`.**
- **Morphological cleanup of the filament run structure (`clean_flat_runs`).** New step in
  `profileProcessingAlgorithms.py`, run in `profileProcessing.main()` after the PLC join and **before**
  `measure_filament_volume` / `measure_run_lengths`, that de-noises the segment/defect runs the
  categorisation produces: it **bridges** gaps shorter than `SEGMENT_MERGE_GAP_MM = 5 mm` (flanked by
  segment — a single stray flat profile no longer splits a segment) and then **drops** segments shorter
  than `MIN_SEGMENT_LENGTH_MM = 10 mm` (isolated non-flat blips). Bridging first, so a real segment split
  by a tiny gap is rejoined before the length test. Result stored in `isSegment` (see above); `floorMask`
  and the per-profile geometry are unchanged. Effect on Exp1: **182 → 120 filament segments**, and the
  per-speed counts match a visual count (e.g. rollerbandSpeed 0.16: 52 → 27, 0.14: 29 → 18). **Reprocess
  (`python profileProcessing.py`) to apply** — `isSegment`/`segmentVolume`/`sliceVolume`/`segmentLength`/
  `defectLength` in the cache change. Uses physical distances (`profile_advance_distances`), so it must
  run post-join.
- **Feature-vs-PLC correlation plot (`featurePlcTrends.plot_feature_plc_trends`).** New matplotlib
  module putting a PLC channel on x and per-profile feature(s) on y, to see whether a geometry feature
  tracks a machine input. Built for one channel family at a time via a `kind` argument, so it opens as
  **two `dataAnalysis.py` cells**: `kind="stepwise"` for the discrete/setpoint channels
  (`STEPWISE_CHANNELS` = `rollerbandSpeed`, `printHeadMixxingSpeed`, the two viscotec pumps) and
  `kind="continuous"` for the analogue signals (`CONTINUOUS_CHANNELS` = `pressurePrintHead`,
  `printHeadTorque`, `pressurePipeEnd`, `pressurePipeStart`); `mortarPumpFlow` and `rollerbandHeight`
  are omitted. One feature shown → the rich single-feature view in real units — a **box or violin** per
  level (stepwise; a **box/violin shape radio**) or a **hexbin density + central-per-quantile-bin trend
  + spread band** (continuous) — with a **Spearman r**; two or more → each collapses to one **normalised
  0–1** central±band trend line, with each feature's r in the legend. A **median/mean radio** switches
  the central statistic everywhere (median+IQR ↔ mean+std) and, in the single stepwise view, moves the
  central line on the box/violin. Live `CheckButtons` toggle features and a `RadioButtons` list picks
  the x-channel; **constant channels are force-shown** (the viscotec pumps read 0 in the current cache),
  and a constant channel's Spearman shows `n/a`. Offered features are the per-profile geometry measures
  plus the broadcast segment/defect aggregates (`segmentVolume`/`segmentLength`/`defectLength`). In the
  **stepwise** view these `RUN_FEATURES` are aggregated **per run** (one datapoint per segment/defect,
  via `_contiguous_runs` on the full profile list), not per profile — fixing the profile-count weighting
  — with, per level: an **`n=<count>` annotation** above every level, the box/violin drawn only where a
  level has **> 3** runs (`MIN_RUNS_FOR_BOX`) else the count alone, and runs **excluded** when longer
  than **0.5 m** (`MAX_RUN_LENGTH_MM`) or when they **span more than one level** of the active channel.
  The stepwise **x-axis is fixed** to the channel's full level set (stable across feature switches /
  empty levels). Idle profiles (`rollerbandSpeed == 0`) are excluded. Reuses `featureComparison`'s
  `_feature_values`/`_scale_range`/`_normalise` and `FEATURE_DISPLAY` (Spearman via `scipy.stats`, guarded
  for constant inputs). Needs a GUI backend (like `compare_features`). No pipeline/cache change.
- **Filament-segment length + pure-floor defect length (mm heat-map features).** New
  `measure_run_lengths()` in `profileProcessingAlgorithms.py` sums the along-path advance
  (`profile_advance_distances`) over each maximal run of like profiles and broadcasts the total onto
  every profile in the run: `segmentLength` on a filament segment (run of non-flat profiles) and
  `defectLength` on a pure-floor gap (run of flat, no-filament profiles). Both are stored in distance
  units and shown in **mm** (factor 0.01), each its own heat-map colour group. Runs in
  `profileProcessing.main()` after the PLC join (needs `rollerbandSpeed` for physical distances), next
  to `measure_filament_volume`. `segmentLength` is selectable in the filament heat-map; because
  `defectLength` lives on flat profiles (which have no filament points), `plot_feature_heatmap` gained
  a `category` argument ("profile" default / "floor") and a new `dataAnalysis.py` cell colours the
  floor points by `defectLength`. Reprocess to populate the new fields.
- **Live colour-scale modes for the feature heat-map.** `plot_feature_heatmap` gained a second radio
  group (bottom-right corner) that sets the active feature's colour scale in real time — **linear**
  (full min–max, default, unchanged), **log** (`LookupTable.log_scale`; range from the smallest
  positive value, falling back to linear for a feature with no positive values), **clip** (2–98th
  percentile over the *distinct* per-segment values, so one outlier segment saturates instead of
  squashing the rest — size-unbiased), and **rank** (dense rank 0–1, tie-safe so a profile's points
  stay one colour). Fixes a single large `segmentVolume` making the smaller segments indistinguishable.
  Each mode is precomputed per feature in `_build_feature_cloud` (a `__rank` companion array + a
  per-mode clim map) and applied by the new `_apply_scale`; the colour-bar label announces the mode.
  All in `profile3Dplotting.py`; no cache/pipeline change.
- **Filament-segment volume (two measures, cm³ heat-map features).** New `measure_filament_volume()`
  in `profileProcessingAlgorithms.py` integrates each profile's cross-section (`shoelaceArea`) along the
  print path over each **filament segment** — a run of consecutive non-flat profiles bounded by flat,
  pure-floor profiles (`isFlat`). It sets two per-profile fields (native units area-unit×distance-unit;
  `FEATURE_DISPLAY` converts by `1e-9` to **cm³**): `segmentVolume` (the segment's total, `Σ shoelaceArea
  × inter-profile advance`, broadcast onto every profile in the run so a segment reads as one colour) and
  `sliceVolume` (each profile's own slab `shoelaceArea × its gap`, so the heat-map shows variation along a
  filament). Both are selectable heat-map features, each in its own colour group (a segment total is
  ~10²–10³× a single slice, so they don't share a `clim`). It runs in `profileProcessing.main()` **after**
  the PLC join (it needs `rollerbandSpeed` for the physical inter-profile distances) rather than inside
  `process_profiles`. The distance helper `profile_advance_distances` (with `METERS_TO_PROFILE_UNITS` /
  `UNIFORM_PROFILE_DISTANCE`) moved from `profile3Dplotting.py` to `profileProcessingAlgorithms.py` so both
  the 3D layout and the volume step share it (no plotting→processing back-edge, no duplication); the layout
  imports it back and is unchanged. Reprocess to populate the new fields.
- **Store the leveling transform so the before/after overlay needs no raw reload.**
  `rotate_and_shift_uniform` now returns its uniform `(angle, offset)`, `process_profiles` passes it
  out, and `profileProcessing` saves it on the cache as `level_angle`/`level_offset`. The new
  `unlevel_profiles()` inverts that transform to reconstruct each profile's pre-leveling x/z exactly
  (the transform is uniform, applied first, and the only x/z mutator), so `dataAnalysis` builds the
  grey "before" overlay from `processed` alone — the multi-minute second load of the raw file is
  gone (with a fallback to the old raw-slice load for caches predating the attrs). Reprocess to
  populate them.
- **Optional voxel downsampling for the 3D clouds.** `plottingClass(profiles, voxel_size=...)` keeps
  one point per `voxel_size`-cubed voxel (profile units; `None` = off, default) via a new
  `_maybe_voxel` helper (`np.unique` on binned coords + `PolyData.extract_points`, which carries the
  heat-map's per-feature scalar arrays along). Cuts the point count/overdraw so the now-densely-spaced
  layout stays responsive on large sets; composes with `profile_step`/`point_step`. Wired into
  `dataAnalysis.py` via a `VOXEL_SIZE` knob. (A `points_gaussian` splat mapper was evaluated too but
  renders invisibly with solid-colour clouds in this PyVista/VTK, so it was not included.)
- **Central dataset config.** New `datasetConfig.py` holds the active experiment's `RAW_FILE` /
  `PLC_FILE` (and the derived `PROCESSED_FILE`) in one place, imported by `profileProcessing.py` and
  `dataAnalysis.py` so their paths can't drift; other datasets are kept as commented blocks for a
  one-line switch. Switched the active dataset to Exp1 (velocity alterations).
- **Interactive feature-comparison plot.** New `featureComparison.py` (`compare_features`, backed by
  a `FeatureComparisonPlot` class) overlays several per-profile features (geometry measures and/or
  joined PLC channels) as time series on one matplotlib axis, each robustly normalised to 0–1 by
  percentile clipping (`clip_percentile`, default 1–99) so an outlier spike doesn't flatten the
  curve. Each legend entry shows both `scale[lo–hi unit]` (the range that maps to 0→1) and
  `true[min–max]` (the real extremes, so a clipped outlier is visible as `true max` ≫ `scale hi`),
  and every feature's colour stays in the legend regardless of visibility (bold text marks the
  currently-shown curves). Left-panel checkboxes toggle each curve ("show") and moving-average-smooth
  selected curves ("smooth", NaN-aware, fixed scale so the legend doesn't shift), with a slider
  setting the smoothing window. The x-axis is time-from-start (from `arrivalTime`). Reuses
  `FEATURE_DISPLAY` for units; new `dataAnalysis.py` cell. Needs a GUI backend for interactivity (as
  with the PyVista window).
- **PLC machine-log join by timestamp.** New `plcData.py` loads the machine PLC log
  (`waterContenExp2PLC.csv`) — parsing Windows FILETIME → Unix, correcting the PLC clock offset
  (`PLC_CLOCK_OFFSET_S = 564 s`) — into a staging `PlcLog`, and `join_plc_to_profiles()` attaches
  the **nearest-in-time** PLC sample to each profile. The two streams run at different rates
  (~226 Hz profiles vs 50 Hz PLC), so the join is by absolute time, not index: profiles carry a new
  `arrivalTime` field (mapped from the raw sensor `arrival_time` in `load_profiles`) and the 10 PLC
  channels become flat `profileData` fields, auto-persisted by the generic I/O and selectable as
  heat-map features (each its own colour range). Profiles outside the streams' mutual overlap are
  dropped; `profileProcessing` records the surviving raw span as `raw_start`/`raw_end` file attrs
  (read via the new `read_file_attrs()`), so `dataAnalysis` loads the matching raw slice
  automatically. Runs in `profileProcessing.main` after the geometry pipeline (full range by
  default); reprocess to populate.
- **Robust bead height (two measures).** `measure_bead_height()` in `profileProcessingAlgorithms.py`
  sets `profileData.beadHeight` (95th percentile, `HEIGHT_PERCENTILE`, of the bead points' z) and
  `beadHeightSmooth` (max of a median-smoothed profile, `MEDIAN_SMOOTH_WINDOW`) above the shared
  `z = 0` median floor — both robust to outlier/noise spikes, unlike a raw max, and cross-checking.
  NaN for flat profiles. Runs in `process_profiles` and both are selectable heat-map features.
  Reprocess to populate.
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
