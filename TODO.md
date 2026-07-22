# TODO / Backlog

The single home for **suggested** improvements. Nothing here has been applied yet —
this pass was documentation-only and behaviour-preserving. Work top-down; move
completed items to [CHANGELOG.md](CHANGELOG.md). Add newly discovered debt here.

Priorities: **P1** = correctness / blocks future work · **P2** = maintainability ·
**P3** = polish / nice-to-have.

---

## P1 — Correctness (do before relying on the affected code)

- [ ] **Fix `.y` → `.z` in the registration path.** `profileRegistration.py`
      (`generateJoinedProfile`, `translateInX`, `registerAndShiftProfiles`) reads a `.y`
      field that `profileData` no longer has. Dormant today, but will `AttributeError`
      the instant registration is re-enabled. (The other `.y` users — the CSV loaders and
      the `dataAnalysis.py` commented block — have since been removed.)
- [x] ~~**Resolve the `ProfileData` / `profileData` name collision.**~~ Done — the
      wire-format class in `rawProfileUdpCapturing.py` was renamed to `ProfileDataRaw`. See
      [CHANGELOG.md](CHANGELOG.md).

## P2 — Maintainability

- [ ] **Introduce a `config.py` (or constants module)** for the unit-dependent
      magic numbers, keeping values **identical**:
      smoothing windows `15/9/5/5/65/55/15/5`; peak `height=0.15`, `distance=50`;
      border/height threshold `20`; baseline `borderPoints=30`, error `50`;
      print-path geometry `2000`, `5000`, `80000`; UDP `192.168.0.251:1234`.
      Document each with its meaning and unit (~0.01 mm).
- [ ] **Replace wildcard imports** (`from x import *`) with explicit imports,
      module by module as files are touched. Start with the entry points.
- [ ] **Add type hints + docstrings** incrementally to the processing functions in
      `profilePointsClass.py` and the plotting methods — when editing them, not in
      one giant pass.
- [ ] **Parameterise hardcoded paths & the sensor IP/port** (CLI args or a small
      config) so the scripts run on another machine. Remove the absolute path in
      `loadProfiles()`.
- [ ] **Harden the UDP boundary** in `rawProfileUdpCapturing.py`: replace the broad
      `except Exception` with specific exceptions, and bound/expire the
      `measurement_dict` / `zprofile_dict` buffers so out-of-order packets can't
      grow memory without limit.

## P3 — Polish & tooling

- [x] ~~**Remove dead code & unused imports** in the active pipeline.~~ Done — deleted the CSV
      loaders (`loadProfiles`/`plotProfiles`/`groupProfiles`), `find_border_points`, the
      `borderPoints` field, and unused imports; see [CHANGELOG.md](CHANGELOG.md). Remaining
      (out of scope): dead code in legacy `profileRegistration.py`. The `profilePointsClass.py`
      area/max-height block is intentionally kept for later.
- [ ] **Migrate `print` → `logging`** in `rawProfileUdpCapturing.py` (and elsewhere) with a
      module-level logger.
- [ ] **Add a `pytest` suite** under `tests/`, starting with the pure functions
      (`moving_average`, `get_baseline_from_profileBorder`, width detection on a
      synthetic bump). No sensor/network/display dependence.
- [ ] **Add lint/type-check config** (`ruff`, `mypy`) and a `pyproject.toml` once
      tooling is wanted. Wire the commands listed in [CLAUDE.md](CLAUDE.md) §17.
- [ ] **Clarify the coordinate mapping** in plotting (height → PyVista y slot) —
      either rename for clarity or document inline; see ARCHITECTURE.md §5.
- [ ] **Consume HDF5 metadata** on the analysis side (timestamps, encoder, quality)
      — currently written by the capturer but ignored by `load_profiles`.
- [x] ~~**Avoid loading the whole raw file when only a slice is needed.**~~ Done —
      `load_profiles(fileName, start, end)` reads only the range and `count_profiles()`
      gives a fast pre-check. See [CHANGELOG.md](CHANGELOG.md).
- [ ] **Persist smoothed arrays in the processed cache if a "plot smoothed profile"
      view is wanted.** `save_profiles` skips the bulky intermediates listed in
      `_TRANSIENT_FIELDS` (`ySmooth`/`ySlopeSmooth`/`ySlope`); drop one from that set to
      persist it.
- [ ] **Smoother large-cloud rendering beyond subsampling.** `plot()` now supports
      `profile_step`/`point_step` decimation (~37M points lags otherwise). If full
      detail with smooth interaction is needed, investigate a VTK level-of-detail
      actor (full resolution when still, decimated while interacting).

---

## Source TODOs already living in the code (carried over)

- `profileProcessingAlgorithms.py`: several smoothing / peak-detection constants remain
  empirical (now named — `MIN_PROFILE_POINTS`, `SLOPE_PEAK_MIN_HEIGHT` / `_DISTANCE`, the
  smoothing windows).
- `profile3Dplotting.py`: the height → PyVista-y coordinate mapping is noted as confusing
  (kept as an inline design note).
- `profileRegistration.py`: "make a class again from this?"; joined-profile logic
  only handles <20% overhang; point ordering for area calc.
- `rawProfileUdpCapturing.py`: NTP time sync strategy undecided; Python-side time sync.
