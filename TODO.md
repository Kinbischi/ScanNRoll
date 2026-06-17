# TODO / Backlog

The single home for **suggested** improvements. Nothing here has been applied yet —
this pass was documentation-only and behaviour-preserving. Work top-down; move
completed items to [CHANGELOG.md](CHANGELOG.md). Add newly discovered debt here.

Priorities: **P1** = correctness / blocks future work · **P2** = maintainability ·
**P3** = polish / nice-to-have.

---

## P1 — Correctness (do before relying on the affected code)

- [ ] **Fix `.y` → `.z` in the registration path.** `profileRegistration.py`
      (`generateJoinedProfile`, `translateInX`, `registerAndShiftProfiles`),
      `profileLoading.py` (`plotProfiles`, `loadProfiles`), and the commented
      block in `LidarProfileAnalysis.py` read a `.y` field that `profileData` no
      longer has. Dormant today, but will `AttributeError` the instant
      registration is re-enabled. Fix all references together.
- [x] ~~**Resolve the `ProfileData` / `profileData` name collision.**~~ Done — the
      wire-format class in `udpCapturing.py` was renamed to `ProfileDataRaw`. See
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
- [ ] **Harden the UDP boundary** in `udpCapturing.py`: replace the broad
      `except Exception` with specific exceptions, and bound/expire the
      `measurement_dict` / `zprofile_dict` buffers so out-of-order packets can't
      grow memory without limit.

## P3 — Polish & tooling

- [ ] **Remove dead code & unused imports**: the large commented blocks in
      `LidarProfileAnalysis.py`, `profileRegistration.py`, `profilePointsClass.py`;
      unused `copy`, `NearestNeighbors`, and `pyvista` (in `profilePointsClass.py`);
      unused legacy `loadProfiles` / `plotProfiles` if confirmed obsolete.
- [ ] **Migrate `print` → `logging`** in `udpCapturing.py` (and elsewhere) with a
      module-level logger.
- [ ] **Add a `pytest` suite** under `tests/`, starting with the pure functions
      (`moving_average`, `get_baseline_from_profileBorder`, width detection on a
      synthetic bump). No sensor/network/display dependence.
- [ ] **Add lint/type-check config** (`ruff`, `mypy`) and a `pyproject.toml` once
      tooling is wanted. Wire the commands listed in [CLAUDE.md](CLAUDE.md) §17.
- [ ] **Clarify the coordinate mapping** in plotting (height → PyVista y slot) —
      either rename for clarity or document inline; see ARCHITECTURE.md §5.
- [ ] **Consume HDF5 metadata** on the analysis side (timestamps, encoder, quality)
      — currently written by the capturer but ignored by `load_hdf5_profiles`.

---

## Source TODOs already living in the code (carried over)

- `profilePointsClass.py`: `profileNumber` not always parsed correctly;
  several "empirical value" thresholds need justification.
- `plottingProfiles3D.py`: width-points plotting marked "not working"; coordinate
  system noted as confusing; "check whether real profiles need 180° inversion".
- `profileRegistration.py`: "make a class again from this?"; joined-profile logic
  only handles <20% overhang; point ordering for area calc.
- `udpCapturing.py`: NTP time sync strategy undecided; Python-side time sync.
