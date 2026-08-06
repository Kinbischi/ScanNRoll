# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository. Read this
before editing. It is written to keep behaviour **consistent across models and
sessions** (Sonnet, Opus, future versions). When in doubt, prefer the simplest
change that preserves behaviour.

> **Guiding principle: do not overcomplicate.** This is a single-developer PhD
> research codebase. Favour small, readable, incremental changes over clever
> abstractions or large rewrites. If a task seems to require a big refactor,
> stop and propose it first (see *Large refactors*).

---

## 1. Project purpose

Capture, store, process and visualise 2D laser-line **profiles** from a Baumer
OX200 LIDAR sensor for additive-manufacturing ("Rollerband") research.

**Domain primer (one read and you understand the data):**
- A **profile** = one cross-sectional scan: arrays `x` (across-track position)
  and `z` (height). Units: 1 unit ≈ 0.01 mm.
- The **baseline / floor** = the flat substrate the printed filament sits on; estimated from
  each profile's left/right edges (`get_baseline_from_profileBorder`). Leveling applies
  **one median** rotation + shift to the whole set (keeping the real relative heights
  between profiles) and stores each profile's own floor fit in `m`/`b`.
- **Floor vs filament categorization** = a per-point `floorMask` splitting each profile into the
  flat **floor** (substrate) and the raised **filament** (printed material). Most downstream
  features build on it; a profile with no filament points is **flat**.
- The **width** = how wide the filament is, measured two ways: between the two outermost
  smoothed-slope peaks, and directly as the x-span between the outer filament points.
- The **area** = the filament's cross-section above the shared `z = 0` median floor, measured two
  ways as a cross-check: Simpson integration (`area`) and the shoelace polygon (`shoelaceArea`).
- The **volume / length** = per-run aggregates along the print path (a *segment* = a run of filament
  profiles between flat ones): segment volume (`segmentVolume`, cross-section integrated over the
  segment) and per-profile slab (`sliceVolume`); segment length (`segmentLength`) and the length of
  each pure-floor gap between segments (`defectLength`). Computed after the PLC join (they need the
  physical inter-profile spacing from `rollerbandSpeed`).
- **Registration** (currently dormant) = aligning overlapping left/centre/right
  profiles in `x` and joining them into one combined profile.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline and module map.

---

## 2. Folder & file structure

```
Python/
├── rawProfileUdpCapturing.py             # ACTIVE  acquisition: sensor → HDF5
├── datasetConfig.py                      # ACTIVE  central dataset paths (RAW / PLC / derived PROCESSED)
├── profileProcessing.py                  # ACTIVE  process entry point: raw → processed cache
├── dataAnalysis.py                       # ACTIVE  plot workbench (# %% cells): cache → 3D plot (raw "before" reconstructed)
├── profilePointsClass.py                 # ACTIVE  profileData dataclass ONLY (pure data model)
├── profileProcessingAlgorithms.py        # ACTIVE  processing algorithms (rotate/level/smooth/width/flatness)
├── profileLoading.py                     # ACTIVE  load_profiles/save_profiles (HDF5 columnar-cache I/O)
├── plcData.py                            # ACTIVE  PLC machine-log CSV load + timestamp join
├── profile3Dplotting.py                  # ACTIVE  PyVista 3D plotting + print-path geometry
├── featureComparison.py                  # ACTIVE  matplotlib 2D feature-vs-time comparison plot
├── profileRegistration.py                # LEGACY  alignment/joining (dormant, has .y bug)
├── HDf5data/  ProfileData/  Pics/        # data & outputs (git-ignored)
├── old/  RandomOther/  testGIthubCircleSquare/  # scratch / experiments (incl. old entry points)
└── *.md, requirements.txt, .gitignore    # docs & scaffolding
```

**Active vs legacy matters.** Do **not** "fix" or refactor legacy/dormant code
unless explicitly asked — it is kept for reference and is not on the live path.
If you touch legacy code, say so explicitly and explain why.

---

## 3. The one convention you must not break

**Height lives in `z`, position in `x`.** The analysis `profileData` dataclass
has fields `x` and `z` and **no `y`**. Any code reading `.y` off a `profileData`
is legacy and incorrect (it predates the rename). Never introduce new `.y` access
on `profileData`. (In 3D plotting, height is currently mapped into PyVista's y
slot — that is a plotting detail, documented in ARCHITECTURE.md, not a data field.)

There are **two** distinct profile dataclasses — keep them separate:
- `profilePointsClass.profileData` — the analysis model (`x`, `z`, derived fields).
- `rawProfileUdpCapturing.ProfileDataRaw` — the wire/storage model (packet + measurement fields).

Be explicit about which one you mean. Do not merge them in a drive-by edit.

---

## 4. Coding standards

- **Python style:** PEP 8. 4-space indent, `snake_case` functions/variables,
  `PascalCase` classes.
  - *Existing exception:* the class is named `profileData` (lowercase). Do **not**
    mass-rename it — too many call sites. New classes use PascalCase.
- **Imports:** for **new or edited** modules, use explicit imports
  (`from profilePointsClass import profileData, find_smooth_slope`) rather than
  `from x import *`. Don't do a repo-wide import rewrite unless asked; just don't
  add new wildcard imports.
- **Type hints:** add them to **new and modified** functions — parameters and
  return types. Use `list[profileData]`, `np.ndarray`, `Optional[...]`. Don't
  back-fill the entire codebase in one pass.
- **Docstrings:** required on new/modified **public** functions and classes. One
  short summary line; add Args/Returns only when not obvious. Keep math-heavy
  functions' docstrings focused on *what* and *units*, not line-by-line *how*.
- **Readability over cleverness.** Prefer a clear loop to an obscure vectorised
  one-liner unless performance demands it (see *Performance*).
- **Functions stay focused.** If a function does loading *and* parsing *and*
  plotting, that's a smell — but only split it when you're already editing it for
  another reason. No speculative refactors.

---

## 5. Magic constants

The code is full of unit-dependent empirical constants (smoothing window sizes,
peak thresholds, the `20` height/border cutoff, `30`/`50` baseline parameters,
print-path geometry `2000/5000/80000`, UDP address/port, parser byte offsets).
Units are roughly hundredths of a millimetre (the comments note "20 = 0.20 mm").

- When you **add** a tunable number, give it a named constant with a comment
  stating its **meaning and unit**, near the top of the module (or in a future
  `config.py` — see [TODO.md](TODO.md)).
- When you **touch** code containing a magic number, it is fine (encouraged) to
  lift it into a named constant — but **keep the value identical** to preserve
  behaviour.
- Do not invent new "reasonable" values for existing constants; they are tuned to
  real sensor data.

---

## 6. Error handling

- **Boundaries** (file I/O, UDP parsing, HDF5 reads) should fail loudly and
  specifically. Prefer catching concrete exceptions over bare `except Exception`.
  The existing broad `except` in `rawProfileUdpCapturing.run_udp_listener` is known debt —
  don't copy that pattern into new code.
- **Pure processing functions** may assume valid numpy input but should guard the
  documented edge cases the code already handles (e.g. empty / too-short profiles:
  see the `x.shape[0] < 10` guard). Preserve those guards when editing.
- Validate at the point of entry, not deep in the call stack.

---

## 7. Logging

- New code should use the stdlib `logging` module, not `print`. Module-level
  `logger = logging.getLogger(__name__)`.
- Existing `print` calls in `rawProfileUdpCapturing.py` are acceptable to leave; migrate
  them only when doing related work (tracked in [TODO.md](TODO.md)).
- Log levels: `DEBUG` for per-packet/per-profile detail, `INFO` for lifecycle
  ("loaded N profiles"), `WARNING` for recoverable anomalies, `ERROR` for failures.

---

## 8. Testing philosophy

- No test suite exists yet. When adding tests, use **pytest** under a `tests/`
  directory.
- Start with the **pure, deterministic functions** — highest value, easiest to
  test: `moving_average`, `get_baseline_from_profileBorder`, and width detection
  on synthetic profiles (build a known bump, assert the measured width).
- Tests must not require the sensor, a network, or a display. Generate synthetic
  `profileData` in fixtures.
- A change to processing logic should come with a test that would have caught the
  old behaviour, when practical.

---

## 9. Performance considerations

- Data is numpy arrays; profiles are modest in size. **Do not pre-optimise.**
- The known hot spot is `registerAndShiftProfiles` (`scipy.optimize.minimize`
  with a KDTree query per profile) — but it's dormant. Don't optimise dormant code.
- A few loops build arrays with repeated `np.append` (O(n²)); fine for current
  sizes. If you're already rewriting such a loop, prefer preallocation or list +
  single `np.array`, but don't make that the goal of an unrelated change.
- Profile correctness first, speed only when a real bottleneck is measured.

---

## 10. Architectural principles

- `profilePointsClass` is the **base layer** — it holds only the `profileData` data
  model and must not import other project modules. Keep it dependency-free internally.
- `profileProcessingAlgorithms` sits one layer up: pure processing functions that import
  only `profilePointsClass`. The `process_profiles` pipeline that composes them lives in
  the `profileProcessing.py` entry point.
- Acquisition (`rawProfileUdpCapturing.py`) stays **standalone**. Don't couple it to the
  analysis modules.
- One responsibility per module: loading, processing, plotting, registration,
  acquisition stay separate.
- Data flows one direction: acquisition → storage (HDF5) → loading → processing →
  visualisation. Don't create back-edges.

---

## 11. Avoiding duplicate functionality

- **Search before you write.** Before adding a helper, grep for an existing one
  (baseline fitting, smoothing, peak finding, path geometry all already exist).
- Baseline/floor estimation lives **only** in
  `get_baseline_from_profileBorder` (`profileProcessingAlgorithms.py`). Reuse it; do not
  reimplement a second baseline fit.
- Smoothing goes through `moving_average` (`profileProcessingAlgorithms.py`). Print-path/
  tilt geometry goes through `compute_print_path_and_angle` (`profile3Dplotting.py`).
  Don't fork these.

---

## 12. Rules for modifying existing files

1. **Preserve behaviour** unless the task explicitly asks to change it. Numeric
   results, plot output, and HDF5 layout must stay identical otherwise.
2. Make the smallest change that satisfies the request. Don't reformat unrelated
   lines (keeps diffs reviewable).
3. If you spot debt while editing, **note it in [TODO.md](TODO.md)** rather than
   fixing it inline — unless the fix is trivial and directly related.
4. Keep one logical change per commit.

## 13. Rules for adding new modules

- Place new analysis modules at the repo root alongside the existing ones (flat
  layout — no `src/` package yet; that's a deliberate "keep it simple" choice).
- A new module must declare its dependencies with explicit imports and must not
  create a circular import (respect the layering in §10).
- Add the module to the table in [ARCHITECTURE.md](ARCHITECTURE.md).

## 14. Backwards compatibility

- The **HDF5 schema** (group naming `profile_NNNNNN`, dataset/attribute names in
  `HDF5ProfileWriter`) is a contract with already-captured data files. Don't
  rename or drop fields; only add new optional ones.
- The analysis `profileData` field names are read across modules — adding fields
  is safe (they're `Optional` with defaults); renaming/removing is a breaking
  change that requires updating every call site in the same commit.

## 15. Large refactors

Before any change spanning multiple files or altering structure, **post a short
plan first** and get agreement:
1. Your understanding of the current architecture.
2. The specific problems being addressed.
3. The proposed plan (incremental steps).
4. Expected benefits.
5. Files affected.
Then proceed step by step, keeping the project runnable after each step.

## 16. Documentation standards

- Keep [ARCHITECTURE.md](ARCHITECTURE.md) in sync when you change module
  responsibilities or the dependency graph.
- Record notable changes in [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.
- Move completed items out of [TODO.md](TODO.md); add newly discovered debt to it.
- Docs are Markdown at the repo root. No `docs/` directory until the project is
  large enough to need it.

---

## 17. Common commands

```bash
# Environment
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Run acquisition (sensor → HDF5)
python rawProfileUdpCapturing.py

# Process raw profiles into the processed cache (run once / when params change)
python profileProcessing.py

# Run the 3D visualisation from the processed cache (needs a display)
python dataAnalysis.py
```

### Build / test / lint (recommended, not yet wired up)

These tools are **not** configured yet — adding them is tracked in
[TODO.md](TODO.md). When present, the expected commands are:

```bash
pytest                 # run the test suite
ruff check .           # lint
ruff format .          # format
mypy .                 # type-check
```

Until then, the only "build" is running the scripts above.

---

## 18. Checklist for future sessions

- [ ] Read this file and [ARCHITECTURE.md](ARCHITECTURE.md) before editing.
- [ ] Confirm whether the file you're touching is **active** or **legacy** (§2).
- [ ] Preserve behaviour; keep magic-number values identical when lifting them.
- [ ] Add type hints + a docstring to anything new you write.
- [ ] No new wildcard imports; no new `.y` access on `profileData`.
- [ ] Record discovered debt in [TODO.md](TODO.md), notable changes in
      [CHANGELOG.md](CHANGELOG.md).
- [ ] For anything structural, propose a plan first (§15).
- [ ] Don't overcomplicate.
