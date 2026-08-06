# Rollerband — LIDAR Profile Analysis

Tools for capturing, storing, processing and visualising 2D laser-line profiles
from a **Baumer OX200** LIDAR sensor, as part of a PhD research project on
"Rollerband" additive-manufacturing prints.

A *profile* is a single cross-sectional scan: a row of `(x, z)` points where `x`
is the position across the sensor's field of view and `z` is the measured height.
Many profiles taken along a print path reconstruct the 3D shape of a printed filament.
Processing splits each profile into the flat **floor** (the substrate) and the raised
**filament** (the printed material), and measures the filament's width, cross-sectional area,
per-segment volume, and segment/defect lengths. (Units: 1 unit ≈ 0.01 mm.)

---

## What the project does

1. **Capture** — listen for UDP packets from the sensor and write each profile
   (plus its measurement metadata) into an HDF5 file.
2. **Process** — level every profile against the substrate (one median rotation + shift
   for the whole set), categorise each point as **floor** or **filament**, flag substrate-only
   ("flat") profiles, measure the filament **width** two ways (from the smoothed-slope peaks
   and from the outer filament points), its cross-sectional **area** two ways (integration and
   the shoelace formula), and — per filament segment along the print path — its **volume** and
   **length** (plus the **defect length** of each pure-floor gap). Cached to a small processed HDF5.
3. **Visualise** — lay the profiles out along a computed print path and render them as an
   interactive 3D point cloud (floor vs filament in colour, flat profiles highlighted, width
   markers, baselines), plus a feature heat-map coloured by any per-profile measure with a live
   feature selector and colour-scale modes (linear / log / clip / rank).

A separate (currently dormant) *registration* path aligns and joins overlapping
profiles; see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell:  .venv\Scripts\Activate.ps1)
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### Capture profiles from the sensor

```bash
python rawProfileUdpCapturing.py
```
Listens on the UDP address/port set at the top of `rawProfileUdpCapturing.py`
(`192.168.0.251:1234` by default) and appends to an HDF5 file.

### Process, then visualise stored profiles

Processing is decoupled from plotting so the view can be re-run cheaply. Run the
process step once (writes a processed cache), then the plot step as often as you like:

```bash
python profileProcessing.py   # raw HDF5 → process → processed cache (run once)
python dataAnalysis.py      # processed cache → interactive 3D view (run freely)
```
`profileProcessing.py` reads the raw file named at the top of the script (optionally a
profile index range); `dataAnalysis.py` loads the processed cache and opens an
interactive PyVista 3D window (requires a display).

> **Note:** file paths and the sensor IP are currently hardcoded in the scripts.
> See [TODO.md](TODO.md) for the plan to parameterise them.

---

## Data flow

```
 Baumer OX200 ──UDP──> rawProfileUdpCapturing.py ──> HDf5data/*.h5 (raw)
                                              │
                                              ▼
                                  profileProcessing.py ──> *_processed.h5 (cache)
                                                              │
                                                              ▼
                                                        dataAnalysis.py
                                                        (load cache → 3D plot)
```

## Where things live

| Path                  | Contents                                              |
| --------------------- | ----------------------------------------------------- |
| `*.py` (root)         | Source modules and entry-point scripts                |
| `HDf5data/`           | Captured profile data (HDF5) — not version-controlled |
| `ProfileData/`        | Older CSV profile exports (unused; the CSV loaders were removed) |
| `Pics/`               | Saved figures / screenshots                           |
| `old/`, `RandomOther/`, `testGIthubCircleSquare/` | Scratch / legacy experiments      |
| `Baumer_OX200_EN.pdf` | Sensor datasheet & UDP protocol reference             |

---

## Documentation map

| File                                 | Purpose                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| [ARCHITECTURE.md](ARCHITECTURE.md)   | Components, module dependencies, execution flows, risks  |
| [CLAUDE.md](CLAUDE.md)               | Conventions & guidance for AI / future contributors      |
| [TODO.md](TODO.md)                   | Prioritised backlog of suggested improvements            |
| [CHANGELOG.md](CHANGELOG.md)         | Notable changes over time                                |

---

## Entry points at a glance

| Script                              | Role                                              |
| ----------------------------------- | ------------------------------------------------- |
| `rawProfileUdpCapturing.py`                   | Acquisition: sensor → HDF5                         |
| `profileProcessing.py`                | **Active** process step: raw → processed cache     |
| `dataAnalysis.py`                   | **Active** plot step: processed cache → 3D view    |
| `LidarProfileAnalysis_oldRegistration.py` | Legacy analysis using the registration path  |
