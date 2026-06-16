# Rollerband — LIDAR Profile Analysis

Tools for capturing, storing, processing and visualising 2D laser-line profiles
from a **Baumer OX200** LIDAR sensor, as part of a PhD research project on
"Rollerband" additive-manufacturing prints.

A *profile* is a single cross-sectional scan: a row of `(x, z)` points where `x`
is the position across the sensor's field of view and `z` is the measured height.
Many profiles taken along a print path reconstruct the 3D shape of a printed bead.

---

## What the project does

1. **Capture** — listen for UDP packets from the sensor and write each profile
   (plus its measurement metadata) into an HDF5 file.
2. **Process** — for every profile: level it against the floor, rotate it flat,
   smooth it, and measure the bead **width** from the smoothed slope.
3. **Visualise** — lay the profiles out along a computed print path and render
   them as an interactive 3D point cloud.

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
python udpCapturing.py
```
Listens on the UDP address/port set at the top of `udpCapturing.py`
(`192.168.0.251:1234` by default) and appends to an HDF5 file.

### Analyse and visualise stored profiles

```bash
python LidarProfileAnalysis.py
```
Loads the HDF5 file named near the top of the script, processes the profiles, and
opens an interactive PyVista 3D window. Requires a display.

> **Note:** file paths and the sensor IP are currently hardcoded in the scripts.
> See [TODO.md](TODO.md) for the plan to parameterise them.

---

## Data flow

```
 Baumer OX200 ──UDP──> udpCapturing.py ──> HDf5data/*.h5
                                              │
                                              ▼
                                  LidarProfileAnalysis.py
                                   (load → process → plot)
```

## Where things live

| Path                  | Contents                                              |
| --------------------- | ----------------------------------------------------- |
| `*.py` (root)         | Source modules and entry-point scripts                |
| `HDf5data/`           | Captured profile data (HDF5) — not version-controlled |
| `ProfileData/`        | Older CSV profile exports (legacy loader)             |
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
| `udpCapturing.py`                   | Acquisition: sensor → HDF5                         |
| `LidarProfileAnalysis.py`           | **Active** analysis & 3D visualisation pipeline    |
| `LidarProfileAnalysis_oldRegistration.py` | Legacy analysis using the registration path  |
