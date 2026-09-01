"""Export each KEPT filament segment's along-path areaShoelace series to CSV, one file per belt speed.

Layout: one CSV per distinct `rollerbandSpeed` level, written into `SegmentAreaCsv/`. Inside each file every
segment is a **pair of adjacent columns** — its profile capture **time** (`arrivalTime`, seconds) on the LEFT
and its `areaShoelace` on the right — in profile order along the print path (top row = the segment's first
profile), padded with blanks to the longest segment in that file. Only segments that pass the shape sort-out
(`segmentShapeStatus == 0`, i.e. KEPT) are written; a segment's speed is the median `rollerbandSpeed` over its
profiles.

Times are the capture-PC Unix timestamp in **seconds** (float, not a date/time string). Areas are in **mm^2**
(`areaShoelace` * `AREA_UNITS_TO_MM2`), matching every plot's area axis. The sort-out is recomputed in memory
from the current `segmentShape` code (the cached `areaShoelace`/`arrivalTime` values themselves are unchanged),
so the KEPT set is up to date even if the on-disk cache predates a `measure_segment_shape` change.

Run:  python exportSegmentAreas.py
"""
import csv
import logging
import os

import numpy as np

from datasetConfig import PROCESSED_FILE
from profileLoading import load_profiles
from profileProcessingAlgorithms import _contiguous_runs, _segment_mask
from segmentShape import measure_segment_shape

logger = logging.getLogger(__name__)

AREA_UNITS_TO_MM2 = 1e-4      # 1 area unit (profile-unit^2) = 1e-4 mm^2 (matches FEATURE_DISPLAY areaShoelace)
OUTPUT_DIR = "SegmentAreaCsv"  # folder (relative to the project root) the per-speed CSVs are written into
_STATUS_KEPT = 0.0            # segmentShapeStatus code for a segment that passed the shape sort-out
_SPEED_DECIMALS = 2          # rollerbandSpeed rounding used to group segments into per-speed files


def export_segment_area_csvs(processed_file: str = PROCESSED_FILE, output_dir: str = OUTPUT_DIR) -> None:
    """Write one CSV of KEPT segments' areaShoelace series (mm^2) per rollerband-speed level (see module doc)."""
    profiles = load_profiles(processed_file)
    measure_segment_shape(profiles)  # refresh segmentShapeStatus from current code (areaShoelace unchanged)
    os.makedirs(output_dir, exist_ok=True)

    # Collect each KEPT segment as (first-profile index, capture times s, area series mm^2), by median speed.
    by_speed: "dict[float, list[tuple[int, np.ndarray, np.ndarray]]]" = {}
    for run in _contiguous_runs(_segment_mask(profiles)):
        if profiles[run[0]].segmentShapeStatus != _STATUS_KEPT:
            continue
        speeds = [profiles[i].rollerbandSpeed for i in run if profiles[i].rollerbandSpeed is not None]
        if not speeds:
            continue
        spd = round(float(np.median(speeds)), _SPEED_DECIMALS)
        times = np.array([profiles[i].arrivalTime if profiles[i].arrivalTime is not None else np.nan
                          for i in run], dtype=float)
        areas = np.array([profiles[i].areaShoelace * AREA_UNITS_TO_MM2 if profiles[i].areaShoelace is not None
                          else np.nan for i in run], dtype=float)
        by_speed.setdefault(spd, []).append((int(run[0]), times, areas))

    if not by_speed:
        logger.warning("no KEPT segments found in %s — nothing written", processed_file)
        return

    for spd in sorted(by_speed):
        segments = by_speed[spd]
        n_rows = max(a.size for _, _, a in segments)
        headers = []  # per segment: a time column (left) then its area column (right)
        for n, (start, _, _) in enumerate(segments):
            headers += [f"seg{n + 1}_p{start}_t_s", f"seg{n + 1}_p{start}_area_mm2"]
        path = os.path.join(output_dir, f"segmentAreas_speed{spd:.{_SPEED_DECIMALS}f}.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for r in range(n_rows):
                row = []
                for _, t, a in segments:
                    row.append(f"{t[r]:.6f}" if r < t.size and np.isfinite(t[r]) else "")   # capture time (s), left
                    row.append(f"{a[r]:.4f}" if r < a.size and np.isfinite(a[r]) else "")   # areaShoelace (mm^2), right
                writer.writerow(row)
        logger.info("wrote %s  (%d segments, up to %d profiles each)", path, len(segments), n_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_segment_area_csvs()
