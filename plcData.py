"""Load the machine PLC log (CSV) and join it to the LIDAR profiles by timestamp.

The PLC controller exports a YT-Scope CSV: a few metadata header lines, a column header,
then data rows whose first column is a Windows FILETIME timestamp and whose remaining
columns are machine signals sampled at 50 Hz. Profiles carry a Unix-epoch ``arrivalTime``
(the capture PC's wall clock). This module converts FILETIME -> Unix, corrects the PLC
clock offset, and attaches the nearest-in-time PLC sample to each profile.

Data flows one direction: this staging model (`PlcLog`) is parsed here and merged onto the
analysis `profileData`; it never touches the profile HDF5 schema. Kept separate from the
analysis model on purpose, the same way acquisition owns `ProfileDataRaw`.
"""
import csv
import logging
from dataclasses import dataclass

import numpy as np

from profilePointsClass import profileData

logger = logging.getLogger(__name__)

# The 10 PLC signal channels, in the exact order/spelling of the CSV column header. Kept
# verbatim (including the "Mixxing" typo) so the CSV column -> profileData field mapping is
# 1:1 and self-documenting. These names are also profileData field names (see the join).
PLC_COLUMNS: tuple[str, ...] = (
    "mortarPumpFlow",
    "pressurePipeEnd",
    "pressurePipeStart",
    "pressurePrintHead",
    "printHeadMixxingSpeed",
    "printHeadTorque",
    "rollerbandHeight",
    "rollerbandSpeed",
    "viscoPump1_VMAflow",
    "viscoPump2_AcceleratorFlow",
)

# Derived PLC channels: computed from the raw log columns, not parsed from the CSV. Each is a
# `profileData` @property (e.g. `pipePressureDifference = pressurePipeStart - pressurePipeEnd`), so it
# needs no cache slot. `PLC_COLUMNS` alone drives CSV parsing; `ALL_PLC_COLUMNS` is the channel set the
# plots offer (raw + derived) — use it in FEATURE_DISPLAY, the selector grouping, and the plot cells.
DERIVED_PLC_COLUMNS: tuple[str, ...] = ("pipePressureDifference",)
ALL_PLC_COLUMNS: tuple[str, ...] = PLC_COLUMNS + DERIVED_PLC_COLUMNS

# Windows FILETIME counts 100 ns ticks since 1601-01-01 UTC; convert to Unix seconds by
# dividing by the ticks-per-second and shifting the epoch by the 1601->1970 gap.
_FILETIME_TICKS_PER_S = 10_000_000                # 100 ns per tick
_FILETIME_UNIX_EPOCH_OFFSET_S = 11_644_473_600    # seconds between 1601-01-01 and 1970-01-01

# The PLC controller's clock ran 9 min 24 s AHEAD of the capture laptop (and thus the profile
# arrival_time). Subtract this from PLC timestamps to align them to the profile clock.
# Measured manually for this recording; adjust here if a future dataset has a different skew.
PLC_CLOCK_OFFSET_S = 9 * 60 + 24  # = 564 s


def filetime_to_unix(filetime: "int | np.ndarray") -> "float | np.ndarray":
    """Convert a Windows FILETIME (100 ns ticks since 1601-01-01 UTC) to a Unix timestamp (s)."""
    return filetime / _FILETIME_TICKS_PER_S - _FILETIME_UNIX_EPOCH_OFFSET_S


@dataclass
class PlcLog:
    """A parsed PLC log: one absolute time array plus one value array per signal channel.

    ``time`` is Unix-epoch seconds, already corrected for `PLC_CLOCK_OFFSET_S`, so it shares
    the profiles' ``arrivalTime`` clock. ``channels`` maps each `PLC_COLUMNS` name to its
    value array (same length as ``time``, ordered by increasing time).
    """
    time: np.ndarray
    channels: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.time.shape[0])


def load_plc_csv(path: str, offset_s: float = PLC_CLOCK_OFFSET_S) -> PlcLog:
    """Parse a PLC YT-Scope CSV into a `PlcLog`, converting FILETIME->Unix and applying the offset.

    Skips the metadata/column-header lines: a data row is any whose first field is a long
    integer FILETIME (>= 17 digits). Only the `PLC_COLUMNS` channels are kept, in order. The
    returned times are offset-corrected Unix seconds on the profile clock.
    """
    filetimes: list[int] = []
    rows: list[list[float]] = []
    header: list[str] | None = None
    n_cols = len(PLC_COLUMNS)

    with open(path, newline="") as f:
        for record in csv.reader(f):
            if not record or not record[0].strip():
                continue
            first = record[0].strip()
            if first.isdigit() and len(first) >= 17:  # data row: first field is a FILETIME
                if len(record) < 1 + n_cols:
                    raise ValueError(f"PLC data row has {len(record)} fields, expected >= {1 + n_cols}")
                filetimes.append(int(first))
                rows.append([float(v) for v in record[1:1 + n_cols]])
            elif header is None and len(record) > 1 and record[1].strip() == PLC_COLUMNS[0]:
                header = [c.strip() for c in record[1:1 + n_cols]]

    if not filetimes:
        raise ValueError(f"No PLC data rows found in {path}")
    if header is not None and tuple(header) != PLC_COLUMNS:
        raise ValueError(f"PLC column header {header} does not match expected {PLC_COLUMNS}")

    unix = filetime_to_unix(np.array(filetimes, dtype=np.int64)) - offset_s
    values = np.array(rows, dtype=float)  # shape (N, n_cols)
    channels = {name: values[:, i] for i, name in enumerate(PLC_COLUMNS)}
    logger.info("Loaded PLC log: %d rows spanning %.3f s from %s", len(unix), unix[-1] - unix[0], path)
    return PlcLog(time=unix, channels=channels)


def _nearest_index(times: np.ndarray, t: float) -> int:
    """Index of the entry in the sorted array ``times`` closest to ``t``."""
    j = int(np.searchsorted(times, t))
    if j <= 0:
        return 0
    if j >= times.shape[0]:
        return times.shape[0] - 1
    return j - 1 if (t - times[j - 1]) <= (times[j] - t) else j


def join_plc_to_profiles(profiles: list[profileData], plc: PlcLog) -> list[profileData]:
    """Attach the nearest-in-time PLC sample to each profile; drop profiles outside the overlap.

    Restricts to the mutual time overlap of the two streams (either may start first / stop
    last): keeps only profiles whose ``arrivalTime`` falls inside `[plc.time[0], plc.time[-1]]`
    and, for each, sets the `PLC_COLUMNS` fields from the nearest PLC row. PLC samples outside
    the profile span are never a kept profile's nearest neighbour, so they are inherently
    dropped. Profiles must already carry ``arrivalTime`` (mapped from the raw sensor
    ``arrival_time`` in `load_profiles`); any without it is dropped. Returns the kept profiles
    (a contiguous slice when ``arrivalTime`` is monotonic, as it is for a single capture).
    """
    if len(plc) == 0:
        raise ValueError("PLC log is empty")
    plc_start, plc_end = float(plc.time[0]), float(plc.time[-1])

    kept: list[profileData] = []
    for p in profiles:
        if p.arrivalTime is None:
            continue
        t = float(p.arrivalTime)
        if t < plc_start or t > plc_end:
            continue
        j = _nearest_index(plc.time, t)
        for name in PLC_COLUMNS:
            setattr(p, name, float(plc.channels[name][j]))
        kept.append(p)

    logger.info("Joined PLC to profiles: kept %d, dropped %d outside overlap",
                len(kept), len(profiles) - len(kept))
    return kept
