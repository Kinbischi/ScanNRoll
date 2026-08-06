import numpy as np
import h5py
from dataclasses import fields
from typing import cast

from profilePointsClass import profileData
from profileProcessingAlgorithms import FLATNESS_RMS_THRESHOLD  # explicit: used in load_profiles


# The three per-point array fields (same length as x/z per profile), stored padded in `/points`.
_POINT_ARRAY_FIELDS = ("x", "z", "floorMask")
# Fixed-length-2 index-pair fields, stored as `[N, 2]` int with -1 = absent (None).
_INDEX_PAIR_FIELDS = ("peaks", "filamentWidthIdx")


def _gzip(arr: np.ndarray) -> "str | None":
    """gzip compression only for non-empty arrays (h5py can't chunk a zero-size dataset)."""
    return "gzip" if arr.size > 0 else None


def save_profiles(profiles: list[profileData], fileName: str, **file_attrs) -> None:
    """Write profileData objects to an HDF5 file as a columnar "table" (processed-cache layout).

    Every per-profile field is one array keyed by profile index (profile *i* = row *i*), so the
    whole cache loads in a handful of bulk reads instead of ~N tiny per-group reads:

    - `/points/x`, `/points/z`, `/points/floorMask` — padded `[N, L]` matrices (L = max point count);
      `/points/lengths` `[N]` gives each profile's valid point count.
    - `/points/peaks`, `/points/filamentWidthIdx` — `[N, 2]` int (`-1` = absent / None).
    - `/scalars/<field>` — one `[N]` float64 per scalar field (NaN = unset).
    - `/names` — `[N]` strings.

    File-level attributes `layout="columnar"` and `n_profiles` mark the format; any extra keywords
    (e.g. `kind="processed"`, `raw_start=...`) are stored as file attributes too. `load_profiles`
    reads this layout and still understands the legacy per-group-attribute layout (older caches and
    raw acquisition files). Only called on processed profiles, which always carry `floorMask`; a
    None `floorMask` is rejected here (it is only representable in the legacy per-group layout).
    """
    n = len(profiles)
    if any(p.floorMask is None for p in profiles):
        raise ValueError("columnar layout requires floorMask on every profile "
                         "(run categorize_floor_points); a None floorMask needs the legacy layout")

    # scalar fields = every persisted field that isn't a point array, index pair, or the name
    handled = set(_POINT_ARRAY_FIELDS) | set(_INDEX_PAIR_FIELDS) | {"name"}
    scalar_fields: list[str] = []
    for fld in fields(profileData):
        if fld.name in handled:
            continue
        first = next((getattr(p, fld.name) for p in profiles if getattr(p, fld.name) is not None), None)
        if first is None:
            continue  # unset on every profile -> don't persist
        if isinstance(first, np.ndarray):  # guards against a new array field silently truncating
            raise TypeError(f"array field {fld.name!r} not handled by the columnar layout")
        scalar_fields.append(fld.name)

    lengths = np.array([p.x.shape[0] for p in profiles], dtype=np.int32)
    length = int(lengths.max()) if n else 0
    x_mat = np.zeros((n, length), dtype=np.float64)
    z_mat = np.zeros((n, length), dtype=np.float64)
    fmask = np.zeros((n, length), dtype=bool)
    pair_mats = {name: np.full((n, 2), -1, dtype=np.int32) for name in _INDEX_PAIR_FIELDS}
    for i, p in enumerate(profiles):
        li = int(lengths[i])
        x_mat[i, :li] = p.x
        z_mat[i, :li] = p.z
        fmask[i, :li] = p.floorMask
        for name, mat in pair_mats.items():
            value = getattr(p, name)
            if value is not None:
                mat[i] = value

    with h5py.File(fileName, "w") as f:
        for key, value in file_attrs.items():
            f.attrs[key] = value
        f.attrs["layout"] = "columnar"
        f.attrs["n_profiles"] = n
        points = f.create_group("points")
        points.create_dataset("x", data=x_mat, compression=_gzip(x_mat))
        points.create_dataset("z", data=z_mat, compression=_gzip(z_mat))
        points.create_dataset("floorMask", data=fmask, compression=_gzip(fmask))
        points.create_dataset("lengths", data=lengths, compression=_gzip(lengths))
        for name, mat in pair_mats.items():
            points.create_dataset(name, data=mat, compression=_gzip(mat))
        scalars = f.create_group("scalars")
        for name in scalar_fields:
            col = np.full(n, np.nan, dtype=np.float64)
            for j, p in enumerate(profiles):
                value = getattr(p, name)
                if value is not None:
                    col[j] = float(value)
            scalars.create_dataset(name, data=col, compression=_gzip(col))
        names = np.array([p.name for p in profiles], dtype=object)
        f.create_dataset("names", data=names, dtype=h5py.string_dtype(encoding="utf-8"))
    print(f"Wrote {len(profiles)} profiles to {fileName}")


def count_profiles(fileName: str) -> int:
    """Return the number of profiles in an HDF5 file (fast; reads no point data).

    Handles both the columnar layout (from `n_profiles`) and the legacy per-group layout
    (counts `profile_` groups — raw acquisition files and older caches).
    """
    with h5py.File(fileName, "r") as f:
        if f.attrs.get("layout") == "columnar":
            points = cast("h5py.Group", f["points"])
            return int(cast("h5py.Dataset", points["lengths"]).shape[0])
        return sum(1 for name in f.keys() if name.startswith("profile_"))


def read_file_attrs(fileName: str) -> dict:
    """Return an HDF5 file's file-level attributes (e.g. kind, source_file, raw_start/raw_end)."""
    with h5py.File(fileName, "r") as f:
        return {key: f.attrs[key] for key in f.attrs}


def load_profiles(fileName: str, start: int = 0, end: int | None = None) -> list[profileData]:
    """Load profileData objects from an HDF5 file into a list.

    Reads two kinds of file, routed by their file-level attributes:
      * a columnar "table" processed cache (`layout="columnar"`, written by `save_profiles`) — every
        field is one `[N, ...]` array, read in a handful of bulk reads (fast);
      * a raw acquisition file (`profile_NNNNNN` groups, no `layout`) — takes `x`/`z` plus
        `arrival_time`/`timestamp_*`, ignoring the rest of the sensor metadata.
    An old per-group *processed* cache (`kind="processed"` but no `layout`) is rejected — reprocess it
    to the columnar layout. `isFlat` is re-derived from the cached `flatness` so its threshold can be
    tuned without reprocessing.

    Only profiles in the index range [start:end) are read, so a sub-range can be loaded without
    reading the whole file. Defaults load everything.
    """
    with h5py.File(fileName, "r") as f:
        if f.attrs.get("layout") == "columnar":
            known = {fld.name for fld in fields(profileData)}
            profiles = _load_columnar(f, start, end, known)
        elif f.attrs.get("kind") == "processed":
            raise ValueError(
                f"{fileName} is an old per-group processed cache; reprocess it to the columnar "
                "layout with `python profileProcessing.py` (the legacy processed-cache reader "
                "was removed).")
        else:
            profiles = _load_raw(f, start, end)
    print(f"Loaded {len(profiles)} profiles from {fileName}")
    return profiles


def _load_columnar(f: "h5py.File", start: int, end: "int | None", known: set[str]) -> list[profileData]:
    """Load the columnar-table layout: bulk-read the `[start:end]` slice of every `[N, ...]` array,
    then rebuild each profileData from row `i` (sliced to its stored length)."""
    points = cast("h5py.Group", f["points"])
    sl = slice(start, end)
    x_mat = cast("h5py.Dataset", points["x"])[sl]
    z_mat = cast("h5py.Dataset", points["z"])[sl]
    fmask = cast("h5py.Dataset", points["floorMask"])[sl]
    lengths = cast("h5py.Dataset", points["lengths"])[sl]
    pairs = {name: cast("h5py.Dataset", points[name])[sl] for name in _INDEX_PAIR_FIELDS}
    names_col = cast("h5py.Dataset", f["names"])[sl] if "names" in f else None
    scalar_cols: dict[str, np.ndarray] = {}
    if "scalars" in f:
        sg = cast("h5py.Group", f["scalars"])
        scalar_cols = {key: cast("h5py.Dataset", sg[key])[sl] for key in sg if key in known}

    profiles = []
    for r in range(x_mat.shape[0]):
        li = int(lengths[r])
        x = np.array(x_mat[r, :li], dtype=float)
        z = np.array(z_mat[r, :li], dtype=float)
        if names_col is not None:
            nm = names_col[r]
            name = nm.decode() if isinstance(nm, bytes) else str(nm)
        else:
            name = f"profile_{start + r:06d}"
        prof = profileData(str(name), x, z)
        prof.floorMask = np.array(fmask[r, :li], dtype=bool) if li > 0 else None
        for field_name, mat in pairs.items():
            if int(mat[r, 0]) >= 0:  # -1 = absent (None); real indices are >= 0
                setattr(prof, field_name, np.array(mat[r], dtype=int))
        for key, col in scalar_cols.items():
            val = col[r]
            if not np.isnan(val):  # NaN = unset -> leave the field's default (None)
                setattr(prof, key, val)
        if prof.flatness is not None and np.isfinite(prof.flatness):
            prof.isFlat = prof.flatness < FLATNESS_RMS_THRESHOLD
        profiles.append(prof)
    return profiles


def _load_raw(f: "h5py.File", start: int, end: "int | None") -> list[profileData]:
    """Load a raw acquisition file: one `profile_NNNNNN` group each, taking `x`/`z` plus the two
    timing attributes the analysis uses. The rest of the sensor metadata is ignored.

    `arrival_time` -> `arrivalTime` (absolute capture clock, for the PLC join); `timestamp_sec`(+usec)
    -> `sensorTime` (low-jitter sensor clock, for the physical plot spacing). The full geometry /
    scalar fields are produced by processing, not read here.
    """
    profiles = []
    profile_names = sorted(name for name in f.keys() if name.startswith("profile_"))
    for group_name in profile_names[start:end]:
        group = cast("h5py.Group", f[group_name])
        if "x" not in group or "z" not in group:
            continue
        x = cast("h5py.Dataset", group["x"])[:].astype(float)
        z = cast("h5py.Dataset", group["z"])[:].astype(float)
        prof = profileData(group_name, x, z)
        if "arrival_time" in group.attrs:
            prof.arrivalTime = float(group.attrs["arrival_time"])
        if "timestamp_sec" in group.attrs:
            usec = float(group.attrs.get("timestamp_usec", 0))
            prof.sensorTime = float(group.attrs["timestamp_sec"]) + usec / 1e6
        profiles.append(prof)
    return profiles




