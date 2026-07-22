import numpy as np
import h5py
from dataclasses import fields

from profilePointsClass import profileData
from profileProcessingAlgorithms import FLATNESS_RMS_THRESHOLD  # explicit: used in load_profiles


# profileData fields that are bulky per-point intermediates recomputed by processing;
# they are not persisted by save_profiles so cached files stay small.
_TRANSIENT_FIELDS = {"ySmooth", "ySlopeSmooth"}


def save_profiles(profiles: list[profileData], fileName: str, **file_attrs) -> None:
    """Write profileData objects to an HDF5 file, one `profile_NNNNNN` group each.

    Every set (non-None) field is stored automatically: numpy arrays become datasets,
    scalars/strings become group attributes. Bulky intermediate arrays
    (`_TRANSIENT_FIELDS`) are skipped. Optional file-level attributes (e.g.
    `kind="processed"`, `source_file=...`) may be passed as keywords.
    """
    with h5py.File(fileName, "w") as f:
        for key, value in file_attrs.items():
            f.attrs[key] = value
        for i, p in enumerate(profiles):
            group = f.create_group(f"profile_{i:06d}")
            for fld in fields(p):
                if fld.name in _TRANSIENT_FIELDS:
                    continue
                value = getattr(p, fld.name)
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    compression = "gzip" if value.size > 0 else None  # gzip needs chunking
                    group.create_dataset(fld.name, data=value, compression=compression)
                else:
                    group.attrs[fld.name] = value
    print(f"Wrote {len(profiles)} profiles to {fileName}")


def count_profiles(fileName: str) -> int:
    """Return the number of profile_ groups in an HDF5 file (fast; reads no point data)."""
    with h5py.File(fileName, "r") as f:
        return sum(1 for name in f.keys() if name.startswith("profile_"))


def load_profiles(fileName: str, start: int = 0, end: int | None = None) -> list[profileData]:
    """Load profileData objects from an HDF5 file into a list.

    Restores whichever profileData fields are present (arrays from datasets, scalars
    from attributes) and ignores foreign ones, so this reads both files written by
    `save_profiles` and raw acquisition files (only `x`/`z` are taken from those; the
    sensor metadata is skipped). `x`/`z` are required per group. `isFlat` is re-derived
    from the cached `flatness` so its threshold can be tuned without reprocessing.

    Only profiles in the index range [start:end) (sorted by group name) are read, so a
    sub-range can be loaded without reading the whole file. Defaults load everything.
    """
    known = {fld.name for fld in fields(profileData)}
    profiles = []
    with h5py.File(fileName, "r") as f:
        profile_names = sorted(name for name in f.keys() if name.startswith("profile_"))
        profile_names = profile_names[start:end]
        for group_name in profile_names:
            group = f[group_name]
            if "x" not in group or "z" not in group:
                continue
            x = group["x"][:].astype(float)
            z = group["z"][:].astype(float)
            name = group.attrs.get("name", group_name)
            if isinstance(name, bytes):
                name = name.decode()
            prof = profileData(str(name), x, z)
            # restore any other known array fields (datasets)
            for key in group.keys():
                if key in ("x", "z") or key not in known:
                    continue
                arr = group[key][:]
                if key in ("peaks", "beadWidthIdx"):
                    arr = arr.astype(int)
                setattr(prof, key, arr if arr.size > 0 else None)
            # restore any other known scalar fields (attributes)
            for key in group.attrs:
                if key == "name" or key not in known:
                    continue
                setattr(prof, key, group.attrs[key])
            # Re-derive the flat flag from the cached (threshold-independent) metric so
            # FLATNESS_RMS_THRESHOLD can be tuned without reprocessing the cache.
            if prof.flatness is not None and np.isfinite(prof.flatness):
                prof.isFlat = prof.flatness < FLATNESS_RMS_THRESHOLD
            profiles.append(prof)
    print(f"Loaded {len(profiles)} profiles from {fileName}")
    return profiles




