import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
import h5py
from dataclasses import fields

from profilePointsClass import *
from profilePointsClass import FLATNESS_RMS_THRESHOLD  # explicit: used in load_profiles


# profileData fields that are bulky per-point intermediates recomputed by processing;
# they are not persisted by save_profiles so cached files stay small.
_TRANSIENT_FIELDS = {"ySlope", "ySmooth", "ySlopeSmooth", "borderPoints"}


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
                if key == "peaks":
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




def plotProfiles(profileGroups, noFloorPoints = True):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.5))

    for i, ax in enumerate(axes.flat):
        if i >= len(profileGroups):
            break
        group = profileGroups[i]

        for j in range(len(group)):
            p=group[j]
            if noFloorPoints:
                x = p.x.copy()
                y = p.y.copy()
            else:
                x = p.x.copy()
                x = np.append(x,p.borderPoints[0,:])
                y = p.y.copy()
                y = np.append(y,p.borderPoints[1,:])

            ax.scatter(x, y, marker='x', s=1, label=str(p.name))

            ax.legend()
            ax.set_aspect('equal',adjustable='datalim')
    plt.tight_layout()

def loadProfiles():
    path = Path(r"C:/Users/zimme/Documents/A-Phd/Rollerband/Python/ProfileData/Registration")
    profiles = []

    for file_path in path.iterdir():
        if file_path.is_file():  # Skip subfolders
            with open(file_path) as f:
                lines = [line for line in f if line.strip()]
                lines.pop(0)
                lines.pop(0)
                lines.pop(0)
                for elem in lines:
                    elem.replace("\n", "")
                
                profileLength = len(lines)
                xPts=np.empty(profileLength)
                yPts=np.empty(profileLength)
                
                for i in range(profileLength):
                    xVal,yVal = lines[i].split(';')
                    xPts[i]=xVal
                    yPts[i]=yVal

                profileName = str(f.name).replace(str(path), "")
                profileName = profileName.replace(".csv", "").replace("\\", "")
                profiles.append(profileData(profileName,xPts,yPts))
    
    return profiles


def groupProfiles(profiles):
    group = []
    profileGroups = []
    lastNum = profiles[0].profileNumber
    for i in range(len(profiles)):
        if profiles[i].profileNumber == lastNum:
            group.append(profiles[i])
            if i == len(profiles)-1:
                profileGroups.append(group)
        else:
            profileGroups.append(group)
            group = []
            group.append(profiles[i])
            lastNum = profiles[i].profileNumber
    
    sortedGroups = []
    for g in profileGroups:
        pOthers = []
        for p in g:
            if "vertical" not in p.name:
                pOthers.append(p)
            else:
                pVertical = p
        sortGroup = []
        sortGroup.append(pVertical)
        sortGroup.extend(pOthers)
        sortedGroups.append(sortGroup)
    return sortedGroups

