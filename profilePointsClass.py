import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class profileData:
    name: str
    x: np.ndarray
    z: np.ndarray

    # floor baseline fit (slope, intercept) of the levelled profile, set by
    # rotate_and_shift_uniform and reused to draw the baseline (line_points_from_floorSides)
    m: Optional[float] = None
    b: Optional[float] = None

    floorMask: Optional[np.ndarray] = None  # per-point bool: True = floor point, False = profile (filament)
    widthFlankIdx: Optional[np.ndarray] = None  # the two flank-foot indices of widthFlank
    widthFlank: Optional[float] = None      # filament width between the outer smoothed-slope flank feet
    widthOuterIdx: Optional[np.ndarray] = None  # indices of the two outer filament points (filament-edge width)
    widthOuter: Optional[float] = None          # x-span between the two outer filament points
    heightP95: Optional[float] = None         # robust filament height (95th pct of filament z above z=0)
    heightSmooth: Optional[float] = None   # robust filament height (max of median-smoothed z above z=0)
    isFlat: Optional[bool] = None          # raw per-profile flag: True = no filament points (== floorMask.all())
    isSegment: Optional[bool] = None       # cleaned run flag: True = part of a real filament segment (see clean_flat_runs)
    flatness: Optional[float] = None
    areaSimpson: Optional[float] = None    # filament cross-section, Simpson integration (profile-unit^2)
    areaShoelace: Optional[float] = None   # filament cross-section, shoelace polygon (profile-unit^2)
    shoelaceArea2: Optional[float] = None
    segmentVolume: Optional[float] = None  # total volume of this profile's filament segment (area-unit*dist-unit)
    sliceVolume: Optional[float] = None    # this profile's own slab volume, areaShoelace * inter-profile gap
    segmentLength: Optional[float] = None  # along-path length of this profile's filament segment (dist units)
    defectLength: Optional[float] = None   # along-path length of this profile's pure-floor (no-filament) run
    # Per-segment shape features (measure_segment_shape), broadcast onto every profile of the segment.
    # Describe the thinning / startup / rupture of the filament along the print path; stored in physical
    # units (mm, mm^2, %/mm, %, dimensionless). NaN where a phase is absent (e.g. rupture fields on a
    # segment that ended thick); None off-segment or on a too-short segment.
    segmentBodyThinning: Optional[float] = None        # body thinning rate (%/mm; negative = thinning)
    segmentBodyThinningStability: Optional[float] = None  # steadiness of the thinning (Spearman area vs arc-length, -1..1)
    segmentCriticalArea: Optional[float] = None     # cross-section at the rupture start / cliff top (mm^2)
    segmentRuptureLength: Optional[float] = None    # arc-length of the terminal rupture cliff (mm; short = abrupt)
    segmentHeadOvershoot: Optional[float] = None    # startup bulge height over the body level (%)
    segmentRuptures: Optional[float] = None         # 1.0 if the segment ended in a rupture, else 0.0
    segmentSection: Optional[float] = None          # per-profile phase flag (1 body, 2 rupture, 3 peak; else NaN); heat-map debug
    maxSmoothedHeight: Optional[float] = None
    maxSmoothedPlace: Optional[int] = None
    maxHeight: Optional[float] = None
    maxPlace: Optional[int] = None

    # --- Absolute time + PLC machine log, joined by timestamp (see plcData.join_plc_to_profiles) ---
    arrivalTime: Optional[float] = None        # capture-PC Unix timestamp (from raw sensor arrival_time)
    sensorTime: Optional[float] = None         # sensor clock (timestamp_sec+usec); low-jitter dt
    mortarPumpFlow: Optional[float] = None
    pressurePipeEnd: Optional[float] = None
    pressurePipeStart: Optional[float] = None
    pressurePrintHead: Optional[float] = None
    printHeadMixxingSpeed: Optional[float] = None
    printHeadTorque: Optional[float] = None
    rollerbandHeight: Optional[float] = None
    rollerbandSpeed: Optional[float] = None
    viscoPump1_VMAflow: Optional[float] = None
    viscoPump2_AcceleratorFlow: Optional[float] = None

    @property
    def isNotFlat(self) -> Optional[bool]:
        """Inverse of `isFlat` (True = the profile HAS filament points). Derived, not a stored field,
        so it needs no cache slot; used as a 0/1 heat-map feature that shares the "filament = high"
        colour convention with `isSegment` (so the two flags read the same colour on filament)."""
        return None if self.isFlat is None else (not self.isFlat)

    @property
    def pipePressureDifference(self) -> Optional[float]:
        """Pressure drop along the delivery pipe: `pressurePipeStart - pressurePipeEnd`. Derived from the
        two joined PLC channels, not a stored field, so it needs no cache slot and no reprocess (it reads
        the already-cached pressures). None if either pressure is missing."""
        if self.pressurePipeStart is None or self.pressurePipeEnd is None:
            return None
        return self.pressurePipeStart - self.pressurePipeEnd

    """
    # only trust this formula for profiles with monotonically rising x values (not the ones where "points are below each other")
    def integrate_area(self):
        area=np.round(sp.integrate.simpson(self.y,self.x), decimals=2)
        return area


    # Area using shoelace formula --> (points must be ordered!, points do not need to be monotonically increasing in x)
    def shoelace_area(self):
        # chat gpt code
        areaShoelace = 0.5 * abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))

        points= np.vstack((self.x,self.y))
        shifted = np.vstack((points[1:], points[0]))
        cross = points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]
        shoelaceArea2 = 0.5 * abs(np.sum(cross))
        return areaShoelace,shoelaceArea2

    def find_max_height(self):
        self.maxSmoothedHeight = np.round(np.max(self.ySmooth),decimals=2)
        self.maxSmoothedPlace = np.argmax(self.ySmooth)
        self.maxHeight = np.round(np.max(self.y),decimals=2)
        self.maxPlace = np.argmax(self.y)
    """
