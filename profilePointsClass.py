import numpy as np
from dataclasses import dataclass
from typing import Optional

# --- flowVelocity: nozzle geometry + pump-flow unit conversions (see profileData.flowVelocity) ---
NOZZLE_DIAMETER_M = 0.02                                          # round nozzle, 2 cm diameter
NOZZLE_AREA_M2 = float(np.pi * (NOZZLE_DIAMETER_M / 2.0) ** 2)    # nozzle cross-section (~3.1416e-4 m^2)
# Convert each pump channel's raw PLC unit to a COMMON volumetric flow in m^3/s before summing, so the
# total / nozzle area comes out in m/s (directly comparable to rollerbandSpeed). The mortar pump reports
# L/min; both viscotec pumps report mL/min. (In Exp1 the viscotec pumps read 0, so only the mortar pump
# contributes here; the viscotec factors matter for datasets where those pumps run.)
FLOW_TO_M3_PER_S = {
    "mortarPumpFlow": 1e-3 / 60.0,             # L/min  -> m^3/s
    "viscoPump1_VMAflow": 1e-6 / 60.0,         # mL/min -> m^3/s
    "viscoPump2_AcceleratorFlow": 1e-6 / 60.0,  # mL/min -> m^3/s
}

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
    isContinuousFilament: Optional[bool] = None  # True = an isSegment run too long to be a discrete segment (see classify_continuous_filaments); excluded from segment-shape analysis
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
    # Body thinning family: the same taper measured on three signals over the body plateau — area
    # (areaShoelace), width (widthOuter) and height (heightP95). Each is a rate (%/mm; negative = thinning)
    # plus a steadiness (Spearman of the signal vs arc-length over the body, -1..1; -1 = steadily thinning).
    segmentBodyAreaThinning: Optional[float] = None       # body AREA thinning rate (%/mm; negative = thinning)
    segmentBodyAreaSteadiness: Optional[float] = None     # steadiness of the area thinning (Spearman area vs arc-length, -1..1)
    segmentBodyWidthThinning: Optional[float] = None      # body WIDTH (widthOuter) thinning rate (%/mm; negative = narrowing)
    segmentBodyWidthSteadiness: Optional[float] = None    # steadiness of the width thinning (Spearman width vs arc-length, -1..1)
    segmentBodyHeightThinning: Optional[float] = None     # body HEIGHT (heightP95) thinning rate (%/mm; negative = flattening)
    segmentBodyHeightSteadiness: Optional[float] = None   # steadiness of the height thinning (Spearman height vs arc-length, -1..1)
    segmentCriticalArea: Optional[float] = None     # cross-section at the rupture start / cliff top (mm^2)
    segmentCriticalWidth: Optional[float] = None    # outer width at the rupture start / cliff top (mm; same profile as criticalArea)
    segmentRuptureLength: Optional[float] = None    # arc-length of the terminal rupture cliff (mm; short = abrupt)
    segmentHeadOvershoot: Optional[float] = None    # startup bulge height over the body level (%)
    segmentRuptures: Optional[float] = None         # 1.0 if the segment ended in a rupture, else 0.0
    segmentNeck: Optional[float] = None             # per-profile neck marker: 1.0 over a neck span, 0.0 elsewhere in an analysed run, None off-segment; heat-map (the per-speed neck RATE is pooled from this in featureRates)
    segmentSection: Optional[float] = None          # per-profile phase flag (1 body, 2 rupture, 3 peak; else NaN); heat-map debug
    segmentShapeStatus: Optional[float] = None       # per-segment sort-out reason for the shape analysis: 0 kept, 1 too short, 2 continuous filament, 3 degenerate, 4 tiny body, 5 high width change, 6 didn't rupture; heat-map debug
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

    @property
    def flowVelocity(self) -> Optional[float]:
        """Extrusion velocity through the nozzle (m/s): total volumetric pump flow / nozzle cross-section.

        total flow = mortarPumpFlow (L/min) + viscoPump1_VMAflow + viscoPump2_AcceleratorFlow (both
        mL/min), each converted to m^3/s (`FLOW_TO_M3_PER_S`), divided by the round-nozzle area
        (`NOZZLE_AREA_M2`, 2 cm diameter). The result is in m/s, so it is directly comparable to
        `rollerbandSpeed`. Derived from the joined PLC channels, not a stored field, so it needs no cache
        slot and no reprocess. None if all three flow channels are missing (a pump reading 0 counts as 0)."""
        names = ("mortarPumpFlow", "viscoPump1_VMAflow", "viscoPump2_AcceleratorFlow")
        vals = [getattr(self, n) for n in names]
        if all(v is None for v in vals):
            return None
        total = sum(FLOW_TO_M3_PER_S[n] * (v or 0.0) for n, v in zip(names, vals))  # m^3/s
        return total / NOZZLE_AREA_M2                                               # m/s

    @property
    def conveyorExtrusionVelocityDifference(self) -> Optional[float]:
        """Conveyor-minus-extrusion velocity (m/s): `rollerbandSpeed - flowVelocity`. Positive = the belt
        outruns extrusion (stretching / thinning); negative = material leaves the nozzle faster than the belt
        carries it away (over-supply / bead thickens). Both terms are m/s. Derived from the joined PLC
        channels, so it needs no cache slot and no reprocess. None if either velocity is missing."""
        fv = self.flowVelocity
        if fv is None or self.rollerbandSpeed is None:
            return None
        return self.rollerbandSpeed - fv

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
