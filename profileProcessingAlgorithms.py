"""Per-profile processing algorithms for LIDAR profiles.

Pure functions that operate in place on lists of `profileData` (defined in
profilePointsClass): rotate/level to the floor, categorise floor/filament points, smooth,
measure width, flag flatness, plus the shared `moving_average` and
`get_baseline_from_profileBorder` helpers. The `process_profiles` pipeline in
profileProcessing.py composes these in order.

Units: x and z are in profile units where 1 unit = 0.01 mm (so 20 = 0.2 mm, 100 = 1 mm).
"""
import numpy as np
import scipy as sp
import scipy.signal    # ensure sp.signal.find_peaks is available without relying on side-effect imports
import scipy.integrate # ensure sp.integrate.simpson is available (filament area)
import scipy.ndimage   # ensure sp.ndimage.median_filter is available (smoothed height)

from profilePointsClass import profileData

# Profiles with fewer than this many points are too short for a reliable slope/width fit.
MIN_PROFILE_POINTS = 10

# RMS residual (profile units, ~0.01 mm) of a straight-line fit to the whole profile: below
# this the profile is "flat" (substrate only), above it a filament is present. Set in the valley
# of the bimodal flat-vs-filament residual distribution.
FLATNESS_RMS_THRESHOLD = 350

def flag_flat_profiles(profiles: list[profileData], use_line_fit: bool = False,
                       threshold: float = FLATNESS_RMS_THRESHOLD, max_filament_points: int = 0) -> None:
    """Flag each profile as flat (substrate only) or not, setting `isFlat`.

    Default (floor-based): flat when categorisation found essentially no filament points — at most
    `max_filament_points` of them. Uses `floorMask`, so run categorize_floor_points +
    grow_profile_points first; `flatness` is left None (it is a line-fit-only metric). With
    use_line_fit=True the old method is used instead: a straight line is fit to all points and
    `flatness` is set to the RMS residual (small for a substrate-only profile, large for a
    filament), `isFlat` = residual < threshold. Too-short profiles are treated as flat.
    """
    for p in profiles:
        if p.x.shape[0] < MIN_PROFILE_POINTS:
            p.flatness = 0.0
            p.isFlat = True
            continue
        if use_line_fit:
            coeffs, residuals, rank, singular_values, rcond = np.polyfit(p.x, p.z, 1, full=True)
            rms = float(np.sqrt(residuals[0] / p.x.shape[0])) if residuals.size > 0 else 0.0
            p.flatness = rms
            p.isFlat = rms < threshold
        else:
            filament_count = 0 if p.floorMask is None else int((~p.floorMask).sum())
            p.flatness = None
            p.isFlat = filament_count <= max_filament_points

# Cascaded box-smoothing windows (points): first smooth the height, then the |slope|.
HEIGHT_SMOOTH_WINDOWS = (15, 9, 5, 5)
SLOPE_SMOOTH_WINDOWS = (65, 55, 15, 5)
# Peak detection on the smoothed |dz/dx| (each filament flank shows up as a peak):
SLOPE_PEAK_MIN_HEIGHT = 0.15   # min smoothed-slope height to count as a flank peak
SLOPE_PEAK_MIN_DISTANCE = 50   # min points between two peaks
FLANK_FOOT_HEIGHT = 100        # smoothed height (~1 mm above the z=0 floor) marking a flank foot

def _flank_foot(x: np.ndarray, y_smooth: np.ndarray, peak_idx: int, threshold: float,
                toward_smaller_x: bool) -> int:
    """Index of the flank foot: the point nearest the peak, moving toward the filament edge, where the
    smoothed height first drops to `threshold`. Falls back to the extreme edge point on that side if
    the height never gets that low (e.g. raised edge floor)."""
    if toward_smaller_x:
        cand = np.flatnonzero((x < x[peak_idx]) & (y_smooth <= threshold))
        return int(cand[np.argmax(x[cand])]) if cand.size else int(np.argmin(x))
    cand = np.flatnonzero((x > x[peak_idx]) & (y_smooth <= threshold))
    return int(cand[np.argmin(x[cand])]) if cand.size else int(np.argmax(x))

def width_from_smoothed_slope(profiles: list[profileData]) -> None:
    """Filament width between the feet of the two OUTERMOST filament flanks.

    Smooths the height and its |dz/dx| with cascaded box filters (locally — no stored arrays), finds
    the flank peaks in the smoothed slope, keeps the furthest-left and furthest-right (intermediate
    peaks from surface texture are ignored), then walks each flank down to its foot near the floor so
    the width sits at the filament base rather than mid-flank. Sets `width` and `peaks` (the two width-
    edge indices — flank feet); NaN / None when fewer than two flank peaks are found or the profile
    is too short.
    """
    for p in profiles:
        p.peaks = None
        if p.x.shape[0] < MIN_PROFILE_POINTS:
            p.width = np.nan
            continue
        y_smooth = p.z
        for window in HEIGHT_SMOOTH_WINDOWS:
            y_smooth = moving_average(y_smooth, window)
        slope = np.abs(np.gradient(y_smooth, p.x))
        for window in SLOPE_SMOOTH_WINDOWS:
            slope = moving_average(slope, window)

        peaks, _ = sp.signal.find_peaks(slope, height=SLOPE_PEAK_MIN_HEIGHT,
                                        distance=SLOPE_PEAK_MIN_DISTANCE)
        if peaks.size < 2:
            p.width = np.nan
            continue
        left_peak = peaks[np.argmin(p.x[peaks])]    # flank at smallest x
        right_peak = peaks[np.argmax(p.x[peaks])]   # flank at largest x
        left_foot = _flank_foot(p.x, y_smooth, left_peak, FLANK_FOOT_HEIGHT, toward_smaller_x=True)
        right_foot = _flank_foot(p.x, y_smooth, right_peak, FLANK_FOOT_HEIGHT, toward_smaller_x=False)
        p.peaks = np.array([left_foot, right_foot])
        p.width = float(np.round(abs(p.x[right_foot] - p.x[left_foot]), decimals=2))

def width_from_filament_edges(profiles: list[profileData]) -> None:
    """Filament width from the outer filament points: the x-span between the leftmost and rightmost
    filament point (uses `floorMask`). Sets `filamentWidth` and `filamentWidthIdx` (the two point indices).
    Run after grow_profile_points; NaN / None when a profile has fewer than two filament points.
    """
    for p in profiles:
        if p.floorMask is None:
            p.filamentWidth = np.nan
            p.filamentWidthIdx = None
            continue
        filament_idx = np.flatnonzero(~p.floorMask)
        if filament_idx.size < 2:
            p.filamentWidth = np.nan
            p.filamentWidthIdx = None
            continue
        left = int(filament_idx[np.argmin(p.x[filament_idx])])   # filament point at smallest x
        right = int(filament_idx[np.argmax(p.x[filament_idx])])  # filament point at largest x
        p.filamentWidthIdx = np.array([left, right])
        p.filamentWidth = float(np.round(abs(p.x[right] - p.x[left]), decimals=2))

HEIGHT_PERCENTILE = 95     # filament height = this percentile of the filament-point heights (robust to spikes)
MEDIAN_SMOOTH_WINDOW = 15  # median-filter window (points) for the smoothed-height measure

def measure_filament_height(profiles: list[profileData], percentile: float = HEIGHT_PERCENTILE) -> None:
    """Robust filament height above the shared median floor (z = 0 after rotate_and_shift_uniform), two ways.

    `filamentHeight` is the `percentile`-th percentile of the filament points' z (uses `floorMask`), so a lone
    outlier/noise spike above that percentile is ignored. `filamentHeightSmooth` is the max of a
    median-smoothed profile over the filament points — a median filter removes single-point spikes, so its
    peak is robust too; the two cross-check each other. Units: profile units (1 unit = 0.01 mm). NaN
    when a profile has no filament points (no floorMask, or flat). Run after grow_profile_points.
    """
    for p in profiles:
        p.filamentHeight = np.nan
        p.filamentHeightSmooth = np.nan
        if p.floorMask is None:
            continue
        filament = ~p.floorMask
        filament_z = p.z[filament]
        if filament_z.size == 0:
            continue
        p.filamentHeight = float(np.round(np.percentile(filament_z, percentile), decimals=2))
        z_smooth = sp.ndimage.median_filter(p.z, size=MEDIAN_SMOOTH_WINDOW, mode="nearest")
        p.filamentHeightSmooth = float(np.round(z_smooth[filament].max(), decimals=2))

def _filament_area_integration(x: np.ndarray, h: np.ndarray) -> float:
    """Filament cross-section by Simpson integration of the filament height h over x.

    abs() so the result is independent of x direction (profiles are stored with x
    decreasing in index, which would otherwise flip the integral's sign).

    Known issue: Simpson fits parabolas through point triples, so on the rare profile whose
    filament-span x is unevenly spaced, non-monotonic, or has near-coincident points (LIDAR jitter
    at the flanks), the fit overshoots and this area spikes to a wrong value (~1 in 3000 profiles
    seen >50% off). `shoelaceArea` is piecewise-linear and immune, so it is the robust cross-check.
    """
    return abs(float(sp.integrate.simpson(h, x=x)))

def _filament_area_shoelace(x: np.ndarray, h: np.ndarray) -> float:
    """Filament cross-section by the shoelace formula on the closed polygon: the filament surface
    (x, h) plus the floor baseline (h = 0) that connects its two ends."""
    px = np.concatenate([x, [x[-1], x[0]]])   # close along the baseline (two h = 0 corners)
    ph = np.concatenate([h, [0.0, 0.0]])
    return 0.5 * float(abs(np.dot(px, np.roll(ph, 1)) - np.dot(ph, np.roll(px, 1))))

def measure_filament_area(profiles: list[profileData]) -> None:
    """Cross-sectional filament area, two ways, above the shared median floor (z = 0 after
    rotate_and_shift_uniform levels every profile to it — deliberately NOT each profile's own fit).

    Across the filament span (leftmost to rightmost filament point of floorMask) the height above the
    median floor is simply z. `area` integrates it with Simpson's rule; `shoelaceArea` is the
    shoelace area of the polygon bounded by the filament surface and the z = 0 baseline. The two use
    different numerical schemes and should agree closely (a cross-check). Units: profile-unit^2
    (1 unit = 0.01 mm, so 1 area unit = 1e-4 mm^2). NaN when a profile has fewer than two filament
    points (no floorMask, or flat). Run after grow_profile_points.
    """
    for p in profiles:
        p.area = np.nan
        p.shoelaceArea = np.nan
        if p.floorMask is None:
            continue
        filament_idx = np.flatnonzero(~p.floorMask)
        if filament_idx.size < 2:
            continue
        sl = slice(int(filament_idx.min()), int(filament_idx.max()) + 1)  # contiguous filament span, edge to edge
        x, h = p.x[sl], p.z[sl]                                    # h = height above the z = 0 median floor
        p.area = float(np.round(_filament_area_integration(x, h), decimals=2))
        p.shoelaceArea = float(np.round(_filament_area_shoelace(x, h), decimals=2))

# Along-track spacing of profiles. 1 profile unit = 0.01 mm, so 1 m = 1e5 units. Each profile
# advances rollerbandSpeed (m/s) * dt (s) along the print path (see profile_advance_distances);
# when speed/time data is missing, profiles fall back to a uniform gap.
METERS_TO_PROFILE_UNITS = 1e5
UNIFORM_PROFILE_DISTANCE = 2000.0  # fallback along-track gap when speed/time is unavailable

def profile_advance_distances(profiles: list[profileData]) -> np.ndarray:
    """Per-profile along-path advance (profile units) from rollerbandSpeed (m/s) x inter-profile dt.

    dt is taken from the sensor clock (`sensorTime` = timestamp_sec + timestamp_usec, monotonic and
    low-jitter), falling back to `arrivalTime`; the first profile gets 0 (path origin). Missing speed
    or a non-monotonic step yields 0 advance for that profile. If neither a time base nor
    `rollerbandSpeed` is available at all (e.g. raw profiles without the PLC join), returns a uniform
    `UNIFORM_PROFILE_DISTANCE` spacing (the pre-physical behaviour). Used by the 3D layout
    (profile3Dplotting) and by measure_filament_volume.
    """
    n = len(profiles)
    if n == 0:
        return np.empty(0)
    t = np.array([p.sensorTime if p.sensorTime is not None else np.nan for p in profiles], float)
    if not np.any(np.isfinite(t)):  # no sensor clock (e.g. old cache) -> capture-PC clock
        t = np.array([p.arrivalTime if p.arrivalTime is not None else np.nan for p in profiles], float)
    speed = np.array([p.rollerbandSpeed if p.rollerbandSpeed is not None else np.nan for p in profiles], float)
    if not np.any(np.isfinite(t)) or not np.any(np.isfinite(speed)):
        return np.full(n, UNIFORM_PROFILE_DISTANCE)

    dist = np.zeros(n)
    dt = np.diff(t)                             # seconds between consecutive profiles
    speed_mid = 0.5 * (speed[:-1] + speed[1:])  # mean belt speed over each interval
    dist[1:] = speed_mid * dt * METERS_TO_PROFILE_UNITS
    dist[~np.isfinite(dist)] = 0.0             # missing speed/time on a profile -> no advance
    dist[dist < 0.0] = 0.0                     # guard non-monotonic time
    return dist

def measure_filament_volume(profiles: list[profileData]) -> None:
    """Volume of each filament segment (a run of consecutive non-flat profiles bounded by flat ones).

    Volume is the cross-section integrated along the print path: over each segment, the sum of
    `shoelaceArea` x the inter-profile advance (`profile_advance_distances`) — a left-Riemann sum of
    area x d(path). Sets two per-profile fields, both in native units (area-unit * distance-unit;
    convert to cm^3 with 1e-9 via FEATURE_DISPLAY):

    - `sliceVolume`: this profile's own slab, `shoelaceArea_i * dist_i` (NaN if its area is NaN, i.e.
      a degenerate < 2-filament-point profile).
    - `segmentVolume`: the whole segment's total (NaN slices count as 0), broadcast onto every profile
      in the run so the segment reads as one value.

    Flat profiles (and any before/after a segment) keep both as None. Needs the PLC-joined
    `rollerbandSpeed` for physical distances (else the uniform-gap fallback); run after
    `flag_flat_profiles` and `measure_filament_area`.
    """
    dist = profile_advance_distances(profiles)
    for p in profiles:
        p.segmentVolume = None
        p.sliceVolume = None
    mask = np.array([not bool(p.isFlat) for p in profiles])  # True = filament (non-flat) profile
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):  # maximal non-flat runs = segments
        slices = np.array([p.shoelaceArea if p.shoelaceArea is not None else np.nan
                           for p in (profiles[i] for i in run)], float) * dist[run]
        total = float(np.nansum(slices))  # NaN slice (degenerate area) contributes 0 to the total
        for i, s in zip(run, slices):
            profiles[i].sliceVolume = float(s)
            profiles[i].segmentVolume = total

def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    """List of index arrays, one per maximal run of consecutive True values in `mask`."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)

def measure_run_lengths(profiles: list[profileData]) -> None:
    """Along-path length of each filament segment and each pure-floor (defect) run.

    Sums the inter-profile advance (`profile_advance_distances`) over each maximal run of like
    profiles, broadcasting the total onto every profile in the run (so a run reads as one value):

    - `segmentLength`: length of a filament segment — a run of non-flat profiles (`isFlat` False).
    - `defectLength`: length of a pure-floor gap — a run of flat profiles (no filament).

    Both are in distance units (0.01 mm; convert to mm with 0.01 via FEATURE_DISPLAY). Every profile
    gets exactly one of the two set (the other stays None), by whether it is flat. Needs the PLC-joined
    `rollerbandSpeed` for physical distances (else the uniform-gap fallback); run after
    `flag_flat_profiles`.
    """
    dist = profile_advance_distances(profiles)
    for p in profiles:
        p.segmentLength = None
        p.defectLength = None
    flat = np.array([bool(p.isFlat) for p in profiles])
    for run in _contiguous_runs(~flat):  # filament segments
        total = float(np.sum(dist[run]))
        for i in run:
            profiles[i].segmentLength = total
    for run in _contiguous_runs(flat):   # pure-floor defects
        total = float(np.sum(dist[run]))
        for i in run:
            profiles[i].defectLength = total

def translate_floor_to_zero(profiles: list[profileData]):
    for p in profiles:
        m,b = get_baseline_from_profileBorder(p.x,p.z)
        p.z = p.z-b
    return profiles

def rotate_pointcloud(profiles: list[profileData]):
    for p in profiles:
        m,b = get_baseline_from_profileBorder(p.x,p.z)

        angle_deg = np.arctan(m)*180/np.pi

        angle_rad = -np.pi/180 * angle_deg
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        points =np.column_stack((p.x, p.z))
        rotatedPoints = points @ R.T
        p.x = rotatedPoints[:,0]
        p.z = rotatedPoints[:,1]


def rotate_and_shift_uniform(profiles: list[profileData]):
    """Level all profiles with one representative (median) rotation and shift.

    Applies the median per-profile tilt and floor offset to every profile, so the real
    height differences between profiles are preserved (only a common tilt/offset removed).
    Median is robust to the bad baseline fits of profiles containing filament. Returns the uniform transform
    as ``(angle, offset)`` (angle in radians) so it can be inverted later to recover the
    pre-leveling coordinates without reloading the raw file (see ``unlevel_profiles``).
    """
    # median tilt angle across profiles (from each floor slope)
    angles = []
    for p in profiles:
        if p.x.shape[0] < MIN_PROFILE_POINTS:  # too short for a reliable fit
            continue
        m, b = get_baseline_from_profileBorder(p.x, p.z)
        angles.append(-np.arctan(m))  # sign matches rotate_pointcloud
    angle = float(np.median(angles)) if angles else 0.0

    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    for p in profiles:
        pts = np.column_stack((p.x, p.z)) @ R.T
        p.x = pts[:, 0]
        p.z = pts[:, 1]

    # median floor height across the now-rotated profiles; also store each profile's
    # floor fit (m, b) so the baseline plot can reuse it instead of refitting
    offsets = []
    for p in profiles:
        if p.x.shape[0] < MIN_PROFILE_POINTS:  # too short for a reliable fit
            p.m = p.b = None
            continue
        m, b = get_baseline_from_profileBorder(p.x, p.z)
        p.m, p.b = m, b
        offsets.append(b)
    offset = float(np.median(offsets)) if offsets else 0.0
    for p in profiles:
        p.z = p.z - offset
        if p.b is not None:
            p.b -= offset  # keep the stored intercept in the final (shifted) coordinates

    print(f"rotate_and_shift_uniform: angle = {np.degrees(angle):.3f} deg, shift = {offset:.2f}")
    return angle, offset


def unlevel_profiles(profiles: list[profileData], angle: float, offset: float) -> list[profileData]:
    """Invert `rotate_and_shift_uniform` to recover each profile's pre-leveling (raw) x/z.

    The forward leveling is one uniform transform for every profile — rotate by `R` (from `angle`,
    radians) then subtract `offset` from z — so it is exactly invertible: add the offset back, then
    rotate by `R` (`R` is orthogonal, so `@ R` undoes the forward `@ R.T`). Returns new lightweight
    `profileData` (name + raw x/z only) — enough to draw the "before" overlay without reloading the
    raw file. Exact for the uniform leveling only; the dormant per-profile levelling would need
    per-profile parameters instead.
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    restored = []
    for p in profiles:
        pts = np.column_stack((p.x, p.z + offset)) @ R  # unshift z, then unrotate
        restored.append(profileData(p.name, pts[:, 0], pts[:, 1]))
    return restored


# z above this (100~1 mm) seeds a confident filament point; at/below it is floor.
FLOOR_POINT_THRESHOLD = 250
# grow the filament into connected points down to this height (20~0.2 mm); below stays floor.
PROFILE_GROW_THRESHOLD = 150
MIN_SEED_LENGTH = 3   # a filament core must span >= this many points (rejects lone spikes)
PROFILE_FILL_GAP = 5  # fill interior floor gaps up to this many points to solidify the filament

# Positional prior: the filament sits near the scan's middle, so points far from centre need more
# height to count as filament (suppresses raised edge floor being mislabelled). The penalty is 0
# within a central plateau, ramping to EDGE_HEIGHT_PENALTY at the edge; added to the seed and
# grow thresholds. t = |x - centre| / half-width (0 = centre, 1 = edge).
FILAMENT_CENTER_HALFWIDTH = 0.4   # central |t| with no penalty (real filaments fade out by ~0.45)
EDGE_HEIGHT_PENALTY = 400     # z units (~4 mm) added to the filament thresholds at the edge

def position_height_penalty(x: np.ndarray) -> np.ndarray:
    """Per-point height penalty (z units) rising from 0 in the centre to EDGE_HEIGHT_PENALTY
    at the profile edges, so points far from the middle need more height to be classed as filament."""
    lo, hi = float(np.min(x)), float(np.max(x))
    half = 0.5 * (hi - lo)
    if half == 0:
        return np.zeros_like(x, dtype=float)
    t = np.abs((x - 0.5 * (lo + hi)) / half)                          # 0 centre .. 1 edge
    ramp = np.clip((t - FILAMENT_CENTER_HALFWIDTH) / (1.0 - FILAMENT_CENTER_HALFWIDTH), 0.0, 1.0)
    return EDGE_HEIGHT_PENALTY * ramp

def _categorization_height(p: profileData, use_profile_baseline: bool) -> np.ndarray:
    """Point heights used for the floor/filament thresholds.

    Default: the uniform (median) levelled z, which keeps the real height differences between
    profiles (good for visualisation). With use_profile_baseline, heights are measured above
    each profile's OWN floor fit (p.m, p.b) as z - (m*x + b), so every profile's floor sits at
    0 regardless of how it deviates from the dataset median (removes residual per-profile tilt/
    offset). Falls back to z when the profile has no stored fit.
    """
    if use_profile_baseline and p.m is not None and p.b is not None:
        return p.z - (p.m * p.x + p.b)
    return p.z

def categorize_floor_points(profiles: list[profileData], threshold: float = FLOOR_POINT_THRESHOLD,
                            use_profile_baseline: bool = False):
    """Set floorMask (True = floor) by height plus a positional prior: floor where
    height <= threshold + position penalty (edge points need more height to seed filament).

    height is the uniform (median) levelled z by default, or each profile's own floor-relative
    height when use_profile_baseline is set (see _categorization_height). Non-destructive,
    vectorised. Pair with grow_profile_points (same flag) to add the filament flanks.
    """
    for p in profiles:
        height = _categorization_height(p, use_profile_baseline)
        p.floorMask = height <= threshold + position_height_penalty(p.x)


def _runs_at_least(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only contiguous True runs of length >= min_len."""
    out = np.zeros_like(mask)
    idx = np.flatnonzero(mask)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if run.size >= min_len:
                out[run] = True
    return out


def _fill_interior_gaps(filament: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill floor gaps <= max_gap that are flanked by filament on both sides."""
    out = filament.copy()
    idx = np.flatnonzero(~filament)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            lo, hi = run[0], run[-1]
            if 0 < lo and hi < filament.size - 1 and run.size <= max_gap and filament[lo - 1] and filament[hi + 1]:
                out[run] = True
    return out


def grow_profile_points(profiles: list[profileData], low_threshold: float = PROFILE_GROW_THRESHOLD,
                        min_seed_length: int = MIN_SEED_LENGTH, max_gap: int = PROFILE_FILL_GAP,
                        use_profile_baseline: bool = False):
    """Hysteresis: a run of points above the grow threshold touching a filament seed becomes filament.

    Seeds shorter than min_seed_length are ignored (lone spikes); after growing, interior
    floor gaps up to max_gap are filled. Recovers flanks and keeps the filament solid; isolated
    low bumps and true floor stay floor. The grow threshold is raised toward the edges by the
    positional prior, so the filament is not grown into raised edge floor. use_profile_baseline sets
    the grow height basis; it is usually the same as categorize_floor_points's, but may differ
    (seed on one basis, grow on another) — as configured in process_profiles. Assumes ordered
    points. Updates floorMask.
    """
    for p in profiles:
        if p.floorMask is None:
            continue
        seed = _runs_at_least(~p.floorMask, min_seed_length)   # filament cores, lone spikes dropped
        candidate = _categorization_height(p, use_profile_baseline) > low_threshold + position_height_penalty(p.x)
        filament = np.zeros(p.z.shape, dtype=bool)
        if candidate.any():
            idx = np.flatnonzero(candidate)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if seed[run].any():       # run touches a filament core -> whole run is filament
                    filament[run] = True
        p.floorMask = ~_fill_interior_gaps(filament, max_gap)


def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

BASELINE_FIT_ITERS = 50  # lower-envelope passes; enough to descend past the curled edge

def get_baseline_from_profileBorder(x, y, borderPoints=150, iters=BASELINE_FIT_ITERS):
    """Fit the floor line (m, b) from the border points, robust to the raised paper edge.

    Iterative lower-envelope: pull points above the current fit down onto it and refit, so
    the line descends past the curled-up edge (and any filament intruding into the border) onto
    the flat floor instead of averaging through them.
    """
    n = borderPoints
    bx = np.concatenate([x[:n], x[-n:]])
    work = np.concatenate([y[:n], y[-n:]]).astype(float)
    m, b = np.polyfit(bx, work, 1)
    for _ in range(iters):
        work = np.minimum(work, m * bx + b)  # clip raised points down to the fit
        m, b = np.polyfit(bx, work, 1)
    return m, b
