"""Per-profile processing algorithms for LIDAR profiles.

Pure functions that operate in place on lists of `profileData` (defined in
profilePointsClass): rotate/level to the floor, categorise floor/bead points, smooth,
measure width, flag flatness, plus the shared `moving_average` and
`get_baseline_from_profileBorder` helpers. The `process_profiles` pipeline in
profileProcessing.py composes these in order.

Units: x and z are in profile units where 1 unit = 0.01 mm (so 20 = 0.2 mm, 100 = 1 mm).
"""
import numpy as np
import scipy as sp
import scipy.signal    # ensure sp.signal.find_peaks is available without relying on side-effect imports
import scipy.integrate # ensure sp.integrate.simpson is available (bead area)
import scipy.ndimage   # ensure sp.ndimage.median_filter is available (smoothed height)

from profilePointsClass import profileData

# Profiles with fewer than this many points are too short for a reliable slope/width fit.
MIN_PROFILE_POINTS = 10

# RMS residual (profile units, ~0.01 mm) of a straight-line fit to the whole profile: below
# this the profile is "flat" (substrate only), above it a bead is present. Set in the valley
# of the bimodal flat/beaded residual distribution.
FLATNESS_RMS_THRESHOLD = 350

def flag_flat_profiles(profiles: list[profileData], use_line_fit: bool = False,
                       threshold: float = FLATNESS_RMS_THRESHOLD, max_bead_points: int = 0) -> None:
    """Flag each profile as flat (substrate only) or not, setting `isFlat`.

    Default (floor-based): flat when categorisation found essentially no bead points — at most
    `max_bead_points` of them. Uses `floorMask`, so run categorize_floor_points +
    grow_profile_points first; `flatness` is left None (it is a line-fit-only metric). With
    use_line_fit=True the old method is used instead: a straight line is fit to all points and
    `flatness` is set to the RMS residual (small for a substrate-only profile, large for a
    bead), `isFlat` = residual < threshold. Too-short profiles are treated as flat.
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
            bead_count = 0 if p.floorMask is None else int((~p.floorMask).sum())
            p.flatness = None
            p.isFlat = bead_count <= max_bead_points

# Cascaded box-smoothing windows (points): first smooth the height, then the |slope|.
HEIGHT_SMOOTH_WINDOWS = (15, 9, 5, 5)
SLOPE_SMOOTH_WINDOWS = (65, 55, 15, 5)
# Peak detection on the smoothed |dz/dx| (each bead flank shows up as a peak):
SLOPE_PEAK_MIN_HEIGHT = 0.15   # min smoothed-slope height to count as a flank peak
SLOPE_PEAK_MIN_DISTANCE = 50   # min points between two peaks
FLANK_FOOT_HEIGHT = 100        # smoothed height (~1 mm above the z=0 floor) marking a flank foot

def _flank_foot(x: np.ndarray, y_smooth: np.ndarray, peak_idx: int, threshold: float,
                toward_smaller_x: bool) -> int:
    """Index of the flank foot: the point nearest the peak, moving toward the bead edge, where the
    smoothed height first drops to `threshold`. Falls back to the extreme edge point on that side if
    the height never gets that low (e.g. raised edge floor)."""
    if toward_smaller_x:
        cand = np.flatnonzero((x < x[peak_idx]) & (y_smooth <= threshold))
        return int(cand[np.argmax(x[cand])]) if cand.size else int(np.argmin(x))
    cand = np.flatnonzero((x > x[peak_idx]) & (y_smooth <= threshold))
    return int(cand[np.argmin(x[cand])]) if cand.size else int(np.argmax(x))

def width_from_smoothed_slope(profiles: list[profileData]) -> None:
    """Bead width between the feet of the two OUTERMOST bead flanks.

    Smooths the height and its |dz/dx| with cascaded box filters (locally — no stored arrays), finds
    the flank peaks in the smoothed slope, keeps the furthest-left and furthest-right (intermediate
    peaks from surface texture are ignored), then walks each flank down to its foot near the floor so
    the width sits at the bead base rather than mid-flank. Sets `width` and `peaks` (the two width-
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

def width_from_bead_edges(profiles: list[profileData]) -> None:
    """Bead width from the outer bead points: the x-span between the leftmost and rightmost
    bead point (uses `floorMask`). Sets `beadWidth` and `beadWidthIdx` (the two point indices).
    Run after grow_profile_points; NaN / None when a profile has fewer than two bead points.
    """
    for p in profiles:
        if p.floorMask is None:
            p.beadWidth = np.nan
            p.beadWidthIdx = None
            continue
        bead_idx = np.flatnonzero(~p.floorMask)
        if bead_idx.size < 2:
            p.beadWidth = np.nan
            p.beadWidthIdx = None
            continue
        left = int(bead_idx[np.argmin(p.x[bead_idx])])   # bead point at smallest x
        right = int(bead_idx[np.argmax(p.x[bead_idx])])  # bead point at largest x
        p.beadWidthIdx = np.array([left, right])
        p.beadWidth = float(np.round(abs(p.x[right] - p.x[left]), decimals=2))

HEIGHT_PERCENTILE = 95     # bead height = this percentile of the bead-point heights (robust to spikes)
MEDIAN_SMOOTH_WINDOW = 15  # median-filter window (points) for the smoothed-height measure

def measure_bead_height(profiles: list[profileData], percentile: float = HEIGHT_PERCENTILE) -> None:
    """Robust bead height above the shared median floor (z = 0 after rotate_and_shift_uniform), two ways.

    `beadHeight` is the `percentile`-th percentile of the bead points' z (uses `floorMask`), so a lone
    outlier/noise spike above that percentile is ignored. `beadHeightSmooth` is the max of a
    median-smoothed profile over the bead points — a median filter removes single-point spikes, so its
    peak is robust too; the two cross-check each other. Units: profile units (1 unit = 0.01 mm). NaN
    when a profile has no bead points (no floorMask, or flat). Run after grow_profile_points.
    """
    for p in profiles:
        p.beadHeight = np.nan
        p.beadHeightSmooth = np.nan
        if p.floorMask is None:
            continue
        bead = ~p.floorMask
        bead_z = p.z[bead]
        if bead_z.size == 0:
            continue
        p.beadHeight = float(np.round(np.percentile(bead_z, percentile), decimals=2))
        z_smooth = sp.ndimage.median_filter(p.z, size=MEDIAN_SMOOTH_WINDOW, mode="nearest")
        p.beadHeightSmooth = float(np.round(z_smooth[bead].max(), decimals=2))

def _bead_area_integration(x: np.ndarray, h: np.ndarray) -> float:
    """Bead cross-section by Simpson integration of the bead height h over x.

    abs() so the result is independent of x direction (profiles are stored with x
    decreasing in index, which would otherwise flip the integral's sign).

    Known issue: Simpson fits parabolas through point triples, so on the rare profile whose
    bead-span x is unevenly spaced, non-monotonic, or has near-coincident points (LIDAR jitter
    at the flanks), the fit overshoots and this area spikes to a wrong value (~1 in 3000 profiles
    seen >50% off). `shoelaceArea` is piecewise-linear and immune, so it is the robust cross-check.
    """
    return abs(float(sp.integrate.simpson(h, x=x)))

def _bead_area_shoelace(x: np.ndarray, h: np.ndarray) -> float:
    """Bead cross-section by the shoelace formula on the closed polygon: the bead surface
    (x, h) plus the floor baseline (h = 0) that connects its two ends."""
    px = np.concatenate([x, [x[-1], x[0]]])   # close along the baseline (two h = 0 corners)
    ph = np.concatenate([h, [0.0, 0.0]])
    return 0.5 * float(abs(np.dot(px, np.roll(ph, 1)) - np.dot(ph, np.roll(px, 1))))

def measure_bead_area(profiles: list[profileData]) -> None:
    """Cross-sectional bead area, two ways, above the shared median floor (z = 0 after
    rotate_and_shift_uniform levels every profile to it — deliberately NOT each profile's own fit).

    Across the bead span (leftmost to rightmost bead point of floorMask) the height above the
    median floor is simply z. `area` integrates it with Simpson's rule; `shoelaceArea` is the
    shoelace area of the polygon bounded by the bead surface and the z = 0 baseline. The two use
    different numerical schemes and should agree closely (a cross-check). Units: profile-unit^2
    (1 unit = 0.01 mm, so 1 area unit = 1e-4 mm^2). NaN when a profile has fewer than two bead
    points (no floorMask, or flat). Run after grow_profile_points.
    """
    for p in profiles:
        p.area = np.nan
        p.shoelaceArea = np.nan
        if p.floorMask is None:
            continue
        bead_idx = np.flatnonzero(~p.floorMask)
        if bead_idx.size < 2:
            continue
        sl = slice(int(bead_idx.min()), int(bead_idx.max()) + 1)  # contiguous bead span, edge to edge
        x, h = p.x[sl], p.z[sl]                                    # h = height above the z = 0 median floor
        p.area = float(np.round(_bead_area_integration(x, h), decimals=2))
        p.shoelaceArea = float(np.round(_bead_area_shoelace(x, h), decimals=2))

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
    Median is robust to the bad baseline fits of beaded profiles.
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
    return profiles


# z above this (~1 mm) seeds a confident bead point; at/below it is floor.
FLOOR_POINT_THRESHOLD = 100
# grow the bead into connected points down to this height (~0.2 mm); below stays floor.
PROFILE_GROW_THRESHOLD = 20
MIN_SEED_LENGTH = 3   # a bead core must span >= this many points (rejects lone spikes)
PROFILE_FILL_GAP = 5  # fill interior floor gaps up to this many points to solidify the bead

# Positional prior: the bead sits near the scan's middle, so points far from centre need more
# height to count as bead (suppresses raised edge floor being mislabelled). The penalty is 0
# within a central plateau, ramping to EDGE_HEIGHT_PENALTY at the edge; added to the seed and
# grow thresholds. t = |x - centre| / half-width (0 = centre, 1 = edge).
BEAD_CENTER_HALFWIDTH = 0.4   # central |t| with no penalty (real beads fade out by ~0.45)
EDGE_HEIGHT_PENALTY = 400     # z units (~4 mm) added to the bead thresholds at the edge

def position_height_penalty(x: np.ndarray) -> np.ndarray:
    """Per-point height penalty (z units) rising from 0 in the centre to EDGE_HEIGHT_PENALTY
    at the profile edges, so points far from the middle need more height to be classed as bead."""
    lo, hi = float(np.min(x)), float(np.max(x))
    half = 0.5 * (hi - lo)
    if half == 0:
        return np.zeros_like(x, dtype=float)
    t = np.abs((x - 0.5 * (lo + hi)) / half)                          # 0 centre .. 1 edge
    ramp = np.clip((t - BEAD_CENTER_HALFWIDTH) / (1.0 - BEAD_CENTER_HALFWIDTH), 0.0, 1.0)
    return EDGE_HEIGHT_PENALTY * ramp

def _categorization_height(p: profileData, use_profile_baseline: bool) -> np.ndarray:
    """Point heights used for the floor/bead thresholds.

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
    height <= threshold + position penalty (edge points need more height to seed bead).

    height is the uniform (median) levelled z by default, or each profile's own floor-relative
    height when use_profile_baseline is set (see _categorization_height). Non-destructive,
    vectorised. Pair with grow_profile_points (same flag) to add the bead flanks.
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


def _fill_interior_gaps(bead: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill floor gaps <= max_gap that are flanked by bead on both sides."""
    out = bead.copy()
    idx = np.flatnonzero(~bead)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            lo, hi = run[0], run[-1]
            if 0 < lo and hi < bead.size - 1 and run.size <= max_gap and bead[lo - 1] and bead[hi + 1]:
                out[run] = True
    return out


def grow_profile_points(profiles: list[profileData], low_threshold: float = PROFILE_GROW_THRESHOLD,
                        min_seed_length: int = MIN_SEED_LENGTH, max_gap: int = PROFILE_FILL_GAP,
                        use_profile_baseline: bool = False):
    """Hysteresis: a run of points above the grow threshold touching a bead seed becomes bead.

    Seeds shorter than min_seed_length are ignored (lone spikes); after growing, interior
    floor gaps up to max_gap are filled. Recovers flanks and keeps the bead solid; isolated
    low bumps and true floor stay floor. The grow threshold is raised toward the edges by the
    positional prior, so the bead is not grown into raised edge floor. use_profile_baseline sets
    the grow height basis; it is usually the same as categorize_floor_points's, but may differ
    (seed on one basis, grow on another) — as configured in process_profiles. Assumes ordered
    points. Updates floorMask.
    """
    for p in profiles:
        if p.floorMask is None:
            continue
        seed = _runs_at_least(~p.floorMask, min_seed_length)   # bead cores, lone spikes dropped
        candidate = _categorization_height(p, use_profile_baseline) > low_threshold + position_height_penalty(p.x)
        bead = np.zeros(p.z.shape, dtype=bool)
        if candidate.any():
            idx = np.flatnonzero(candidate)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if seed[run].any():       # run touches a bead core -> whole run is bead
                    bead[run] = True
        p.floorMask = ~_fill_interior_gaps(bead, max_gap)


def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

BASELINE_FIT_ITERS = 50  # lower-envelope passes; enough to descend past the curled edge

def get_baseline_from_profileBorder(x, y, borderPoints=150, iters=BASELINE_FIT_ITERS):
    """Fit the floor line (m, b) from the border points, robust to the raised paper edge.

    Iterative lower-envelope: pull points above the current fit down onto it and refit, so
    the line descends past the curled-up edge (and any bead intruding into the border) onto
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
