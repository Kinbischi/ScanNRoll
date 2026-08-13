"""Per-segment shape features: how a filament segment's cross-section evolves along the print path.

================================ SEGMENT PROCESSING SUMMARY ================================
A *segment* is a maximal run of `isSegment` profiles (cleaned filament, from `clean_flat_runs`). A printed
filament typically starts a bit wide, thins slightly along a plateau, then ruptures abruptly at the end.
`measure_segment_shape` turns each segment into one signal and measures that behaviour.

THE SIGNAL
  S(s) = `areaShoelace` (filament cross-section) per profile, vs s = along-path arc-length in mm
  (cumulative `profile_advance_distances`, physical from `rollerbandSpeed`). S is median-smoothed over
  `SEGMENT_SHAPE_SMOOTH` (7) profiles so a single jittery profile can't distort it, while the ~5-15 mm
  rupture cliff stays sharp. `bodyLevel` = the median of S over the middle 20-80 % of the segment
  (`SEGMENT_BODYLEVEL_WINDOW`) = the reference "normal" cross-section, used to normalise the features.

PARTITIONING (all boundaries derived on the ±`SEGMENT_BODY_BAND` (15 %) plateau band, so the debug flag
matches exactly what the features use). Along one segment:
  start ramp ─▶ [overshoot peak] ─▶ BODY plateau ─▶ (shoulder) ─▶ RUPTURE cliff
  - body_start = where the ramp first settles into the band and stays for `SETTLE_SPAN_MM` (3 mm), after
    the early overshoot peak (`_settle_index`).
  - BODY = the plateau: body_start .. body_end, where body_end = the last point still inside the band
    before the signal leaves it toward the cliff/end. The taper is fit here.
  - rupture_start = the top of the terminal cliff: the steepest descent in the segment's back half, walked
    left to where the slope is still >= `RUPTURE_START_GFRAC` of that peak steepness (`_rupture_start`).
  - A segment "ruptures" only if its end level (median over the last `SEGMENT_END_SPAN_MM`, 1.5 mm) is
    < `RUPTURE_GATE_FRAC` (40 %) of bodyLevel; otherwise it ended thick (truncated) and the rupture
    features are NaN.
  - The start ramp and the body->rupture shoulder are NOT used by any feature -> left unclassified.

THE FEATURES (broadcast onto every profile of the segment, like segmentVolume; physical units):
  - `segmentBodyThinning` (%/mm)          Theil-Sen slope of S over the BODY plateau, / bodyLevel x100
                                          (negative = thinning). Robust median-slope fit.
  - `segmentBodyThinningStability` (-1..1) Spearman(S, s) over the body: -1 = steadily thinning, ~0 = wavy.
  - `segmentHeadOvershoot` (%)            peak of the start ramp over bodyLevel = the initial bulge height.
  - `segmentCriticalArea` (mm^2)          S at rupture_start = the cross-section at which failure begins
                                          (a lower bound: the last cross-section still detected as filament).
  - `segmentRuptureLength` (mm)           arc-length rupture_start .. end = the cliff length (short = snap).
  - `segmentRuptures` (0/1)               did it rupture (gate above); its per-speed mean = the rupture rate.
  - `segmentSection` (1/2/3, else NaN)    per-profile DEBUG flag (varies within the segment): 1 body,
                                          2 rupture, 3 a small band around the overshoot peak; the start
                                          ramp + shoulder are NaN. Shows exactly the regions used above.

WHY THESE CHOICES (validated on 120 real Exp1 segments):
  - The body is the *actual* plateau (band-based), not a fixed 25-75 % window: the fixed window mis-samples
    (its edges land in dips/ramps for many segments). The band-body taper is self-robust (Spearman 0.99 to
    the band width, 0.998 to the smoothing) and representative.
  - A separate "rupture sharpness" was rejected as smoothing-fragile; `segmentRuptureLength` is the
    abruptness measure. `segmentCriticalWidth` and `segmentSettleLength` were dropped as redundant/soft.
  - Everything is arc-length (mm) based, so the features are speed-robust; a segment's median speed and the
    per-run aggregation live in the feature-vs-PLC stepwise plot (compare rupture/thinning vs belt speed).
============================================================================================

Sits one layer above `profileProcessingAlgorithms` (reuses its run helpers + physical spacing); composed
by the `profileProcessing.py` entry point after the PLC join. Units: `areaShoelace` is in profile-unit^2
(1 unit = 1e-4 mm^2), arc-length in profile units (1 unit = 0.01 mm); features are stored in physical
units (mm, mm^2, %/mm, %, dimensionless).
"""
import numpy as np
from scipy.stats import spearmanr, theilslopes

from profilePointsClass import profileData
from profileProcessingAlgorithms import (
    PROFILE_UNITS_TO_MM,
    _contiguous_runs,
    _segment_mask,
    profile_advance_distances,
)

AREA_UNITS_TO_MM2 = 1e-4          # 1 area unit (profile-unit^2) = 1e-4 mm^2 (matches FEATURE_DISPLAY areaShoelace)
SEGMENT_SHAPE_SMOOTH = 7          # median-filter window (profiles) for the along-segment cross-section signal
SEGMENT_BODYLEVEL_WINDOW = (0.20, 0.80)  # arc-length fraction the robust body level (median scalar) is taken over
SEGMENT_BODY_BAND = 0.15          # the body plateau = where |A - body| <= this fraction of body (settle-in + body-end)
SETTLE_SPAN_MM = 3.0             # the body starts where A first stays inside the band for this many mm (sustained)
RUPTURE_START_GFRAC = 0.25        # cliff top: walk left from the steepest descent while slope <= this fraction of it
RUPTURE_GATE_FRAC = 0.40          # a segment "ruptures" if its end level < this * body level (else it ended thick)
SEGMENT_END_SPAN_MM = 1.5         # end level = median of the signal over the last this-many mm
PEAK_MARK_HALFWIDTH = 5           # profiles each side of the overshoot peak flagged in segmentSection (survives decimation)
MIN_SEGMENT_PROFILES = 40         # segments shorter than this can't support the body + cliff fits -> features stay None

# segmentSection phase codes (NaN elsewhere = start ramp + shoulder, unused by any feature):
_SECTION_BODY, _SECTION_RUPTURE, _SECTION_PEAK = 1.0, 2.0, 3.0

_SEGMENT_SHAPE_FIELDS = ("segmentBodyThinning", "segmentBodyThinningStability", "segmentCriticalArea",
                         "segmentRuptureLength", "segmentHeadOvershoot", "segmentRuptures")

def _median_smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred nan-aware median filter (window in samples): each output is the median of the finite
    values in the window, so isolated NaNs (degenerate profiles) don't spread. NaN where all-NaN."""
    n = y.shape[0]
    out = np.full(n, np.nan)
    h = window // 2
    for i in range(n):
        w = y[max(0, i - h): i + h + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = float(np.median(w))
    return out

def _nearest_finite(arr: np.ndarray, i: int) -> float:
    """The value at index i, or the nearest finite value on either side; NaN if none is finite."""
    if 0 <= i < arr.size and np.isfinite(arr[i]):
        return float(arr[i])
    for d in range(1, arr.size):
        for j in (i - d, i + d):
            if 0 <= j < arr.size and np.isfinite(arr[j]):
                return float(arr[j])
    return np.nan

def _settle_index(s: np.ndarray, A: np.ndarray, body: float) -> "int | None":
    """Body-start index: scanning forward from the early overshoot peak, the first point where the smoothed
    signal enters and stays within SEGMENT_BODY_BAND of the body level for SETTLE_SPAN_MM (where the ramp/
    overshoot settles onto the plateau)."""
    L = s[-1]
    early = np.flatnonzero((s <= 0.5 * L) & np.isfinite(A))
    if early.size == 0:
        return None
    peak = int(early[np.argmax(A[early])])       # the overshoot bulge (early max)
    band = SEGMENT_BODY_BAND * body
    for i in range(peak, s.size):
        if np.isfinite(A[i]) and abs(A[i] - body) <= band:
            j = i
            while j < s.size and s[j] - s[i] < SETTLE_SPAN_MM:
                j += 1
            window = A[i:j][np.isfinite(A[i:j])]
            if window.size and np.all(np.abs(window - body) <= band):
                return i
    return None

def _rupture_start(s: np.ndarray, A: np.ndarray) -> "int | None":
    """Rupture-start index = top of the terminal cliff: find the steepest descent p in the terminal half
    of the segment, then walk left while the slope is still >= RUPTURE_START_GFRAC of the peak steepness.

    (`A` is already median-smoothed, so the single steepest gradient is not a spike — a robustness check
    on the real data found it only ~11% steeper than a locally-averaged version; see the module history.)
    """
    g = np.gradient(A, s)
    idx = np.flatnonzero((s > 0.5 * s[-1]) & np.isfinite(g))
    if idx.size == 0:
        return None
    p = int(idx[np.argmin(g[idx])])
    if not np.isfinite(g[p]) or g[p] >= 0:
        return None
    k = p
    while k > 0 and np.isfinite(g[k]) and g[k] <= RUPTURE_START_GFRAC * g[p]:
        k -= 1
    return k

def _segment_shape_values(s: np.ndarray, A: np.ndarray) -> "tuple[dict, np.ndarray] | None":
    """Compute the per-segment shape features from the smoothed area signal A(s) (arc-length s in mm).
    Returns (feature dict in physical units, per-profile section code array), or None for a degenerate
    segment.

    Regions (all detected on the ±SEGMENT_BODY_BAND plateau band, so the flag matches what the features
    use): the **body** = the plateau from `body_start` (where the ramp settles into the band) to
    `body_end` (where it leaves the band on the way down); the **rupture** = `[rupture_start, end]`
    (cliff); the **start** ramp `[0, body_start)` and the **shoulder** `[body_end, rupture_start)` are
    left unclassified (NaN) — no feature uses them as a whole. The overshoot **peak** (a small band around
    the highest point of the start ramp, which `segmentHeadOvershoot` reads) is flagged separately.

    Taper: Theil-Sen slope of A over the body plateau / body level (%/mm) + its Spearman monotonicity.
    Overshoot: the start ramp's peak, % over body. Rupture (gated on ending thin): critical area at the
    cliff top + the cliff length (mm). `bodyLevel` is a fixed-window (20-80%) robust median scalar.
    """
    L = float(s[-1])
    finite = np.isfinite(A)
    if L <= 0 or finite.sum() < 5:
        return None
    blo, bhi = SEGMENT_BODYLEVEL_WINDOW
    bmask = (s >= blo * L) & (s <= bhi * L) & finite
    body = float(np.median(A[bmask])) if bmask.any() else np.nan
    if not np.isfinite(body) or body <= 0:
        return None
    band = SEGMENT_BODY_BAND * body
    out: dict[str, float] = {}

    # rupture gate + cliff top (needed before the body, since the body ends where the plateau meets the cliff)
    etail = finite & (s >= L - SEGMENT_END_SPAN_MM)
    end_level = float(np.median(A[etail])) if etail.any() else np.nan
    ruptures = bool(np.isfinite(end_level) and end_level < RUPTURE_GATE_FRAC * body)
    rupture_start = _rupture_start(s, A) if ruptures else None

    # body plateau = within the band, from body_start (ramp settles in) to where it leaves the band (before
    # the cliff / segment end). The shoulder between body_end and rupture_start is left unclassified.
    body_start = _settle_index(s, A, body)
    body_end = None
    if body_start is not None:
        upper = rupture_start if rupture_start is not None else A.size - 1
        region = np.arange(body_start, upper + 1)
        inband = region[finite[region] & (np.abs(A[region] - body) <= band)]
        body_end = int(inband[-1]) if inband.size else body_start

    # --- taper over the band-body plateau ---
    if body_end is not None and body_end > body_start:
        idx = np.arange(body_start, body_end + 1)
        idx = idx[finite[idx]]
        if idx.size >= 5:
            out["segmentBodyThinning"] = 100.0 * float(theilslopes(A[idx], s[idx])[0]) / body  # %/mm
            r = spearmanr(s[idx], A[idx])[0]
            out["segmentBodyThinningStability"] = float(r) if np.isfinite(r) else np.nan
        else:
            out["segmentBodyThinning"] = out["segmentBodyThinningStability"] = np.nan
    else:
        out["segmentBodyThinning"] = out["segmentBodyThinningStability"] = np.nan

    # --- overshoot: the peak of the start ramp [0, body_start) vs body ---
    peak = None
    if body_start is not None and body_start > 0:
        ramp = np.arange(0, body_start)[finite[:body_start]]
        if ramp.size:
            peak = int(ramp[np.argmax(A[ramp])])
    out["segmentHeadOvershoot"] = 100.0 * (float(A[peak]) - body) / body if peak is not None else np.nan

    # --- rupture ---
    out["segmentRuptures"] = 1.0 if ruptures else 0.0
    if rupture_start is not None:
        out["segmentCriticalArea"] = _nearest_finite(A, rupture_start) * AREA_UNITS_TO_MM2   # mm^2
        out["segmentRuptureLength"] = L - float(s[rupture_start])                            # mm
    else:
        out["segmentCriticalArea"] = out["segmentRuptureLength"] = np.nan

    # per-profile section flag: NaN start/shoulder; body; rupture; a peak band around the overshoot point
    sections = np.full(s.size, np.nan)
    if body_end is not None:
        sections[body_start:body_end + 1] = _SECTION_BODY
    if rupture_start is not None:
        sections[rupture_start:] = _SECTION_RUPTURE
    if peak is not None:  # small band so it survives heat-map decimation; clamped to the start ramp
        sections[max(0, peak - PEAK_MARK_HALFWIDTH): min(body_start, peak + PEAK_MARK_HALFWIDTH + 1)] = _SECTION_PEAK
    return out, sections

def measure_segment_shape(profiles: list[profileData]) -> None:
    """Per-segment shape features (thinning / startup / rupture), broadcast onto every profile of the
    segment (like segmentVolume / segmentLength). Computed on the along-path cross-section signal
    areaShoelace (smoothed).

    Sets on each segment profile (None off-segment or on a too-short segment; NaN where a phase is
    absent, e.g. the rupture fields on a segment that ended thick):

    - `segmentBodyThinning` (%/mm)          thinning rate over the body plateau (negative = thinning)
    - `segmentBodyThinningStability` (-1..1) how steadily the body thins (Spearman of area vs arc-length)
    - `segmentCriticalArea` (mm^2)       cross-section at the rupture cliff top (last detected before failure)
    - `segmentRuptureLength` (mm)        arc-length of the terminal cliff (short = abrupt snap)
    - `segmentHeadOvershoot` (%)         the start ramp's peak, over the body level ("wider at the start")
    - `segmentRuptures` (0/1)            whether the segment ended in a rupture (vs truncated thick)
    - `segmentSection` (1/2/3, else NaN) per-profile phase flag for the heat map: 1 body, 2 rupture,
                                         3 the overshoot-peak band; the start ramp + shoulder are NaN
                                         (unclassified — no feature uses them as a whole)

    Needs `isSegment` (clean_flat_runs) + areaShoelace (measure_filament_area) + the PLC-joined spacing
    (profile_advance_distances). Run after clean_flat_runs / measure_run_lengths. See the module constants
    for the (empirical) window and threshold values.
    """
    for p in profiles:
        for f in _SEGMENT_SHAPE_FIELDS:
            setattr(p, f, None)
        p.segmentSection = None  # per-profile phase code (varies within a segment), not a broadcast value

    dist = profile_advance_distances(profiles)
    area = np.array([p.areaShoelace if p.areaShoelace is not None else np.nan for p in profiles], float)

    for run in _contiguous_runs(_segment_mask(profiles)):
        if run.size < MIN_SEGMENT_PROFILES:
            continue
        s_mm = np.concatenate([[0.0], np.cumsum(dist[run][1:])]) * PROFILE_UNITS_TO_MM  # arc-length (mm)
        A = _median_smooth(area[run], SEGMENT_SHAPE_SMOOTH)
        result = _segment_shape_values(s_mm, A)
        if result is None:
            continue
        values, sections = result
        for local_i, i in enumerate(run):
            profiles[i].segmentSection = float(sections[local_i])  # per-profile phase code
            for f, v in values.items():                            # per-segment features (broadcast)
                setattr(profiles[i], f, float(v))
