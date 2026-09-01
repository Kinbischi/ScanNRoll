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
  - body_start = where the signal first settles into the band and stays for `SETTLE_SPAN_MM` (3 mm)
    (`_settle_index`). Only a prominent *leading* overshoot — a bulge > the band above body within the first
    `HEAD_OVERSHOOT_FRAC` of the segment — is skipped first; a segment with no/little overshoot (or a
    mid-segment dome/spike) settles from the start, so its body isn't pushed late.
  - BODY = the plateau: body_start .. body_end, where body_end = the last in-band point held back at least
    `SEGMENT_BODY_RUPTURE_MARGIN_MM` (3 mm) before the cliff top (or the segment end if it never ruptures),
    so the pre-rupture roll-off stays out of the taper fit. The taper is fit over this plateau. If a NECK
    (below) occurs inside the plateau, body_end is pulled back to the first neck's onset, so the taper is fit
    only over the clean pre-neck stretch — a plateau left too short by this then falls into the tiny-body sort-out.

NECKING (`_detect_necks`, LOCAL DE-TREND): a NECK is a dip that thins and then RECOVERS (climbs back), unlike
  the terminal rupture (thins and stays low). It is detected against a LOCAL baseline — the signal median-
  filtered over `NECK_LOCAL_BASELINE_MM` (80 mm) of arc-length, which follows slow bulges / tapers — so a neck
  is a dip of the AREA below its local baseline (residual `RA = A - baseline` < 0) whose dip-from-shoulder and
  recovery each exceed `NECK_AREA_PROMINENCE_FRAC` (30 %) of the LOCAL area baseline, CONFIRMED on the WIDTH
  residual (>= `NECK_WIDTH_CONFIRM_FRAC` (10 %) of the local width baseline). Measuring
  against the local trend keeps every neck span local (onset/recovery = where the residual returns to baseline)
  and stops a slow bulge or taper from itself reading as a neck. Necks count only after the body settles; the
  first neck ends the body (above). Each neck is written to the per-profile `segmentNeck` marker (1 over the
  neck span) — also on continuous filaments (otherwise not shape-analysed) — and the per-rollerband-speed neck
  RATE (necks / m) is pooled from that marker later, in `featureRates`. NOTE this is a LOCAL-relative
  definition: a broad, gentle neck the baseline can follow is ABSORBED (not counted) — chosen deliberately for
  robustness on long prints, at the cost of some broad low-speed necks.
  - rupture_start = the top of the terminal cliff: the steepest descent in the segment's back half, walked
    left to where the slope is still >= `RUPTURE_START_GFRAC` of that peak steepness (`_rupture_start`).
  - A segment "ruptures" only if its end level (median over the last `SEGMENT_END_SPAN_MM`, 1.5 mm) is
    < `RUPTURE_GATE_FRAC` (40 %) of bodyLevel; otherwise it ended thick (truncated) and the rupture
    features are NaN.
  - The start ramp and the body->rupture shoulder are NOT used by any feature -> left unclassified.

THE FEATURES (broadcast onto every profile of the segment, like segmentVolume; physical units):
  - body thinning family                  the same taper measured on THREE body signals — area (S =
                                          areaShoelace), width (widthOuter) and height (heightP95):
      `segmentBodyAreaThinning`  / `...WidthThinning`  / `...HeightThinning`  (%/mm)
                                          Theil-Sen slope of the signal over the BODY plateau, / its body
                                          level x100 (negative = thinning). Robust median-slope fit.
      `segmentBodyAreaSteadiness`/ `...WidthSteadiness`/ `...HeightSteadiness` (-1..1)
                                          Spearman(signal, s) over the body: -1 = steadily thinning, ~0 = wavy.
  - `segmentHeadOvershoot` (%)            peak of the start ramp over bodyLevel = the initial bulge height.
  - `segmentCriticalArea` (mm^2)          S at rupture_start = the cross-section at which failure begins
                                          (a lower bound: the last cross-section still detected as filament).
  - `segmentCriticalWidth` (mm)           widthOuter at rupture_start = the outer width at that same profile
                                          (the width analogue of segmentCriticalArea).
  - `segmentRuptureLength` (mm)           arc-length rupture_start .. end = the cliff length (short = snap).
  - `segmentRuptures` (0/1)               did it rupture (gate above); its per-speed mean = the rupture rate.
  - `segmentNeck` (0/1 per profile)       per-profile marker: 1 over each neck span, 0 elsewhere in an analysed
                                          run — the heat-map "where are the necks" view; the per-speed neck RATE
                                          (necks/m) is pooled from it in `featureRates` (see NECKING).
  - `segmentSection` (1/2/3, else NaN)    per-profile DEBUG flag (varies within the segment): 1 body,
                                          2 rupture, 3 a small band around the overshoot peak; the start
                                          ramp + shoulder are NaN. Shows exactly the regions used above.

WEIRD-SEGMENT SORT-OUT (`segmentShapeStatus`, a per-segment 0-6 DEBUG code; the shape features above are set
only for status 0 = KEPT, so weird segments drop out of the thinning plots but you can SEE why each is out):
  0 kept | 1 too short (< 50 mm) | 2 continuous filament (> 600 mm) | 3 degenerate (no body) |
  4 tiny body (plateau < `MIN_BODY_LENGTH_MM` -> a steep, unreliable taper) |
  5 high width change (outer width swings > `WIDTH_CHANGE_MAX_FRAC` of the body width across the body:
    turbulent/spreading, so the area taper misleads) | 6 didn't rupture (ended thick / truncated).
  `segmentRuptures` is the exception — it stays set on every valid-body segment (even sorted-out ones) so the
  rupture *rate* isn't lost. The sort-out is shape-analysis only; length/volume/defect keep every segment.

WHY THESE CHOICES (validated on 120 real Exp1 segments):
  - The body is the *actual* plateau (band-based), not a fixed 25-75 % window: the fixed window mis-samples
    (its edges land in dips/ramps for many segments). The band-body taper is self-robust (Spearman 0.99 to
    the band width, 0.998 to the smoothing) and representative.
  - A separate "rupture sharpness" was rejected as smoothing-fragile; `segmentRuptureLength` is the
    abruptness measure. `segmentSettleLength` was dropped as soft. `segmentCriticalWidth` (the outer width
    at the same rupture profile as `segmentCriticalArea`) is kept for the per-segment proportionality checks.
  - Only *discrete* segments are analysed: length in [`SEGMENT_SHAPE_MIN_LENGTH_MM` (50 mm),
    `MAX_SEGMENT_LENGTH_MM` (600 mm)]. Below 50 mm the taper is noisy (a mm cut, not a profile-count cut, so it
    holds across belt speeds); above 600 mm the run is a continuous filament (`isContinuousFilament`), a
    different object with no single startup/rupture, handled separately.
  - The body ends `SEGMENT_BODY_RUPTURE_MARGIN_MM` before the cliff so the pre-rupture roll-off does not bias
    the taper (previously body_end could touch the cliff top, or run to the last profile on a non-rupturing run).
  - Everything is arc-length (mm) based, so the features are speed-robust; a segment's median speed and the
    per-run aggregation live in the feature-vs-PLC stepwise plot (compare rupture/thinning vs belt speed).
============================================================================================

Sits one layer above `profileProcessingAlgorithms` (reuses its run helpers + physical spacing); composed
by the `profileProcessing.py` entry point after the PLC join. Units: `areaShoelace` is in profile-unit^2
(1 unit = 1e-4 mm^2), arc-length in profile units (1 unit = 0.01 mm); features are stored in physical
units (mm, mm^2, %/mm, %, dimensionless).
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr, theilslopes

from profilePointsClass import profileData
from profileProcessingAlgorithms import (
    MAX_SEGMENT_LENGTH_MM,
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
HEAD_OVERSHOOT_FRAC = 0.25        # a head overshoot is a prominent bulge within this leading fraction; only then is the body started after it (a mid-segment dome/spike is NOT treated as an overshoot -> body starts early)
RUPTURE_START_GFRAC = 0.25        # cliff top: walk left from the steepest descent while slope <= this fraction of it
RUPTURE_GATE_FRAC = 0.40          # a segment "ruptures" if its end level < this * body level (else it ended thick)
SEGMENT_END_SPAN_MM = 1.5         # end level = median of the signal over the last this-many mm
SEGMENT_BODY_RUPTURE_MARGIN_MM = 3.0  # the body plateau must end at least this far before the cliff top / segment end (keeps the pre-rupture roll-off out of the taper fit)
PEAK_MARK_HALFWIDTH = 5           # profiles each side of the overshoot peak flagged in segmentSection (survives decimation)
# Length gating (mm), evaluated on the physical arc-length. Below MIN, a segment is too short to shape-analyse
# stably (its taper is noisy); above MAX_SEGMENT_LENGTH_MM it is a continuous filament, not a discrete segment.
SEGMENT_SHAPE_MIN_LENGTH_MM = 50.0  # segments shorter than this are not shape-analysed (features stay None)
# Weird-segment sort-out (shape analysis only; see segmentShapeStatus). A segment whose plateau is shorter than
# MIN_BODY_LENGTH_MM gives a degenerate (steep, unreliable) taper; one whose outer width swings by more than
# WIDTH_CHANGE_MAX_FRAC of the body width across the body is turbulent/spreading (its area taper misleads).
MIN_BODY_LENGTH_MM = 15.0        # sort out if the body plateau is shorter than this (catches degenerate tapers)
WIDTH_CHANGE_MAX_FRAC = 0.40     # sort out if (max-min widthOuter over the body) / body width exceeds this
# Necking: a NECK is a local dip in the along-segment cross-section that thins then RECOVERS (climbs back),
# unlike the terminal rupture (thins and stays low). Detected by LOCAL DE-TREND: the smoothed signal is
# compared to a rolling-median local baseline (median over NECK_LOCAL_BASELINE_MM of arc-length) that follows
# slow bulges / tapers, and a neck is a dip of the AREA below that local baseline (residual R = A - baseline)
# that recovers, CONFIRMED on the WIDTH residual (width dips ~half as much at a real neck). Measuring against
# the LOCAL trend (not one global body level) keeps every neck span local and stops a slow bulge/taper from
# reading as a neck. The first neck ends the body plateau, so the taper is fit only over the clean pre-neck
# stretch; each neck is written to the per-profile `segmentNeck` marker (the per-speed necks/m rate is pooled
# from it in `featureRates`). NOTE the trade-off (deliberate, for robustness on long prints): a broad, gentle
# neck the baseline can follow is ABSORBED (not counted).
NECK_LOCAL_BASELINE_MM = 80.0     # arc-length window of the rolling-median local baseline necks are measured against
NECK_AREA_PROMINENCE_FRAC = 0.30  # a neck's AREA residual must dip from its shoulder AND recover, each >= this * local area baseline (raise for stricter/fewer necks)
NECK_WIDTH_CONFIRM_FRAC = 0.10    # ... and its widthOuter residual must also dip+recover >= this * local width baseline (width moves less, but a real neck still narrows clearly)
NECK_WIDTH_TOL = 6                # profiles: the confirming width minimum is sought within +-this of the area-neck minimum
NECK_MARK_HALFWIDTH = 5           # profiles each side of a neck minimum flagged in segmentNeck (survives heat-map decimation)

# segmentSection phase codes (NaN elsewhere = start ramp + shoulder, unused by any feature):
_SECTION_BODY, _SECTION_RUPTURE, _SECTION_PEAK = 1.0, 2.0, 3.0

# segmentShapeStatus codes: 0 = kept (shape-analysed); 1-6 = sorted out of the shape analysis, with the reason.
_STATUS_KEPT, _STATUS_TOO_SHORT, _STATUS_CONTINUOUS = 0.0, 1.0, 2.0
_STATUS_DEGENERATE, _STATUS_TINY_BODY, _STATUS_HIGH_WIDTH, _STATUS_NO_RUPTURE = 3.0, 4.0, 5.0, 6.0

_SEGMENT_SHAPE_FIELDS = ("segmentBodyAreaThinning", "segmentBodyAreaSteadiness",
                         "segmentBodyWidthThinning", "segmentBodyWidthSteadiness",
                         "segmentBodyHeightThinning", "segmentBodyHeightSteadiness",
                         "segmentCriticalArea", "segmentCriticalWidth", "segmentRuptureLength",
                         "segmentHeadOvershoot", "segmentRuptures")

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
    """Body-start index: the first point where the smoothed signal enters and stays within
    SEGMENT_BODY_BAND of the body level for SETTLE_SPAN_MM (where it settles onto the plateau).

    Only a genuine *head overshoot* — a prominent bulge (> the band above body) within the leading
    HEAD_OVERSHOOT_FRAC of the segment — makes the search start *after* the bulge, so the overshoot isn't
    counted as body. A segment with no/little overshoot (or whose maximum is a mid-segment dome or a stray
    spike) is scanned from the start, so its body isn't pushed late by a peak that isn't a leading overshoot.
    """
    L = s[-1]
    finite = np.isfinite(A)
    band = SEGMENT_BODY_BAND * body
    early = np.flatnonzero((s <= HEAD_OVERSHOOT_FRAC * L) & finite)
    scan_from = 0
    if early.size:
        peak = int(early[np.argmax(A[early])])
        if A[peak] - body > band:            # a prominent leading bulge -> start the body after it
            scan_from = peak
    for i in range(scan_from, s.size):
        if finite[i] and abs(A[i] - body) <= band:
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

def _fill_nans(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior NaNs so peak-finding sees a continuous signal (unchanged if <2 finite)."""
    finite = np.isfinite(y)
    if finite.all() or finite.sum() < 2:
        return y
    out = y.copy()
    out[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    return out

def _dip_recovery(y: np.ndarray, a: int, m: int) -> "tuple[float, float]":
    """Depth of the dip at index m below its preceding shoulder (max over [a, m)) and its recovery afterwards
    (max over (m, end]). (0, 0) at an edge. A terminal descent (dips and stays low to the end) has ~0 recovery,
    so requiring both distinguishes a neck (thins then recovers) from the final rupture."""
    if m <= a or m >= y.size - 1:
        return 0.0, 0.0
    return float(np.max(y[a:m])) - float(y[m]), float(np.max(y[m + 1:])) - float(y[m])

def _detect_necks(s: np.ndarray, A: np.ndarray, W: np.ndarray, body_start: "int | None"
                  ) -> "list[tuple[int, int, int]]":
    """Interior necks of a run by LOCAL DE-TREND: a dip of the AREA below its local baseline that recovers,
    confirmed by a coincident WIDTH dip below its local baseline.

    The local baseline of each signal = the (already-smoothed) signal median-filtered over
    `NECK_LOCAL_BASELINE_MM` of arc-length `s`, i.e. the slowly-varying trend (it follows a bulge / taper
    without following a neck). On the AREA residual `RA = A - baseline`, a neck is an interior minimum with
    `RA < 0` (below the local trend) whose dip-from-shoulder and recovery each exceed
    `NECK_AREA_PROMINENCE_FRAC` * (local area baseline) — a % of the LOCAL cross-section — and whose nearest
    WIDTH residual minimum (within `NECK_WIDTH_TOL`) likewise dips and recovers by at least
    `NECK_WIDTH_CONFIRM_FRAC` * (local width baseline). Only minima at/after `body_start` count. onset/recovery
    = where the residual returns to its baseline (`RA >= 0`) around the dip, so a dip inside a broad bulge does
    NOT span the bulge. Returns [(onset, min, recovery), ...] in local indices. (Trade-off: a broad neck the
    baseline can follow is absorbed — see the NECKING note in the module constants.)"""
    if body_start is None or s.size < 3:
        return []
    Af, Wf = _fill_nans(A), _fill_nans(W)
    dt = float(np.median(np.diff(s)))
    if not np.isfinite(dt) or dt <= 0:
        return []
    win = max(5, round(NECK_LOCAL_BASELINE_MM / dt) | 1)  # odd window (profiles) for the local baseline
    BA, BW = _median_smooth(Af, win), _median_smooth(Wf, win)  # local baselines (slow trend)
    RA, RW = Af - BA, Wf - BW                                   # residuals (deviation from the local trend)
    ba_scale = float(np.nanmedian(BA))
    if not np.isfinite(ba_scale) or ba_scale <= 0:
        return []
    a = int(body_start)
    above = RA >= 0
    peaks, _ = find_peaks(-RA, prominence=0.5 * NECK_AREA_PROMINENCE_FRAC * ba_scale)  # candidate residual minima
    necks: list[tuple[int, int, int]] = []
    for m in peaks:
        m = int(m)
        if m <= a or m >= RA.size - 1 or RA[m] >= 0:                 # must dip BELOW the local baseline
            continue
        ad, ar = _dip_recovery(RA, a, m)
        if min(ad, ar) < NECK_AREA_PROMINENCE_FRAC * BA[m]:         # AREA must dip AND recover (local scale)
            continue
        lo = max(a + 1, m - NECK_WIDTH_TOL)
        hi = min(Wf.size - 1, m + NECK_WIDTH_TOL + 1)
        if hi <= lo:
            continue
        mw = int(lo + np.argmin(RW[lo:hi]))
        wd, wr = _dip_recovery(RW, a, mw)
        if min(wd, wr) < NECK_WIDTH_CONFIRM_FRAC * BW[mw]:          # WIDTH must confirm (lighter threshold)
            continue
        left = np.flatnonzero(above[a:m])
        right = np.flatnonzero(above[m + 1:])
        onset = int(left[-1] + a) if left.size else a
        recovery = int(right[0] + m + 1) if right.size else RA.size - 1
        necks.append((onset, m, recovery))
    return necks

def _neck_marker(necks: "list[tuple[int, int, int]]", n: int) -> np.ndarray:
    """Per-profile 0/1 marker of length n: 1.0 over each neck span (onset..recovery, widened to
    +-`NECK_MARK_HALFWIDTH` around the minimum so a thin neck survives heat-map decimation), else 0.0."""
    marker = np.zeros(n)
    for onset, m, recovery in necks:
        lo = max(0, min(onset, m - NECK_MARK_HALFWIDTH))
        hi = min(n, max(recovery, m + NECK_MARK_HALFWIDTH) + 1)
        marker[lo:hi] = 1.0
    return marker

def _run_neck_info(s: np.ndarray, A: np.ndarray, W: np.ndarray
                   ) -> "tuple[np.ndarray | None, int | None]":
    """Per-profile neck marker + first-neck onset for one run's smoothed area/width signals.

    Establishes the run's settle point (`body_start`, via the 20-80% area body level) and then calls
    `_detect_necks` (local de-trend). Used for BOTH discrete segments (whose first neck ends the body plateau)
    and continuous filaments (necks marked + shown, though they are not otherwise shape-analysed), so the
    definition is identical. Returns (marker, first_onset); (None, None) if no body / settle point exists.
    The per-speed neck *rate* is pooled from the marker later (`featureRates`), so no count is stored here."""
    L = float(s[-1])
    finite = np.isfinite(A)
    if L <= 0 or finite.sum() < 5:
        return None, None
    blo, bhi = SEGMENT_BODYLEVEL_WINDOW
    bmask = (s >= blo * L) & (s <= bhi * L) & finite
    area_body = float(np.median(A[bmask])) if bmask.any() else np.nan
    if not (np.isfinite(area_body) and area_body > 0):
        return None, None
    body_start = _settle_index(s, A, area_body)
    if body_start is None:
        return None, None
    necks = _detect_necks(s, A, W, body_start)
    onset = necks[0][0] if necks else None
    return _neck_marker(necks, s.size), onset

def _thinning_pair(sig: np.ndarray, s: np.ndarray, idx: np.ndarray, level: float) -> "tuple[float, float]":
    """(thinning rate %/mm, steadiness) of one signal over the body indices `idx`, normalised by `level`.

    Thinning = Theil-Sen slope of `sig` vs arc-length `s` over `idx`, / `level` x100 (negative = thinning).
    Steadiness = Spearman(`sig`, `s`) over the same points (-1 = steadily thinning, ~0 = wavy). Both NaN if
    fewer than 5 finite samples or `level` is non-positive. Used identically for area / width / height.
    """
    j = idx[np.isfinite(sig[idx])]
    if j.size < 5 or not np.isfinite(level) or level <= 0:
        return np.nan, np.nan
    thin = 100.0 * float(theilslopes(sig[j], s[j])[0]) / level
    r = spearmanr(s[j], sig[j])[0]
    return thin, (float(r) if np.isfinite(r) else np.nan)

def _segment_shape_values(s: np.ndarray, A: np.ndarray, W: np.ndarray, H: np.ndarray,
                          neck_onset: "int | None" = None
                          ) -> "tuple[float, float | None, dict | None, np.ndarray | None]":
    """Compute the per-segment shape features (and sort-out status) from the smoothed area signal A(s),
    outer-width signal W(s) and height signal H(s) (arc-length s in mm). `neck_onset` (from `_run_neck_info`)
    is the onset index of the segment's first neck, if any: the body plateau ends there so the taper is fit
    only over the clean pre-neck stretch (an over-short result falls into the tiny-body sort-out). Returns:

    - `status`: a `segmentShapeStatus` code (`_STATUS_*`) — 0 kept, else the reason it was sorted out of
      the shape analysis (3 degenerate, 4 tiny body, 5 high width change, 6 didn't rupture).
    - `ruptures_flag`: 1.0/0.0 (the rupture gate) when a valid body level exists, else None. Set on every
      valid-body segment (even sorted-out ones) so the rupture *rate* is preserved; None only when the
      segment is too degenerate to even establish a body level.
    - `values`: the thinning/shape features (physical units) — the area/width/height thinning + steadiness
      pairs, the overshoot and the two rupture features — only for a KEPT segment, else None.
    - `sections`: the per-profile `segmentSection` phase code array — only for a KEPT segment, else None.

    Regions (all detected on the ±SEGMENT_BODY_BAND plateau band, so the flag matches what the features
    use): the **body** = the plateau from `body_start` (where the ramp settles into the band) to
    `body_end` (last in-band point, held SEGMENT_BODY_RUPTURE_MARGIN_MM before the cliff top / segment end);
    the **rupture** = `[rupture_start, end]` (cliff); the **start** ramp `[0, body_start)` and the
    **shoulder** `[body_end, rupture_start)` are left unclassified (NaN). The overshoot **peak** (a small
    band around the highest point of the start ramp, which `segmentHeadOvershoot` reads) is flagged
    separately. `bodyLevel` is a fixed-window (20-80%) robust median scalar.
    """
    L = float(s[-1])
    finite = np.isfinite(A)
    if L <= 0 or finite.sum() < 5:
        return _STATUS_DEGENERATE, None, None, None
    blo, bhi = SEGMENT_BODYLEVEL_WINDOW
    bmask = (s >= blo * L) & (s <= bhi * L) & finite
    body = float(np.median(A[bmask])) if bmask.any() else np.nan
    if not np.isfinite(body) or body <= 0:
        return _STATUS_DEGENERATE, None, None, None
    band = SEGMENT_BODY_BAND * body

    # rupture gate + cliff top (needed before the body, since the body ends where the plateau meets the cliff)
    etail = finite & (s >= L - SEGMENT_END_SPAN_MM)
    end_level = float(np.median(A[etail])) if etail.any() else np.nan
    ruptures = bool(np.isfinite(end_level) and end_level < RUPTURE_GATE_FRAC * body)
    ruptures_flag = 1.0 if ruptures else 0.0    # the rupture gate: valid once a body level exists
    rupture_start = _rupture_start(s, A) if ruptures else None

    # body plateau = within the band, from body_start (ramp settles in) to body_end, held back at least
    # SEGMENT_BODY_RUPTURE_MARGIN_MM before the cliff top (or the segment end, if it never ruptures).
    body_start = _settle_index(s, A, body)
    body_end = None
    if body_start is not None:
        end_bound = (s[rupture_start] if rupture_start is not None else L) - SEGMENT_BODY_RUPTURE_MARGIN_MM
        upper = rupture_start if rupture_start is not None else A.size - 1
        region = np.arange(body_start, upper + 1)
        inband = region[finite[region] & (np.abs(A[region] - body) <= band) & (s[region] <= end_bound)]
        body_end = int(inband[-1]) if inband.size else body_start
        # end the body plateau at the FIRST neck (a thin-and-recover event), so the taper isn't fit through the
        # dip + recovery. neck_onset >= body_start (necks are detected after settle), so this only shortens the
        # plateau; an over-short result then falls into the tiny-body sort-out below.
        if body_end is not None and neck_onset is not None and neck_onset < body_end:
            body_end = neck_onset
    if body_start is None or body_end is None or body_end <= body_start:
        return _STATUS_DEGENERATE, ruptures_flag, None, None
    idx = np.arange(body_start, body_end + 1)
    idx = idx[finite[idx]]
    if idx.size < 5:
        return _STATUS_DEGENERATE, ruptures_flag, None, None

    # --- weird-segment sort-out (priority: tiny body -> high width change -> didn't rupture) ---
    if float(s[body_end] - s[body_start]) < MIN_BODY_LENGTH_MM:
        return _STATUS_TINY_BODY, ruptures_flag, None, None       # plateau too short -> unreliable taper
    w_body = W[idx][np.isfinite(W[idx])]
    if w_body.size >= 2:
        w_med = float(np.median(w_body))
        if w_med > 0 and (w_body.max() - w_body.min()) / w_med > WIDTH_CHANGE_MAX_FRAC:
            return _STATUS_HIGH_WIDTH, ruptures_flag, None, None  # turbulent / spreading width over the body
    if not ruptures:
        return _STATUS_NO_RUPTURE, ruptures_flag, None, None      # ended thick (truncated), not a real segment life

    # --- KEPT: compute the thinning family (area/width/height) + rupture + overshoot + the section flag ---
    out: dict[str, float] = {}
    # body thinning family: the same taper measured on area, width (widthOuter) and height (heightP95) over the
    # body plateau `idx`, each normalised by its own body level (median over the 20-80% window `bmask`).
    out["segmentBodyAreaThinning"], out["segmentBodyAreaSteadiness"] = _thinning_pair(A, s, idx, body)
    w_win = W[bmask][np.isfinite(W[bmask])]
    out["segmentBodyWidthThinning"], out["segmentBodyWidthSteadiness"] = _thinning_pair(
        W, s, idx, float(np.median(w_win)) if w_win.size else np.nan)
    h_win = H[bmask][np.isfinite(H[bmask])]
    out["segmentBodyHeightThinning"], out["segmentBodyHeightSteadiness"] = _thinning_pair(
        H, s, idx, float(np.median(h_win)) if h_win.size else np.nan)
    # overshoot: the peak of the start ramp [0, body_start) vs body
    peak = None
    if body_start > 0:
        ramp = np.arange(0, body_start)[finite[:body_start]]
        if ramp.size:
            peak = int(ramp[np.argmax(A[ramp])])
    out["segmentHeadOvershoot"] = 100.0 * (float(A[peak]) - body) / body if peak is not None else np.nan
    # rupture (a KEPT segment ruptured): critical cross-section + outer width at the cliff top + the cliff length
    if rupture_start is not None:
        out["segmentCriticalArea"] = _nearest_finite(A, rupture_start) * AREA_UNITS_TO_MM2    # mm^2
        out["segmentCriticalWidth"] = _nearest_finite(W, rupture_start) * PROFILE_UNITS_TO_MM  # mm
        out["segmentRuptureLength"] = L - float(s[rupture_start])                             # mm
    else:
        out["segmentCriticalArea"] = out["segmentCriticalWidth"] = out["segmentRuptureLength"] = np.nan

    # per-profile section flag: NaN start/shoulder; body; rupture; a peak band around the overshoot point
    sections = np.full(s.size, np.nan)
    sections[body_start:body_end + 1] = _SECTION_BODY
    if rupture_start is not None:
        sections[rupture_start:] = _SECTION_RUPTURE
    if peak is not None:  # small band so it survives heat-map decimation; clamped to the start ramp
        sections[max(0, peak - PEAK_MARK_HALFWIDTH): min(body_start, peak + PEAK_MARK_HALFWIDTH + 1)] = _SECTION_PEAK
    return _STATUS_KEPT, ruptures_flag, out, sections

def measure_segment_shape(profiles: list[profileData]) -> None:
    """Per-segment shape features (thinning / startup / rupture), broadcast onto every profile of the
    segment (like segmentVolume / segmentLength). Computed on the along-path cross-section signal
    areaShoelace (smoothed).

    Every `isSegment` run gets a `segmentShapeStatus` code (0 kept, 1-6 = sorted out of the shape analysis;
    see the module constants). A sorted-out segment keeps `segmentRuptures` (the rupture gate, so the
    rupture *rate* survives) but its thinning/shape features + `segmentSection` stay None, so it drops
    out of the thinning plots while the status records why. Only the shape analysis is gated this way —
    `segmentLength` / `segmentVolume` / `defectLength` still cover every run.

    Sets on each segment profile (None off-segment or on a sorted-out segment; NaN where a phase is
    absent):

    - `segmentBody{Area,Width,Height}Thinning` (%/mm)     thinning rate over the body plateau, measured on
                                         area (areaShoelace) / width (widthOuter) / height (heightP95); negative = thinning
    - `segmentBody{Area,Width,Height}Steadiness` (-1..1)  how steadily each thins (Spearman of the signal vs arc-length)
    - `segmentCriticalArea` (mm^2)       cross-section at the rupture cliff top (last detected before failure)
    - `segmentCriticalWidth` (mm)        outer width at the rupture cliff top (same profile as criticalArea)
    - `segmentRuptureLength` (mm)        arc-length of the terminal cliff (short = abrupt snap)
    - `segmentHeadOvershoot` (%)         the start ramp's peak, over the body level ("wider at the start")
    - `segmentRuptures` (0/1)            whether the segment ended in a rupture (set on every valid-body run)
    - `segmentSection` (1/2/3, else NaN) per-profile phase flag for the heat map: 1 body, 2 rupture,
                                         3 the overshoot-peak band; the start ramp + shoulder are NaN
    - `segmentNeck` (0/1, else None)     per-profile neck marker for the heat map: 1 over each neck span, 0
                                         elsewhere in an analysed run (discrete or continuous); None off-segment.
                                         The per-rollerband-speed neck RATE (necks/m) is pooled from it in `featureRates`
    - `segmentShapeStatus` (0-6)         per-segment sort-out reason (heat-map debug): 0 kept, 1 too short,
                                         2 continuous filament, 3 degenerate, 4 tiny body, 5 high width
                                         change, 6 didn't rupture

    The first neck ends the body plateau (so the thinning taper is fit over the clean pre-neck stretch, not
    through a dip + recovery); a segment whose plateau is then too short falls into the tiny-body sort-out.

    Needs `isSegment` (clean_flat_runs) + areaShoelace + widthOuter (measure_filament_area / width) + the
    PLC-joined spacing (profile_advance_distances). Run after clean_flat_runs / measure_run_lengths. See the
    module constants for the (empirical) window and threshold values.
    """
    for p in profiles:
        for f in _SEGMENT_SHAPE_FIELDS:
            setattr(p, f, None)
        p.segmentSection = None       # per-profile phase code (varies within a segment)
        p.segmentNeck = None          # per-profile neck marker (varies within a segment)
        p.segmentShapeStatus = None   # per-segment sort-out reason (None off-segment)

    dist = profile_advance_distances(profiles)
    area = np.array([p.areaShoelace if p.areaShoelace is not None else np.nan for p in profiles], float)
    width = np.array([p.widthOuter if p.widthOuter is not None else np.nan for p in profiles], float)
    height = np.array([p.heightP95 if p.heightP95 is not None else np.nan for p in profiles], float)

    for run in _contiguous_runs(_segment_mask(profiles)):
        s_mm = np.concatenate([[0.0], np.cumsum(dist[run][1:])]) * PROFILE_UNITS_TO_MM  # arc-length (mm)
        L_mm = float(s_mm[-1])
        neck_marker = None
        # Length gate first (a run outside the discrete-segment band is sorted out before any taper fitting).
        if L_mm < SEGMENT_SHAPE_MIN_LENGTH_MM:
            status, ruptures_flag, values, sections = _STATUS_TOO_SHORT, None, None, None
        else:
            A = _median_smooth(area[run], SEGMENT_SHAPE_SMOOTH)
            W = _median_smooth(width[run], SEGMENT_SHAPE_SMOOTH)
            # necks are marked on discrete AND continuous runs (same definition); the first neck's onset ends
            # the discrete body plateau (passed into _segment_shape_values below).
            neck_marker, neck_onset = _run_neck_info(s_mm, A, W)
            if L_mm > MAX_SEGMENT_LENGTH_MM:
                status, ruptures_flag, values, sections = _STATUS_CONTINUOUS, None, None, None
            else:
                H = _median_smooth(height[run], SEGMENT_SHAPE_SMOOTH)
                status, ruptures_flag, values, sections = _segment_shape_values(s_mm, A, W, H, neck_onset)
        for local_i, i in enumerate(run):
            profiles[i].segmentShapeStatus = float(status)         # sort-out reason (broadcast per segment)
            if ruptures_flag is not None:
                profiles[i].segmentRuptures = float(ruptures_flag)  # rupture gate: kept even when sorted out
            if neck_marker is not None:
                profiles[i].segmentNeck = float(neck_marker[local_i])  # per-profile neck marker (0/1)
            if values is not None and sections is not None:         # KEPT: broadcast the shape features + phase
                profiles[i].segmentSection = float(sections[local_i])
                for f, v in values.items():
                    setattr(profiles[i], f, float(v))
