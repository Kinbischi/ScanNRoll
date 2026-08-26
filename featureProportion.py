"""Interactive per-segment proportionality plot: is one composed quantity proportional to another?

One datapoint per filament **segment** (a run of `isSegment` profiles, collapsed to a single scalar per
subject like the stepwise run features). Each axis is a **product of up to two `feature ^ power` terms**,
built live from radios, so you can test relationships such as

    rollerbandSpeed * segmentCriticalWidth   ∝   segmentCriticalArea ** 2      (the default config)

and read off whether they line up. Two readouts, both always shown:
- a **through-origin fit** `y = k·x` with its R² and Pearson r — the direct "is P ∝ Q" check;
- a **log-log fitted exponent** `m` (slope of log y vs log x) — the *measured* power between the two axes,
  so "to the power of two" is measured, not assumed. A linear/log-log axes toggle draws the matching line.

Per-segment values: a broadcast segment feature (segmentCriticalArea/…/segmentVolume) is constant over the
run, so its per-segment value is just that constant; a per-profile subject (widthOuter, rollerbandSpeed, a
PLC channel) is aggregated as the **nan-median over the segment's non-idle profiles**. A segment is dropped
if any active term is non-finite for it (so the rupture/critical tests fall to the ruptured segments only)
or if it is longer than `MAX_RUN_LENGTH_MM` (a continuous filament, not a discrete segment).

Complements `featureComparison` (feature-vs-time) and `featurePlcTrends` (feature-vs-PLC): those are
per-profile; this one is per-segment and multiplicative.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons

from featureComparison import _feature_values  # shared per-profile physical-value helper
from profile3Dplotting import FEATURE_DISPLAY
from profileProcessingAlgorithms import _contiguous_runs, _segment_mask
from profilePointsClass import profileData

logger = logging.getLogger(__name__)

IDLE_SPEED = 0.0            # rollerbandSpeed value meaning "machine stopped" — excluded from the medians
MAX_RUN_LENGTH_MM = 500.0  # drop runs longer than 0.5 m (a continuous filament, not a discrete segment)
DISABLED = "—"             # term-2 feature entry that switches the second product term off
_POWERS: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0, 2.0, 3.0)  # exponent choices offered per term
_AXES_MODES = ("linear", "log-log")  # scatter axes scale (log-log draws the fitted power line)

# The subjects offered on each term (curated pool; pass `subjects=` to override). Segment-broadcast shape /
# run features first, then the per-profile geometry + PLC channels (median-aggregated per segment). All must
# be in FEATURE_DISPLAY (for the unit factor). Order sets the radio order.
PROPORTION_SUBJECTS: tuple[str, ...] = (
    "segmentCriticalArea", "segmentCriticalWidth", "segmentRuptureLength", "segmentHeadOvershoot",
    "segmentVolume", "segmentLength",
    "widthOuter", "widthFlank", "heightP95", "areaShoelace",
    "rollerbandSpeed", "flowVelocity", "conveyorExtrusionVelocityDifference",
    "pressurePrintHead", "printHeadTorque",
)
# Default axis terms = the user's hypothesis: x = segmentCriticalArea²,  y = rollerbandSpeed × segmentCriticalWidth.
DEFAULT_X_TERMS: tuple[tuple[str, float], ...] = (("segmentCriticalArea", 2.0),)
DEFAULT_Y_TERMS: tuple[tuple[str, float], ...] = (("rollerbandSpeed", 1.0), ("segmentCriticalWidth", 1.0))


def _expr_label(terms: list[list]) -> str:
    """A human-readable expression string for an axis, e.g. 'rollerbandSpeed · segmentCriticalWidth' or
    'segmentCriticalArea^2'; a disabled (None-feature) term is skipped."""
    parts = []
    for feat, power in terms:
        if feat is None:
            continue
        parts.append(feat if power == 1.0 else f"{feat}^{power:g}")
    return " · ".join(parts) if parts else "—"


def _through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope k of y = k·x (through the origin) and its uncentered R² (1 - Σ(y-kx)²/Σy²)."""
    sxx = float(np.sum(x * x))
    sst = float(np.sum(y * y))
    if sxx == 0.0 or sst == 0.0:
        return np.nan, np.nan
    k = float(np.sum(x * y)) / sxx
    r2 = 1.0 - float(np.sum((y - k * x) ** 2)) / sst
    return k, r2


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit log y = m·log x + c over the strictly-positive pairs; returns (m, c, r) — the empirical exponent
    between the two axes, the intercept, and the Pearson r on the logs. NaNs if fewer than 3 positive pairs."""
    m = (x > 0) & (y > 0)
    if int(np.count_nonzero(m)) < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(x[m]), np.log(y[m])
    slope, intercept = np.polyfit(lx, ly, 1)
    r = float(np.corrcoef(lx, ly)[0, 1])
    return float(slope), float(intercept), r


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of x and y (NaN if fewer than 3 points or either side constant)."""
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


class FeatureProportionPlot:
    """Per-segment scatter of two composed expressions, each a product of up to two `feature ^ power` terms,
    with a through-origin fit and a log-log fitted exponent. Four radio columns (x·term1, x·term2, y·term1,
    y·term2) build the axes live; an axes-scale radio switches linear ↔ log-log."""

    def __init__(self, profiles: list[profileData], x_terms: tuple[tuple[str, float], ...] = DEFAULT_X_TERMS,
                 y_terms: tuple[tuple[str, float], ...] = DEFAULT_Y_TERMS,
                 subjects: tuple[str, ...] = PROPORTION_SUBJECTS, idle_speed: float = IDLE_SPEED,
                 max_run_length_mm: float = MAX_RUN_LENGTH_MM) -> None:
        unknown = [s for s in subjects if s not in FEATURE_DISPLAY]
        if unknown:
            raise ValueError(f"unknown subjects (not in FEATURE_DISPLAY): {unknown}")
        for name, terms in (("x_terms", x_terms), ("y_terms", y_terms)):
            for feat, _ in terms:
                if feat not in subjects:
                    raise ValueError(f"{name} feature {feat!r} not in subjects")
        self._subjects = tuple(subjects)
        self._axes_mode = "linear"
        # two mutable [feature, power] slots per axis (term 2 defaults to disabled if only one term given)
        self._x_terms = self._init_terms(x_terms)
        self._y_terms = self._init_terms(y_terms)

        self._build_segment_table(profiles, idle_speed, max_run_length_mm)
        logger.info("Feature proportionality: %d segments, %d subjects", self._n, len(self._subjects))
        self._build_figure()
        self._render()

    @staticmethod
    def _init_terms(terms: tuple[tuple[str, float], ...]) -> list[list]:
        """Two [feature, power] slots; a missing second term is the disabled (None-feature) slot."""
        slots = [[f, float(p)] for f, p in terms][:2]
        while len(slots) < 2:
            slots.append([None, 1.0])
        return slots

    def _build_segment_table(self, profiles: list[profileData], idle_speed: float,
                             max_run_length_mm: float) -> None:
        """One scalar per subject per segment: nan-median over the segment's non-idle profiles (a broadcast
        feature is constant, so its median = the value). Also stores the per-segment rollerbandSpeed (colour)
        and drops runs longer than `max_run_length_mm`."""
        seg = _segment_mask(profiles)
        idle = np.array([p.rollerbandSpeed == idle_speed for p in profiles])
        phys = {s: _feature_values(profiles, s) for s in self._subjects}
        speed = _feature_values(profiles, "rollerbandSpeed")
        length_factor = FEATURE_DISPLAY["segmentLength"][1]
        table: dict[str, list[float]] = {s: [] for s in self._subjects}
        speed_col: list[float] = []
        for idx in _contiguous_runs(seg):
            head = profiles[int(idx[0])]
            if head.segmentLength is not None and float(head.segmentLength) * length_factor > max_run_length_mm:
                continue  # continuous filament / over-long run — not a discrete segment
            use = idx[~idle[idx]]
            if use.size == 0:
                use = idx
            for s in self._subjects:
                v = phys[s][use]
                table[s].append(float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan)
            sv = speed[use]
            speed_col.append(float(np.nanmedian(sv)) if np.isfinite(sv).any() else np.nan)
        self._table = {s: np.array(v, dtype=float) for s, v in table.items()}
        self._speed = np.array(speed_col, dtype=float)
        self._n = len(speed_col)

    def _axis_values(self, terms: list[list]) -> np.ndarray:
        """Product of the enabled `feature ^ power` terms over all segments (NaN if the axis has no term)."""
        out = np.ones(self._n)
        used = False
        for feat, power in terms:
            if feat is None:
                continue
            used = True
            out = out * np.power(self._table[feat], power)
        return out if used else np.full(self._n, np.nan)

    # --- figure / widgets -----------------------------------------------------------------------
    def _build_figure(self) -> None:
        """Scatter on top; four term-builder radio columns (x·1, x·2, y·1, y·2) along the bottom, plus a
        linear/log-log axes-scale radio."""
        self.fig = plt.figure(figsize=(15, 9.5))
        self.ax = self.fig.add_axes((0.07, 0.46, 0.64, 0.48))
        self.cax = self.fig.add_axes((0.735, 0.46, 0.014, 0.48))  # rollerbandSpeed colour bar

        # axes-scale (linear / log-log) radio, upper right (clear of the colour-bar label)
        self._scale_ax = self.fig.add_axes((0.85, 0.70, 0.13, 0.16), frame_on=True)
        self._scale_ax.set_title("axes", fontsize=11)
        self._scale_radio = RadioButtons(self._scale_ax, list(_AXES_MODES), active=0)
        self._scale_radio.on_clicked(self._on_scale)

        # four control columns along the bottom: (title, terms-list, term index)
        cols = (("x · term 1", self._x_terms, 0), ("x · term 2", self._x_terms, 1),
                ("y · term 1", self._y_terms, 0), ("y · term 2", self._y_terms, 1))
        col_w = 0.20
        col_x = (0.025, 0.265, 0.505, 0.745)
        self._feature_radios = []
        self._power_radios = []
        for (title, terms, i), x0 in zip(cols, col_x):
            allow_off = (i == 1)  # only the second term can be switched off
            labels = ([DISABLED] if allow_off else []) + list(self._subjects)
            cur_feat = terms[i][0] if terms[i][0] is not None else DISABLED
            self.fig.text(x0, 0.375, title, fontsize=11, fontweight="bold", ha="left")
            f_ax = self.fig.add_axes((x0, 0.055, col_w, 0.31), frame_on=True)
            f_radio = RadioButtons(f_ax, labels, active=labels.index(cur_feat))
            for t in f_radio.labels:
                t.set_fontsize(8)
            f_radio.on_clicked(self._make_feature_cb(terms, i))
            self._feature_radios.append(f_radio)
            # power radio, small, to the right-top of the column
            p_labels = [f"{p:g}" for p in _POWERS]
            p_ax = self.fig.add_axes((x0 + col_w + 0.006, 0.055, 0.03, 0.31), frame_on=True)
            p_ax.set_title("pow", fontsize=8)
            p_radio = RadioButtons(p_ax, p_labels, active=_POWERS.index(terms[i][1]))
            for t in p_radio.labels:
                t.set_fontsize(8)
            p_radio.on_clicked(self._make_power_cb(terms, i))
            self._power_radios.append(p_radio)

    def _make_feature_cb(self, terms: list[list], i: int):
        def cb(label: str) -> None:
            terms[i][0] = None if label == DISABLED else label
            self._render()
        return cb

    def _make_power_cb(self, terms: list[list], i: int):
        def cb(label: str) -> None:
            terms[i][1] = float(label)
            self._render()
        return cb

    def _on_scale(self, label: str) -> None:
        self._axes_mode = label
        self._render()

    # --- rendering ------------------------------------------------------------------------------
    def _render(self) -> None:
        """Recompute the two axes, scatter one point per segment, draw the fit line and annotate the
        through-origin (k, R², r) and log-log (exponent m) readouts."""
        self.ax.clear()
        self.cax.clear()
        x_all, y_all = self._axis_values(self._x_terms), self._axis_values(self._y_terms)
        log_mode = self._axes_mode == "log-log"
        m = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(self._speed)
        if log_mode:
            m &= (x_all > 0) & (y_all > 0)
        x, y, c = x_all[m], y_all[m], self._speed[m]
        xlabel, ylabel = _expr_label(self._x_terms), _expr_label(self._y_terms)

        if x.size < 2:
            self.ax.text(0.5, 0.5, "no segments with both axes finite", ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=13, color="0.4")
            self.cax.set_visible(False)
        else:
            sc = self.ax.scatter(x, y, c=c, cmap="viridis", s=45, edgecolor="0.25", linewidth=0.4)
            self.cax.set_visible(True)
            self.fig.colorbar(sc, cax=self.cax, label="rollerbandSpeed")
            self._draw_fit_and_readout(x, y, log_mode)

        if log_mode:
            self.ax.set_xscale("log"); self.ax.set_yscale("log")
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(f"{ylabel}   vs   {xlabel}   ({self._n} segments)", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()

    def _draw_fit_and_readout(self, x: np.ndarray, y: np.ndarray, log_mode: bool) -> None:
        """Draw the fit line for the current axes mode and annotate both fits."""
        k, r2 = _through_origin(x, y)
        r = _pearson(x, y)
        slope, intercept, rll = _loglog_fit(x, y)
        xs = np.linspace(max(x.min(), 1e-12) if log_mode else 0.0, x.max(), 200)
        if log_mode and np.isfinite(slope):  # power-law line y = e^c · x^m
            self.ax.plot(xs, np.exp(intercept) * xs ** slope, color="crimson", lw=2.0,
                         label=f"log-log fit  m={slope:+.2f}")
        elif not log_mode and np.isfinite(k):  # proportional line y = k·x
            self.ax.plot(xs, k * xs, color="crimson", lw=2.0, label=f"y = k·x  k={k:.3g}")
        self.ax.legend(loc="lower right", fontsize=9)
        lines = [f"n = {x.size}",
                 f"y = k·x:  k={k:.3g}   R²={r2:+.2f}" if np.isfinite(k) else "y = k·x:  n/a",
                 f"Pearson r = {r:+.2f}" if np.isfinite(r) else "Pearson r = n/a",
                 f"log-log slope m = {slope:+.2f}  (r={rll:+.2f})" if np.isfinite(slope) else "log-log: n/a"]
        self.ax.text(0.02, 0.98, "\n".join(lines), transform=self.ax.transAxes, va="top", ha="left",
                     fontsize=11, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"})


def plot_feature_proportion(profiles: list[profileData],
                            x_terms: tuple[tuple[str, float], ...] = DEFAULT_X_TERMS,
                            y_terms: tuple[tuple[str, float], ...] = DEFAULT_Y_TERMS,
                            subjects: tuple[str, ...] = PROPORTION_SUBJECTS, idle_speed: float = IDLE_SPEED,
                            max_run_length_mm: float = MAX_RUN_LENGTH_MM, show: bool = True) -> plt.Figure:
    """Per-segment proportionality scatter: is one composed quantity proportional to another?

    One point per filament segment. Each axis is a product of up to two `feature ^ power` terms, built live
    from radios (x·term1/2 and y·term1/2 columns), so you can test relationships like
    `rollerbandSpeed · segmentCriticalWidth ∝ segmentCriticalArea²` (the default). Two readouts: a
    through-origin fit `y = k·x` (k, R², Pearson r) and the log-log fitted exponent m (the measured power
    between the axes); a linear/log-log axes toggle draws the matching line.

    Args:
        profiles: processed, PLC-joined profiles (need `isSegment`, the segment-shape features + the
            aggregated subjects, and `rollerbandSpeed`).
        x_terms, y_terms: initial axis terms as `((feature, power), ...)` (1 or 2 per axis).
        subjects: the pool offered on each term radio (all must be in `FEATURE_DISPLAY`).
        idle_speed: `rollerbandSpeed` value treated as idle/stopped and excluded from the per-segment medians.
        max_run_length_mm: runs longer than this (continuous filaments) are dropped.
        show: call `plt.show()` before returning (set False for headless use).

    Returns the Figure. Note: the radios need a GUI backend (run as a script or `%matplotlib qt`);
    VS Code's inline backend renders a static image.
    """
    fp = FeatureProportionPlot(profiles, x_terms=x_terms, y_terms=y_terms, subjects=subjects,
                               idle_speed=idle_speed, max_run_length_mm=max_run_length_mm)
    fp.fig._feature_proportion = fp  # keep the instance (and its widgets) alive
    if show:
        plt.show()
    return fp.fig
