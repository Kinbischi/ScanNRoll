"""Interactive 2D comparison of several per-profile features over the run.

A matplotlib line plot: each selected `profileData` feature (geometry measure or joined PLC channel)
is drawn against time. With two or more shown they are robustly normalised to 0-1 so features on very
different scales (torque ~1.5, width ~4000 units, flow ~0-1) share one axis; with exactly one shown it
is drawn in its **real units** on a self-scaled axis. A left panel grouped by category (Geometry /
Segment / PLC) toggles each curve's visibility ("show", tinted to match its curve) and whether it is
smoothed ("smooth"); a slider sets the smoothing window. Complements the 3D filament heat-map
(`profile3Dplotting.plot_feature_heatmap`), which colours one feature at a time in space; here many
features are compared as time series.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, Slider

from profile3Dplotting import FEATURE_DISPLAY, group_by_category  # unit factor/label + shared grouping
from profilePointsClass import profileData

logger = logging.getLogger(__name__)

_SECONDS_PER = {"s": 1.0, "min": 60.0}  # time_unit -> seconds per unit


def _feature_values(profiles: list[profileData], feature: str) -> np.ndarray:
    """Per-profile feature values in physical units (None -> NaN so flat-profile gaps break lines)."""
    factor = FEATURE_DISPLAY[feature][1]
    raw = [getattr(p, feature, None) for p in profiles]
    return np.array([np.nan if v is None else float(v) * factor for v in raw], dtype=float)


def _scale_range(vals: np.ndarray, clip_percentile: tuple[float, float]) -> tuple[float, float]:
    """The (lo, hi) percentile range that maps to [0, 1]; (nan, nan) if too few finite points."""
    if np.count_nonzero(np.isfinite(vals)) < 2:
        return np.nan, np.nan
    lo, hi = np.nanpercentile(vals, clip_percentile)
    return float(lo), float(hi)


def _normalise(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale to [0, 1] over [lo, hi] (outliers clipped); NaNs preserved; constant -> flat 0.5."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.full_like(vals, np.nan)
    if hi <= lo:  # constant / degenerate feature: draw it flat rather than divide by ~0
        return np.where(np.isfinite(vals), 0.5, np.nan)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


def _smooth(vals: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centred moving average (window in samples); window <= 1 returns vals unchanged.

    Normalised convolution: averages only the finite neighbours in each window, so NaN gaps (flat-
    profile features) don't spread and the ends aren't attenuated. Original-NaN positions stay NaN.
    Unlike `profileProcessingAlgorithms.moving_average` this is NaN-aware and edge-correct.
    """
    if window <= 1:
        return vals
    finite = np.isfinite(vals)
    kernel = np.ones(int(window))
    num = np.convolve(np.where(finite, vals, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.full_like(vals, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    out[~finite] = np.nan
    return out


class FeatureComparisonPlot:
    """Several per-profile features/PLC channels on a shared time axis (real units for a lone visible
    curve, else a robust-normalised 0-1 overlay), with live controls: show/hide each curve, smooth
    selected curves (moving average), and a window slider.

    Per-feature scale is fixed from the raw values, so smoothing changes only the curve shape, not
    the axis meaning or the `scale[..]` / `true[..]` legend (which keeps flagging raw outliers).
    """

    def __init__(self, profiles: list[profileData], features: tuple[str, ...],
                 initial: tuple[str, ...] | None = None,
                 initial_smooth: tuple[str, ...] | None = None,
                 clip_percentile: tuple[float, float] = (1.0, 99.0),
                 time_unit: str = "min", profile_step: int = 1,
                 smooth_window_max: int = 151, smooth_window_init: int = 1) -> None:
        if time_unit not in _SECONDS_PER:
            raise ValueError(f"time_unit must be one of {list(_SECONDS_PER)}, got {time_unit!r}")
        if not features:
            raise ValueError("no features given to compare")
        self.features = tuple(features)
        initial = self.features if initial is None else tuple(initial)
        initial_smooth = () if initial_smooth is None else tuple(initial_smooth)

        profs = profiles[::profile_step]
        times = np.array([np.nan if p.arrivalTime is None else float(p.arrivalTime) for p in profs])
        if not np.any(np.isfinite(times)):
            raise ValueError("profiles have no arrivalTime; reprocess the cache to populate it")
        self._x = (times - np.nanmin(times)) / _SECONDS_PER[time_unit]

        # per-feature state (smoothing state + window set BEFORE the curve loop so a feature can open
        # pre-smoothed — the curves are drawn through _series, which honours _smoothed/_window)
        self._phys: dict[str, np.ndarray] = {}   # physical values
        self._scale: dict[str, tuple[float, float]] = {}  # (lo, hi) from RAW values, fixed
        self._lines: dict[str, Line2D] = {}
        self._smoothed: dict[str, bool] = {f: f in initial_smooth for f in self.features}
        self._window = max(1, int(smooth_window_init))
        self._clip = clip_percentile  # for the normalised-overlay y-label

        # --- figure + axes: grouped checkbox panel (smooth | show columns) on the left, plot centre,
        #     slider below, legend right. The panel is inset from the left edge (no more clipping) and
        #     built after the curves so each show-box can be tinted with its curve's colour.
        self.fig = plt.figure(figsize=(17, 8.5))
        self.ax = self.fig.add_axes((0.30, 0.17, 0.35, 0.75))  # centre; room for panel left + wide legend right
        slider_ax = self.fig.add_axes((0.30, 0.06, 0.35, 0.025))

        # --- curves (fixed raw scale; drawn via _series so pre-smoothed features open smoothed) ---
        labels = []
        for f in self.features:
            phys = _feature_values(profs, f)
            self._phys[f] = phys
            self._scale[f] = _scale_range(phys, clip_percentile)
            (line,) = self.ax.plot(self._x, self._series(f), lw=2.0)
            line.set_visible(f in initial)
            self._lines[f] = line
            labels.append(self._label(f))

        self.ax.set_xlabel(f"time from start ({time_unit})", fontsize=12)
        self.ax.margins(x=0)
        self.ax.tick_params(labelsize=11)
        self.ax.grid(True, alpha=0.3)
        # the y-axis (label + limits) is set by _apply_display: real units for a lone visible curve,
        # else the shared normalised 0-1 overlay.

        # --- legend: always-visible proxy handles (colours never disappear); bold text = visible ---
        handles = [Line2D([], [], color=self._lines[f].get_color(), lw=2.5) for f in self.features]
        self._legend = self.ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                                      borderaxespad=0.0, prop={"family": "monospace", "size": 9},
                                      frameon=False)
        self._legend_texts = dict(zip(self.features, self._legend.get_texts()))
        for f in self.features:
            self._style_legend(f)

        # --- grouped selector panel (per-category smooth/show columns) + smoothing slider ---
        self._build_selector_panel(group_by_category(self.features), initial, initial_smooth)
        self._slider = Slider(slider_ax, "smoothing window", 1, smooth_window_max,
                              valinit=self._window, valstep=1)
        self._slider.label.set_fontsize(11)
        self._slider.on_changed(self._on_window)

        self._apply_display()  # set the initial y-axis (real units vs normalised) for the initial selection
        logger.info("Comparing %d features over %d profiles (step %d)",
                    len(self.features), len(profs), profile_step)

    def _build_selector_panel(self, groups: list[tuple[str, list[str]]],
                              initial: tuple[str, ...], initial_smooth: tuple[str, ...],
                              box_size: int = 90) -> None:
        """Draw the left selector: a 'smooth' + 'show' CheckButtons pair per category, stacked top-to-
        bottom under bold category headers. Show-boxes are tinted with each curve's colour so the panel
        maps visually to the plot; smooth-boxes stay neutral. Each group's axes height is proportional to
        its feature count, giving a uniform row height across groups so the two columns stay row-aligned."""
        smooth_x, smooth_w = 0.028, 0.030   # left inset (was 0.008 -> clipped); column x + width (fig frac)
        show_x, show_w = 0.080, 0.150       # gap widened so the two column headers no longer collide
        y_top, y_bot = 0.88, 0.10
        h_head, gap = 0.028, 0.020          # per-category header height and inter-group gap

        total_rows = sum(len(feats) for _, feats in groups)
        n_groups = len(groups)
        avail = (y_top - y_bot) - n_groups * h_head - max(0, n_groups - 1) * gap
        row_h = avail / total_rows if total_rows else avail

        # column headers, one each over the smooth / show box columns (well separated so they don't merge)
        self.fig.text(smooth_x + 0.006, y_top + 0.028, "smooth", fontsize=10, ha="center", color="0.35")
        self.fig.text(show_x + 0.022, y_top + 0.028, "show", fontsize=10, ha="center", color="0.35")

        self._show_groups: list[tuple[CheckButtons, list[str]]] = []
        self._smooth_groups: list[tuple[CheckButtons, list[str]]] = []
        top = y_top
        for name, feats in groups:
            n = len(feats)
            self.fig.text(smooth_x, top - h_head / 2, name, fontsize=11, fontweight="bold",
                          va="center", ha="left", color="0.2")
            body_bot = top - h_head - n * row_h
            body_h = n * row_h
            smooth_panel = self.fig.add_axes((smooth_x, body_bot, smooth_w, body_h), frame_on=False)
            show_panel = self.fig.add_axes((show_x, body_bot, show_w, body_h), frame_on=False)
            for a in (smooth_panel, show_panel):
                a.set_xticks([]); a.set_yticks([])
            colours = [self._lines[f].get_color() for f in feats]
            show_cb = CheckButtons(show_panel, list(feats), [f in initial for f in feats],
                                   frame_props={"s": box_size},
                                   check_props={"s": box_size, "facecolor": colours})
            smooth_cb = CheckButtons(smooth_panel, [""] * n, [f in initial_smooth for f in feats],
                                     frame_props={"s": box_size}, check_props={"s": box_size})
            for txt in show_cb.labels:
                txt.set_fontsize(11)
            show_cb.on_clicked(self._on_show)
            smooth_cb.on_clicked(self._on_smooth)
            self._show_groups.append((show_cb, feats))
            self._smooth_groups.append((smooth_cb, feats))
            top = body_bot - gap

    def _label(self, f: str) -> str:
        """Legend label: the scale range that maps to 0-1 plus the true min-max (reveals outliers)."""
        lo, hi = self._scale[f]
        unit = FEATURE_DISPLAY[f][2]
        suffix = f" {unit}" if unit else ""
        phys = self._phys[f]
        tmin, tmax = np.nanmin(phys), np.nanmax(phys)
        return f"{f}   scale[{lo:.3g}-{hi:.3g}{suffix}]   true[{tmin:.3g}-{tmax:.3g}]"

    def _phys_series(self, f: str) -> np.ndarray:
        """Physical values for f (real units), smoothed if its smooth toggle is on; no normalisation."""
        return (_smooth(self._phys[f], self._window)
                if (self._smoothed[f] and self._window > 1) else self._phys[f])

    def _series(self, f: str) -> np.ndarray:
        """The plotted normalised (0-1) series: the physical series scaled by the feature's fixed range."""
        lo, hi = self._scale[f]
        return _normalise(self._phys_series(f), lo, hi)

    def _apply_display(self) -> None:
        """Set every curve's data + the y-axis for the current selection. A lone visible curve is drawn in
        its own real units (self-scaled y-axis, labelled with its unit); two or more share the normalised
        0-1 overlay. Called on every show / smooth / window change."""
        visible = [f for f in self.features if self._lines[f].get_visible()]
        if len(visible) == 1:  # single curve -> real units on its own scale
            f = visible[0]
            yv = self._phys_series(f)
            self._lines[f].set_ydata(yv)
            unit = FEATURE_DISPLAY[f][2]
            self.ax.set_ylabel(f"{f} [{unit}]" if unit else f, fontsize=12)
            finite = yv[np.isfinite(yv)]
            if finite.size:
                lo, hi = float(np.min(finite)), float(np.max(finite))
                pad = 0.04 * (hi - lo) if hi > lo else (abs(hi) * 0.04 or 1.0)
                self.ax.set_ylim(lo - pad, hi + pad)
        else:  # 0 or 2+ curves -> shared normalised overlay
            for f in self.features:
                self._lines[f].set_ydata(self._series(f))
            p_lo, p_hi = self._clip
            self.ax.set_ylabel(f"normalised per feature (robust {p_lo:g}-{p_hi:g} pct -> 0-1)", fontsize=12)
            self.ax.set_ylim(-0.03, 1.03)

    def _style_legend(self, f: str) -> None:
        """Bold black text for a visible curve, normal grey for a hidden one (colour swatch stays)."""
        txt = self._legend_texts[f]
        visible = self._lines[f].get_visible()
        txt.set_fontweight("bold" if visible else "normal")
        txt.set_color("black" if visible else "0.6")

    def _on_show(self, _label: str) -> None:
        """Sync curve visibility from every group's show checkboxes, then re-apply the y-axis (a lone
        visible curve switches to real units, several back to the normalised overlay)."""
        for show_cb, feats in self._show_groups:
            for vis, f in zip(show_cb.get_status(), feats):
                if vis != self._lines[f].get_visible():
                    self._lines[f].set_visible(vis)
                    self._style_legend(f)
        self._apply_display()
        self.fig.canvas.draw_idle()

    def _on_smooth(self, _label: str) -> None:
        """Sync per-feature smoothing from every group's smooth checkboxes and re-plot."""
        for smooth_cb, feats in self._smooth_groups:
            for on, f in zip(smooth_cb.get_status(), feats):
                if on != self._smoothed[f]:
                    self._smoothed[f] = on
        self._apply_display()
        self.fig.canvas.draw_idle()

    def _on_window(self, val: float) -> None:
        """Slider moved: re-smooth (and re-scale, in the single-curve view) the plotted curves."""
        self._window = int(val)
        self._apply_display()
        self.fig.canvas.draw_idle()


def compare_features(profiles: list[profileData], features: tuple[str, ...],
                     initial: tuple[str, ...] | None = None,
                     initial_smooth: tuple[str, ...] | None = None,
                     clip_percentile: tuple[float, float] = (1.0, 99.0),
                     time_unit: str = "min", profile_step: int = 1,
                     smooth_window_max: int = 151, smooth_window_init: int = 1,
                     show: bool = True) -> plt.Figure:
    """Compare several per-profile features / PLC channels on a shared time axis, with live controls.

    Builds a `FeatureComparisonPlot`: with two or more curves shown each is scaled to 0-1 over its
    `clip_percentile` range (outliers clipped; `(0, 100)` = exact min-max); with exactly one shown it is
    drawn in its real units on a self-scaled y-axis. The left panel groups the
    features by category (Geometry / Segment / PLC / Other) with, per group, a "show" checkbox column
    (tinted to match each curve) and a neutral "smooth" column; the slider sets the smoothing window
    (samples, 1 = raw). Each legend entry shows both `scale[lo-hi unit]` (what maps to 0-1) and
    `true[min-max]` (real extremes, so a clipped outlier shows as true max >> scale hi), and its text is
    bold while the curve is shown. x-axis is time from the first profile (`arrivalTime`).

    Args:
        profiles: processed profiles (need `arrivalTime`).
        features: features to build curves + checkboxes for.
        initial: which features start visible (default: all).
        initial_smooth: which features start smoothed (their smooth box pre-checked); needs
            `smooth_window_init > 1` to visibly smooth. Default none.
        clip_percentile: (low, high) percentiles mapped to 0-1.
        time_unit: "min" or "s" for the x-axis.
        profile_step: draw every Nth profile (decimation for responsiveness).
        smooth_window_max: slider's maximum smoothing window.
        smooth_window_init: initial slider position / smoothing window (samples; 1 = raw).
        show: call `plt.show()` before returning (set False for headless use).

    Returns the Figure. Note: the widgets need a GUI backend (run as a script or `%matplotlib qt`);
    VS Code's inline backend renders a static image (as with the PyVista `notebook=False` note).
    """
    fc = FeatureComparisonPlot(profiles, features, initial=initial, initial_smooth=initial_smooth,
                               clip_percentile=clip_percentile, time_unit=time_unit,
                               profile_step=profile_step, smooth_window_max=smooth_window_max,
                               smooth_window_init=smooth_window_init)
    fc.fig._feature_comparison = fc  # keep the instance (and its widgets) alive
    if show:
        plt.show()
    return fc.fig
