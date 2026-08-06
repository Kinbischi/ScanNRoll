"""Interactive 2D comparison of several per-profile features over the run.

A matplotlib line overlay: each selected `profileData` feature (geometry measure or joined PLC
channel) is drawn against time, robustly normalised to 0-1 so features on very different scales
(torque ~1.5, width ~4000 units, flow ~0-1) share one axis. Left-panel checkboxes toggle each
curve's visibility ("show") and whether it is smoothed ("smooth"); a slider sets the smoothing
window. Complements the 3D filament heat-map (`profile3Dplotting.plot_feature_heatmap`), which colours
one feature at a time in space; here many features are compared as time series.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, Slider

from profile3Dplotting import FEATURE_DISPLAY  # single source of truth for unit factor + label
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
    """Overlay of several per-profile features on one robust-normalised time axis, with live
    controls: show/hide each curve, smooth selected curves (moving average), and a window slider.

    Per-feature scale is fixed from the raw values, so smoothing changes only the curve shape, not
    the axis meaning or the `scale[..]` / `true[..]` legend (which keeps flagging raw outliers).
    """

    def __init__(self, profiles: list[profileData], features: tuple[str, ...],
                 initial: tuple[str, ...] | None = None,
                 clip_percentile: tuple[float, float] = (1.0, 99.0),
                 time_unit: str = "min", profile_step: int = 1,
                 smooth_window_max: int = 151) -> None:
        if time_unit not in _SECONDS_PER:
            raise ValueError(f"time_unit must be one of {list(_SECONDS_PER)}, got {time_unit!r}")
        if not features:
            raise ValueError("no features given to compare")
        self.features = tuple(features)
        initial = self.features if initial is None else tuple(initial)

        profs = profiles[::profile_step]
        times = np.array([np.nan if p.arrivalTime is None else float(p.arrivalTime) for p in profs])
        if not np.any(np.isfinite(times)):
            raise ValueError("profiles have no arrivalTime; reprocess the cache to populate it")
        self._x = (times - np.nanmin(times)) / _SECONDS_PER[time_unit]

        # per-feature state
        self._phys: dict[str, np.ndarray] = {}   # physical values
        self._scale: dict[str, tuple[float, float]] = {}  # (lo, hi) from RAW values, fixed
        self._lines: dict[str, Line2D] = {}
        self._smoothed: dict[str, bool] = {f: False for f in self.features}
        self._window = 1

        # --- figure + axes: two checkbox columns (smooth | show) left, plot centre, slider below,
        #     legend right. The "smooth" boxes are unlabelled (headers name the columns); rows align
        #     with the labelled "show" boxes since both axes share the same y-extent and count.
        self.fig = plt.figure(figsize=(17, 8.5))
        smooth_ax = self.fig.add_axes((0.008, 0.17, 0.026, 0.71), frame_on=False)
        show_ax = self.fig.add_axes((0.038, 0.17, 0.150, 0.71), frame_on=False)
        for a in (smooth_ax, show_ax):
            a.set_xticks([]); a.set_yticks([])
        self.fig.text(0.021, 0.89, "smooth", fontsize=11, ha="center")
        self.fig.text(0.062, 0.89, "show", fontsize=11, ha="left")
        self.ax = self.fig.add_axes((0.28, 0.17, 0.35, 0.75))  # narrower -> room for the wide legend
        slider_ax = self.fig.add_axes((0.28, 0.06, 0.35, 0.025))

        # --- curves (fixed raw scale) ---
        labels = []
        for f in self.features:
            phys = _feature_values(profs, f)
            lo, hi = _scale_range(phys, clip_percentile)
            self._phys[f] = phys
            self._scale[f] = (lo, hi)
            (line,) = self.ax.plot(self._x, _normalise(phys, lo, hi), lw=1.0)
            line.set_visible(f in initial)
            self._lines[f] = line
            labels.append(self._label(f))

        p_lo, p_hi = clip_percentile
        self.ax.set_xlabel(f"time from start ({time_unit})", fontsize=12)
        self.ax.set_ylabel(f"normalised per feature (robust {p_lo:g}-{p_hi:g} pct -> 0-1)", fontsize=12)
        self.ax.set_ylim(-0.03, 1.03)
        self.ax.margins(x=0)
        self.ax.tick_params(labelsize=11)
        self.ax.grid(True, alpha=0.3)

        # --- legend: always-visible proxy handles (colours never disappear); bold text = visible ---
        handles = [Line2D([], [], color=self._lines[f].get_color(), lw=2.5) for f in self.features]
        self._legend = self.ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                                      borderaxespad=0.0, prop={"family": "monospace", "size": 9},
                                      frameon=False)
        self._legend_texts = dict(zip(self.features, self._legend.get_texts()))
        for f in self.features:
            self._style_legend(f)

        # --- widgets --- (enlarge the boxes/marks via *_props; label font set after, which is
        # robust across matplotlib versions where label_props wants list-valued props)
        box = {"s": 120}
        self._show = CheckButtons(show_ax, list(self.features), [f in initial for f in self.features],
                                  frame_props=box, check_props=box)
        self._smooth_cb = CheckButtons(smooth_ax, [""] * len(self.features), [False] * len(self.features),
                                       frame_props=box, check_props=box)
        for txt in self._show.labels:
            txt.set_fontsize(12)
        self._slider = Slider(slider_ax, "smoothing window", 1, smooth_window_max, valinit=1, valstep=1)
        self._slider.label.set_fontsize(11)
        self._show.on_clicked(self._on_show)
        self._smooth_cb.on_clicked(self._on_smooth)
        self._slider.on_changed(self._on_window)

        logger.info("Comparing %d features over %d profiles (step %d)",
                    len(self.features), len(profs), profile_step)

    def _label(self, f: str) -> str:
        """Legend label: the scale range that maps to 0-1 plus the true min-max (reveals outliers)."""
        lo, hi = self._scale[f]
        unit = FEATURE_DISPLAY[f][2]
        suffix = f" {unit}" if unit else ""
        phys = self._phys[f]
        tmin, tmax = np.nanmin(phys), np.nanmax(phys)
        return f"{f}   scale[{lo:.3g}-{hi:.3g}{suffix}]   true[{tmin:.3g}-{tmax:.3g}]"

    def _series(self, f: str) -> np.ndarray:
        """The plotted (normalised) series for a feature: smoothed if enabled, else raw."""
        vals = _smooth(self._phys[f], self._window) if (self._smoothed[f] and self._window > 1) else self._phys[f]
        lo, hi = self._scale[f]
        return _normalise(vals, lo, hi)

    def _refresh(self, f: str) -> None:
        self._lines[f].set_ydata(self._series(f))

    def _style_legend(self, f: str) -> None:
        """Bold black text for a visible curve, normal grey for a hidden one (colour swatch stays)."""
        txt = self._legend_texts[f]
        visible = self._lines[f].get_visible()
        txt.set_fontweight("bold" if visible else "normal")
        txt.set_color("black" if visible else "0.6")

    def _on_show(self, _label: str) -> None:
        """Sync curve visibility from the show checkboxes (robust to the label; empty smooth labels)."""
        for vis, f in zip(self._show.get_status(), self.features):
            if vis != self._lines[f].get_visible():
                self._lines[f].set_visible(vis)
                self._style_legend(f)
        self.fig.canvas.draw_idle()

    def _on_smooth(self, _label: str) -> None:
        """Sync per-feature smoothing from the smooth checkboxes and re-plot changed curves."""
        for on, f in zip(self._smooth_cb.get_status(), self.features):
            if on != self._smoothed[f]:
                self._smoothed[f] = on
                self._refresh(f)
        self.fig.canvas.draw_idle()

    def _on_window(self, val: float) -> None:
        """Slider moved: re-smooth every feature currently marked for smoothing."""
        self._window = int(val)
        for f in self.features:
            if self._smoothed[f]:
                self._refresh(f)
        self.fig.canvas.draw_idle()


def compare_features(profiles: list[profileData], features: tuple[str, ...],
                     initial: tuple[str, ...] | None = None,
                     clip_percentile: tuple[float, float] = (1.0, 99.0),
                     time_unit: str = "min", profile_step: int = 1,
                     smooth_window_max: int = 151, show: bool = True) -> plt.Figure:
    """Overlay several per-profile features on one robust-normalised time axis, with live controls.

    Builds a `FeatureComparisonPlot`: each feature in `features` is a curve, scaled to 0-1 over its
    `clip_percentile` range (outliers clipped; `(0, 100)` = exact min-max). Left-panel checkboxes
    toggle each curve's visibility ("show") and smoothing ("smooth"); the slider sets the smoothing
    window (samples, 1 = raw). Each legend entry shows both `scale[lo-hi unit]` (what maps to 0-1)
    and `true[min-max]` (real extremes, so a clipped outlier shows as true max >> scale hi), and its
    text is bold while the curve is shown. x-axis is time from the first profile (`arrivalTime`).

    Args:
        profiles: processed profiles (need `arrivalTime`).
        features: features to build curves + checkboxes for.
        initial: which features start visible (default: all).
        clip_percentile: (low, high) percentiles mapped to 0-1.
        time_unit: "min" or "s" for the x-axis.
        profile_step: draw every Nth profile (decimation for responsiveness).
        smooth_window_max: slider's maximum smoothing window.
        show: call `plt.show()` before returning (set False for headless use).

    Returns the Figure. Note: the widgets need a GUI backend (run as a script or `%matplotlib qt`);
    VS Code's inline backend renders a static image (as with the PyVista `notebook=False` note).
    """
    fc = FeatureComparisonPlot(profiles, features, initial=initial, clip_percentile=clip_percentile,
                               time_unit=time_unit, profile_step=profile_step,
                               smooth_window_max=smooth_window_max)
    fc.fig._feature_comparison = fc  # keep the instance (and its widgets) alive
    if show:
        plt.show()
    return fc.fig
