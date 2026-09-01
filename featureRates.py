"""Interactive 2D bar plot of POOLED per-rollerband-speed RATES.

Unlike the feature-vs-PLC *stepwise* plot — which shows a DISTRIBUTION of per-segment values per speed level
(box/violin) and drops runs > 0.5 m + straddlers — this plot shows a single POOLED RATE per speed: a ratio of
sums over ALL profiles/runs at that speed, with no length or straddler filter. That is the right object for
sparse events (necks) and for continuous filaments: e.g. the low-speed necks live in the long continuous
filaments the stepwise plot filters out, so only a pooled rate surfaces them. It is also distance-normalised
(per metre), so a 4 m filament and a 0.2 m segment are directly comparable regardless of how much was printed.

Each metric is a small function in `RATE_METRICS` mapping the shared `_RateContext` to `{speed: (value, n)}`;
add an entry to extend the plot (a reusable "channel" for any per-speed rate/fraction). Shipped metrics:
- `neckRate`         necks per metre of filament (pooled from the per-profile `segmentNeck` marker)
- `breakRate`        bead breaks (defect gaps) per metre of print
- `ruptureFraction`  fraction of segments that ended in a rupture

Complements `featurePlcTrends` (per-segment distributions), `featureComparison` (feature-vs-time) and the 3D
heat map. Needs the processed cache; `neckRate` reads the `segmentNeck` marker (a reprocess field) and shows 0
until the cache is reprocessed, while `breakRate` / `ruptureFraction` use fields already in the cache.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons

from profilePointsClass import profileData
from profileProcessingAlgorithms import (
    PROFILE_UNITS_TO_MM, _contiguous_runs, _segment_mask, profile_advance_distances,
)

logger = logging.getLogger(__name__)

IDLE_SPEED = 0.0          # rollerbandSpeed value meaning "machine stopped" — excluded before pooling
_SPEED_DECIMALS = 4       # rounding used to group profiles/runs into discrete speed levels
_BAR_COLOUR = "indianred"


class _RateContext:
    """Per-profile arrays + run structure for the rate metrics, computed once and shared by all of them."""

    def __init__(self, profiles: list[profileData], idle_speed: float = IDLE_SPEED) -> None:
        self.profiles = profiles
        self.dist = profile_advance_distances(profiles)          # inter-profile advance (profile units)
        self.speed = np.array([p.rollerbandSpeed if p.rollerbandSpeed is not None else np.nan
                               for p in profiles], float)
        self.seg = _segment_mask(profiles)                       # cleaned isSegment membership
        self.neck = np.array([p.segmentNeck if p.segmentNeck is not None else 0.0 for p in profiles], float)
        self.ruptures = [p.segmentRuptures for p in profiles]    # per-profile broadcast rupture flag (0/1/None)
        self.non_idle = np.isfinite(self.speed) & (self.speed != idle_speed)
        self.levels = sorted({round(float(s), _SPEED_DECIMALS) for s in self.speed[self.non_idle]})

    def level_of(self, value: float) -> "float | None":
        """The speed level a value belongs to (rounded), or None if idle / missing."""
        if not np.isfinite(value) or value == IDLE_SPEED:
            return None
        return round(float(value), _SPEED_DECIMALS)

    def length_m(self, mask: "np.ndarray | None" = None) -> "dict[float, float]":
        """Metres of print at each level, over the non-idle profiles selected by `mask` (default: all)."""
        out = {lv: 0.0 for lv in self.levels}
        sel = self.non_idle if mask is None else (self.non_idle & mask)
        for i in np.flatnonzero(sel):
            out[self.level_of(self.speed[i])] += float(self.dist[i]) * PROFILE_UNITS_TO_MM / 1000.0
        return out

    def run_level(self, run: np.ndarray) -> "float | None":
        """The (median) speed level of a run, over its non-idle profiles; None if the run is entirely idle."""
        sp = self.speed[run]
        sp = sp[np.isfinite(sp) & (sp != IDLE_SPEED)]
        return self.level_of(float(np.median(sp))) if sp.size else None


def _neck_rate(ctx: _RateContext) -> "dict[float, tuple[float, int]]":
    """Necks per metre of filament: `segmentNeck` spans (by their median speed) / filament length per speed."""
    fil = ctx.length_m(ctx.seg)
    count = {lv: 0 for lv in ctx.levels}
    for span in _contiguous_runs(ctx.neck == 1.0):
        lv = ctx.run_level(span)
        if lv in count:
            count[lv] += 1
    return {lv: (count[lv] / fil[lv] if fil[lv] > 0 else np.nan, count[lv]) for lv in ctx.levels}


def _break_rate(ctx: _RateContext) -> "dict[float, tuple[float, int]]":
    """Bead breaks per metre: defect (pure-floor gap) runs by median speed / total print length per speed."""
    tot = ctx.length_m()
    count = {lv: 0 for lv in ctx.levels}
    for run in _contiguous_runs(~ctx.seg):
        lv = ctx.run_level(run)
        if lv in count:
            count[lv] += 1
    return {lv: (count[lv] / tot[lv] if tot[lv] > 0 else np.nan, count[lv]) for lv in ctx.levels}


def _rupture_fraction(ctx: _RateContext) -> "dict[float, tuple[float, int]]":
    """Fraction of segments that ended in a rupture (segmentRuptures == 1 / segments with a flag), per speed."""
    rupt = {lv: 0 for lv in ctx.levels}
    total = {lv: 0 for lv in ctx.levels}
    for run in _contiguous_runs(ctx.seg):
        lv = ctx.run_level(run)
        if lv not in total:
            continue
        flag = ctx.ruptures[int(run[0])]
        if flag is None:
            continue
        total[lv] += 1
        if flag == 1.0:
            rupt[lv] += 1
    return {lv: (rupt[lv] / total[lv] if total[lv] > 0 else np.nan, total[lv]) for lv in ctx.levels}


# Registry: name -> (display label, y-unit, metric fn(ctx) -> {level: (value, n)}). Add an entry to show
# another pooled per-speed rate/fraction as bars — the reusable "channel" for future bar-wise features.
RATE_METRICS: "dict[str, tuple[str, str, object]]" = {
    "neckRate":        ("neck rate", "necks / m", _neck_rate),
    "breakRate":       ("break rate", "breaks / m", _break_rate),
    "ruptureFraction": ("rupture fraction", "fraction", _rupture_fraction),
}


class FeatureRatePlot:
    """Interactive bar plot: one pooled RATE per rollerband speed, switchable via a left radio (RATE_METRICS)."""

    def __init__(self, profiles: list[profileData], metrics: dict = RATE_METRICS,
                 initial: str = "neckRate", idle_speed: float = IDLE_SPEED) -> None:
        if not metrics:
            raise ValueError("no rate metrics given")
        self._ctx = _RateContext(profiles, idle_speed)
        if not self._ctx.levels:
            raise ValueError("no non-idle profiles (all rollerbandSpeed == idle or missing)")
        self._metrics = metrics
        self._names = list(metrics)
        self._active = initial if initial in metrics else self._names[0]
        self._results = {name: fn(self._ctx) for name, (_, _, fn) in metrics.items()}  # precompute all metrics
        logger.info("Feature rates: %d speed levels, %d metrics", len(self._ctx.levels), len(self._names))
        self._build_figure()
        self._render()

    def _build_figure(self) -> None:
        """Main bar axes + a left radio to pick the metric (one entry per RATE_METRICS)."""
        self.fig = plt.figure(figsize=(11, 6))
        self.ax = self.fig.add_axes((0.30, 0.12, 0.66, 0.80))
        self._radio_ax = self.fig.add_axes((0.02, 0.35, 0.22, 0.32), frame_on=True)
        self._radio_ax.set_title("rate metric", fontsize=9)
        labels = [self._metrics[n][0] for n in self._names]
        self._radio = RadioButtons(self._radio_ax, labels, active=self._names.index(self._active))
        self._radio.on_clicked(self._on_pick)

    def _on_pick(self, label: str) -> None:
        """Radio callback: switch the active metric by its display label and re-render."""
        for name in self._names:
            if self._metrics[name][0] == label:
                self._active = name
                break
        self._render()

    def _render(self) -> None:
        """Draw the active metric as one bar per speed level, annotated with its value and n (event/sample count)."""
        self.ax.clear()
        display, unit, _ = self._metrics[self._active]
        result = self._results[self._active]
        levels = self._ctx.levels
        xs = np.arange(len(levels))
        vals = np.array([result[lv][0] for lv in levels], float)
        counts = [result[lv][1] for lv in levels]
        self.ax.bar(xs, np.where(np.isfinite(vals), vals, 0.0), color=_BAR_COLOUR, width=0.7)
        top = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        for x, v, n in zip(xs, vals, counts):
            txt = f"{v:.1f}\n(n={n})" if np.isfinite(v) else f"n/a\n(n={n})"
            self.ax.text(x, (v if np.isfinite(v) else 0.0) + top * 0.02, txt, ha="center", va="bottom", fontsize=8)
        self.ax.set_xticks(xs)
        self.ax.set_xticklabels([f"{lv:g}" for lv in levels])
        self.ax.set_xlabel("rollerbandSpeed (m/s)")
        self.ax.set_ylabel(f"{display}  ({unit})")
        self.ax.set_title(f"Per-speed {display}: pooled over all filament/runs at each speed  (n = event / sample count)")
        self.ax.margins(y=0.15)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def plot_feature_rates(profiles: list[profileData], metrics: dict = RATE_METRICS,
                       initial: str = "neckRate") -> FeatureRatePlot:
    """Open the interactive per-rollerband-speed rate bar plot (see module docstring). Returns the plot object."""
    plot = FeatureRatePlot(profiles, metrics=metrics, initial=initial)
    plot.show()
    return plot
