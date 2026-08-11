"""Interactive 2D feature-vs-PLC-channel correlation plot.

Correlates per-profile geometry features and PLC machine-log channels. The plot is built for one
channel family at a time via the `kind` argument, so `dataAnalysis.py` opens it as two cells:
- `kind="stepwise"` — a discrete/setpoint channel on x (`rollerbandSpeed`, `printHeadMixxingSpeed`, the
  two viscotec pumps): the feature is aggregated **per level** as a box or a violin.
- `kind="continuous"` — a **hexbin density + trend line**, with **any subject on either axis**: a shared
  pool of all features + channels feeds both a single-select x-picker and a multi-select y-panel, so you
  can plot feature-vs-channel, channel-vs-channel, or feature-vs-feature (e.g. torque vs pressure).

Live controls: toggle the y feature(s) (`show`), pick the x subject, a median/mean statistic toggle,
and — in the stepwise view — a box/violin shape toggle. In the continuous view both selectors are
category-grouped (Geometry / Segment / PLC). Complements `featureComparison` (feature-vs-time) and the
3D heat-map (`profile3Dplotting.plot_feature_heatmap`).

Single vs multiple features (auto-switch):
- **one** feature shown -> the rich single-feature view in real units: a box or violin per level
  (stepwise), or a hexbin density + central-per-bin trend + spread band (continuous).
- **two or more** features shown -> each collapses to one **normalised (0-1)** central+/-band trend line
  on a shared axis, since several boxes/densities can't overlay legibly. Each feature's Spearman
  correlation with the channel is shown in the legend.

A **median/mean toggle** switches the central statistic everywhere (median with a 25-75 IQR band, or
mean with a +/-std band); in the single stepwise view it just moves the central line on the box/violin.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons
from scipy.stats import spearmanr

from featureComparison import _feature_values, _normalise, _scale_range  # shared per-profile helpers
from profile3Dplotting import FEATURE_DISPLAY, group_by_category  # unit factor/label + shared grouping
from profileProcessingAlgorithms import _contiguous_runs, _segment_mask  # runs + cleaned segment membership
from profilePointsClass import profileData

logger = logging.getLogger(__name__)

IDLE_SPEED = 0.0            # rollerbandSpeed value meaning "machine stopped" — excluded before plotting
N_TREND_BINS = 20          # quantile bins for the continuous median-per-bin trend line
CLIP_PERCENTILE = (1.0, 99.0)  # robust range mapped to 0-1 for the normalised multi-feature overlay

# The two channel families, each opened as its own cell via `kind`. mortarPumpFlow (binary) and
# rollerbandHeight (constant here) are intentionally omitted. Constant channels (e.g. the viscotec
# pumps at 0 in the current dataset) are still shown — see the force-show note in __init__.
STEPWISE_CHANNELS: tuple[str, ...] = (
    "rollerbandSpeed", "printHeadMixxingSpeed", "viscoPump1_VMAflow", "viscoPump2_AcceleratorFlow",
)
CONTINUOUS_CHANNELS: tuple[str, ...] = (
    "pressurePrintHead", "printHeadTorque", "pressurePipeEnd", "pressurePipeStart",
)

# Default y-features: per-profile geometry plus the broadcast segment/defect aggregates. NOTE the last
# three (segmentVolume, segmentLength, defectLength) are one value repeated across a whole run, not
# independent per-profile samples: aggregating them per level weights each segment by its profile count,
# and a filament segment can span several speed levels. defectLength is only set on flat (pure-floor)
# profiles, so it plots on a disjoint profile subset from the filament features.
DEFAULT_FEATURES: tuple[str, ...] = (
    "widthFlank", "widthOuter", "heightP95", "heightSmooth",
    "areaSimpson", "areaShoelace", "sliceVolume",
    "segmentVolume", "segmentLength", "defectLength",
)
_STATS = ("median", "mean")  # central statistic for the boxes / trend / overlay lines (stat toggle)
_SHAPES = ("box", "violin")  # single-feature stepwise distribution shape (shape toggle)

# "Run" features are one value broadcast across a whole run, so in the stepwise view they are aggregated
# per run (one datapoint per segment/defect), not per profile. Everything else stays per-profile.
SEGMENT_FEATURES = ("segmentVolume", "segmentLength")  # per filament segment (run of non-flat profiles)
DEFECT_FEATURES = ("defectLength",)                    # per defect (run of flat / pure-floor profiles)
RUN_FEATURES = SEGMENT_FEATURES + DEFECT_FEATURES
MAX_RUN_LENGTH_MM = 500.0   # runs longer than 0.5 m are dropped (a long continuous filament/defect isn't interesting)
MIN_RUNS_FOR_BOX = 4        # a level needs > 3 surviving runs to draw a box/violin; else only the n= count is shown


class FeaturePlcPlot:
    """Interactive feature-vs-PLC plot for one `kind`: box/violin per level (stepwise) or hexbin density
    + trend (continuous), with a median/mean statistic toggle and — for stepwise — a box/violin shape
    toggle. Stepwise has feature show-toggles + a channel radio (x); continuous offers a shared
    feature+channel pool on both axes via category-grouped y (multi) and x (single) panels.

    Per-subject scale is fixed once from the (idle-excluded) values, so the normalised multi-feature
    overlay stays comparable as the x subject / stat / selection change.
    """

    def __init__(self, profiles: list[profileData], kind: str, features: tuple[str, ...] = DEFAULT_FEATURES,
                 channels: tuple[str, ...] | None = None, initial_feature: str = "widthFlank",
                 initial_channel: str | None = None, profile_step: int = 1,
                 idle_speed: float = IDLE_SPEED) -> None:
        if kind not in ("stepwise", "continuous"):
            raise ValueError(f"kind must be 'stepwise' or 'continuous', got {kind!r}")
        if channels is None:
            channels = STEPWISE_CHANNELS if kind == "stepwise" else CONTINUOUS_CHANNELS
        if not features:
            raise ValueError("no features given")
        unknown = [k for k in (*features, *channels) if k not in FEATURE_DISPLAY]
        if unknown:
            raise ValueError(f"unknown features/channels (not in FEATURE_DISPLAY): {unknown}")
        if initial_feature not in features:
            raise ValueError(f"initial_feature {initial_feature!r} not in {features}")

        # exclude idle (machine stopped) so averages/trends reflect actual printing
        kept = [p for p in profiles if p.rollerbandSpeed is not None and p.rollerbandSpeed != idle_speed]
        if not kept:
            raise ValueError("no non-idle profiles (all rollerbandSpeed == idle_speed or missing)")
        profs = kept[::profile_step]

        self._kind = kind
        self._stat = "median"
        self._shape = "box"
        self._suppress = False  # re-entrancy guard for the continuous x-panel's single-select behaviour

        if kind == "continuous":
            # Any subject on either axis: one shared pool of features + channels. x = single-select
            # (self._channel holds the current x key, which may be a feature OR a channel), y = multi.
            subjects = tuple(dict.fromkeys((*features, *channels)))  # union, order preserved, deduped
            self._features = subjects           # the y-options drive `visible` in _render
            self._x_options = subjects
            self._y_options = subjects
            self._phys = {k: _feature_values(profs, k) for k in subjects}
            self._scale = {k: _scale_range(self._phys[k], CLIP_PERCENTILE) for k in subjects}
            self._channel = (initial_channel if initial_channel in subjects
                             else initial_feature if initial_feature in subjects else subjects[0])
            self._visible = {k: (k == initial_feature) for k in subjects}
        else:  # stepwise: discrete channels on x (box/violin per level), features on y
            self._features = tuple(features)
            self._y_options = self._features
            # per-feature physical values (None -> NaN) and a fixed robust 0-1 scale for the overlay
            self._phys = {f: _feature_values(profs, f) for f in self._features}
            self._scale = {f: _scale_range(self._phys[f], CLIP_PERCENTILE) for f in self._features}
            # per-channel physical values; force-show every configured channel (constants included), drop
            # only a channel with no finite value at all (nothing to plot).
            chan_vals = {c: _feature_values(profs, c) for c in channels}
            self._chan = {c: v for c, v in chan_vals.items() if np.isfinite(v).any()}
            self._channels = [c for c in channels if c in self._chan]
            if not self._channels:
                raise ValueError("no PLC channels with any data")
            self._x_options = tuple(self._channels)
            self._channel = initial_channel if initial_channel in self._chan else self._channels[0]
            self._visible = {f: (f == initial_feature) for f in self._features}
            # Precompute one datapoint per run (segment/defect) for the RUN_FEATURES, from the FULL profile
            # list (idle-exclusion + decimation break run contiguity). Also keep the full channel arrays +
            # idle mask so each run's channel level(s) can be looked up per active channel.
            self._build_run_table(profiles, channels)

        logger.info("Feature-vs-PLC (%s): %d non-idle profiles (step %d), %d x-options, %d y-options",
                    kind, len(profs), profile_step, len(self._x_options), len(self._y_options))
        self._build_figure()
        self._render()

    def _build_run_table(self, profiles: list[profileData], channels: tuple[str, ...]) -> None:
        """Precompute per-run datapoints for the RUN_FEATURES and the full-list lookups they need.

        Builds `self._runs = {"segment": [...], "defect": [...]}` where each entry is
        `(idx, length_mm, {feature: display_value})`; `self._full_chan` (full channel arrays) and
        `self._full_idle` (rollerbandSpeed == 0). Also overwrites each run feature's normalisation scale
        with one derived from its per-run values (≤ 0.5 m), so the overlay 0-1 mapping matches the plot.
        """
        seg = _segment_mask(profiles)  # cleaned isSegment — segments vs defect gaps
        self._full_chan = {c: _feature_values(profiles, c) for c in channels}
        self._full_idle = np.array([p.rollerbandSpeed == IDLE_SPEED for p in profiles])
        self._runs = {"segment": [], "defect": []}
        for kind_key, mask, feats in (("segment", seg, SEGMENT_FEATURES), ("defect", ~seg, DEFECT_FEATURES)):
            for idx in _contiguous_runs(mask):
                head = profiles[int(idx[0])]
                values = {f: getattr(head, f) for f in feats}  # broadcast -> same across the run
                if any(v is None for v in values.values()):
                    continue
                values = {f: float(v) * FEATURE_DISPLAY[f][1] for f, v in values.items()}
                length_mm = values["segmentLength" if kind_key == "segment" else "defectLength"]
                self._runs[kind_key].append((idx, length_mm, values))
        # per-run normalisation scale for the run features (values within the length cutoff)
        for f in self._features:
            if f in RUN_FEATURES:
                bucket = "segment" if f in SEGMENT_FEATURES else "defect"
                vals = np.array([v[f] for _, length_mm, v in self._runs[bucket]
                                 if length_mm <= MAX_RUN_LENGTH_MM], dtype=float)
                if vals.size:
                    self._scale[f] = _scale_range(vals, CLIP_PERCENTILE)

    # --- statistics -----------------------------------------------------------------------------
    def _spearman(self, xv: np.ndarray, yv: np.ndarray) -> float:
        """Spearman rank correlation over the pairs finite in both (NaN if too few or x is constant)."""
        m = np.isfinite(xv) & np.isfinite(yv)
        if int(np.count_nonzero(m)) < 3:
            return float("nan")
        xm, ym = xv[m], yv[m]
        if np.all(xm == xm[0]) or np.all(ym == ym[0]):  # either side constant -> undefined; skip scipy warn
            return float("nan")
        r, _ = spearmanr(xm, ym)
        return float(r)

    def _central(self, values: np.ndarray) -> tuple[float, float, float]:
        """Central statistic + spread band per the stat toggle: (mean, mean-std, mean+std) or
        (median, 25th pct, 75th pct)."""
        if self._stat == "mean":
            m, s = float(np.mean(values)), float(np.std(values))
            return m, m - s, m + s
        return (float(np.median(values)), float(np.percentile(values, 25)),
                float(np.percentile(values, 75)))

    def _levels(self, channel: str) -> np.ndarray:
        """The channel's fixed set of x-axis levels: its sorted distinct non-idle values."""
        v = self._chan[channel]
        return np.unique(np.round(v[np.isfinite(v)], 6))

    def _run_level_values(self, feature: str, channel: str) -> tuple[dict, dict]:
        """Per-run datapoints of a RUN_FEATURE grouped by the active channel's level.

        Returns `(per_level, counts)` mapping level -> list of per-run values / surviving-run count.
        A run is dropped if it is longer than `MAX_RUN_LENGTH_MM` or spans more than one non-idle level
        of `channel` (or is entirely idle) — so each surviving run sits at exactly one level.
        """
        bucket = "segment" if feature in SEGMENT_FEATURES else "defect"
        chan, idle = self._full_chan[channel], self._full_idle
        per_level: dict[float, list[float]] = {}
        counts: dict[float, int] = {}
        for idx, length_mm, values in self._runs[bucket]:
            if length_mm > MAX_RUN_LENGTH_MM:
                continue
            ch = np.round(chan[idx], 6)
            ch = ch[~idle[idx] & np.isfinite(ch)]
            lv = np.unique(ch)
            if lv.size != 1:  # straddles >1 level, or entirely idle
                continue
            key = float(lv[0])
            per_level.setdefault(key, []).append(values[feature])
            counts[key] = counts.get(key, 0) + 1
        return per_level, counts

    def _binned_trend(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, ...]:
        """Central stat (+ spread band, per the stat toggle) of ys within `N_TREND_BINS` quantile bins
        of xs; returns (bin centres, central, band_low, band_high)."""
        edges = np.unique(np.quantile(xs, np.linspace(0.0, 1.0, N_TREND_BINS + 1)))
        if edges.size < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])
        idx = np.clip(np.digitize(xs, edges[1:-1]), 0, edges.size - 2)
        centres, central, low, high = [], [], [], []
        for k in range(edges.size - 1):
            sel = idx == k
            if np.any(sel):
                c, band_lo, band_hi = self._central(ys[sel])
                centres.append(float(np.median(xs[sel])))
                central.append(c); low.append(band_lo); high.append(band_hi)
        return np.array(centres), np.array(central), np.array(low), np.array(high)

    def _legend_label(self, feature: str, channel_vals: np.ndarray) -> str:
        """Overlay legend entry: feature name + its Spearman r against the current channel."""
        r = self._spearman(channel_vals, self._phys[feature])
        return f"{feature}  r={r:+.2f}" if np.isfinite(r) else f"{feature}  r=n/a"

    # --- figure / widgets -----------------------------------------------------------------------
    def _build_figure(self) -> None:
        """Lay out the main axes + colour-bar axes, the stat toggle, and the feature/channel selectors.

        Stepwise keeps two single widgets (feature checkboxes on the left, a channel radio on the right)
        plus a box/violin shape radio. Continuous instead offers a shared feature+channel pool on BOTH
        axes, via two category-grouped CheckButtons panels: y (multi-select) left, x (single-select,
        enforced in `_on_xpick`) right."""
        self.fig = plt.figure(figsize=(16, 8.5))
        self.ax = self.fig.add_axes((0.26, 0.12, 0.52, 0.80))
        self.cax = self.fig.add_axes((0.795, 0.12, 0.015, 0.80))  # hexbin colour bar (continuous only)
        self._stat_ax = self.fig.add_axes((0.855, 0.08, 0.14, 0.15), frame_on=True)
        self._stat_ax.set_title("stat", fontsize=11)
        self._stat_radio = RadioButtons(self._stat_ax, list(_STATS), active=_STATS.index(self._stat))
        self._stat_radio.on_clicked(self._on_stat)

        self._shape_radio = None
        if self._kind == "continuous":
            # any subject on either axis -> grouped panels, y (multi) left, x (single) right
            self._y_groups = self._build_grouped_panel(
                (0.008, 0.05, 0.165, 0.90), self._y_options, self._on_show,
                {k: self._visible[k] for k in self._y_options}, "y: show (multi)")
            self._x_groups = self._build_grouped_panel(
                (0.828, 0.26, 0.168, 0.66), self._x_options, self._on_xpick,
                {k: (k == self._channel) for k in self._x_options}, "x: pick one")
        else:
            self._show_ax = self.fig.add_axes((0.015, 0.30, 0.15, 0.60), frame_on=True)
            self._chan_ax = self.fig.add_axes((0.855, 0.30, 0.14, 0.60), frame_on=True)
            self._show_ax.set_title("features (show)", fontsize=11)
            self._chan_ax.set_title("channel (x)", fontsize=11)
            box = {"s": 90}
            self._show = CheckButtons(self._show_ax, list(self._features),
                                      [self._visible[f] for f in self._features],
                                      frame_props=box, check_props=box)
            for txt in self._show.labels:
                txt.set_fontsize(10)
            self._chan_radio = RadioButtons(self._chan_ax, self._channels,
                                            active=self._channels.index(self._channel))
            self._show.on_clicked(self._on_show)
            self._chan_radio.on_clicked(self._on_channel)
            # shape (box/violin) only applies to the per-level view
            self._shape_ax = self.fig.add_axes((0.015, 0.08, 0.15, 0.15), frame_on=True)
            self._shape_ax.set_title("shape", fontsize=11)
            self._shape_radio = RadioButtons(self._shape_ax, list(_SHAPES), active=_SHAPES.index(self._shape))
            self._shape_radio.on_clicked(self._on_shape)

    def _build_grouped_panel(self, region: tuple[float, float, float, float], keys: tuple[str, ...],
                             on_click, states: dict[str, bool], title: str,
                             box_size: int = 34) -> list[tuple[CheckButtons, list[str]]]:
        """Draw a category-grouped CheckButtons panel inside `region` (x, y, w, h in figure fractions):
        one CheckButtons per non-empty SELECTOR_CATEGORIES group under a bold header. Returns
        `[(CheckButtons, [keys]), ...]`; multi- vs single-select is enforced by `on_click`. Row height is
        uniform across groups (each group's axes height ∝ its key count)."""
        groups = group_by_category(keys)
        rx, ry, rw, rh = region
        self.fig.text(rx, ry + rh + 0.012, title, fontsize=11, ha="left", fontweight="bold")
        total_rows = sum(len(g) for _, g in groups)
        n_groups = len(groups)
        h_head, gap = 0.024, 0.012
        avail = rh - n_groups * h_head - max(0, n_groups - 1) * gap
        row_h = avail / total_rows if total_rows else avail
        out: list[tuple[CheckButtons, list[str]]] = []
        top = ry + rh
        for name, gkeys in groups:
            n = len(gkeys)
            self.fig.text(rx, top - h_head / 2, name, fontsize=10, fontweight="bold",
                          va="center", ha="left", color="0.2")
            body_bot = top - h_head - n * row_h
            panel = self.fig.add_axes((rx, body_bot, rw, n * row_h), frame_on=False)
            panel.set_xticks([]); panel.set_yticks([])
            cb = CheckButtons(panel, list(gkeys), [states.get(k, False) for k in gkeys],
                              frame_props={"s": box_size}, check_props={"s": box_size})
            for txt in cb.labels:
                txt.set_fontsize(9)
            cb.on_clicked(on_click)
            out.append((cb, list(gkeys)))
            top = body_bot - gap
        return out

    def _on_show(self, _label: str) -> None:
        if self._kind == "continuous":  # multi-select over every grouped y-panel
            for cb, keys in self._y_groups:
                for k, status in zip(keys, cb.get_status()):
                    self._visible[k] = bool(status)
        else:
            for f, status in zip(self._features, self._show.get_status()):
                self._visible[f] = bool(status)
        self._render()

    def _on_xpick(self, label: str) -> None:
        """Continuous x-panel single-select: make `label` the sole checked x key across all groups.
        Re-entrant (each corrective `set_active` re-fires this), so guard with `self._suppress`."""
        if self._suppress:
            return
        self._channel = label
        self._suppress = True
        for cb, keys in self._x_groups:  # force exactly the clicked key checked, everything else off
            for i, (cur, want) in enumerate(zip(cb.get_status(), [k == label for k in keys])):
                if cur != want:
                    cb.set_active(i)
        self._suppress = False
        self._render()

    def _on_channel(self, label: str) -> None:
        self._channel = label
        self._render()

    def _on_stat(self, label: str) -> None:
        self._stat = label
        self._render()

    def _on_shape(self, label: str) -> None:
        self._shape = label
        self._render()

    # --- rendering ------------------------------------------------------------------------------
    def _render(self) -> None:
        """Clear and rebuild the main axes for the current channel and visible features.

        The render kind is fixed by `self._kind`; the plot *type* still changes with the selection
        (single vs multiple features, box vs violin), so the axes are rebuilt from scratch each call.
        """
        self.ax.clear()
        self.cax.clear()
        self.cax.set_visible(False)
        channel = self._channel
        # x from the active x key: a channel array (stepwise) or any subject's values (continuous)
        x = self._chan[channel] if self._kind == "stepwise" else self._phys[channel]
        visible = [f for f in self._features if self._visible[f]]

        if not visible:
            self.ax.text(0.5, 0.5, "no feature selected", ha="center", va="center",
                         transform=self.ax.transAxes, fontsize=13, color="0.4")
        elif self._kind == "stepwise":
            self._render_stepwise(x, visible)
        else:
            self._render_continuous(x, visible)

        chan_unit = FEATURE_DISPLAY[channel][2]
        self.ax.set_xlabel(f"{channel} [{chan_unit}]" if chan_unit else channel, fontsize=12)
        self.ax.set_title(f"{('  '.join(visible)) or '—'}  vs  {channel}   ({self._kind})", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()

    def _render_stepwise(self, x: np.ndarray, visible: list[str]) -> None:
        """One channel level per x-tick. Single feature: a per-profile box/violin (regular) or a per-run
        box/violin with n= counts (run features). Several: normalised central+/-band lines. The x-axis is
        fixed to the channel's full level set regardless of which levels actually have a datapoint."""
        channel = self._channel
        xr = np.round(x, 6)
        levels = self._levels(channel)
        if len(visible) == 1:
            f = visible[0]
            if f in RUN_FEATURES:
                self._draw_run_single(f, channel, levels)
            else:
                yv = self._phys[f]
                groups = [(float(lv), yv[(xr == lv) & np.isfinite(yv)]) for lv in levels]
                groups = [(lv, d) for lv, d in groups if d.size]
                if groups:
                    self._draw_box_or_violin([lv for lv, _ in groups], [d for _, d in groups])
                self._set_single_ylabel(f, self._spearman(x, yv))
        else:
            for f in visible:
                if f in RUN_FEATURES:
                    pos, cen, blo, bhi = self._run_level_central(f, channel)
                else:
                    pos, cen, blo, bhi = self._regular_level_central(f, xr, levels)
                if not pos:
                    continue
                lo, hi = self._scale[f]
                pos = np.array(pos)
                line, = self.ax.plot(pos, _normalise(np.array(cen), lo, hi), marker="o", lw=1.5,
                                     label=self._step_legend_label(f, channel))
                self.ax.fill_between(pos, _normalise(np.array(blo), lo, hi),
                                     _normalise(np.array(bhi), lo, hi),
                                     color=line.get_color(), alpha=0.15)
            self._set_overlay_ylabel()
        self._apply_fixed_xaxis(levels)

    def _draw_box_or_violin(self, positions: list[float], data: list[np.ndarray]) -> None:
        """Draw a box or violin (shape toggle) per level, its central line following the stat toggle."""
        width = 0.6 * float(np.min(np.diff(positions))) if len(positions) > 1 else 0.5
        if self._shape == "violin" and all(d.size >= 2 for d in data):  # KDE needs >=2 points
            self.ax.violinplot(data, positions=positions, widths=width, showextrema=True,
                               showmedians=(self._stat == "median"), showmeans=(self._stat == "mean"))
        else:
            box_kwargs = dict(positions=positions, widths=width, showfliers=True, manage_ticks=False)
            if self._stat == "mean":  # show the mean line, hide the (default) median line
                box_kwargs.update(showmeans=True, meanline=True, medianprops={"linewidth": 0})
            self.ax.boxplot(data, **box_kwargs)

    def _draw_run_single(self, feature: str, channel: str, levels: np.ndarray) -> None:
        """Single run feature: box/violin from per-run values where a level has > 3 runs, plus an n=
        count above every level (shown even where the box is suppressed)."""
        per_level, counts = self._run_level_values(feature, channel)
        positions = [float(lv) for lv in levels if counts.get(float(lv), 0) >= MIN_RUNS_FOR_BOX]
        if positions:
            self._draw_box_or_violin(positions, [np.array(per_level[p]) for p in positions])
        self._set_single_ylabel(feature, self._feature_spearman(feature, channel))
        trans = self.ax.get_xaxis_transform()  # x in data coords, y in axes fraction
        for lv in levels:
            self.ax.text(float(lv), 0.99, f"n={counts.get(float(lv), 0)}", transform=trans,
                         ha="center", va="top", fontsize=9, color="0.35")

    def _regular_level_central(self, feature: str, xr: np.ndarray, levels: np.ndarray):
        """Per-level (central, band-low, band-high) of a per-profile feature over its profiles."""
        yv = self._phys[feature]
        pos, cen, blo, bhi = [], [], [], []
        for lv in levels:
            sel = (xr == lv) & np.isfinite(yv)
            if np.any(sel):
                c, band_lo, band_hi = self._central(yv[sel])
                pos.append(float(lv)); cen.append(c); blo.append(band_lo); bhi.append(band_hi)
        return pos, cen, blo, bhi

    def _run_level_central(self, feature: str, channel: str):
        """Per-level (central, band) of a run feature over its per-run values, at levels with > 3 runs."""
        per_level, counts = self._run_level_values(feature, channel)
        pos, cen, blo, bhi = [], [], [], []
        for lv in sorted(per_level):
            if counts[lv] >= MIN_RUNS_FOR_BOX:
                c, band_lo, band_hi = self._central(np.array(per_level[lv]))
                pos.append(lv); cen.append(c); blo.append(band_lo); bhi.append(band_hi)
        return pos, cen, blo, bhi

    def _feature_spearman(self, feature: str, channel: str) -> float:
        """Spearman r vs the channel: per-run points for a run feature, else per-profile."""
        if feature in RUN_FEATURES:
            per_level, _ = self._run_level_values(feature, channel)
            xs = np.array([lv for lv, vals in per_level.items() for _ in vals])
            ys = np.array([v for vals in per_level.values() for v in vals])
            return self._spearman(xs, ys)
        return self._spearman(self._chan[channel], self._phys[feature])

    def _step_legend_label(self, feature: str, channel: str) -> str:
        """Overlay legend entry for the stepwise view: feature + its (run-aware) Spearman r."""
        r = self._feature_spearman(feature, channel)
        return f"{feature}  r={r:+.2f}" if np.isfinite(r) else f"{feature}  r=n/a"

    def _apply_fixed_xaxis(self, levels: np.ndarray) -> None:
        """Tick the channel's full level set with a fixed xlim, so the axis doesn't move between
        features or when some levels have no datapoint."""
        if levels.size == 0:
            return
        self.ax.set_xticks(levels)
        self.ax.set_xticklabels([f"{lv:g}" for lv in levels])
        gap = float(np.min(np.diff(levels))) if levels.size > 1 else 1.0
        self.ax.set_xlim(float(levels[0]) - 0.7 * gap, float(levels[-1]) + 0.7 * gap)

    def _render_continuous(self, x: np.ndarray, visible: list[str]) -> None:
        """Hexbin density + central trend for a single feature (real units); normalised central lines
        for several. The central statistic (median/IQR or mean/std) follows the stat toggle."""
        if len(visible) == 1:
            f = visible[0]
            yv = self._phys[f]
            m = np.isfinite(x) & np.isfinite(yv)
            xs, ys = x[m], yv[m]
            hb = self.ax.hexbin(xs, ys, gridsize=40, cmap="Blues", mincnt=1)
            self.cax.set_visible(True)
            self.fig.colorbar(hb, cax=self.cax, label="count")
            centres, central, band_lo, band_hi = self._binned_trend(xs, ys)
            if centres.size:
                self.ax.fill_between(centres, band_lo, band_hi, color="crimson", alpha=0.20)
                self.ax.plot(centres, central, color="crimson", lw=2.0, label=f"{self._stat} / bin")
                self.ax.legend(loc="best", fontsize=9)
            self._set_single_ylabel(f, self._spearman(x, yv))
        else:
            for f in visible:
                yv = self._phys[f]
                lo, hi = self._scale[f]
                m = np.isfinite(x) & np.isfinite(yv)
                centres, central, _, _ = self._binned_trend(x[m], yv[m])
                if centres.size:
                    self.ax.plot(centres, _normalise(central, lo, hi), marker=".", lw=1.5,
                                 label=self._legend_label(f, x))
            self._set_overlay_ylabel()

    def _set_single_ylabel(self, feature: str, r: float) -> None:
        """y-axis label in the feature's real unit, plus a Spearman-r annotation in the corner."""
        unit = FEATURE_DISPLAY[feature][2]
        self.ax.set_ylabel(f"{feature} [{unit}]" if unit else feature, fontsize=12)
        text = f"Spearman r = {r:+.2f}" if np.isfinite(r) else "Spearman r = n/a"
        self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes, va="top", ha="left",
                     fontsize=11, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

    def _set_overlay_ylabel(self) -> None:
        """Shared normalised y-axis for the multi-feature overlay."""
        p_lo, p_hi = CLIP_PERCENTILE
        self.ax.set_ylabel(f"normalised per feature (robust {p_lo:g}-{p_hi:g} pct -> 0-1)", fontsize=12)
        self.ax.set_ylim(-0.03, 1.03)
        self.ax.legend(loc="best", fontsize=9)


def plot_feature_plc_trends(profiles: list[profileData], kind: str,
                            features: tuple[str, ...] = DEFAULT_FEATURES,
                            channels: tuple[str, ...] | None = None, initial_feature: str = "widthFlank",
                            initial_channel: str | None = None, profile_step: int = 1,
                            idle_speed: float = IDLE_SPEED, show: bool = True) -> plt.Figure:
    """Interactive feature-vs-PLC-channel correlation plot for one channel family.

    `kind="stepwise"` puts a discrete PLC channel on x and aggregates the feature per level (a box or
    violin). `kind="continuous"` draws a hexbin density + trend line and offers a **shared pool of all
    `features` + `channels` on both axes** (single-select x-picker, multi-select y-panel), so any subject
    can be x or y — feature-vs-channel, channel-vs-channel, or feature-vs-feature. One subject shown on y
    -> the rich single view in real units with a Spearman r; two or more -> normalised 0-1 trend lines
    with each subject's r in the legend. A median/mean toggle sets the central statistic (median+IQR or
    mean+std); the stepwise view also has a box/violin shape toggle. Idle profiles
    (`rollerbandSpeed == idle_speed`) are excluded.

    Args:
        profiles: processed, PLC-joined profiles (need `rollerbandSpeed` + the channels).
        kind: "stepwise" (discrete channels → box/violin) or "continuous" (analogue channels → density).
        features: geometry/segment features to offer; default = per-profile geometry + the broadcast
            segment aggregates (`segmentVolume`/`segmentLength`/`defectLength` — one value per run). In
            the continuous kind these join `channels` in the shared any-axis pool (offered on x and y).
        channels: PLC channels to offer; defaults to `STEPWISE_CHANNELS` / `CONTINUOUS_CHANNELS` for the
            `kind`. Stepwise: the x-axis channels (constant channels shown, only all-NaN dropped).
            Continuous: added to the shared pool so they can sit on either axis.
        initial_feature: the subject shown first on y.
        initial_channel: the subject shown first on x (defaults to the first offered x option; in the
            continuous kind this may be a feature or a channel).
        profile_step: use every Nth non-idle profile (decimation for responsiveness).
        idle_speed: `rollerbandSpeed` value treated as idle/stopped and excluded.
        show: call `plt.show()` before returning (set False for headless use).

    Returns the Figure. Note: the widgets need a GUI backend (run as a script or `%matplotlib qt`);
    VS Code's inline backend renders a static image.
    """
    fp = FeaturePlcPlot(profiles, kind, features=features, channels=channels,
                        initial_feature=initial_feature, initial_channel=initial_channel,
                        profile_step=profile_step, idle_speed=idle_speed)
    fp.fig._feature_plc = fp  # keep the instance (and its widgets) alive
    if show:
        plt.show()
    return fp.fig
