from typing import cast

import numpy as np
import pyvista as pv
from plcData import PLC_COLUMNS
from profilePointsClass import profileData
from profileProcessingAlgorithms import _is_segment, profile_advance_distances

# Heat-map feature display: feature -> (group, unit factor, unit label). 1 profile unit = 0.01 mm,
# so lengths scale to mm and areas to mm^2. Features sharing a group share one colour range (clim),
# so paired measures are directly comparable; grouping also sets the button order.
FEATURE_DISPLAY = {
    "widthFlank":       ("width",  0.01, "mm"),
    "widthOuter":       ("width",  0.01, "mm"),
    "heightP95":       ("height", 0.01, "mm"),
    "heightSmooth": ("height", 0.01, "mm"),
    "areaSimpson":      ("area",   1e-4, "mm^2"),
    "areaShoelace":     ("area",   1e-4, "mm^2"),
    # Volumes: area-unit * distance-unit -> cm^3 (1e-4 mm^2 * 0.01 mm = 1e-6 mm^3 = 1e-9 cm^3). Each
    # its own colour group: a segment total is ~10^2-10^3x a single slice, so they must not share a clim.
    "segmentVolume":    ("segmentVolume", 1e-9, "cm^3"),
    "sliceVolume":      ("sliceVolume",   1e-9, "cm^3"),
    # Run lengths along the print path: profile units -> mm (0.01). Own colour groups (a macro cm-scale
    # length, unlike the mm-scale bead widths). defectLength lives on pure-floor profiles, so it only
    # shows in a floor-category heat-map (`plot_feature_heatmap(..., category="floor")`).
    "segmentLength":    ("segmentLength", 0.01, "mm"),
    "defectLength":     ("defectLength",  0.01, "mm"),
    # Per-profile 0/1 flags (share one 0-1 colour group; 1 = filament/segment -> same high colour):
    # `isSegment` = part of a cleaned filament segment (clean_flat_runs), `isNotFlat` = raw "has filament
    # points" (inverse of isFlat). View over ALL points (`plot_feature_heatmap(..., category=None)`) so
    # bridged floor-only gaps are visible.
    "isSegment":        ("segFlag", 1.0, ""),
    "isNotFlat":        ("segFlag", 1.0, ""),
    # PLC machine-log channels (joined by timestamp): each its own colour group, since their
    # magnitudes differ widely. Shown in the PLC's native engineering units (factor 1.0; the
    # unit label is left blank as the physical units aren't recorded in the CSV).
    **{name: (name, 1.0, "") for name in PLC_COLUMNS},
}

# Heat-map colour-scale modes, cycled live by the scale button (see plottingClass._add_scale_button).
# "linear" (default) keeps the full min-max range; the others tame an outlier that would otherwise
# squash every smaller value into one end of the colormap.
_SCALE_MODES = ("linear", "log", "clip", "rank")
CLIP_PERCENTILES = (2.0, 98.0)  # "clip" mode maps this percentile range to the colormap (outliers saturate)
_RANK_SUFFIX = "__rank"          # per-feature companion array holding the [0, 1] dense rank (rank mode)

# Heat-map point set per feature: the unified heat-map prebuilds one cloud per distinct point set and
# swaps the visible one when the active feature changes. These features live on floor/gap profiles (no
# filament points) so they are coloured over ALL points; every other feature colours the filament points.
_ALL_POINT_FEATURES = frozenset({"isSegment", "isNotFlat", "defectLength"})


def _feature_pointset(feature: str) -> "str | None":
    """Which points a feature is coloured on: None = all points, "profile" = filament points."""
    return None if feature in _ALL_POINT_FEATURES else "profile"


# For the selector: a paired row shows the measure name once + a short per-method button label.
_GROUP_DISPLAY = {"segFlag": "flags"}  # measure label of a paired row (else the FEATURE_DISPLAY group name)


def _method_label(feature: str) -> str:
    """Short button label for a paired feature: the method suffix after its colour-group name
    (widthFlank -> 'flank', areaSimpson -> 'simpson'), or the full name when it has no such prefix."""
    group = FEATURE_DISPLAY[feature][0]
    if feature.lower().startswith(group.lower()) and len(feature) > len(group):
        suffix = feature[len(group):]
        return suffix[0].lower() + suffix[1:]
    return feature


# Feature-selector button-panel categories (grouping + headers only; independent of the FEATURE_DISPLAY
# colour groups and the point-set groups above). Features not in any list fall under "Other". Public so the
# 2D feature-comparison selector (featureComparison.py) groups identically — single source of truth (§11).
SELECTOR_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Geometry", ("widthFlank", "widthOuter", "heightP95", "heightSmooth", "areaSimpson", "areaShoelace", "sliceVolume")),
    ("Segment", ("segmentVolume", "segmentLength", "defectLength", "isSegment", "isNotFlat")),
    ("PLC", PLC_COLUMNS),
)


def group_by_category(keys: "tuple[str, ...] | list[str]") -> list[tuple[str, list[str]]]:
    """Group `keys` under the SELECTOR_CATEGORIES headers (category member order within each; empty
    categories skipped), with any unlisted key collected under a trailing 'Other'. Shared grouping for
    the 2D selector panels (featureComparison + featurePlcTrends), matching the 3D heat-map layout."""
    present = set(keys)
    listed = {m for _, members in SELECTOR_CATEGORIES for m in members}
    groups: list[tuple[str, list[str]]] = []
    for name, members in SELECTOR_CATEGORIES:
        members_present = [m for m in members if m in present]
        if members_present:
            groups.append((name, members_present))
    other = [k for k in keys if k not in listed]
    if other:
        groups.append(("Other", other))
    return groups


class plottingClass:
    def __init__(self, profiles: list[profileData], voxel_size: float | None = None):
        """Build the 3D print path from the profiles' physical along-track advance (see
        profile_advance_distances). `profiles` must be the same list (order/length) later passed to
        `plot()`, so each profile lands at its path point. Prefer the processed (PLC-joined) profiles,
        which carry `rollerbandSpeed`; without it the layout falls back to a uniform gap.

        `voxel_size` (default None = off) downsamples the dense clouds to one point per cube of a 3-D
        grid — cube edge = `voxel_size` in profile units (1 unit = 0.01 mm) — cutting the point count
        (and overdraw/memory) to keep large sets responsive. It bins all axes INCLUDING height, so it
        is not a uniform on-screen spacing: steep features (filament flanks) keep points stacked
        ~`voxel_size` apart in height. Off ⇒ renders identically to before.
        """
        self.plotter = pv.Plotter(window_size=[1280, 860])  # roomy default so the tall left feature panel fits
        self._voxel_size = voxel_size

        distances = profile_advance_distances(profiles)
        self.pathPoints, self.tiltAngles = compute_print_path_and_angle(distances)
        self.rotation_matrices = [np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]) for theta in self.tiltAngles]

    def _maybe_voxel(self, cloud: "pv.DataSet") -> "pv.DataSet":
        """Downsample a point cloud to one point per `self._voxel_size` voxel (off when None).

        Bins points on a grid and keeps the first point in each occupied voxel. `extract_points`
        carries every point-data array along, so a heat-map cloud's per-feature scalars stay aligned.

        The 3-D voxel index is flattened to a single int64 key (a bijection via `ravel_multi_index`)
        so uniqueness is a fast 1-D sort instead of a 3-column lexsort. Output is identical to the
        row-wise `np.unique(axis=0)` — same first point per voxel — because the flatten is one-to-one
        and `np.unique(return_index=True)` returns first occurrences. Falls back to the row-wise
        unique in the pathological case where the flattened index space would overflow int64.
        """
        if not self._voxel_size or cloud.n_points == 0:
            return cloud
        key = np.floor(cloud.points / self._voxel_size).astype(np.int64)
        key -= key.min(axis=0)  # shift to a non-negative grid so ravel_multi_index is valid
        dims = key.max(axis=0) + 1
        if int(dims[0]) * int(dims[1]) * int(dims[2]) < 2**63:  # flat index fits -> fast 1-D unique
            flat = np.ravel_multi_index((key[:, 0], key[:, 1], key[:, 2]), dims)
            _, idx = np.unique(flat, return_index=True)
        else:  # pathological extent: exact row-wise fallback
            _, idx = np.unique(key, axis=0, return_index=True)
        return cast("pv.DataSet", cloud.extract_points(np.sort(idx)))

    def show(self):
        self.plotter.add_camera_orientation_widget()
        self.plotter.show()

    def plot(self, profiles: list[profileData], plotSubject:str, colour:str, size=5,
             profile_step: int = 1, point_step: int = 1, flat_colour: str | None = None,
             category: str | None = None, spheres: bool = False) -> None:
        """Add one subject to the 3D scene: "profile", "baseline", "zeroBaseline",
        "widthFlankPoints" (slope-peak method, uses `widthFlankIdx`), or "widthOuterPoints" (outer-filament-
        point method, uses `widthOuterIdx`).

        profile_step / point_step subsample the dense "profile" cloud so interaction
        stays responsive on very large datasets: plot every profile_step-th profile and
        every point_step-th point. Both default to 1 (plot everything) and only affect
        "profile". size / spheres set point size and sphere rendering for the point subjects
        (enlarge + spheres=True on the width markers so the chosen points stand out).

        flat_colour (profiles only): if set, gap profiles (not in a cleaned filament segment,
        `~isSegment`) are drawn in this colour and the rest in
        `colour`; if None, every profile uses `colour`.

        category (profiles only): "floor" or "profile" draws only points of that category
        (uses `floorMask`); None draws all points. Call twice with different category +
        colour to show floor vs filament in two colours.
        """
        match plotSubject:
            case "profile":
                if flat_colour is None:
                    self.add_3d_points_to_plot(
                        get_profile_points_for_plot(profiles, profile_step, point_step, category=category), colour, size)
                else:
                    self.add_3d_points_to_plot(
                        get_profile_points_for_plot(profiles, profile_step, point_step, want_flat=False, category=category), colour, size)
                    self.add_3d_points_to_plot(
                        get_profile_points_for_plot(profiles, profile_step, point_step, want_flat=True, category=category), flat_colour, size)
            case "baseline":
                self.add_lines_to_plot(line_points_from_floorSides(profiles), colour)
            case "zeroBaseline":
                # flat z = 0 reference on each profile (the uniform leveling target)
                self.add_lines_to_plot(line_points_from_zero(profiles), colour)
            case "widthFlankPoints":
                # slope-peak width method: mark the two flank-foot points (from `widthFlankIdx`)
                self.add_3d_points_to_plot(width_point_arrays(profiles, "widthFlankIdx"), colour, size, spheres=spheres)
            case "widthOuterPoints":
                # filament-edge width method: mark the two outer filament points (from `widthOuterIdx`)
                self.add_3d_points_to_plot(width_point_arrays(profiles, "widthOuterIdx"), colour, size, spheres=spheres)

    def add_3d_points_to_plot(self,points, colour = 'green', point_size=5, spheres=False):
        # Collect every profile's transformed points and add them as a single actor, placed along the
        # shared physical print path (self.pathPoints, built in __init__). One add_points call instead
        # of one per profile is far faster for many profiles.
        transformed = []
        for i, prof in enumerate(points):
            if prof.shape[0] > 0:
                prof = prof @ self.rotation_matrices[i].T  # rotate to match path direction
                transformed.append(self.pathPoints[i] + prof)

        if transformed:
            cloud = pv.PolyData(np.vstack(transformed))
            if not spheres:  # dense cloud: downsample; markers (spheres) stay full so both show
                cloud = self._maybe_voxel(cloud)
            self.plotter.add_points(cloud, color=colour, point_size=point_size, render_points_as_spheres=spheres)

    def add_lines_to_plot(self, linePoints, colour = 'green'):
        # Collect every line's two transformed endpoints and add them all as a single mesh, placed
        # along the shared physical print path. One add_mesh call is far faster for many profiles.
        endpoints = []
        for i in range(len(linePoints)):
            rot_matrix = self.rotation_matrices[i]
            p0 = np.asarray(linePoints[i][0], dtype=float) @ rot_matrix.T + self.pathPoints[i]  # rotate + translate to path
            p1 = np.asarray(linePoints[i][1], dtype=float) @ rot_matrix.T + self.pathPoints[i]
            endpoints.append(p0)
            endpoints.append(p1)

        if endpoints:
            # points ordered as segment pairs (p0, p1, p0, p1, ...) -> one line per pair
            lines = pv.line_segments_from_points(np.array(endpoints))
            self.plotter.add_mesh(lines, color = colour, line_width=5)

    def plot_feature_heatmap(self, profiles: list[profileData],
                             features: tuple[str, ...] = ("widthFlank", "widthOuter", "heightP95", "heightSmooth", "areaSimpson", "areaShoelace"),
                             initial: str = "widthOuter", cmap: str = "viridis",
                             point_size: int = 6, profile_step: int = 1, point_step: int = 1) -> None:
        """Colour the cloud by a per-profile scalar feature, with a clickable button panel to switch the
        active feature live.

        Each feature's scalar is broadcast to its points (converted to physical units per
        FEATURE_DISPLAY). Features in the same group (width / height / area) share one colour range so
        the paired measures are directly comparable, and the colour bar is labelled in mm / mm^2. A radio
        group at the bottom-right sets the colour-scale mode of the active feature live — linear (full
        min-max, default), log, clip (2-98 pct, so an outlier doesn't squash the rest), rank (dense rank
        0-1) — see `_apply_scale`.

        Features live on different point sets, so the heat-map **prebuilds one cloud per point set** and
        swaps the visible one when the active feature changes (see `_feature_pointset`): filament
        geometry + PLC channels colour the **filament** points, while the `isSegment` / `isNotFlat` flags
        and `defectLength` (which live on floor/gap profiles) colour **all** points. Switching within a
        point set only repoints the mapper (instant); switching across point sets swaps which cloud is
        drawn, so the displayed points change (filament-only ↔ all) while the camera is kept. NaN feature
        values render in the NaN colour. Interactive-window only (buttons need a live VTK interactor).
        """
        self._build_feature_cloud(profiles, features, initial, cmap, point_size, profile_step, point_step)
        self._add_feature_selector()
        self._add_scale_selector()

    def _build_feature_cloud(self, profiles: list[profileData], features: tuple[str, ...],
                             initial: str, cmap: str, point_size: int,
                             profile_step: int, point_step: int) -> None:
        """Prebuild one point cloud per distinct point set among `features` (see `_feature_pointset`),
        each carrying its features' scalar + rank arrays and a hidden actor, and store the state the
        selector callbacks mutate. Only the active feature's cloud is shown (via `_apply_scale`).
        Separated from the widget wiring so it can be exercised headless."""
        if initial not in features:
            raise ValueError(f"initial feature {initial!r} not in {features}")
        self._feature_names = list(features)
        self._feature_initial = initial
        self._active_feature = initial
        self._scale_mode = "linear"
        self._feature_pointset = {f: _feature_pointset(f) for f in features}
        self._feature_unit = {f: FEATURE_DISPLAY[f][2] for f in features}
        self._clouds: dict = {}
        self._mappers: dict = {}
        self._actors: dict = {}
        self._feature_scale_clim: dict = {}
        self._scalar_bar_ps = "<none>"  # which point set the single scalar bar is currently tied to

        by_pointset: dict = {}
        for f in features:
            by_pointset.setdefault(self._feature_pointset[f], []).append(f)
        for pointset, feats in by_pointset.items():
            built = self._build_one_cloud(profiles, feats, cmap, point_size, profile_step, point_step, pointset)
            if built is not None:
                self._clouds[pointset], self._mappers[pointset], self._actors[pointset], scale_clim = built
                self._feature_scale_clim.update(scale_clim)
        if not self._clouds:
            return  # nothing to draw (e.g. every profile flat)
        if self._feature_pointset[initial] not in self._clouds:  # initial's cloud is empty -> pick a built one
            self._active_feature = next(f for f in features if self._feature_pointset[f] in self._clouds)
        self._apply_scale(self._active_feature, "linear")

    def _build_one_cloud(self, profiles: list[profileData], features: list[str], cmap: str,
                         point_size: int, profile_step: int, point_step: int, pointset: str | None):
        """Build one cloud for `features` on `pointset` (category for get_profile_points_for_plot): attach
        each feature's scalar + rank array, voxel-downsample, add a hidden points actor. Returns
        (cloud, mapper, actor, {feature: {mode: clim}}), or None if the point set has no points."""
        per_profile = get_profile_points_for_plot(profiles, profile_step, point_step, category=pointset)
        transformed = []
        columns: dict[str, list[np.ndarray]] = {f: [] for f in features}
        for i, prof_pts in enumerate(per_profile):
            if prof_pts.shape[0] == 0:  # skipped, empty, or no points of this category
                continue
            transformed.append(self.pathPoints[i] + prof_pts @ self.rotation_matrices[i].T)
            for f in features:
                val = getattr(profiles[i], f)
                factor = FEATURE_DISPLAY[f][1]  # profile units -> physical (mm / mm^2)
                columns[f].append(np.full(prof_pts.shape[0], np.nan if val is None else float(val) * factor))
        if not transformed:
            return None

        cloud = pv.PolyData(np.vstack(transformed))
        for f in features:
            cloud[f] = np.concatenate(columns[f])
        cloud = self._maybe_voxel(cloud)  # downsample before clim/store so the reduced cloud is coloured
        for f in features:
            cloud[f + _RANK_SUFFIX] = _rank01(np.asarray(cloud[f]))

        # features in the same group share one colour range, over the group's values (all in this cloud)
        groups: dict[str, list[str]] = {}
        for f in features:
            groups.setdefault(FEATURE_DISPLAY[f][0], []).append(f)
        group_vals = {g: np.concatenate([cloud[f] for f in feats]) for g, feats in groups.items()}
        group_linear = {g: _finite_clim(v) for g, v in group_vals.items()}
        group_log = {g: _positive_clim(v) for g, v in group_vals.items()}
        group_clip = {g: _percentile_clim(v, CLIP_PERCENTILES) for g, v in group_vals.items()}
        scale_clim = {
            f: {"linear": group_linear[g], "log": group_log[g], "clip": group_clip[g], "rank": [0.0, 1.0]}
            for f, g in ((f, FEATURE_DISPLAY[f][0]) for f in features)
        }
        actor = self.plotter.add_points(
            cloud, scalars=features[0], cmap=cmap, clim=scale_clim[features[0]]["linear"],
            nan_color="lightgray", point_size=point_size, render_points_as_spheres=False,
            show_scalar_bar=False,  # one shared bar is managed centrally in _apply_scale
        )
        actor.SetVisibility(False)
        return cloud, actor.mapper, actor, scale_clim

    def _scale_label(self, feature: str, mode: str) -> str:
        """Text above the colour bar: the feature's unit annotated with the active scale mode."""
        unit = self._feature_unit[feature]
        if mode == "rank":
            return "rank 0-1"
        if mode == "log":
            return f"{unit} (log)".strip()
        if mode == "clip":
            lo, hi = CLIP_PERCENTILES
            return f"{unit} (clip {lo:g}-{hi:g}%)".strip()
        return unit  # linear

    def _set_feature_labels(self, feature: str, mode: str) -> None:
        """Feature name at the top edge, and its unit + active scale mode centred just above the
        horizontal colour bar (a separate text actor so it is not cramped against the bar's numbers)."""
        self.plotter.add_text(feature, name="feature_title", position="upper_edge", font_size=16)
        # anchor at the bar's centre (x = 0.50) with centred justification so the label stays centred
        unit_actor = self.plotter.add_text(self._scale_label(feature, mode), name="feature_unit",
                                           position=(0.50, 0.17), viewport=True, font_size=16)
        unit_actor.GetTextProperty().SetJustificationToCentered()

    def _apply_scale(self, feature: str, mode: str) -> None:
        """Colour `feature` with scale `mode` live: show its point-set cloud (swapping the visible actor
        + re-tying the shared colour bar when the point set changes), pick the value/rank array, set the
        log flag and colour range, relabel and re-render. A mode with no valid range (e.g. log on a
        feature with no positive values) renders as linear, but the requested `mode` is kept so cycling
        still advances."""
        self._active_feature = feature
        self._scale_mode = mode
        pointset = self._feature_pointset[feature]
        if pointset not in self._clouds:  # this feature's cloud is empty -> nothing to show
            return
        cloud, mapper = self._clouds[pointset], self._mappers[pointset]

        if self._scalar_bar_ps != pointset:  # point set changed -> swap visible actor + re-tie the bar
            for ps, actor in self._actors.items():
                actor.SetVisibility(ps == pointset)
            if self._scalar_bar_ps != "<none>":
                self.plotter.remove_scalar_bar(title="")
            self.plotter.add_scalar_bar(title="", mapper=mapper, label_font_size=14,
                                        position_x=0.33, position_y=0.10, width=0.34, height=0.05)
            self._scalar_bar_ps = pointset

        clim = self._feature_scale_clim[feature][mode]
        render_mode = mode
        if clim is None:  # mode unavailable for this feature -> fall back to linear for rendering
            render_mode = "linear"
            clim = self._feature_scale_clim[feature]["linear"]
        array = feature + _RANK_SUFFIX if render_mode == "rank" else feature
        cloud.set_active_scalars(array)
        mapper.array_name = array
        mapper.lookup_table.log_scale = (render_mode == "log")
        if clim is not None:
            mapper.scalar_range = clim
        self._set_feature_labels(feature, render_mode)
        self.plotter.render()

    def _set_feature(self, feature: str) -> None:
        """Switch the active feature, re-applying the currently-selected colour-scale mode."""
        self._apply_scale(feature, self._scale_mode)

    def _add_feature_selector(self, size: int = 26, gap: int = 8,
                              label_col: int = 84, pair_offset: int = 140) -> None:
        """Add a toggle button per feature in a single left-edge column, grouped under category headers
        (Geometry / Segment / PLC / Other). Paired measures — the two features sharing a FEATURE_DISPLAY
        colour group — sit on one row as `measure  [method] [method]` (e.g. `area  [simpson] [shoelace]`)
        to save height; singletons show the full name. Clicking one makes it the active feature and
        deselects the others (radio behaviour). Bottom-anchored (survives a resize); reads top-to-bottom.

        `label_col` (px) is the measure-name column width before the first paired button — wide enough to
        clear the widest measure label ("height"). `pair_offset` (px) is the per-method column pitch —
        wide enough that a button plus its (up to ~100 px) method label clears the next button."""
        x = 12
        listed = {m for _, members in SELECTOR_CATEGORIES for m in members}
        other = tuple(f for f in self._feature_names if f not in listed)

        # rows top-to-bottom: a header per non-empty category, then one row per colour group (1-2 features)
        rows: list[tuple] = []
        for name, members in (*SELECTOR_CATEGORIES, ("Other", other)):
            feats = [f for f in members if f in self._feature_names]
            if not feats:
                continue
            rows.append(("header", name))
            by_group: dict[str, list[str]] = {}
            for f in feats:  # group paired measures (same colour group) onto one row, preserving order
                by_group.setdefault(FEATURE_DISPLAY[f][0], []).append(f)
            for group_feats in by_group.values():
                rows.append(("features", group_feats))
            rows.append(("spacer",))
        if rows and rows[-1][0] == "spacer":
            rows.pop()  # no trailing spacer

        self._feature_buttons = []
        idx = 0  # each button's index into self._feature_buttons (for the radio)
        n = len(rows)
        for r, item in enumerate(rows):
            y = 12 + (n - 1 - r) * (size + gap)  # first row highest
            if item[0] == "header":
                self.plotter.add_text(f"{item[1]}:", position=(x, y + 3), font_size=15,
                                      color="cyan", shadow=True)  # bright + shadow -> visible on any bg
            elif item[0] == "features":
                group_feats = item[1]
                if len(group_feats) == 1:  # singleton -> button + full feature name
                    feature = group_feats[0]
                    widget = self.plotter.add_checkbox_button_widget(
                        self._make_feature_callback(feature, idx),
                        value=(feature == self._feature_initial),
                        position=(x + 2, y), size=size, color_on="green", color_off="grey",
                    )
                    self._feature_buttons.append(widget); idx += 1
                    self.plotter.add_text(feature, position=(x + 2 + size + 6, y + 5), font_size=10)
                else:  # paired measure -> measure name once, then a short-labelled button per method
                    group = FEATURE_DISPLAY[group_feats[0]][0]
                    self.plotter.add_text(_GROUP_DISPLAY.get(group, group), position=(x, y + 5), font_size=12)
                    for j, feature in enumerate(group_feats):
                        bx = x + label_col + j * pair_offset  # buttons start after the measure label
                        widget = self.plotter.add_checkbox_button_widget(
                            self._make_feature_callback(feature, idx),
                            value=(feature == self._feature_initial),
                            position=(bx + 2, y), size=size, color_on="green", color_off="grey",
                        )
                        self._feature_buttons.append(widget); idx += 1
                        self.plotter.add_text(_method_label(feature), position=(bx + 2 + size + 4, y + 5), font_size=10)

    def _make_feature_callback(self, feature: str, idx: int):
        """Build the click callback for one feature button: enforce single-selection (radio) and
        switch the active feature. Setting the other buttons' state does not re-fire callbacks."""
        def callback(state: bool) -> None:
            for j, widget in enumerate(self._feature_buttons):
                widget.GetRepresentation().SetState(1 if j == idx else 0)
            self._set_feature(feature)
        return callback

    def _add_scale_selector(self, size: int = 26, gap: int = 8) -> None:
        """Add a radio group of colour-scale buttons pinned to the bottom-RIGHT corner (linear / log /
        clip / rank), single-select like the feature selector; clicking one re-scales the active feature
        live.

        The group sits bottom-right (pixels from the lower-left window corner) so it clears the tall
        left-edge feature panel. VTK button widgets take fixed pixel positions with no right-edge anchor,
        so a window-resize (`ConfigureEvent`) observer re-places the buttons + labels to track the right
        edge (`_reposition_scale_selector`). The vertical y's are bottom-anchored and never change.
        """
        n = len(_SCALE_MODES)
        self._scale_size = size
        self._scale_button_ys = [12 + (n - 1 - j) * (size + gap) for j in range(n)]  # linear top .. rank bottom
        self._scale_header_y = 12 + n * (size + gap) + 4
        x = self._scale_selector_x()
        self._scale_header_actor = self.plotter.add_text("scale:", position=(x, self._scale_header_y),
                                                         font_size=11)
        self._scale_buttons = []
        self._scale_label_actors = []
        for j, mode in enumerate(_SCALE_MODES):
            y = self._scale_button_ys[j]
            widget = self.plotter.add_checkbox_button_widget(
                self._make_scale_callback(mode, j),
                value=(mode == self._scale_mode),
                position=(x, y), size=size, color_on="blue", color_off="grey",
            )
            self._scale_buttons.append(widget)
            self._scale_label_actors.append(
                self.plotter.add_text(mode, position=(x + size + 8, y + 4), font_size=10))
        if self.plotter.iren is not None:  # keep it pinned to the corner on resize (needs an interactor)
            self.plotter.iren.add_observer("ConfigureEvent", self._on_window_resize)

    def _scale_selector_x(self) -> int:
        """Left x (pixels) of the bottom-right scale group, tracking the current window width."""
        return int(self.plotter.window_size[0]) - 130  # column width leaves room for labels to its right

    def _reposition_scale_selector(self) -> None:
        """Re-place the scale buttons + labels at the current window's right edge (y is unchanged)."""
        x, size = self._scale_selector_x(), self._scale_size
        self._scale_header_actor.SetPosition(x, self._scale_header_y)
        for widget, actor, y in zip(self._scale_buttons, self._scale_label_actors, self._scale_button_ys):
            widget.GetRepresentation().PlaceWidget([x, x + size, y, y + size, 0.0, 0.0])
            actor.SetPosition(x + size + 8, y + 4)

    def _on_window_resize(self, *args) -> None:
        """ConfigureEvent handler: keep the scale group pinned to the bottom-right corner."""
        self._reposition_scale_selector()
        self.plotter.render()

    def _make_scale_callback(self, mode: str, idx: int):
        """Build the click callback for one scale button: enforce single-selection (radio) and
        re-scale the active feature to `mode`. Setting the other buttons' state does not re-fire."""
        def callback(state: bool) -> None:
            for j, widget in enumerate(self._scale_buttons):
                widget.GetRepresentation().SetState(1 if j == idx else 0)
            self._apply_scale(self._active_feature, mode)
        return callback


def get_profile_points_for_plot(profiles: list[profileData], profile_step: int = 1,
                                point_step: int = 1, want_flat: bool | None = None,
                                category: str | None = None):
    """Build one (N, 3) point array per profile (height goes in the plot's y slot).

    Returns one entry per profile so the result stays index-aligned with the print
    path; skipped profiles (every profile not on profile_step) and empty profiles
    contribute an empty (0, 3) array, which the plotter skips. point_step subsamples
    points within each kept profile. If want_flat is set, only profiles whose gap status
    matches it are kept — a "gap" being a profile *not* in a cleaned filament segment
    (`~isSegment`); None = no such filter. If category is "floor" or "profile", only points
    of that category are kept (uses `floorMask`; ignored when it is None).
    """
    points = []
    for i, profile in enumerate(profiles):
        include = i % profile_step == 0 and profile.x.shape[0] > 0
        if want_flat is not None and (not _is_segment(profile)) != want_flat:  # gap = not a segment
            include = False
        if include:
            xs, zs = profile.x, profile.z
            if category is not None and profile.floorMask is not None:
                keep = profile.floorMask if category == "floor" else ~profile.floorMask
                xs, zs = xs[keep], zs[keep]
            xs = xs[::point_step]
            zs = zs[::point_step]
            points.append(np.column_stack((xs, zs, np.zeros_like(xs))))
        else:
            points.append(np.empty((0, 3)))
    return points

def width_point_arrays(profiles: list[profileData], idx_attr: str):
    """One (2, 3) point array per profile from a 2-index attribute ("widthFlankIdx" or "widthOuterIdx").

    One entry per profile keeps alignment with the print path; a profile without exactly two
    indices contributes an empty (0, 3) array (skipped on plot).
    """
    out = []
    for p in profiles:
        idx = getattr(p, idx_attr)
        if idx is not None and len(idx) == 2:
            i0, i1 = int(idx[0]), int(idx[1])
            out.append(np.array([[p.x[i0], p.z[i0], 0], [p.x[i1], p.z[i1], 0]]))
        else:
            out.append(np.empty((0, 3)))
    return out

def _finite_clim(values: np.ndarray) -> "list[float] | None":
    """[min, max] over the finite entries of `values`, or None if none are finite.

    A degenerate (min == max) range is widened by 1 so the colormap is not singular.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    return [lo, hi + 1.0] if lo == hi else [lo, hi]

def _positive_clim(values: np.ndarray) -> "list[float] | None":
    """[smallest positive, max] over the finite entries — the range for a log10 colour scale.

    Returns None if no finite value is > 0 (the caller then falls back to a linear scale). A
    degenerate (min == max) range is widened multiplicatively so the log colormap is not singular.
    """
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return None
    lo, hi = float(positive.min()), float(finite.max())
    return [lo, lo * 10.0] if hi <= lo else [lo, hi]

def _percentile_clim(values: np.ndarray, pct: tuple[float, float]) -> "list[float] | None":
    """[low, high] percentile range (outliers clip to the ends), over the DISTINCT finite values.

    Same robust idea as `featureComparison._scale_range`, but percentiles are taken over the distinct
    values, so each per-profile value (≈ each filament segment) counts once regardless of how many
    points hold it — otherwise one big but long segment (many points) could not be clipped out. None
    if nothing is finite; a degenerate range is widened by 1 (as in `_finite_clim`).
    """
    finite = np.unique(values[np.isfinite(values)])  # unique -> per-segment weighting, size-unbiased
    if finite.size == 0:
        return None
    lo, hi = (float(v) for v in np.percentile(finite, pct))
    return [lo, hi + 1.0] if lo == hi else [lo, hi]

def _rank01(values: np.ndarray) -> np.ndarray:
    """Dense rank of each finite value mapped to [0, 1]; NaN entries stay NaN.

    Tie-safe: identical values get an identical rank (so a profile's broadcast points keep one
    colour), and distinct values are spread evenly regardless of how many points hold them — the
    point of the rank scale. A single distinct value maps to 0.0.
    """
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    v = values[finite]
    if v.size == 0:
        return out
    _, inverse = np.unique(v, return_inverse=True)  # inverse in [0, n_unique - 1], ties share an index
    n_unique = int(inverse.max()) + 1
    out[finite] = inverse / max(n_unique - 1, 1)
    return out

def line_points_from_floorSides(profiles: list[profileData]):
    """Endpoints of each profile's floor baseline from its stored fit (m, b).

    Reuses the fit cached by rotate_and_shift_uniform (no refit here). One entry per
    profile keeps alignment with the print path; a missing fit falls back to z = 0.
    """
    linesPoints = []
    for p in profiles:
        m = p.m if p.m is not None else 0.0
        b = p.b if p.b is not None else 0.0
        p0 = (p.x[0], m * p.x[0] + b, 0)
        p1 = (p.x[-1], m * p.x[-1] + b, 0)
        linesPoints.append((p0, p1))
    return linesPoints

def line_points_from_zero(profiles: list[profileData]):
    """Endpoints of the flat z = 0 line on every profile (the uniform leveling target).

    One entry per profile keeps alignment with the print path. Height (z) goes in the
    plot's y slot, so a levelled floor sitting at z = 0 lines up with this reference.
    """
    return [((p.x[0], 0.0, 0), (p.x[-1], 0.0, 0)) for p in profiles]

#TODO: currently, print path is in xz plane and profile height in y plane
# --> this is confusing --> change profile output to y for height
# also think about unit and label all unit dep. empirical constants
def compute_print_path_and_angle(distances):
    
    # path parameters
    path_radius = 5000.0 # radius of the curved sweep in XY plane
    totalCurveDist = path_radius * np.pi
    totalStraightDist = 80000

    #initializations
    cx=cz=0
    transitionPoint = np.array([0,0,0])
    currentPoint = np.array([0,0,0])
    addedStraightDist = 0
    addedAngledDist = 0
    zDir = 1
    movingStraight = True
    
    pathpoints=[]
    tiltAngles=[]

    for i in range(len(distances)):
        if movingStraight:
            addedStraightDist = addedStraightDist + distances[i]
            if addedStraightDist < totalStraightDist: # straight path points
                currentPoint = transitionPoint + np.array([0, 0, zDir*addedStraightDist])
            else: # first path point on curve
                movingStraight = False
                transitionPoint = transitionPoint + np.array([0, 0, zDir*totalStraightDist])

                addedAngledDist = addedStraightDist - totalStraightDist
                phi = (addedAngledDist / totalCurveDist) * np.pi
                cx = path_radius * np.cos(phi) - path_radius
                cz = zDir * path_radius * np.sin(phi)
                currentPoint = transitionPoint + np.array([cx, 0, cz])
        else: # curved path points
            addedAngledDist = addedAngledDist + distances[i]
            if addedAngledDist < totalCurveDist:
                phi = (addedAngledDist / totalCurveDist) * np.pi
                cx = path_radius * np.cos(phi) - path_radius
                cz = zDir*path_radius * np.sin(phi)
                currentPoint = transitionPoint + np.array([cx, 0, cz])
            else: # first path point on straight after curve
                movingStraight = True
                transitionPoint = transitionPoint - np.array([2*path_radius,0,0])
                addedStraightDist = addedAngledDist - totalCurveDist

                zDir = -zDir
                currentPoint = transitionPoint + np.array([0, 0, zDir*addedStraightDist])

        if movingStraight:
            if zDir == 1: # straight path, direction up
                alpha = 0
            else: # straight path, direction down
                alpha = np.pi 
        else:
            if zDir == 1: # curved path, clockwise
                alpha = -zDir*phi
            else: # curved path, anti-clockwise
                alpha = np.pi-zDir*phi
        
        pathpoints.append(currentPoint)
        tiltAngles.append(alpha)
    return pathpoints, tiltAngles
    