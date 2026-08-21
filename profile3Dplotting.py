from typing import cast

import numpy as np
import pyvista as pv
from plcData import ALL_PLC_COLUMNS
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
    # segmentVolume: area-unit * distance-unit -> cm^3 (1e-4 mm^2 * 0.01 mm = 1e-6 mm^3 = 1e-9 cm^3), its own
    # colour group. (sliceVolume, the per-profile slab, is a backend-only field — not shown in any plot.)
    "segmentVolume":    ("segmentVolume", 1e-9, "cm^3"),
    # Run lengths along the print path: profile units -> mm (0.01). Own colour groups (a macro cm-scale
    # length, unlike the mm-scale bead widths). defectLength lives on pure-floor profiles, so it only
    # shows in a floor-category heat-map (`plot_feature_heatmap(..., category="floor")`).
    "segmentLength":    ("segmentLength", 0.01, "mm"),
    "defectLength":     ("defectLength",  0.01, "mm"),
    # Per-segment shape features (measure_segment_shape), broadcast over the segment. Already in physical
    # units (factor 1.0), each its own colour group. NaN off a rupture / on a too-short segment.
    # Body thinning family: the taper measured on area / width (widthOuter) / height (heightP95) — a rate
    # (%/mm; each its own colour scale) + a steadiness (Spearman, -1..1). In the heat-map selector these six
    # sit behind the `bodyThinning` parent (EXPANDER_GROUPS); in the 2D plots they are ordinary features.
    "segmentBodyAreaThinning":    ("segmentBodyAreaThinning", 1.0, "%/mm"),
    "segmentBodyWidthThinning":   ("segmentBodyWidthThinning", 1.0, "%/mm"),
    "segmentBodyHeightThinning":  ("segmentBodyHeightThinning", 1.0, "%/mm"),
    "segmentBodyAreaSteadiness":  ("segmentBodyAreaSteadiness", 1.0, ""),
    "segmentBodyWidthSteadiness": ("segmentBodyWidthSteadiness", 1.0, ""),
    "segmentBodyHeightSteadiness":("segmentBodyHeightSteadiness", 1.0, ""),
    "segmentCriticalArea":      ("segmentCriticalArea", 1.0, "mm^2"),
    "segmentRuptureLength":     ("segmentRuptureLength", 1.0, "mm"),
    "segmentHeadOvershoot":     ("segmentHeadOvershoot", 1.0, "%"),
    "segmentRuptures":          ("segmentRuptures", 1.0, ""),
    "segmentSection":           ("segmentSection", 1.0, ""),   # per-profile phase code (1/2/3); debug view
    "segmentShapeStatus":       ("segmentShapeStatus", 1.0, ""),  # per-segment sort-out reason (0-6); debug view
    # Per-profile 0/1 segmentation flags (share one 0-1 colour group; 1 = filament/segment = high colour):
    # isNotFlat = raw "has filament points", isSegment = in a cleaned segment, isContinuousFilament = a
    # segment run too long to be discrete. Coloured over ALL points so bridged floor-only gaps are visible.
    # In the heat-map selector they sit behind the `segFlags` parent (EXPANDER_GROUPS); in the 2D plots they
    # are ordinary Segment features.
    "isNotFlat":            ("segFlag", 1.0, ""),
    "isSegment":            ("segFlag", 1.0, ""),
    "isContinuousFilament": ("segFlag", 1.0, ""),
    # PLC machine-log channels (joined by timestamp) + derived ones (e.g. pipePressureDifference):
    # each its own colour group, since their magnitudes differ widely. Shown in the PLC's native
    # engineering units (factor 1.0; the unit label is left blank as the units aren't recorded in the CSV).
    **{name: (name, 1.0, "") for name in ALL_PLC_COLUMNS},
}

# Discrete/categorical features: their values are CODES for distinct categories, not a continuous scale, so
# they are drawn with a distinct-colour (categorical) colormap + a checkbox panel instead of the gradient
# colour bar (a viridis ramp over the codes would imply a false ordering). Map: feature -> {code: label}.
# Codes are assumed contiguous (lo..hi); the palette below colours code `lo+i` with entry i.
CATEGORICAL_FEATURES: dict[str, dict[int, str]] = {
    "segmentShapeStatus":   {0: "kept", 1: "too short", 2: "continuous", 3: "degenerate",
                             4: "tiny body", 5: "high width", 6: "no rupture"},
    # Only `segmentShapeStatus` (the sort-out reason) uses the multicolour checkbox grey-out panel — its
    # values are unordered category codes. Everything else (incl. the 0/1 flags, the phase code, and the
    # bodyThinning members) keeps the plain gradient rendering (viridis + colour bar) and is NOT listed here.
}
# Distinct qualitative colours (RGB 0-1), one per category index; entry 0 (green) reads as the "normal" code
# (segmentShapeStatus 0 = kept). Off-category points (NaN) render grey (the LUT's nan_color).
_CATEGORICAL_PALETTE = [
    (0.20, 0.63, 0.17), (0.89, 0.10, 0.11), (0.22, 0.49, 0.72), (1.00, 0.55, 0.00),
    (0.60, 0.31, 0.64), (0.55, 0.34, 0.16), (0.95, 0.80, 0.20), (0.30, 0.75, 0.75),
]
# A categorical feature is shown with a checkbox panel (one toggle per code, coloured to match) in place of
# the gradient colour bar; unchecking a code greys those points out (remapped to NaN in a companion mask
# array). Slot j always represents the j-th code, so its fixed button colour = palette[j] for every feature.
_CATMASK_SUFFIX = "__catmask"
_MAX_CATEGORIES = max(len(v) for v in CATEGORICAL_FEATURES.values())

# Heat-map colour-scale modes, cycled live by the scale button (see plottingClass._add_scale_button).
# "linear" (default) keeps the full min-max range; the others tame an outlier that would otherwise
# squash every smaller value into one end of the colormap. (Ignored for CATEGORICAL_FEATURES.)
_SCALE_MODES = ("linear", "log", "clip", "rank")
CLIP_PERCENTILES = (2.0, 98.0)  # "clip" mode maps this percentile range to the colormap (outliers saturate)
_RANK_SUFFIX = "__rank"          # per-feature companion array holding the [0, 1] dense rank (rank mode)

# Heat-map point set per feature: the unified heat-map prebuilds one cloud per distinct point set and
# swaps the visible one when the active feature changes. These features live on floor/gap profiles (no
# filament points) so they are coloured over ALL points; every other feature colours the filament points.
# `segmentSection` (the startup/body/rupture phase code) is coloured over ALL points too, so each segment's
# phase bands show full-width in the print context (the off-segment floor is NaN = the NaN colour).
_ALL_POINT_FEATURES = frozenset({"isSegment", "isNotFlat", "isContinuousFilament", "defectLength",
                                 "segmentSection", "segmentShapeStatus"})


def _feature_pointset(feature: str) -> "str | None":
    """Which points a feature is coloured on: None = all points, "profile" = filament points."""
    return None if feature in _ALL_POINT_FEATURES else "profile"


# Heat-map selector "expander" groups: a parent button (left panel) that, when active, reveals its member
# features as a plain-gradient radio sub-panel at the bottom-centre. Each member is rendered exactly like a
# normal gradient feature — same green/grey button style and its own colour scale — NOT the categorical
# grey-out panel (which stays exclusive to segmentShapeStatus, the only CATEGORICAL_FEATURES member; that
# path rebuilds a mask on every toggle and is laggier). The members are ordinary features elsewhere: the 2D
# plots list them individually under Segment (they are in SELECTOR_CATEGORIES); only the 3D heat-map selector
# collapses each group to its parent button. Member order here sets the sub-panel's top-to-bottom order.
EXPANDER_GROUPS: dict[str, tuple[str, ...]] = {
    "bodyThinning": ("segmentBodyAreaThinning", "segmentBodyWidthThinning", "segmentBodyHeightThinning",
                     "segmentBodyAreaSteadiness", "segmentBodyWidthSteadiness", "segmentBodyHeightSteadiness"),
    "segFlags": ("isNotFlat", "isSegment", "isContinuousFilament"),
}
_MEMBER_TO_PARENT = {m: p for p, members in EXPANDER_GROUPS.items() for m in members}
_MAX_MEMBERS = max(len(v) for v in EXPANDER_GROUPS.values())
# Short sub-panel button labels (member -> label); the parent button itself carries the group name.
_MEMBER_LABEL = {
    "segmentBodyAreaThinning": "areaThinning", "segmentBodyWidthThinning": "widthThinning",
    "segmentBodyHeightThinning": "heightThinning", "segmentBodyAreaSteadiness": "areaSteadiness",
    "segmentBodyWidthSteadiness": "widthSteadiness", "segmentBodyHeightSteadiness": "heightSteadiness",
    "isNotFlat": "notFlat", "isSegment": "isSeg", "isContinuousFilament": "contFil",
}


def _method_label(feature: str) -> str:
    """Short button label for a paired feature: the method suffix after its colour-group name
    (widthFlank -> 'flank', areaSimpson -> 'simpson'), or the full name when it has no such prefix."""
    group = FEATURE_DISPLAY[feature][0]
    if feature.lower().startswith(group.lower()) and len(feature) > len(group):
        suffix = feature[len(group):]
        return suffix[0].lower() + suffix[1:]
    return feature


def _selector_tokens(feats: list[str]) -> list[str]:
    """Collapse expander-group members in `feats` to a single parent token at the first member's position,
    leaving non-member features unchanged. Used only by the 3D heat-map selector so a group of members
    (e.g. the bodyThinning family) shows as ONE parent button that reveals its members at the bottom-centre."""
    tokens: list[str] = []
    seen: set[str] = set()
    for f in feats:
        parent = _MEMBER_TO_PARENT.get(f)
        if parent is None:
            tokens.append(f)
        elif parent not in seen:
            seen.add(parent)
            tokens.append(parent)
    return tokens


# Feature-selector button-panel categories (grouping + headers only; independent of the FEATURE_DISPLAY
# colour groups and the point-set groups above). Features not in any list fall under "Other". Public so the
# 2D feature-comparison selector (featureComparison.py) groups identically — single source of truth (§11).
SELECTOR_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Geometry", ("widthFlank", "widthOuter", "heightP95", "heightSmooth", "areaSimpson", "areaShoelace")),
    # NOTE the body-thinning members and the 0/1 flags are listed here individually (so the 2D plots group
    # them under Segment), but the 3D heat-map selector collapses each EXPANDER_GROUP to a single parent
    # button (bodyThinning / segFlags) via `_selector_tokens`. Member ORDER here places the parent button.
    ("Segment", ("segmentVolume", "segmentLength", "defectLength",                    # run aggregates (top row)
                 "segmentHeadOvershoot",                                               # startup phase
                 "segmentBodyAreaThinning", "segmentBodyWidthThinning", "segmentBodyHeightThinning",       # body thinning \
                 "segmentBodyAreaSteadiness", "segmentBodyWidthSteadiness", "segmentBodyHeightSteadiness",  # + steadiness -> bodyThinning parent
                 "segmentCriticalArea", "segmentRuptureLength", "segmentRuptures",    # rupture phase
                 "isNotFlat", "isSegment", "isContinuousFilament",                    # seg flags -> segFlags parent
                 "segmentShapeStatus", "segmentSection")),                            # debug: sortout / phase
    ("PLC", ALL_PLC_COLUMNS),
)

# Selector UI overrides, decoupled from the FEATURE_DISPLAY colour groups (which set each feature's heat-map
# clim). feature -> (row group, short button label): features in the same row group render on ONE row
# `rowlabel [btn:label] [btn:label] …` (like the width/area pairs) but keep their OWN colour scale. Used to
# pack the many segment-shape features into per-phase rows so the panel doesn't overflow the window.
FEATURE_UI: dict[str, tuple[str, str]] = {
    "segmentVolume":            ("runs", "seg vol"),     # run aggregates -> top row of the Segment category
    "segmentLength":            ("runs", "seg len"),
    "defectLength":             ("runs", "defect"),
    "segmentHeadOvershoot":     ("startup", "overshoot"),
    "bodyThinning":             ("body", "Thinning"),      # parent token -> expands to the thinning/steadiness members ("body" is the row label)
    "segmentCriticalArea":      ("rupture", "critArea"),   # cross-section at the rupture start (segmentRuptures
    "segmentRuptureLength":     ("rupture", "length"),     # dropped from the heat map -> read it off `sortout`)
    "segFlags":                 ("debug", "segFlags"),    # parent token -> expands to notFlat/isSeg/contFil
    "segmentShapeStatus":       ("debug", "sortout"),     # per-segment sort-out reason (0-6)
    "segmentSection":           ("debug", "phase"),       # per-profile phase code (1/2/3)
}
# PLC channels are packed a few per row with short labels so the (many) channels don't overflow the panel.
_PLC_LABEL: dict[str, str] = {
    "mortarPumpFlow": "mortarFlow", "pressurePipeEnd": "pPipeEnd", "pressurePipeStart": "pPipeStart",
    "pressurePrintHead": "pPrintHead", "printHeadMixxingSpeed": "mixSpeed", "printHeadTorque": "torque",
    "rollerbandHeight": "rbHeight", "rollerbandSpeed": "rbSpeed", "viscoPump1_VMAflow": "visco1",
    "viscoPump2_AcceleratorFlow": "visco2", "pipePressureDifference": "pPipeDiff",
}
_PLC_PER_ROW = 2       # PLC channels packed this many per selector row
_SELECTOR_CHAR_PX = 11  # approx px per character (button labels, font 10) for sizing each row's button pitch


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
        widget = self.plotter.add_camera_orientation_widget()
        try:  # default anchor is upper-right, which now clashes with the sub-panels -> move it to upper-left
            widget.GetRepresentation().AnchorToUpperLeft()
        except AttributeError:
            pass
        self.plotter.show()

    def plot(self, profiles: list[profileData], plotSubject:str, colour:str, size=5,
             profile_step: int = 1, point_step: int = 1, flat_colour: str | None = None,
             category: str | None = None, spheres: bool = False) -> "object | None":
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

        Returns the added actor (or a list of actors for the two-colour `flat_colour` case, or None if
        nothing was drawn) so the caller can toggle its visibility, e.g. via `add_layer_toggles`.
        """
        match plotSubject:
            case "profile":
                if flat_colour is None:
                    return self.add_3d_points_to_plot(
                        get_profile_points_for_plot(profiles, profile_step, point_step, category=category), colour, size)
                actors = [
                    self.add_3d_points_to_plot(
                        get_profile_points_for_plot(profiles, profile_step, point_step, want_flat=False, category=category), colour, size),
                    self.add_3d_points_to_plot(
                        get_profile_points_for_plot(profiles, profile_step, point_step, want_flat=True, category=category), flat_colour, size),
                ]
                return [a for a in actors if a is not None]
            case "baseline":
                return self.add_lines_to_plot(line_points_from_floorSides(profiles), colour)
            case "zeroBaseline":
                # flat z = 0 reference on each profile (the uniform leveling target)
                return self.add_lines_to_plot(line_points_from_zero(profiles), colour)
            case "widthFlankPoints":
                # slope-peak width method: mark the two flank-foot points (from `widthFlankIdx`)
                return self.add_3d_points_to_plot(width_point_arrays(profiles, "widthFlankIdx"), colour, size, spheres=spheres)
            case "widthOuterPoints":
                # filament-edge width method: mark the two outer filament points (from `widthOuterIdx`)
                return self.add_3d_points_to_plot(width_point_arrays(profiles, "widthOuterIdx"), colour, size, spheres=spheres)
        return None

    def add_3d_points_to_plot(self, points: list, colour: str = 'green', point_size: int = 5,
                              spheres: bool = False) -> "object | None":
        # Collect every profile's transformed points and add them as a single actor, placed along the
        # shared physical print path (self.pathPoints, built in __init__). One add_points call instead
        # of one per profile is far faster for many profiles. Returns the actor (None if no points).
        transformed = []
        for i, prof in enumerate(points):
            if prof.shape[0] > 0:
                prof = prof @ self.rotation_matrices[i].T  # rotate to match path direction
                transformed.append(self.pathPoints[i] + prof)

        if transformed:
            cloud = pv.PolyData(np.vstack(transformed))
            if not spheres:  # dense cloud: downsample; markers (spheres) stay full so both show
                cloud = self._maybe_voxel(cloud)
            return self.plotter.add_points(cloud, color=colour, point_size=point_size, render_points_as_spheres=spheres)
        return None

    def add_lines_to_plot(self, linePoints: list, colour: str = 'green') -> "object | None":
        # Collect every line's two transformed endpoints and add them all as a single mesh, placed
        # along the shared physical print path. One add_mesh call is far faster for many profiles.
        # Returns the line actor (None if there were no lines).
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
            return self.plotter.add_mesh(lines, color = colour, line_width=5)
        return None

    def add_layer_toggles(self, layers: "list[tuple[str, object, str, bool]]",
                          size: int = 26, gap: int = 10) -> None:
        """Add a left-edge checkbox per named layer to show/hide it live (independent multi-select).

        `layers`: list of `(label, actors, colour, initial_on)` — `actors` is one actor or a list of actors
        (as returned by `plot`; `None` entries are ignored), `colour` tints the label to match the layer, and
        `initial_on` sets both the checkbox and the actors' initial visibility. Bottom-anchored on the left
        edge (below the camera-orientation gizmo). Needs a live interactor (interactive window only).
        """
        x, n = 12, len(layers)
        self._toggle_buttons = []  # keep the widget refs alive for the lifetime of the plotter
        for j, (label, actors, colour, on) in enumerate(layers):
            acts = [a for a in (actors if isinstance(actors, list) else [actors]) if a is not None]
            for a in acts:
                a.SetVisibility(on)
            y = 12 + (n - 1 - j) * (size + gap)  # first layer highest
            widget = self.plotter.add_checkbox_button_widget(
                self._make_toggle_callback(acts), value=on,
                position=(x, y), size=size, color_on="green", color_off="grey")
            self._toggle_buttons.append(widget)
            # shadow keeps a pale label (yellow / grey) legible against a light background
            self.plotter.add_text(label, position=(x + size + 6, y + 4), font_size=12, color=colour, shadow=True)

    def _make_toggle_callback(self, actors: list):
        """Click handler for one layer checkbox: show/hide that layer's actor(s)."""
        def callback(state: bool) -> None:
            for a in actors:
                a.SetVisibility(state)
            self.plotter.render()
        return callback

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
        geometry + PLC channels colour the **filament** points, while the 0/1 segmentation flags and
        `defectLength` (which live on floor/gap profiles) colour **all** points. Switching within a
        point set only repoints the mapper (instant); switching across point sets swaps which cloud is
        drawn, so the displayed points change (filament-only ↔ all) while the camera is kept. NaN feature
        values render in the NaN colour. Interactive-window only (buttons need a live VTK interactor).

        Some features are grouped behind an EXPANDER_GROUPS parent button (bodyThinning / segFlags): the
        parent sits in the left panel and, when active, reveals its members as a plain-gradient radio
        sub-panel at the bottom-centre (`_add_member_selector`).
        """
        self._build_feature_cloud(profiles, features, initial, cmap, point_size, profile_step, point_step)
        self._add_feature_selector()
        self._add_scale_selector()
        self._add_category_selector()  # categorical grey-out checkboxes (shown only for a categorical feature)
        self._add_member_selector()    # expander sub-panel (shown only for a bodyThinning/segFlags member)

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
        self._gradient_luts: dict = {}   # per point set: the original (viridis) LUT, restored for continuous features
        self._cat_luts: dict = {}        # per categorical feature: a cached discrete LUT
        self._scalar_bar_ps = "<none>"   # which point set the single scalar bar is currently tied to
        self._visible_ps = "<none>"      # which point set's actor is currently visible
        self._bar_active = False         # is the gradient colour bar currently shown
        self._cat_buttons: list = []     # category-toggle checkbox widgets (built by _add_category_selector)
        self._cat_labels: list = []      # their coloured text labels
        self._cat_slot_codes: list = []  # code currently assigned to each checkbox slot (None = unused)
        self._cat_hidden: set = set()    # category codes currently greyed out
        self._cat_panel_active = False   # is the category checkbox panel currently shown
        self._sub_buttons: list = []     # expander sub-panel radio buttons (built by _add_member_selector)
        self._sub_labels: list = []      # their text labels
        self._sub_slot_features: list = []  # member feature assigned to each sub-panel slot (None = unused)
        self._member_panel_active = False   # is the expander sub-panel currently shown
        self._feature_buttons: list = []    # left-panel feature/parent buttons (built by _add_feature_selector)
        self._button_targets: list = []     # parallel to _feature_buttons: ("feature", name) | ("parent", name)
        self._group_current = {p: EXPANDER_GROUPS[p][0] for p in EXPANDER_GROUPS}  # last member picked per group

        by_pointset: dict = {}
        for f in features:
            by_pointset.setdefault(self._feature_pointset[f], []).append(f)
        for pointset, feats in by_pointset.items():
            built = self._build_one_cloud(profiles, feats, cmap, point_size, profile_step, point_step, pointset)
            if built is not None:
                self._clouds[pointset], self._mappers[pointset], self._actors[pointset], scale_clim = built
                self._feature_scale_clim.update(scale_clim)
                self._gradient_luts[pointset] = self._mappers[pointset].lookup_table  # keep the gradient LUT
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

    def _apply_scale(self, feature: str, mode: str) -> None:
        """Colour `feature` live. Continuous features use the viridis gradient + the shared colour bar with
        the requested scale `mode` (linear/log/clip/rank; an unavailable mode renders linear but is kept so
        cycling advances). Categorical features (CATEGORICAL_FEATURES) use a distinct-colour discrete LUT + a
        checkbox panel instead, and the scale mode is ignored. Swaps the visible point-set actor on change."""
        self._active_feature = feature
        self._scale_mode = mode
        pointset = self._feature_pointset[feature]
        if pointset not in self._clouds:  # this feature's cloud is empty -> nothing to show
            return
        cloud, mapper = self._clouds[pointset], self._mappers[pointset]
        if self._visible_ps != pointset:  # point set changed -> show only this cloud's actor
            for ps, actor in self._actors.items():
                actor.SetVisibility(ps == pointset)
            self._visible_ps = pointset

        self.plotter.add_text(feature, name="feature_title", position="upper_edge", font_size=16)
        if feature in CATEGORICAL_FEATURES:
            self._apply_categorical(feature, mapper)
        else:
            self._apply_gradient(feature, mode, cloud, mapper)
        self._sync_left_radio(feature)  # keep the left panel's parent/feature buttons in sync with the active feature
        self.plotter.render()

    def _apply_gradient(self, feature: str, mode: str, cloud, mapper) -> None:
        """Continuous feature: the viridis gradient LUT + the shared colour bar, coloured by the value (or
        rank) array with the mode's colour range and log flag. An expander-group member also shows its
        bottom-centre sub-panel; any other feature hides it."""
        self._hide_category_panel()
        if feature in _MEMBER_TO_PARENT:
            self._group_current[_MEMBER_TO_PARENT[feature]] = feature
            self._show_member_panel(feature)
        else:
            self._hide_member_panel()
        mapper.lookup_table = self._gradient_luts[self._visible_ps]  # restore the gradient LUT
        clim = self._feature_scale_clim[feature][mode]
        render_mode = mode
        if clim is None:  # mode unavailable for this feature -> render linear (mode kept for cycling)
            render_mode = "linear"
            clim = self._feature_scale_clim[feature]["linear"]
        array = feature + _RANK_SUFFIX if render_mode == "rank" else feature
        cloud.set_active_scalars(array)
        mapper.array_name = array
        mapper.lookup_table.log_scale = (render_mode == "log")
        if clim is not None:
            mapper.scalar_range = clim
        self._ensure_scalar_bar(mapper)
        self._set_unit_label(self._scale_label(feature, render_mode))

    def _apply_categorical(self, feature: str, mapper) -> None:
        """Categorical feature: a discrete distinct-colour LUT keyed by the integer code + a bottom-centre
        checkbox panel (one toggle per category, coloured to match). Unchecking a category greys it out
        (remapped to NaN in a companion mask array). The gradient colour bar + any expander sub-panel are hidden."""
        self._remove_scalar_bar()
        self._hide_member_panel()
        items = sorted(CATEGORICAL_FEATURES[feature].items())  # [(code, label), ...]; codes contiguous
        lo, hi = items[0][0], items[-1][0]
        mapper.lookup_table = self._categorical_lut(feature, lo, hi, len(items))
        mapper.scalar_range = (lo - 0.5, hi + 0.5)
        self._cat_hidden = set()             # all categories visible on (re)entry
        self._show_category_panel(feature)   # bottom-centre checkboxes + coloured labels
        self._set_cat_mask()                 # point the cloud at the (initially full) masked scalar
        self._set_unit_label("")             # the panel is the legend; clear the bar's unit text

    def _categorical_lut(self, feature: str, lo: int, hi: int, n: int):
        """A cached discrete LookupTable: `n` distinct colours over the code range [lo-0.5, hi+0.5], so each
        integer code renders as one solid palette colour (index = code - lo)."""
        if feature not in self._cat_luts:
            from matplotlib.colors import ListedColormap
            colors = [_CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)] for i in range(n)]
            lut = pv.LookupTable(cmap=ListedColormap(colors), n_values=n)
            lut.scalar_range = (lo - 0.5, hi + 0.5)
            lut.nan_color = "lightgray"
            self._cat_luts[feature] = lut
        return self._cat_luts[feature]

    def _ensure_scalar_bar(self, mapper) -> None:
        """Show the shared gradient colour bar tied to `mapper` (re-tie if the point set changed or it was
        hidden for a categorical feature)."""
        if self._bar_active and self._scalar_bar_ps == self._visible_ps:
            return
        if self._bar_active:
            self.plotter.remove_scalar_bar(title="")
        self.plotter.add_scalar_bar(title="", mapper=mapper, label_font_size=14,
                                    position_x=0.33, position_y=0.10, width=0.34, height=0.05)
        self._scalar_bar_ps = self._visible_ps
        self._bar_active = True

    def _remove_scalar_bar(self) -> None:
        """Hide the shared gradient colour bar (for a categorical feature)."""
        if self._bar_active:
            self.plotter.remove_scalar_bar(title="")
            self._bar_active = False
            self._scalar_bar_ps = "<none>"

    # --- side sub-panels (category + expander; upper-RIGHT, top-anchored, so they clear the centre print-
    #     path cloud, the bottom-centre colour bar, and the bottom-right scale selector) -----------------
    def _side_panel_x(self) -> int:
        """Left x (pixels) of the upper-right sub-panels, leaving room for the long member labels before the
        window's right edge (tracks the current window width)."""
        return int(self.plotter.window_size[0]) - 250

    def _side_panel_top_y(self) -> int:
        """Top y (pixels, from the bottom) of the upper-right sub-panels: near the top edge, stacking
        downward, so they sit clear of the cloud and the bottom-right scale selector."""
        return int(self.plotter.window_size[1]) - 70

    def _show_category_panel(self, feature: str) -> None:
        """Assign the active categorical feature's codes to the checkbox pool: position + label the first N
        slots at the upper-right (top-anchored, first code highest) and show them, parking the rest
        off-screen. No-op until the pool exists. Slot j's fixed button colour is palette[j]."""
        if not self._cat_buttons:
            return
        items = sorted(CATEGORICAL_FEATURES[feature].items())
        n, x, size, top_y = len(items), self._side_panel_x(), self._cat_size, self._side_panel_top_y()
        for j, (btn, lbl) in enumerate(zip(self._cat_buttons, self._cat_labels)):
            if j < n:
                code, label = items[j]
                self._cat_slot_codes[j] = code
                y = top_y - j * (size + 8)  # first code highest (top-anchored, stacking downward)
                btn.GetRepresentation().PlaceWidget([x, x + size, y, y + size, 0.0, 0.0])
                btn.GetRepresentation().SetState(0 if code in self._cat_hidden else 1)
                btn.GetRepresentation().SetVisibility(True)
                btn.On()
                lbl.SetInput(label)
                lbl.SetPosition(x + size + 8, y + 4)
                lbl.GetTextProperty().SetColor(*_CATEGORICAL_PALETTE[j % len(_CATEGORICAL_PALETTE)])
                lbl.SetVisibility(True)
            else:
                self._cat_slot_codes[j] = None
                self._hide_one_checkbox(btn, lbl)
        self._cat_panel_active = True

    @staticmethod
    def _hide_one_checkbox(btn, lbl) -> None:
        """Hide one checkbox + label: SetVisibility(False) on the button rep actually hides it (Off/SetEnabled
        and an off-screen PlaceWidget both leave the rep drawn, clamped to (0,0)); Off() stops phantom clicks."""
        btn.GetRepresentation().SetVisibility(False)
        btn.Off()
        lbl.SetVisibility(False)

    def _hide_category_panel(self) -> None:
        """Hide every category checkbox + label (for a continuous feature)."""
        if not self._cat_buttons:
            return
        for btn, lbl in zip(self._cat_buttons, self._cat_labels):
            self._hide_one_checkbox(btn, lbl)
        self._cat_panel_active = False

    def _set_cat_mask(self) -> None:
        """Point the active categorical cloud at a masked scalar: hidden codes -> NaN (grey), the rest keep
        their code (colour). No render (callers render)."""
        feature = self._active_feature
        pointset = self._feature_pointset[feature]
        if pointset not in self._clouds:
            return
        cloud, mapper = self._clouds[pointset], self._mappers[pointset]
        base = np.asarray(cloud[feature], dtype=float)
        masked = np.where(np.isin(base, list(self._cat_hidden)), np.nan, base) if self._cat_hidden else base
        cloud[feature + _CATMASK_SUFFIX] = masked
        cloud.set_active_scalars(feature + _CATMASK_SUFFIX)
        mapper.array_name = feature + _CATMASK_SUFFIX

    def _make_cat_callback(self, slot: int):
        """Click handler for one category checkbox: toggle whether that category's code is greyed out."""
        def callback(state: bool) -> None:
            code = self._cat_slot_codes[slot]
            if code is None:
                return
            self._cat_hidden.discard(code) if state else self._cat_hidden.add(code)
            self._set_cat_mask()
            self.plotter.render()
        return callback

    def _set_unit_label(self, text: str) -> None:
        """Centred text just above the colour bar (unit + scale mode for a gradient; empty for categorical)."""
        actor = self.plotter.add_text(text, name="feature_unit", position=(0.50, 0.17),
                                       viewport=True, font_size=16)
        actor.GetTextProperty().SetJustificationToCentered()

    def _set_feature(self, feature: str) -> None:
        """Switch the active feature, re-applying the currently-selected colour-scale mode."""
        self._apply_scale(feature, self._scale_mode)

    def _add_feature_selector(self, size: int = 26, gap: int = 8, label_col: int = 94) -> None:
        """Add a toggle button per feature in a single left-edge column, grouped under category headers
        (Geometry / Segment / PLC / Other). Related features share one row as `rowlabel [lbl] [lbl] …`
        to save height: FEATURE_DISPLAY colour pairs (width/area) as `measure [method] [method]`, the
        segment-shape features by phase via FEATURE_UI (startup / body / rupture / runs / flags), and the
        PLC channels packed `_PLC_PER_ROW` per row with short labels; other features get a full-name row.
        Clicking one makes it the active feature and deselects the others (radio). Bottom-anchored
        (survives a resize); reads top-to-bottom.

        `label_col` (px) is the row-label column width before the first button. Each row's button pitch is
        sized to fit its widest label (`_SELECTOR_CHAR_PX`), so short-labelled rows stay compact."""
        x = 12
        listed = {m for _, members in SELECTOR_CATEGORIES for m in members}
        other = tuple(f for f in self._feature_names if f not in listed)

        # rows top-to-bottom: a header per category, then feature rows as (row_label_or_None, [(feature, label), …])
        rows: list[tuple] = []
        for name, members in (*SELECTOR_CATEGORIES, ("Other", other)):
            # collapse any expander-group members to their single parent token (bodyThinning / segFlags)
            feats = _selector_tokens([f for f in members if f in self._feature_names])
            if not feats:
                continue
            rows.append(("header", name))
            if name == "PLC":  # pack channels a few per row (short labels, no row label)
                for i in range(0, len(feats), _PLC_PER_ROW):
                    rows.append(("features", None, [(f, _PLC_LABEL.get(f, f)) for f in feats[i:i + _PLC_PER_ROW]]))
            else:
                groups: dict[str, list[tuple[str, "str | None"]]] = {}
                for f in feats:  # group by FEATURE_UI row group, else by FEATURE_DISPLAY colour group
                    key, lbl = FEATURE_UI[f] if f in FEATURE_UI else (FEATURE_DISPLAY[f][0], None)
                    groups.setdefault(key, []).append((f, lbl))
                for key, items in groups.items():
                    if all(lbl is None for _, lbl in items):        # colour-group features (width/area/singletons)
                        if len(items) >= 2:                          # paired measures -> measure + method labels
                            rows.append(("features", key,
                                         [(f, _method_label(f)) for f, _ in items]))
                        else:                                        # singleton -> full name, no row label
                            rows.append(("features", None, [(items[0][0], items[0][0])]))
                    else:                                            # FEATURE_UI phase group -> short labels
                        rows.append(("features", key, list(items)))
            rows.append(("spacer",))
        if rows and rows[-1][0] == "spacer":
            rows.pop()  # no trailing spacer

        self._feature_buttons = []
        self._button_targets = []  # parallel: ("feature", name) | ("parent", name) — drives _sync_left_radio
        n = len(rows)
        for r, item in enumerate(rows):
            y = 12 + (n - 1 - r) * (size + gap)  # first row highest
            if item[0] == "header":
                self.plotter.add_text(f"{item[1]}:", position=(x, y + 3), font_size=15,
                                      color="cyan", shadow=True)  # bright + shadow -> visible on any bg
            elif item[0] == "features":
                _, row_label, items = item
                pitch = size + 8 + int(max(len(lbl) for _, lbl in items) * _SELECTOR_CHAR_PX) + 8
                bx0 = x + (label_col if row_label else 2)
                if row_label:
                    self.plotter.add_text(row_label, position=(x, y + 5), font_size=12)
                for j, (feature, lbl) in enumerate(items):
                    bx = bx0 + j * pitch
                    if feature in EXPANDER_GROUPS:  # parent token: reveals its members at the bottom-centre
                        callback = self._make_parent_callback(feature)
                        on = self._active_feature in EXPANDER_GROUPS[feature]
                        target = ("parent", feature)
                    else:
                        callback = self._make_feature_callback(feature)
                        on = (feature == self._feature_initial)
                        target = ("feature", feature)
                    widget = self.plotter.add_checkbox_button_widget(
                        callback, value=on,
                        position=(bx + 2, y), size=size, color_on="green", color_off="grey",
                    )
                    self._feature_buttons.append(widget)
                    self._button_targets.append(target)
                    self.plotter.add_text(lbl, position=(bx + 2 + size + 4, y + 5), font_size=10)

    def _sync_left_radio(self, feature: str) -> None:
        """Set the left-panel buttons so exactly the active feature (or the parent whose group contains it)
        reads as selected. Called whenever the active feature changes; a no-op before the panel is built."""
        if not getattr(self, "_button_targets", None):
            return
        for widget, (kind, key) in zip(self._feature_buttons, self._button_targets):
            on = (kind == "feature" and key == feature) or (kind == "parent" and feature in EXPANDER_GROUPS[key])
            widget.GetRepresentation().SetState(1 if on else 0)

    def _make_feature_callback(self, feature: str):
        """Build the click callback for one plain feature button: switch the active feature. The left-panel
        radio is re-synced centrally by `_sync_left_radio` (via `_apply_scale`)."""
        def callback(state: bool) -> None:
            self._set_feature(feature)
        return callback

    def _make_parent_callback(self, parent: str):
        """Build the click callback for an expander parent button: activate the group's current member (the
        last one picked, else the first), which reveals the member sub-panel at the bottom-centre."""
        def callback(state: bool) -> None:
            member = self._group_current.get(parent) or EXPANDER_GROUPS[parent][0]
            if self._feature_pointset.get(member) not in self._clouds:  # its cloud is empty -> first built member
                member = next((m for m in EXPANDER_GROUPS[parent]
                               if self._feature_pointset.get(m) in self._clouds), member)
            self._set_feature(member)
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

    def _add_category_selector(self, size: int = 26) -> None:
        """Create the reusable pool of category-toggle checkboxes (bottom-centre, one per category code, up
        to `_MAX_CATEGORIES`). All are parked off-screen; `_show_category_panel` positions + labels the ones
        the active categorical feature needs. Slot j's button colour is fixed to palette[j] (so it always
        matches the j-th category). Needs a live interactor, so it runs after the cloud + other selectors."""
        self._cat_size = size
        self._cat_buttons = []
        self._cat_labels = []
        self._cat_slot_codes = [None] * _MAX_CATEGORIES
        for j in range(_MAX_CATEGORIES):
            widget = self.plotter.add_checkbox_button_widget(
                self._make_cat_callback(j), value=True,
                position=(-100.0, -100.0), size=size,
                color_on=_CATEGORICAL_PALETTE[j % len(_CATEGORICAL_PALETTE)], color_off="darkgray")
            self._cat_buttons.append(widget)
            # shadow so the colour-matched label stays legible where it overlaps a same-colour segment
            lbl = self.plotter.add_text("", position=(-100, -100), font_size=11, shadow=True)
            lbl.SetVisibility(False)
            self._cat_labels.append(lbl)
        self._hide_category_panel()  # park the pool off-screen (else the creation position clamps to 0,0)
        if self._active_feature in CATEGORICAL_FEATURES:  # initial feature is categorical -> show + mask now
            self._show_category_panel(self._active_feature)
            self._set_cat_mask()

    # --- expander sub-panel (bottom-centre; a plain-gradient radio, shown only for a bodyThinning/segFlags
    #     member — the members render exactly like a normal left-panel feature, unlike the categorical panel) -
    def _add_member_selector(self, size: int = 26) -> None:
        """Create the reusable pool of expander sub-panel radio buttons (bottom-centre, up to `_MAX_MEMBERS`).
        All are parked off-screen; `_show_member_panel` positions + labels the ones the active parent needs.
        Plain green/grey (same style as the left buttons). Needs a live interactor, so it runs last."""
        self._member_size = size
        self._sub_buttons = []
        self._sub_labels = []
        self._sub_slot_features = [None] * _MAX_MEMBERS
        for j in range(_MAX_MEMBERS):
            widget = self.plotter.add_checkbox_button_widget(
                self._make_member_callback(j), value=False,
                position=(-100.0, -100.0), size=size, color_on="green", color_off="grey")
            self._sub_buttons.append(widget)
            lbl = self.plotter.add_text("", position=(-100, -100), font_size=11, shadow=True)
            lbl.SetVisibility(False)
            self._sub_labels.append(lbl)
        self._hide_member_panel()  # park the pool off-screen (else the creation position clamps to 0,0)
        if self._active_feature in _MEMBER_TO_PARENT:  # initial feature is a group member -> show it now
            self._show_member_panel(self._active_feature)

    def _show_member_panel(self, feature: str) -> None:
        """Assign the active member's parent-group features to the sub-panel slots: position + label the first
        N at the upper-right (top-anchored, first member highest) with the active one selected (radio),
        parking the rest off-screen. No-op until the pool exists. Members with no built cloud are skipped."""
        if not self._sub_buttons:
            return
        parent = _MEMBER_TO_PARENT[feature]
        members = [m for m in EXPANDER_GROUPS[parent] if self._feature_pointset.get(m) in self._clouds]
        n, x, size, top_y = len(members), self._side_panel_x(), self._member_size, self._side_panel_top_y()
        for j, (btn, lbl) in enumerate(zip(self._sub_buttons, self._sub_labels)):
            if j < n:
                m = members[j]
                self._sub_slot_features[j] = m
                y = top_y - j * (size + 8)  # first member highest (top-anchored, stacking downward)
                btn.GetRepresentation().PlaceWidget([x, x + size, y, y + size, 0.0, 0.0])
                btn.GetRepresentation().SetState(1 if m == feature else 0)
                btn.GetRepresentation().SetVisibility(True)
                btn.On()
                lbl.SetInput(_MEMBER_LABEL.get(m, m))
                lbl.SetPosition(x + size + 8, y + 4)
                lbl.SetVisibility(True)
            else:
                self._sub_slot_features[j] = None
                self._hide_one_checkbox(btn, lbl)
        self._member_panel_active = True

    def _hide_member_panel(self) -> None:
        """Hide every expander sub-panel button + label (for a non-member / categorical feature)."""
        if not self._sub_buttons:
            return
        for btn, lbl in zip(self._sub_buttons, self._sub_labels):
            self._hide_one_checkbox(btn, lbl)
        self._member_panel_active = False

    def _make_member_callback(self, slot: int):
        """Click handler for one sub-panel button: make that member the active feature (radio within the
        group; the left parent stays selected via `_sync_left_radio`)."""
        def callback(state: bool) -> None:
            member = self._sub_slot_features[slot]
            if member is not None:
                self._set_feature(member)
        return callback

    def _on_window_resize(self, *args) -> None:
        """ConfigureEvent handler: keep the bottom-anchored selectors pinned as the window resizes."""
        self._reposition_scale_selector()
        if self._cat_panel_active:  # re-centre the category panel at the new width
            self._show_category_panel(self._active_feature)
        if self._member_panel_active:  # re-centre the expander sub-panel at the new width
            self._show_member_panel(self._active_feature)
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
    