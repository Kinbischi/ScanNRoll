from typing import cast

import numpy as np
import pyvista as pv
from plcData import PLC_COLUMNS
from profilePointsClass import profileData

# Heat-map feature display: feature -> (group, unit factor, unit label). 1 profile unit = 0.01 mm,
# so lengths scale to mm and areas to mm^2. Features sharing a group share one colour range (clim),
# so paired measures are directly comparable; grouping also sets the button order.
FEATURE_DISPLAY = {
    "width":            ("width",  0.01, "mm"),
    "beadWidth":        ("width",  0.01, "mm"),
    "beadHeight":       ("height", 0.01, "mm"),
    "beadHeightSmooth": ("height", 0.01, "mm"),
    "area":             ("area",   1e-4, "mm^2"),
    "shoelaceArea":     ("area",   1e-4, "mm^2"),
    # PLC machine-log channels (joined by timestamp): each its own colour group, since their
    # magnitudes differ widely. Shown in the PLC's native engineering units (factor 1.0; the
    # unit label is left blank as the physical units aren't recorded in the CSV).
    **{name: (name, 1.0, "") for name in PLC_COLUMNS},
}

# Along-track spacing of profiles in the 3D layout. 1 profile unit = 0.01 mm, so 1 m = 1e5 units.
# Each profile advances rollerbandSpeed (m/s) * dt (s) along the print path (see
# profile_advance_distances); when speed/time data is missing, profiles fall back to a uniform gap.
METERS_TO_PROFILE_UNITS = 1e5
UNIFORM_PROFILE_DISTANCE = 2000.0  # fallback along-track gap when speed/time is unavailable


def profile_advance_distances(profiles: list[profileData]) -> np.ndarray:
    """Per-profile along-path advance (profile units) from rollerbandSpeed (m/s) x inter-profile dt.

    dt is taken from the sensor clock (`sensorTime` = timestamp_sec + timestamp_usec, monotonic and
    low-jitter), falling back to `arrivalTime`; the first profile gets 0 (path origin). Missing speed
    or a non-monotonic step yields 0 advance for that profile. If neither a time base nor
    `rollerbandSpeed` is available at all (e.g. raw profiles without the PLC join), returns a uniform
    `UNIFORM_PROFILE_DISTANCE` spacing (the pre-physical behaviour).
    """
    n = len(profiles)
    if n == 0:
        return np.empty(0)
    t = np.array([p.sensorTime if p.sensorTime is not None else np.nan for p in profiles], float)
    if not np.any(np.isfinite(t)):  # no sensor clock (e.g. old cache) -> capture-PC clock
        t = np.array([p.arrivalTime if p.arrivalTime is not None else np.nan for p in profiles], float)
    speed = np.array([p.rollerbandSpeed if p.rollerbandSpeed is not None else np.nan for p in profiles], float)
    if not np.any(np.isfinite(t)) or not np.any(np.isfinite(speed)):
        return np.full(n, UNIFORM_PROFILE_DISTANCE)

    dist = np.zeros(n)
    dt = np.diff(t)                             # seconds between consecutive profiles
    speed_mid = 0.5 * (speed[:-1] + speed[1:])  # mean belt speed over each interval
    dist[1:] = speed_mid * dt * METERS_TO_PROFILE_UNITS
    dist[~np.isfinite(dist)] = 0.0             # missing speed/time on a profile -> no advance
    dist[dist < 0.0] = 0.0                     # guard non-monotonic time
    return dist


class plottingClass:
    def __init__(self, profiles: list[profileData], voxel_size: float | None = None):
        """Build the 3D print path from the profiles' physical along-track advance (see
        profile_advance_distances). `profiles` must be the same list (order/length) later passed to
        `plot()`, so each profile lands at its path point. Prefer the processed (PLC-joined) profiles,
        which carry `rollerbandSpeed`; without it the layout falls back to a uniform gap.

        `voxel_size` (default None = off) downsamples the dense clouds to one point per voxel — a
        float in profile units (1 unit = 0.01 mm) — cutting the point count (and overdraw/memory) to
        keep large sets responsive. Off ⇒ renders identically to before.
        """
        self.plotter = pv.Plotter()
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

        Bins points on a grid and keeps the first in each occupied voxel. `extract_points` carries
        every point-data array along, so a heat-map cloud's per-feature scalars stay aligned.
        """
        if not self._voxel_size:
            return cloud
        key = np.floor(cloud.points / self._voxel_size).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        return cast("pv.DataSet", cloud.extract_points(np.sort(idx)))

    def show(self):
        self.plotter.add_camera_orientation_widget()
        self.plotter.show()

    def plot(self, profiles: list[profileData], plotSubject:str, colour:str, size=5,
             profile_step: int = 1, point_step: int = 1, flat_colour: str | None = None,
             category: str | None = None, spheres: bool = False) -> None:
        """Add one subject to the 3D scene: "profile", "baseline", "zeroBaseline",
        "widthPoints" (slope-peak method, uses `peaks`), or "beadWidthPoints" (outer-bead-point
        method, uses `beadWidthIdx`).

        profile_step / point_step subsample the dense "profile" cloud so interaction
        stays responsive on very large datasets: plot every profile_step-th profile and
        every point_step-th point. Both default to 1 (plot everything) and only affect
        "profile". size / spheres set point size and sphere rendering for the point subjects
        (enlarge + spheres=True on the width markers so the chosen points stand out).

        flat_colour (profiles only): if set, profiles flagged `isFlat` are drawn in this
        colour and the rest in `colour`; if None, every profile uses `colour`.

        category (profiles only): "floor" or "profile" draws only points of that category
        (uses `floorMask`); None draws all points. Call twice with different category +
        colour to show floor vs bead in two colours.
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
            case "widthPoints":
                # slope-peak width method: mark the two peak points (from `peaks`)
                self.add_3d_points_to_plot(width_point_arrays(profiles, "peaks"), colour, size, spheres=spheres)
            case "beadWidthPoints":
                # bead-edge width method: mark the two outer bead points (from `beadWidthIdx`)
                self.add_3d_points_to_plot(width_point_arrays(profiles, "beadWidthIdx"), colour, size, spheres=spheres)

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
                             features: tuple[str, ...] = ("width", "beadWidth", "beadHeight", "beadHeightSmooth", "area", "shoelaceArea"),
                             initial: str = "beadWidth", cmap: str = "viridis",
                             point_size: int = 6, profile_step: int = 1, point_step: int = 1) -> None:
        """Colour the bead cloud by a per-profile scalar feature, with a clickable button panel
        to switch the active feature live.

        Each feature in `features` is attached to the cloud as its own point-data array (each
        profile's scalar broadcast to its bead points, converted to physical units per
        FEATURE_DISPLAY), so switching only repoints the mapper and rescales the colour bar — no
        recompute. Features in the same group (width / height / area) share one colour range so the
        paired measures are directly comparable, and the colour bar is labelled in mm / mm^2. Bead
        points only (`~floorMask`); flat profiles contribute none. NaN feature values (e.g. width
        with < 2 flanks) render in the NaN colour. Interactive-window only (buttons need a live VTK
        interactor).
        """
        self._build_feature_cloud(profiles, features, initial, cmap, point_size,
                                  profile_step, point_step)
        self._add_feature_selector()

    def _build_feature_cloud(self, profiles: list[profileData], features: tuple[str, ...],
                             initial: str, cmap: str, point_size: int,
                             profile_step: int, point_step: int) -> None:
        """Build and add the bead cloud carrying one scalar array per feature, and store the
        state the selector callback mutates. Separated from the widget wiring so it can be
        exercised without a live interactor (headless tests)."""
        if initial not in features:
            raise ValueError(f"initial feature {initial!r} not in {features}")
        per_profile = get_profile_points_for_plot(profiles, profile_step, point_step, category="profile")
        transformed = []
        columns: dict[str, list[np.ndarray]] = {f: [] for f in features}
        for i, prof_pts in enumerate(per_profile):
            if prof_pts.shape[0] == 0:  # skipped, empty, or flat (no bead points)
                continue
            transformed.append(self.pathPoints[i] + prof_pts @ self.rotation_matrices[i].T)
            for f in features:
                val = getattr(profiles[i], f)
                factor = FEATURE_DISPLAY[f][1]  # profile units -> physical (mm / mm^2)
                columns[f].append(np.full(prof_pts.shape[0], np.nan if val is None else float(val) * factor))
        if not transformed:
            return  # nothing to draw (e.g. every profile flat)

        cloud = pv.PolyData(np.vstack(transformed))
        for f in features:
            cloud[f] = np.concatenate(columns[f])
        # Downsample before clim/store so the reduced cloud is what's coloured and live-switched;
        # extract_points carries the per-feature scalar arrays along.
        cloud = self._maybe_voxel(cloud)
        cloud.set_active_scalars(initial)

        # features in the same group share one colour range, computed over the whole group's values
        groups: dict[str, list[str]] = {}
        for f in features:
            groups.setdefault(FEATURE_DISPLAY[f][0], []).append(f)
        group_clim = {g: _finite_clim(np.concatenate([cloud[f] for f in feats]))
                      for g, feats in groups.items()}

        self._feature_names = list(features)
        self._feature_initial = initial
        self._feature_clim = {f: group_clim[FEATURE_DISPLAY[f][0]] for f in features}
        self._feature_unit = {f: FEATURE_DISPLAY[f][2] for f in features}
        self._feature_cloud = cloud
        actor = self.plotter.add_points(
            cloud, scalars=initial, cmap=cmap, clim=self._feature_clim[initial],
            nan_color="lightgray", point_size=point_size, render_points_as_spheres=False,
            scalar_bar_args={"title": "", "label_font_size": 14, "position_x": 0.33,
                             "position_y": 0.10, "width": 0.34, "height": 0.05},
        )
        self._feature_mapper = actor.mapper
        self._set_feature_labels(initial)

    def _set_feature_labels(self, feature: str) -> None:
        """Feature name at the top edge, and its unit centred just above the horizontal colour bar
        (a separate text actor so the unit is not cramped against the bar and its numbers)."""
        self.plotter.add_text(feature, name="feature_title", position="upper_edge", font_size=16)
        # anchor at the bar's centre (x = 0.50) with centred justification so mm / mm^2 stay centred
        unit_actor = self.plotter.add_text(self._feature_unit[feature], name="feature_unit",
                                           position=(0.50, 0.17), viewport=True, font_size=16)
        unit_actor.GetTextProperty().SetJustificationToCentered()

    def _set_feature(self, feature: str) -> None:
        """Switch the active feature: repoint the mapper, rescale to the group's shared range
        (physical units), and relabel the feature name + unit."""
        self._feature_cloud.set_active_scalars(feature)
        self._feature_mapper.array_name = feature
        clim = self._feature_clim[feature]
        if clim is not None:
            self._feature_mapper.scalar_range = clim
        self._set_feature_labels(feature)
        self.plotter.render()

    def _add_feature_selector(self, size: int = 26, gap: int = 8) -> None:
        """Add one toggle button per feature (a left-edge panel); clicking one makes it the active
        feature and visually deselects the others (radio behaviour)."""
        self._feature_buttons = []
        for idx, feature in enumerate(self._feature_names):
            y = 12 + idx * (size + gap)
            widget = self.plotter.add_checkbox_button_widget(
                self._make_feature_callback(feature, idx),
                value=(feature == self._feature_initial),
                position=(12, y), size=size, color_on="green", color_off="grey",
            )
            self._feature_buttons.append(widget)
            self.plotter.add_text(feature, position=(12 + size + 8, y + 4), font_size=10)

    def _make_feature_callback(self, feature: str, idx: int):
        """Build the click callback for one feature button: enforce single-selection (radio) and
        switch the active feature. Setting the other buttons' state does not re-fire callbacks."""
        def callback(state: bool) -> None:
            for j, widget in enumerate(self._feature_buttons):
                widget.GetRepresentation().SetState(1 if j == idx else 0)
            self._set_feature(feature)
        return callback


def get_profile_points_for_plot(profiles: list[profileData], profile_step: int = 1,
                                point_step: int = 1, want_flat: bool | None = None,
                                category: str | None = None):
    """Build one (N, 3) point array per profile (height goes in the plot's y slot).

    Returns one entry per profile so the result stays index-aligned with the print
    path; skipped profiles (every profile not on profile_step) and empty profiles
    contribute an empty (0, 3) array, which the plotter skips. point_step subsamples
    points within each kept profile. If want_flat is set, only profiles whose `isFlat`
    matches it are kept (None = no flatness filter). If category is "floor" or "profile",
    only points of that category are kept (uses `floorMask`; ignored when it is None).
    """
    points = []
    for i, profile in enumerate(profiles):
        include = i % profile_step == 0 and profile.x.shape[0] > 0
        if want_flat is not None and bool(profile.isFlat) != want_flat:
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
    """One (2, 3) point array per profile from a 2-index attribute ("peaks" or "beadWidthIdx").

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
    