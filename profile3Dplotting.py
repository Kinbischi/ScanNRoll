import numpy as np
import pyvista as pv
from profilePointsClass import profileData


class plottingClass:
    def __init__(self, numOfprofiles):
        self.plotter = pv.Plotter()
        
        distances = np.ones(numOfprofiles) * 2000  # 2.0 units between each profile
        self.pathPoints, self.tiltAngles = compute_print_path_and_angle(distances)
        self.rotation_matrices = [np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]) for theta in self.tiltAngles]
        
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
        distances = np.ones(len(points)) * 2000  # 2.0 units between each profile

        pathPoints,tiltAngles = compute_print_path_and_angle(distances)

        # Collect every profile's transformed points and add them as a single actor.
        # One add_points call instead of one per profile is far faster for many profiles.
        transformed = []
        for i, prof in enumerate(points):
            if prof.shape[0] > 0:
                prof = prof @ self.rotation_matrices[i].T  # rotate to match path direction
                transformed.append(pathPoints[i] + prof)

        if transformed:
            cloud = pv.PolyData(np.vstack(transformed))
            self.plotter.add_points(cloud, color=colour, point_size=point_size, render_points_as_spheres=spheres)
            
    def add_lines_to_plot(self, linePoints, colour = 'green'):
        distances = np.ones(len(linePoints)) * 2000  # 2.0 units between each profile

        pathPoints,tiltAngles = compute_print_path_and_angle(distances)

        # Collect every line's two transformed endpoints and add them all as a single mesh.
        # One add_mesh call instead of one per line is far faster for many profiles.
        endpoints = []
        for i in range(len(linePoints)):
            rot_matrix = self.rotation_matrices[i]
            p0 = np.asarray(linePoints[i][0], dtype=float) @ rot_matrix.T + pathPoints[i]  # rotate + translate to path
            p1 = np.asarray(linePoints[i][1], dtype=float) @ rot_matrix.T + pathPoints[i]
            endpoints.append(p0)
            endpoints.append(p1)

        if endpoints:
            # points ordered as segment pairs (p0, p1, p0, p1, ...) -> one line per pair
            lines = pv.line_segments_from_points(np.array(endpoints))
            self.plotter.add_mesh(lines, color = colour, line_width=5)


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
    