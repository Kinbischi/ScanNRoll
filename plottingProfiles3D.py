import h5py
import numpy as np
import pyvista as pv
from profilePointsClass import *

"""
# artificial profile
r = 5  # radius fake profile

theta = np.linspace(0, np.pi/2, 100)  # half-circle in local profile coords
xCircle = r * np.cos(theta)
yCircle = r * np.sin(theta)
xBox1 = np.linspace(0, -1*r, 50)
yBox1 = np.ones_like(xBox1)*r
xBox2 = np.repeat(-1*r, 50)
yBox2 = np.linspace(1,0,50)*r
x = np.concatenate((xCircle, xBox1, xBox2))
y = np.concatenate((yCircle, yBox1, yBox2))
points_3d = np.column_stack((x, y, np.zeros_like(x)))
"""

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
        
        #TODO: add option for getting out smoothed profiles
        # why are width points shifted?
        # take border points out?
    def plot(self, profiles: list[profileData], plotSubject:str, colour:str, size=5) -> None:
        match plotSubject:
            case "profile":
                self.add_3d_points_to_plot(get_profile_points_for_plot(profiles), colour, size)
            case "baseline":
                self.add_lines_to_plot(line_points_from_floorSides(profiles), colour)
            case "widthPoints":
                # one entry per profile so width points land on the correct path slot;
                # profiles without two peaks get an empty (0, 3) array (skipped on plot)
                widthPoints = []
                for p in profiles:
                    if p.peaks is not None and len(p.peaks) == 2:
                        wp = np.array([[p.x[p.peaks[0]], p.z[p.peaks[0]], 0],
                                       [p.x[p.peaks[1]], p.z[p.peaks[1]], 0]])
                    else:
                        wp = np.empty((0, 3))
                    widthPoints.append(wp)
                self.add_3d_points_to_plot(widthPoints, colour, size)

    def add_3d_points_to_plot(self,points, colour = 'green', point_size=5):
        distances = np.ones(len(points)) * 2000  # 2.0 units between each profile
        #totalDistances = np.cumsum(distances)

        pathPoints,tiltAngles = compute_print_path_and_angle(distances)
        
        #TODO: check how the real profiles are set (do they need to be inverted 180 deg?)
        for i,prof in enumerate(points):
            # rotate profile to match path direction
            prof = prof @ self.rotation_matrices[i].T
            
            if prof.shape[0] > 0:
                cloud = pv.PolyData(pathPoints[i]+prof)
                self.plotter.add_points(cloud, color = colour, point_size=point_size, render_points_as_spheres=False)
            
    def add_lines_to_plot(self, linePoints, colour = 'green'):
        distances = np.ones(len(linePoints)) * 2000  # 2.0 units between each profile

        pathPoints,tiltAngles = compute_print_path_and_angle(distances)
        
        for i in range(len(linePoints)):
            p0 = linePoints[i][0]
            p1 = linePoints[i][1]
            # rotate profile to match path direction
            rot_matrix = self.rotation_matrices[i]
            p0 = p0 @ rot_matrix.T+pathPoints[i]    #rotate points and translate to path
            p1 = p1 @ rot_matrix.T+pathPoints[i]

            if p0.shape[0] > 0:
                line = pv.Line(p0, p1)
                self.plotter.add_mesh(line, color = colour, line_width=5) # size was 5


def get_profile_points_for_plot(profiles: list[profileData]):
    
    return [np.column_stack((profile.x, profile.z, np.zeros_like(profile.x))) for profile in profiles]

def line_points_from_floorSides(profiles: list[profileData]):
    linesPoints=[]
    for p in profiles:
        m,b = get_baseline_from_profileBorder(p.x,p.z)
        p0=(p.x[0],m*p.x[0]+b,0)
        p1=(p.x[-1],m*p.x[-1]+b,0)
        
        linesPoints.append((p0,p1))
    return linesPoints

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
    