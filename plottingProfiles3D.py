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

def get_profile_points_for_plot(profiles: list[profileData]):
    
    profile_points = []
    for i in range(len(profiles)):
        profile = profiles[i]
        prof = np.column_stack((profile.x, profile.z, np.zeros_like(profile.x)))
        profile_points.append(prof)
    return profile_points
    #return [np.column_stack((profile.x, profile.z, np.zeros_like(profile.x))) for profile in profiles]

def add_3d_points_to_plot(points, plotter, colour = 'green'):
    distances = np.ones(len(points)) * 2000  # 2.0 units between each profile
    #totalDistances = np.cumsum(distances)

    pathPoints,tiltAngles = compute_print_path_and_angle(distances)
    
    #TODO: check how the real profiles are set (do they need to be inverted 180 deg?)
    for i in range(len(points)):
        prof = points[i]
        # rotate profile to match path direction
        rot_matrix = np.array([
        [np.cos(tiltAngles[i]), 0, np.sin(tiltAngles[i])],
        [0, 1, 0],
        [-np.sin(tiltAngles[i]), 0, np.cos(tiltAngles[i])]
        ])
        prof = prof @ rot_matrix.T
        
        if prof.shape[0] > 0:
            cloud = pv.PolyData(pathPoints[i]+prof)
            plotter.add_points(cloud, color = colour, point_size=5, render_points_as_spheres=True) # size was 5
            #plotter.add_points(cloud.points[0], point_size=6, render_points_as_spheres=True,color='red') # size was 5
        
    return plotter


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
    