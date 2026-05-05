import h5py
import numpy as np
import pyvista as pv

HDF5_FILE = "HDf5data/TestExperiments/udp_profiles_772profiles_same.h5"

def load_hdf5_profiles():
    with h5py.File(HDF5_FILE, "r") as f:
        profile_names = [name for name in f.keys() if name.startswith("profile_")]
        profile_names.sort()  # Sort by name, assuming sequential
        
        print(f"Found {len(profile_names)} profiles")
        
        # Read all profiles first
        profiles_data = []
        for profile_name in profile_names:
            group = f[profile_name]
            if "x" in group and "z" in group:
                x = group["x"][:].astype(float)
                z = group["z"][:].astype(float)
                # Create 3D points: x, 0, z
                points_3d = np.column_stack((x, z, np.zeros_like(x)))
                # points_3d = np.column_stack((x, np.zeros(len(x), dtype=float), z))
                
                profiles_data.append(points_3d)
        
        print(f"Loaded {len(profiles_data)} profiles with data")
        return profiles_data

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

profiles = load_hdf5_profiles()

distances = np.ones(len(profiles)) * 2000  # 2.0 units between each profile
#distances = np.tile(np.array([1,3]),300) # distances to NEXT profile
totalDistances = np.cumsum(distances)
n_profiles = len(distances)
#profiles = np.stack([points_3d] * n_profiles)

# path parameters
path_radius = 5000.0 # radius of the curved sweep in XY plane
totalCurveDist = path_radius * np.pi
totalStraightDist = 80000

#initializations
cx=cy=cz=0
transitionPoint = np.array([0,0,0])
currentPoint = np.array([0,0,0])
addedStraightDist = 0
addedAngledDist = 0
zDir = 1
movingStraight = True

plotter = pv.Plotter()

#TODO: check how the real profiles are set (do they need to be inverted 180 deg?)
for i in range(n_profiles):
    prof = profiles[i]
    
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

    # rotate profile to match path direction
    rot_matrix = np.array([
    [np.cos(alpha), 0, np.sin(alpha)],
    [0, 1, 0],
    [-np.sin(alpha), 0, np.cos(alpha)]
    ])
    prof = prof @ rot_matrix.T

    cloud = pv.PolyData(currentPoint+prof)
    plotter.add_points(cloud, point_size=5, render_points_as_spheres=True) # size was 5
    
plotter.show()
