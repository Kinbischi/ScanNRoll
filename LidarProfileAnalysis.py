import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time

#from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP

import profileLoading
from profilePointsClass import *

profiles = profileLoading.loadProfiles()
profileGroups = profileLoading.groupProfiles(profiles)

# performed on every profile
for group in profileGroups:
    for p in group.profiles:
        p.rotate_pointcloud()
        p.translate_floor_to_zero()
        p.find_border_points()
    
    registerResult = group.registerProfiles()
    #print(registerResult.x)

fig, axes = plt.subplots(2, 2, figsize=(15.2, 7.5))

for i, ax in enumerate(axes.flat):
    group = profileGroups[i]

    p = group.pOther[0]
    shift = group.shift
    pV= group.pV

    ax.scatter(p.x, p.y, marker='x', s=1, label=str(p.name))
    ax.scatter(p.x-shift, p.y, marker='x', s=1, label=str(p.name))
    ax.scatter(pV.x, pV.y, marker='x', s=1, label=str(p.name))
    #ax.scatter(p.x[p.profilePoints], p.y[p.profilePoints], marker='x', s=1, label=str(p.name))

    """
    for j in range(len(group.profiles)):
        p=group.profiles[j]
        #ax.scatter(p.x[p.profilePoints], p.y[p.profilePoints], marker='x', s=1, label=str(p.name))
        ax.legend()
        ax.set_aspect('equal',adjustable='datalim')
    """

plt.tight_layout()
plt.show()

"""
fig, axes = plt.subplots(2, 2, figsize=(15.2, 7.5))

for i, ax in enumerate(axes.flat):
    group = profileGroups[0]
    pVert = group[3]
    des = np.column_stack((pVert.x, pVert.y,np.random.rand(len(pVert.x)))) #
    for j in range(len(group)-1):
        #ax.scatter(group[j].x, group[j].y, s=1)
        p=group[j]
        #p.xTrans = p.xTrans - np.mean(p.xTrans)
        #numbers = np.array([10, 20, 30, 40, 50])
        #p.borderPoints = np.array([True, False, True, False, True])
        #result = numbers[p.borderPoints]

        sr = np.column_stack((p.x, p.y,np.random.rand(len(p.x)))) #

        #tx_total, source = icp_point_to_plane_tx(sr, des)
        
        # Example point clouds
        target = np.random.rand(100, 3)  # Nx3 point numpy array
        #scan = np.random.rand(80, 3)    # Mx3 point numpy array
        icp = VPlaneICP(voxel_size=0.5, max_iter=30, max_dist=2, tol=1e-3)
        icp.set_target(des)  # Set the target point cloud
        T_new = icp.align(sr, init_T=np.eye(4))  # Fit the scan to the target
        print("Estimated Transform matrix:\n", T_new)
        
        #ax.scatter(source[0,:],source[1,:], marker='x', s=1, label=str(p.name))
        ax.scatter(p.x[p.profilePoints]-tx_total, p.y[p.profilePoints], marker='x', s=1, label=str(p.name))

        #ax.scatter(p.x, p.heights,marker='x', s=1, label=str(p.name))
        #ax.scatter(p.x, p.y, marker='x', s=1, label=str(p.name))
        ax.legend()
        ax.set_aspect('equal',adjustable='datalim')
plt.tight_layout()
plt.show()
"""



"""
fig, axes = plt.subplots(4, 4, figsize=(15.2, 7.5))

for i, ax in enumerate(axes.flat):
    p=profiles[i]
    #ax.plot(p.x, p.y)
    #ax.plot(p.x,p.heights)
    ax.scatter(p.x, p.y,marker='x',s=1)
    ax.scatter(p.rotatedPoints[:,0],p.rotatedPoints[:,1],marker='x',s=1)
    
    #ax.plot(p.x,p.smoothedHeight)
    ax.plot(p.x,p.smoothedGradients)
    ax.plot(p.x[p.peaks], p.heights[p.peaks], "ro", label="Width: "+str(p.width))
    ax.plot(p.x[p.maxSmoothedPlace], p.heights[p.maxSmoothedPlace], "yo", label="Max sm: "+str(p.maxSmoothedHeight))
    ax.plot(p.x[p.maxPlace], p.heights[p.maxPlace], "co", label="Max: "+str(p.maxHeight))
    yLine = p.m * p.x + p.b
    #ax.plot(p.x, yLine,linestyle='dashed', label="base line fit")
    ax.text(-39, 25, "Area: "+str(p.area))

    ax.set_title(p.name)
    ax.set_xlim(-40,40)
    ax.set_ylim(-10,50)
    ax.set_aspect('equal',adjustable='datalim')
    ax.legend()

plt.tight_layout()
plt.show()




testP = profiles[0]

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

x_data = []
y_data = []

for x, y in zip(testP.x, testP.y):
    x_data.append(x)
    y_data.append(y)
    
    ax.clear()  # Clear previous frame
    ax.plot(x_data, y_data, marker='o')
    
    ax.set_xlim(-40,40)
    ax.set_ylim(-10, 40)
    ax.set_title("Points appearing one by one")
    
    plt.draw()
    plt.pause(0.05)  # Pause to create animation effect

plt.ioff()
plt.show()


"""
# todo: calculate 3rd heighest point
# width through differentiations
