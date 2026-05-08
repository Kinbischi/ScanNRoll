import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
import pyvista as pv

#from point_cloud_registration import ICP, PlaneICP, NDT, VPlaneICP

import profileLoading
import plottingProfiles3D
from profilePointsClass import *
from profileRegistration import *

HDF5_FILE = "HDf5data/TestExperiments/udp_profiles_772profiles_same.h5"
profiles = profileLoading.load_hdf5_profiles(HDF5_FILE)

plotter = pv.Plotter()

plotter = plottingProfiles3D.add_3d_points_to_plot(plottingProfiles3D.get_profile_points_for_plot(profiles), plotter, 'blue')

plottingProfiles3D.add_lines_to_plot(line_points_from_floorSides(profiles), plotter, 'red')
rotate_pointcloud(profiles)
translate_floor_to_zero(profiles)
#find_border_points(profiles)
#p.find_smooth_slope()
        #p.width_from_smoothed_slope()
plotter = plottingProfiles3D.add_3d_points_to_plot(plottingProfiles3D.get_profile_points_for_plot(profiles), plotter, 'green')

plotter.add_camera_orientation_widget()
plotter.show()

"""
#profiles = profileLoading.loadProfiles()
profileGroups = profileLoading.groupProfiles(profiles)

# performed on every profile
allShifts = []
newX = []
newY = []
joinedProfiles = []
for group in profileGroups:
    for p in group:
        p.rotate_pointcloud()
        p.translate_floor_to_zero()
        p.find_border_points()
    
    registerAndShiftProfiles(group)
    nX,nY,pTesting = generateJoinedProfile(group)
    joinedProfiles.append(pTesting)
    newX.append(nX)
    newY.append(nY)
 

# TODO:
# make sure that points are sorted (along profile line) for area algo


# TODO:
# think of whether profileGroups should be class with obj joined profile
# what obj belongs to groups what to profile
# what do you need in terms of workflow? 
# can profiles remain shifted --> yes, can we discard floor points -->?
# you will anyways only use joinedprofile

areaI =[]
areaI2 =[]
area1 = []
area2 = []
for group in profileGroups:
    areaI.append(group[0].integrate_area())

for j in joinedProfiles:
    areaI2.append(j.integrate_area())
    a1,a2 = j.shoelace_area()
    area1.append(a1)
    area2.append(a2)

testP = joinedProfiles[0] #profiles[0]

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

x_data = []
y_data = []

for x, y in zip(testP.x, testP.y):
    x_data.append(x)
    y_data.append(y)
    
    ax.clear()  # Clear previous frame
    ax.plot(x_data, y_data, marker='x')
    
    ax.set_xlim(-40,40)
    ax.set_ylim(-10, 40)
    ax.set_title("Points appearing one by one")
    
    plt.draw()
    plt.pause(0.05)  # Pause to create animation effect

plt.ioff()
plt.show()


"""