import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time

from profilePointsClass import *
from profileRegistration import *

def plotProfiles(profileGroups, noFloorPoints = False, withShift = False):
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.5))

    for i, ax in enumerate(axes.flat):
        if i >= len(profileGroups):
            break
        group = profileGroups[i]

        #ax.scatter(p.x[p.profilePoints], p.y[p.profilePoints], marker='x', s=1, label=str(p.name))
        #ax.scatter(p.x, p.y, marker='x', s=1, label=str(p.name))
        

        for j in range(len(group)):
            p=group[j]
            if noFloorPoints:
                x = p.x[p.profilePoints].copy()
                y = p.y[p.profilePoints].copy()
            else:
                x = p.x.copy()
                y = p.y.copy()

            if withShift & hasattr(p, 'shift'):
                x=x-p.shift
            ax.scatter(x, y, marker='x', s=1, label=str(p.name))

            ax.legend()
            ax.set_aspect('equal',adjustable='datalim')
    plt.tight_layout()

def loadProfiles():
    path = Path(r"C:/Users/zimme/Documents/A-Phd/Rollerband/Python/ProfileData/Registration")
    profiles = []

    for file_path in path.iterdir():
        if file_path.is_file():  # Skip subfolders
            with open(file_path) as f:
                lines = [line for line in f if line.strip()]
                lines.pop(0)
                lines.pop(0)
                lines.pop(0)
                for elem in lines:
                    elem.replace("\n", "")
                
                profileLength = len(lines)
                xPts=np.empty(profileLength)
                yPts=np.empty(profileLength)
                
                for i in range(profileLength):
                    xVal,yVal = lines[i].split(';')
                    xPts[i]=xVal
                    yPts[i]=yVal

                profileName = str(f.name).replace(str(path), "")
                profileName = profileName.replace(".csv", "").replace("\\", "")
                profiles.append(profileData(profileName,xPts,yPts))
    
    return profiles


def groupProfiles(profiles):
    group = []
    profileGroups = []
    lastNum = profiles[0].profileNumber
    for i in range(len(profiles)):
        if profiles[i].profileNumber == lastNum:
            group.append(profiles[i])
            if i == len(profiles)-1:
                profileGroups.append(group)
        else:
            profileGroups.append(group)
            group = []
            group.append(profiles[i])
            lastNum = profiles[i].profileNumber
    
    sortedGroups = []
    for g in profileGroups:
        pOthers = []
        for p in g:
            if "vertical" not in p.name:
                pOthers.append(p)
            else:
                pVertical = p
        sortGroup = []
        sortGroup.append(pVertical)
        sortGroup.extend(pOthers)
        sortedGroups.append(sortGroup)
    return sortedGroups

