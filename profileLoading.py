import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import time
from profilePointsClass import *

from profileGroupClass import *


def loadProfiles():
    path = Path(r"C:/Users/zimme/Documents/A-Phd/Rollerband/Python/ProfileData/Registration")

    fileNames = os.listdir(path)
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
                g=profileGroupClass(group)
                profileGroups.append(g)
        else:
            g=profileGroupClass(group)
            profileGroups.append(g)
            group = []
            group.append(profiles[i])
            lastNum = profiles[i].profileNumber
    
    return profileGroups

