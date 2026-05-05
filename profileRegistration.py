import numpy as np
import matplotlib.pyplot as plt
import copy

from profilePointsClass import *
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.optimize import minimize

#TODO make a class again from this? ;)
#detect whether left or right profile
#define new profile as: 20% left of vertical profile is the left profile, 20% right of V is the right profile

def find_left_right_center_profile(profiles):
    pVert = profiles[0]
    leftProfiles = []
    rightProfiles = []
    weirdVertProfiles = []

    for i in range(1,len(profiles)):
        p = profiles[i]
        avg = np.mean(p.x)
        if avg < 0.5:
            leftProfiles.append(p)
        elif avg > 0.5:
            rightProfiles.append(p)
        else:
            weirdVertProfiles.append(p)
        
    if len(leftProfiles) > 0:
         left = leftProfiles[0]
    else:
        left = pVert
    if len(rightProfiles) > 0:
        right = rightProfiles[0]
    else:
        right = pVert
    return pVert, left,right

def generateJoinedProfile(profiles):
    pVert, pL, pR = find_left_right_center_profile(profiles)
    min = np.min(pVert.x)
    max = np.max(pVert.x)

    newProfileX = np.empty(0)
    newProfileY = np.empty(0)
    
    #TODO: make so that "all points" from left side get attached? --> currently only works if less then 20% "going in" at bottom
    for i in range(len(pL.x)):
        if pL.x[i] < min*0.8:
            newProfileX = np.append(newProfileX,pL.x[i])
            newProfileY = np.append(newProfileY,pL.y[i])
    
    for i in range(len(pVert.x)):
        if (pVert.x[i] > min*0.8) & (pVert.x[i] < max*0.8):
            newProfileX = np.append(newProfileX,pVert.x[i])
            newProfileY = np.append(newProfileY,pVert.y[i])

    for i in range(len(pR.x)):
        if pR.x[i] > max*0.8:
            newProfileX = np.append(newProfileX,pR.x[i])
            newProfileY = np.append(newProfileY,pR.y[i])
            
    return newProfileX,newProfileY, profileData("Franz",newProfileX,newProfileY)
    
def translateInX(xTranslation,p,pVertical):
    x = p.x
    y = p.y

    xV = pVertical.x
    yV= pVertical.y
    
    xtemp=x.copy()-xTranslation
    pointsV = np.column_stack((xV,yV))
    points = np.column_stack((xtemp,y))
    #points = np.array([(1, 2), (3, 4), (6, 1)])
    tree = KDTree(pointsV)

    #query = np.array([2, 3])
    distances = np.empty(len(points))
    for i in range(len(points)):
        testPoint = points[i]
        dist, idx = tree.query(testPoint)
        nearestPoint = pointsV[idx]
        distances[i]=dist
    sortedDistances = np.sort(distances)
    end = len(sortedDistances)
    part = end//5
    sortedDist = sortedDistances[part:end]  #:len(sortedDistances)//1.1]
    partSumDist = np.sum(sortedDist)

    return partSumDist

def registerAndShiftProfiles(profiles):
    pVert = profiles[0]
    pVert.shift = np.mean(pVert.x)
    pVert.x = pVert.x-pVert.shift # center vertical profile points at x = 0
    pVert.borderPoints[0,:] = pVert.borderPoints[0,:] - pVert.shift
    for i in range(1,len(profiles)):
        x0 = 0
        resu = minimize(translateInX,x0=x0,args=(profiles[i],pVert))
        profiles[i].shift = resu.x
        profiles[i].x = profiles[i].x - profiles[i].shift
        profiles[i].borderPoints[0,:] = profiles[i].borderPoints[0,:] - profiles[i].shift

def estimate_normals(points):
    # simple finite-difference normals for ordered 2D profile
    normals = []
    for i in range(len(points)):
        if i == 0:
            tangent = points[i+1] - points[i]
        elif i == len(points)-1:
            tangent = points[i] - points[i-1]
        else:
            tangent = points[i+1] - points[i-1]

        # perpendicular vector
        normal = np.array([-tangent[1], tangent[0]])
        normal /= np.linalg.norm(normal) + 1e-8
        normals.append(normal)

    return np.array(normals)


def icp_point_to_plane_tx(src, dst, max_iter=20):
    src = src.copy()
    tree = cKDTree(dst)

    normals = estimate_normals(dst)

    tx_total = 0.0

    for _ in range(max_iter):
        dist, idx = tree.query(src)

        q = dst[idx]
        n = normals[idx]

        px, py = src[:, 0], src[:, 1]
        qx, qy = q[:, 0], q[:, 1]
        nx, ny = n[:, 0], n[:, 1]

        # Solve least squares for tx
        A = nx
        b = nx * (qx - px) + ny * (qy - py)

        # least squares solution
        tx = np.sum(A * b) / (np.sum(A * A) + 1e-8)

        # apply update
        src[:, 0] += tx
        tx_total += tx

    return tx_total, src

"""
    # todo: Trimmed ICP
    # idea: just rotate and set baseplane to z=0 --> then shift in x direction
    def translation_icp(A, B, iterations=10):
        A_current = A.copy()
        t_total = np.zeros(2)

        nbrs = NearestNeighbors(n_neighbors=1).fit(B)

        for _ in range(iterations):
            distances, indices = nbrs.kneighbors(A_current)
            B_matched = B[indices[:, 0]]

            # Compute translation
            t = B_matched.mean(axis=0) - A_current.mean(axis=0)

            # Apply
            A_current += t
            t_total += t

        return t_total, A_current
    """