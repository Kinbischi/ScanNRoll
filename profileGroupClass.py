import numpy as np
import matplotlib.pyplot as plt

from profilePointsClass import *
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.optimize import minimize


class profileGroupClass:

    def __init__(self, profiles):
        self.profiles = profiles
        self.pOther = []
        for p in self.profiles:
            if "vertical" not in p.name:
                self.pOther.append(p)
            else:
                self.pV = p

    def translateInX(self,xTranslation):
        p=self.pOther[0]
        x = p.x[p.profilePoints]
        y = p.y[p.profilePoints]

        xV = self.pV.x
        yV= self.pV.y
        
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
        sortedDist = sortedDistances[:len(sortedDistances)//4]
        partSumDist = np.sum(sortedDist)

        return partSumDist

    def registerProfiles(self):
        #x0 = np.array([-1, 0.5, 0, 0.5, 1])
        x0 = 0
        resu = minimize(self.translateInX,x0=x0)    #args=(self.pOther[0].x,self.pOther[0].y,self.pV.x,self.pV.y)
        self.shift = resu.x

        #resu = translateInX(2,self.pOther[0].x,self.pOther[0].y,self.pV.x,self.pV.y)
        return resu


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