import numpy as np

from profilePointsClass import *
from scipy.spatial import cKDTree

class profileGroupClass:
    def __init__(self, profiles):
        self.profiles = profiles
        
    def registerProfiles(self):
        pOther = []
        for p in self.profiles:
            if "vertical" not in p.name:
                pOther.append(p)
            else:
                pVert = p
        
        pVmeanX = np.mean(pVert.x)
        for p in pOther:
            xMean = np.mean(p.x)
            p.x = p.x-xMean+pVmeanX

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