import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class profileData:
    name: str
    x: np.ndarray
    z: np.ndarray

    # floor baseline fit (slope, intercept) of the levelled profile, set by
    # rotate_and_shift_uniform and reused to draw the baseline (line_points_from_floorSides)
    m: Optional[float] = None
    b: Optional[float] = None

    floorMask: Optional[np.ndarray] = None  # per-point bool: True = floor point, False = profile (bead)
    ySmooth: Optional[np.ndarray] = None
    ySlopeSmooth: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    width: Optional[float] = None
    beadWidthIdx: Optional[np.ndarray] = None  # indices of the two outer bead points (bead-edge width)
    beadWidth: Optional[float] = None          # x-span between the two outer bead points
    isFlat: Optional[bool] = None
    flatness: Optional[float] = None
    area: Optional[float] = None           # bead cross-section, Simpson integration (profile-unit^2)
    shoelaceArea: Optional[float] = None   # bead cross-section, shoelace polygon (profile-unit^2)
    shoelaceArea2: Optional[float] = None
    maxSmoothedHeight: Optional[float] = None
    maxSmoothedPlace: Optional[int] = None
    maxHeight: Optional[float] = None
    maxPlace: Optional[int] = None

    """
    # only trust this formula for profiles with monotonically rising x values (not the ones where "points are below each other")
    def integrate_area(self):
        area=np.round(sp.integrate.simpson(self.y,self.x), decimals=2)
        return area


    # Area using shoelace formula --> (points must be ordered!, points do not need to be monotonically increasing in x)
    def shoelace_area(self):
        # chat gpt code
        shoelaceArea = 0.5 * abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))

        points= np.vstack((self.x,self.y))
        shifted = np.vstack((points[1:], points[0]))
        cross = points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]
        shoelaceArea2 = 0.5 * abs(np.sum(cross))
        return shoelaceArea,shoelaceArea2

    def find_max_height(self):
        self.maxSmoothedHeight = np.round(np.max(self.ySmooth),decimals=2)
        self.maxSmoothedPlace = np.argmax(self.ySmooth)
        self.maxHeight = np.round(np.max(self.y),decimals=2)
        self.maxPlace = np.argmax(self.y)
    """
