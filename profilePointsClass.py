import numpy as np
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class profileData:
    name: str
    x: np.ndarray
    z: np.ndarray

    # currently unused
    m: Optional[float] = None
    b: Optional[float] = None

    profileNumber: Optional[int] = None
    borderPoints: Optional[np.ndarray] = None
    ySlope: Optional[np.ndarray] = None
    ySmooth: Optional[np.ndarray] = None
    ySlopeSmooth: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None
    width: Optional[float] = None
    isFlat: Optional[bool] = None
    flatness: Optional[float] = None
    area: Optional[float] = None
    shoelaceArea: Optional[float] = None
    shoelaceArea2: Optional[float] = None
    maxSmoothedHeight: Optional[float] = None
    maxSmoothedPlace: Optional[int] = None
    maxHeight: Optional[float] = None
    maxPlace: Optional[int] = None

    #TODO: profileNumber not correctly taken
    def __post_init__(self):
        profileNumberMatch = re.search(r"profile_(\d{1})", self.name)
        if profileNumberMatch:
            self.profileNumber = int(profileNumberMatch.group(1))


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
