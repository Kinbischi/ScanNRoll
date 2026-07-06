import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors
import re
from dataclasses import dataclass
from typing import Optional
import pyvista as pv

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

def process_profiles(profiles: list[profileData]) -> list[profileData]:
    """Run the full per-profile processing pipeline in place and return the list.

    Order matters: level and rotate to the floor first, then smooth and measure
    width on the levelled profile.
    """
    rotate_pointcloud(profiles)
    translate_floor_to_zero(profiles)
    find_smooth_slope(profiles)
    width_from_smoothed_slope(profiles)
    flag_flat_profiles(profiles)
    return profiles

# RMS residual of a straight-line fit to the whole profile, in profile units (~0.01 mm).
# Below this a profile is "flat" (substrate only); above it a bead/curve is present.
# Set in the valley of the bimodal distribution: the flat mode sits below ~200 and the
# beaded mode above ~500. 250 separates them while still catching small/ramping beads
# (e.g. the first profiles after a flat run, whose residuals cluster near 350-400 and
# were wrongly marked flat at the previous threshold of 400).
FLATNESS_RMS_THRESHOLD = 250

def flag_flat_profiles(profiles: list[profileData], threshold: float = FLATNESS_RMS_THRESHOLD) -> None:
    """Label each profile flat or not by how well a straight line fits all its points.

    A flat (substrate-only) profile is essentially a tilted line, so its line-fit RMS
    residual is small; a printed bead deviates from any line, giving a large residual.
    Sets `flatness` (the RMS residual) and `isFlat` (residual < threshold). Too-short
    profiles are treated as flat (no detectable curve).
    """
    for p in profiles:
        if p.x.shape[0] < 10: # TODO empirical value
            p.flatness = 0.0
            p.isFlat = True
            continue
        coeffs, residuals, rank, singular_values, rcond = np.polyfit(p.x, p.z, 1, full=True)
        rms = float(np.sqrt(residuals[0] / p.x.shape[0])) if residuals.size > 0 else 0.0
        p.flatness = rms
        p.isFlat = rms < threshold

def find_smooth_slope(profiles: list[profileData]):
    for p in profiles:
        if p.x.shape[0] < 10: # TODO empirical value
            continue #too-short profile --> skip this one
        
        p.ySlope = abs(np.gradient(p.z,p.x))

        ySmooth = moving_average(p.z,15)
        ySmooth = moving_average(ySmooth,9)
        ySmooth = moving_average(ySmooth,5)
        ySmooth = moving_average(ySmooth,5)
        p.ySmooth = ySmooth

        ySlopeSmooth = abs(np.gradient(ySmooth,p.x))
        ySlopeSmooth = moving_average(ySlopeSmooth,65)
        ySlopeSmooth = moving_average(ySlopeSmooth,55)
        ySlopeSmooth = moving_average(ySlopeSmooth,15)
        ySlopeSmooth = moving_average(ySlopeSmooth,5)
        p.ySlopeSmooth = ySlopeSmooth

def width_from_smoothed_slope(profiles: list[profileData]):
    for p in profiles:
        if p.x.shape[0] < 10: # TODO empirical value
            p.width = np.nan
            continue #too-short profile --> skip this one
        
        p.peaks, properties = sp.signal.find_peaks(p.ySlopeSmooth,height=0.15,distance=50)
        if len(p.peaks) != 2:
            p.width = np.nan
        else:
            p.width = np.round(abs(p.x[p.peaks[0]]-p.x[p.peaks[1]]), decimals=2)

def translate_floor_to_zero(profiles: list[profileData]):
    for p in profiles:
        m,b = get_baseline_from_profileBorder(p.x,p.z)
        p.z = p.z-b
        p.x = p.x
    return profiles

def rotate_pointcloud(profiles: list[profileData]):
    for p in profiles:
        m,b = get_baseline_from_profileBorder(p.x,p.z)

        angle_deg = np.arctan(m)*180/np.pi

        angle_rad = -np.pi/180 * angle_deg
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        points =np.column_stack((p.x, p.z))
        rotatedPoints = points @ R.T
        p.x = rotatedPoints[:,0]
        p.z = rotatedPoints[:,1]


#TODO: unit is currently: 20 is 0.20mm aka 200 microns --> change?
def find_border_points(profiles: list[profileData]):
    for p in profiles:
        borderPoints = np.empty(len(p.x), dtype=bool)
        profilePoints = np.empty(len(p.z), dtype=bool)
        for i in range(len(p.z)):
            if p.z[i] > 20: # if height is lower than 0.2mm --> set to 0 --> assumed baseline
                profilePoints[i] = True
                borderPoints[i] = False
            else:
                profilePoints[i] = False
                borderPoints[i] = True
        p.borderPoints = np.array([p.x[borderPoints],p.z[borderPoints]])
        p.x = p.x[profilePoints]
        p.z = p.z[profilePoints]

def moving_average(arr, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

def get_baseline_from_profileBorder(x ,y, borderPoints=30):
        n=borderPoints
        profileBordersX = np.concatenate([x[:n], x[-n:]])
        profileBordersY = np.concatenate([y[:n], y[-n:]])

        coeffs, residuals, rank, singular_values, rcond = np.polyfit(profileBordersX, profileBordersY, 1, full=True)
        m=coeffs[0]
        b=coeffs[1]
        LSerror = np.sqrt(residuals[0])

        #if error is not low enough --> not both sides of the floor were caught
        if LSerror > 50:        #take care --> this value depends on the unit (e.g. mm or um)
            profileBordersX1 = x[:n]
            profileBordersY1 = y[:n]
            profileBordersX2 = x[-n:]
            profileBordersY2 = y[-n:]
            coeffs1, residuals1, rank, singular_values, rcond = np.polyfit(profileBordersX1, profileBordersY1, 1, full=True)
            coeffs2, residuals2, rank, singular_values, rcond = np.polyfit(profileBordersX2, profileBordersY2, 1, full=True)

            # set baseline at side with which gives lower least squares error
            if residuals1 > residuals2:
                m=coeffs2[0]
                b=coeffs2[1]
            else:
                m=coeffs1[0]
                b=coeffs1[1]
        return m,b