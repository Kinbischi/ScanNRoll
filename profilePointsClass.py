import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors
import re

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
        if LSerror > 0.5:
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


class profileData:
    
    def __init__(self,name,Xcord,Ycord,):
        self.name=name
        self.x=Xcord
        self.y=Ycord
        profileNumberMatch = re.search(r"Profile(\d{1})", name)
        if profileNumberMatch:
            self.profileNumber = int(profileNumberMatch.group(1))

    def rotate_pointcloud(self):
        m,b = get_baseline_from_profileBorder(self.x,self.y)

        angle_deg = np.arctan(m)*180/np.pi

        angle_rad = -np.pi/180 * angle_deg
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        points =np.column_stack((self.x, self.y))
        rotatedPoints = points @ R.T
        self.x = rotatedPoints[:,0]
        self.y = rotatedPoints[:,1]

    def translate_floor_to_zero(self):
        m,b = get_baseline_from_profileBorder(self.x,self.y)
        self.y = self.y-b
        self.x = self.x

    def find_border_points(self):
        borderPoints = np.empty(len(self.x), dtype=bool)
        profilePoints = np.empty(len(self.y), dtype=bool)
        for i in range(len(self.y)):
            if self.y[i] > 0.2: # if height is lower than 0.2mm --> set to 0 --> assumed baseline
                profilePoints[i] = True
                borderPoints[i] = False
            else:
                profilePoints[i] = False
                borderPoints[i] = True
        self.borderPoints = np.array([self.x[borderPoints],self.y[borderPoints]])
        self.x = self.x[profilePoints]
        self.y = self.y[profilePoints]

    def find_smooth_slope(self):
        self.ySlope = abs(np.gradient(self.y,self.x))

        ySmooth = moving_average(self.y,15)
        ySmooth = moving_average(ySmooth,9)
        ySmooth = moving_average(ySmooth,5)
        ySmooth = moving_average(ySmooth,5)
        self.ySmooth = ySmooth

        ySlopeSmooth = abs(np.gradient(ySmooth,self.x))
        ySlopeSmooth = moving_average(ySlopeSmooth,65)
        ySlopeSmooth = moving_average(ySlopeSmooth,55)
        ySlopeSmooth = moving_average(ySlopeSmooth,15)
        ySlopeSmooth = moving_average(ySlopeSmooth,5)
        self.ySlopeSmooth = ySlopeSmooth

    def width_from_smoothed_slope(self):
        self.peaks, properties = sp.signal.find_peaks(self.ySlopeSmooth,height=0.15,distance=50)
        if len(self.peaks) != 2:
            self.width = np.nan
        else:
            self.width = np.round(abs(self.x[self.peaks[0]]-self.x[self.peaks[1]]), decimals=2)

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
