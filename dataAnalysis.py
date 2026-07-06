import profile3Dplotting
from profileLoading import load_profiles

# Plot-only entry point. Processing now lives in profileProcessing.py, which writes the
# processed HDF5 cache that this script loads. Run profileProcessing.py first if the
# cache is missing or the raw data / processing parameters have changed.
PROCESSED_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor_processed.h5"

# The full dataset is ~37M points, which makes interaction lag. Subsample the profile
# cloud to keep rotate/zoom smooth: plot every PROFILE_STEP-th profile and every
# POINT_STEP-th point (points drawn ≈ total / (PROFILE_STEP * POINT_STEP)). Increase
# either if it still lags; set both to 1 to draw every point.
PROFILE_STEP = 3
POINT_STEP = 3

profiles = load_profiles(PROCESSED_FILE)

plotter = profile3Dplotting.plottingClass(len(profiles))
# flat profiles (substrate only, no bead) are drawn red; the rest green
plotter.plot(profiles, "profile", 'green', profile_step=PROFILE_STEP, point_step=POINT_STEP, flat_colour='red')
plotter.plot(profiles, "widthPoints", 'yellow', 10)
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