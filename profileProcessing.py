"""Process raw LIDAR profiles once and cache the result to a processed HDF5 file.

Run this whenever the raw data or the processing parameters change. The plotting
entry point (dataAnalysis.py) then loads the small processed cache instead
of re-running the (heavy raw load +) processing pipeline every time.
"""
import logging

from profileLoading import count_profiles, load_profiles, save_profiles
from profilePointsClass import profileData
from profileProcessingAlgorithms import (
    categorize_floor_points,
    flag_flat_profiles,
    grow_profile_points,
    measure_bead_area,
    measure_bead_height,
    rotate_and_shift_uniform,
    width_from_bead_edges,
    width_from_smoothed_slope,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor.h5"
PROCESSED_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor_processed.h5"

# Process only the profiles in the index range [START_PROFILE:END_PROFILE) (0-based,
# END exclusive). START_PROFILE = 0 and END_PROFILE = None processes everything.
START_PROFILE = 58000
END_PROFILE = None

# Floor/bead categorisation height basis, chosen separately for the seed and grow steps.
# True: use each profile's OWN floor fit (floor sits at 0, robust to per-profile tilt/offset).
# False: use the uniform median-levelled height. The DATA stays median-levelled either way
# (relative heights kept for visualisation); only the categorisation height basis changes.
SEED_USE_PROFILE_BASELINE = True   # categorize_floor_points
GROW_USE_PROFILE_BASELINE = False  # grow_profile_points


def process_profiles(profiles: list[profileData]) -> list[profileData]:
    """Run the full per-profile processing pipeline in place and return the list.

    Order matters: level and rotate to the floor first, then smooth and measure
    width on the levelled profile.
    """
    # uniform: one median rotation + shift for all profiles (keeps relative heights)
    rotate_and_shift_uniform(profiles)
    # seed floor vs bead points, then grow the bead into its connected lower flanks
    categorize_floor_points(profiles, use_profile_baseline=SEED_USE_PROFILE_BASELINE)
    grow_profile_points(profiles, use_profile_baseline=GROW_USE_PROFILE_BASELINE)
    flag_flat_profiles(profiles)  # flat = no bead points (uses floorMask; set after grow)
    # width, two ways: smoothed-slope flank feet and the outer bead points
    width_from_smoothed_slope(profiles)  # -> peaks, width (smooths internally)
    width_from_bead_edges(profiles)      # -> beadWidthIdx, beadWidth
    measure_bead_height(profiles)        # -> beadHeight, beadHeightSmooth (robust bead heights)
    # cross-sectional bead area above the median floor, two ways (integration + shoelace)
    measure_bead_area(profiles)          # -> area, shoelaceArea
    # Optional steps in profileProcessingAlgorithms (import + call to enable): rotate_pointcloud
    # + translate_floor_to_zero (per-profile levelling).
    return profiles


def main() -> None:
    # Peek the profile count first (fast, reads no point data) so an out-of-range
    # request fails immediately instead of after the slow full load.
    total = count_profiles(RAW_FILE)
    start = START_PROFILE
    end = total if END_PROFILE is None else END_PROFILE

    logger.info("Processing profiles [%d:%d] of %d", start, end, total)
    profiles = load_profiles(RAW_FILE, start, end)
    process_profiles(profiles)
    save_profiles(profiles, PROCESSED_FILE, kind="processed", source_file=RAW_FILE)
    logger.info("Done: wrote %d processed profiles to %s", len(profiles), PROCESSED_FILE)


if __name__ == "__main__":
    main()
