"""Process raw LIDAR profiles once and cache the result to a processed HDF5 file.

Run this whenever the raw data or the processing parameters change. The plotting
entry point (dataAnalysis.py) then loads the small processed cache instead
of re-running the (heavy raw load +) processing pipeline every time.
"""
import logging

from profileLoading import count_profiles, load_profiles, save_profiles
from profilePointsClass import process_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor.h5"
PROCESSED_FILE = "HDf5data/RealExperiments/ClayAndWater_2026_05_28/Exp3/watercontentchangeExp2Sensor_processed.h5"

# Process only the profiles in the index range [START_PROFILE:END_PROFILE) (0-based,
# END exclusive). START_PROFILE = 0 and END_PROFILE = None processes everything.
START_PROFILE = 50000
END_PROFILE = None


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
