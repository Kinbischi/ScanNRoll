"""Process raw LIDAR profiles once and cache the result to a processed HDF5 file.

Run this whenever the raw data or the processing parameters change. The plotting
entry point (dataAnalysis.py) then loads the small processed cache instead
of re-running the (heavy raw load +) processing pipeline every time.
"""
import logging

from datasetConfig import PLC_FILE, PROCESSED_FILE, RAW_FILE
from plcData import join_plc_to_profiles, load_plc_csv
from profileLoading import count_profiles, load_profiles, save_profiles
from profilePointsClass import profileData
from profileProcessingAlgorithms import (
    categorize_floor_points,
    classify_continuous_filaments,
    clean_flat_runs,
    flag_flat_profiles,
    grow_profile_points,
    measure_filament_area,
    measure_filament_height,
    measure_filament_volume,
    measure_run_lengths,
    rotate_and_shift_uniform,
    width_from_filament_edges,
    width_from_smoothed_slope,
)
from segmentShape import measure_segment_shape
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# RAW_FILE / PROCESSED_FILE / PLC_FILE come from datasetConfig (switch datasets there).

# Process only the profiles in the index range [START_PROFILE:END_PROFILE) (0-based,
# END exclusive). START_PROFILE = 0 and END_PROFILE = None processes everything. The PLC
# join then trims this to the profiles overlapping the PLC log (see main).
START_PROFILE = 0
END_PROFILE = None

# Floor/filament categorisation height basis, chosen separately for the seed and grow steps.
# True: use each profile's OWN floor fit (floor sits at 0, robust to per-profile tilt/offset).
# False: use the uniform median-levelled height. The DATA stays median-levelled either way
# (relative heights kept for visualisation); only the categorisation height basis changes.
SEED_USE_PROFILE_BASELINE = True   # categorize_floor_points
GROW_USE_PROFILE_BASELINE = False  # grow_profile_points


def process_profiles(profiles: list[profileData]) -> tuple[list[profileData], float, float]:
    """Run the full per-profile processing pipeline in place; return (profiles, angle, offset).

    Order matters: level and rotate to the floor first, then smooth and measure width on the
    levelled profile. The returned (angle, offset) is the uniform leveling transform (angle in
    radians) — saved to the cache so the before/after overlay can be reconstructed by inverting it.
    """
    # uniform: one median rotation + shift for all profiles (keeps relative heights)
    level_angle, level_offset = rotate_and_shift_uniform(profiles)
    # seed floor vs filament points, then grow the filament into its connected lower flanks
    categorize_floor_points(profiles, use_profile_baseline=SEED_USE_PROFILE_BASELINE)
    grow_profile_points(profiles, use_profile_baseline=GROW_USE_PROFILE_BASELINE)
    flag_flat_profiles(profiles)  # flat = no filament points (uses floorMask; set after grow)
    # width, two ways: smoothed-slope flank feet and the outer filament points
    width_from_smoothed_slope(profiles)  # -> widthFlankIdx, widthFlank (smooths internally)
    width_from_filament_edges(profiles)      # -> widthOuterIdx, widthOuter
    measure_filament_height(profiles)        # -> heightP95, heightSmooth (robust filament heights)
    # cross-sectional filament area above the median floor, two ways (integration + shoelace)
    measure_filament_area(profiles)          # -> areaSimpson, areaShoelace
    # Optional steps in profileProcessingAlgorithms (import + call to enable): rotate_pointcloud
    # + translate_floor_to_zero (per-profile levelling).
    return profiles, level_angle, level_offset


def main() -> None:
    # Peek the profile count first (fast, reads no point data) so an out-of-range
    # request fails immediately instead of after the slow full load.
    total = count_profiles(RAW_FILE)
    start = START_PROFILE
    end = total if END_PROFILE is None else END_PROFILE

    logger.info("Processing profiles [%d:%d] of %d", start, end, total)
    profiles = load_profiles(RAW_FILE, start, end)
    profiles, level_angle, level_offset = process_profiles(profiles)

    # Join the machine PLC log by timestamp. This drops profiles outside the mutual overlap;
    # record the surviving raw index span (a contiguous head/tail trim, arrival_time being
    # monotonic) so dataAnalysis can load the matching raw slice for the overlay cell.
    plc = load_plc_csv(PLC_FILE)
    covered = [i for i, p in enumerate(profiles)
               if p.arrivalTime is not None and plc.time[0] <= p.arrivalTime <= plc.time[-1]]
    if not covered:
        raise ValueError("No profiles overlap the PLC log time window")
    raw_start, raw_end = start + covered[0], start + covered[-1] + 1
    profiles = join_plc_to_profiles(profiles, plc)
    assert raw_end - raw_start == len(profiles), "PLC coverage is not a contiguous profile block"

    # Clean the segment/defect run structure before measuring it: bridge sub-5 mm floor gaps and drop
    # sub-10 mm filament blips from the noisy per-profile categorisation. Needs physical distances
    # (rollerbandSpeed, from the join), so it runs here — before the run-based measures below.
    clean_flat_runs(profiles)
    # Label over-long segment runs as continuous filament (not discrete segments); shape analysis skips them.
    classify_continuous_filaments(profiles)  # -> isContinuousFilament

    # Filament-segment volume runs here (not in process_profiles) because it needs the physical
    # inter-profile distances, which depend on rollerbandSpeed — only populated by the PLC join above.
    measure_filament_volume(profiles)  # -> segmentVolume, sliceVolume
    measure_run_lengths(profiles)      # -> segmentLength (filament runs), defectLength (pure-floor runs)
    # Per-segment shape (thinning / startup / rupture) along the print path; needs the cleaned segments
    # and the physical spacing, so it runs here alongside the other run-based measures. Analyses only
    # discrete segments (SEGMENT_SHAPE_MIN_LENGTH_MM <= length <= MAX_SEGMENT_LENGTH_MM).
    measure_segment_shape(profiles)    # -> segmentBody{Area,Width,Height}{Thinning,Steadiness}, segmentCriticalArea,
                                       #    segmentRuptureLength, segmentHeadOvershoot, segmentRuptures, segmentSection

    save_profiles(profiles, PROCESSED_FILE, kind="processed", source_file=RAW_FILE,
                  raw_start=raw_start, raw_end=raw_end,
                  level_angle=level_angle, level_offset=level_offset)
    logger.info("Done: wrote %d processed profiles to %s (raw span [%d:%d])",
                len(profiles), PROCESSED_FILE, raw_start, raw_end)


if __name__ == "__main__":
    main()
