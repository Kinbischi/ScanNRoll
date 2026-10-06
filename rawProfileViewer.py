"""Standalone raw-profile viewer (SIDE TOOL — off the processing pipeline).

Loads a *raw* acquisition HDF5 (the ``profile_NNNNNN`` layout written by
``rawProfileUdpCapturing.py``) and shows every profile as one 3D point cloud, stacked
along the print direction — no processing, no PLC, no ``datasetConfig``, just ``x``/``z``
straight from the sensor.

The along-track spacing between profiles is not known from a raw capture (it needs the belt
speed). So this viewer lets you *try* a hypothetical constant rollerband speed: each profile
is placed at ``speed (m/s) x (its capture time - t0)`` along the print axis. A slider picks
the speed in realistic discrete steps; moving it rescales only the along-track axis (an actor
transform), so it stays smooth even for large files.

Each profile is placed at its own **absolute** capture time (no forced-monotonic guard), so
several sensors triggered a few ms apart cluster correctly instead of piling on one spot.
The clock is chosen automatically: the sensor clock when it is usable (single sensor, or
multiple sensors that are NTP time-synced), else the common PC ``arrivalTime`` (the only
shared timeline for multiple *unsynced* sensors). Multi-sensor files are coloured by sensor.

Axes (all in mm): X = across-track, Y = height, Z = along-track (belt travel).

Run:
    python rawProfileViewer.py udp_profiles.h5
    python rawProfileViewer.py path/to/capture.h5 --speed 0.08 --colour-by height

In a VS Code ``# %%`` cell:
    from rawProfileViewer import view_raw_profiles
    view_raw_profiles("udp_profiles.h5")            # slider for the speed
    view_raw_profiles("udp_profiles.h5", speed=0.08, use_slider=False)  # fixed number
"""
from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np
import pyvista as pv

from profileLoading import count_profiles, load_profiles
from profilePointsClass import profileData

logger = logging.getLogger(__name__)

# --- units (match profileProcessingAlgorithms) -----------------------------------------
PROFILE_UNITS_TO_MM = 0.01      # 1 profile unit = 0.01 mm
M_PER_S_TO_MM_PER_S = 1000.0    # belt speed m/s -> mm/s (along-track axis is drawn in mm)

# --- realistic belt-speed slider (Exp1 PLC rollerbandSpeed: 0.02..0.16 m/s in 0.02 steps) ---
SPEED_MIN = 0.02      # m/s
SPEED_MAX = 0.20      # m/s
SPEED_STEP = 0.02     # m/s — discrete slider increment
SPEED_DEFAULT = 0.06  # m/s

# --- large-file guard: cap the drawn point budget; auto-thin profiles beyond it --------
MAX_POINTS = 3_000_000  # keeps rotation smooth on a typical GPU

DEFAULT_PATH = "udp_profiles.h5"

# Sensor identity by source IP (edit to match your mounting). The three OX200s view the same
# line from different angles; .101 has been the centre throughout. left/right are an assumption.
SENSOR_LABELS = {"192.168.0.101": "center", "192.168.0.102": "left", "192.168.0.103": "right"}
SENSOR_COLOURS = {"192.168.0.101": (0.86, 0.20, 0.18),   # red
                  "192.168.0.102": (0.15, 0.55, 0.82),   # blue
                  "192.168.0.103": (0.30, 0.69, 0.29)}   # green
_FALLBACK_COLOURS = [(0.60, 0.40, 0.70), (0.90, 0.60, 0.0), (0.40, 0.40, 0.40)]


def _snap_speed(v: float) -> float:
    """Snap a raw slider value to the nearest discrete step, clamped to the realistic range."""
    v = round(v / SPEED_STEP) * SPEED_STEP
    return float(min(max(v, SPEED_MIN), SPEED_MAX))


def _read_sensor_meta(path: str, start: int, end: int | None) -> tuple[list[str], bool] | tuple[None, None]:
    """Read per-profile `source_ip` (in load order) and whether the file is time-synced.

    Returns `(source_ips, synced)`, or `(None, None)` for a non-raw file (e.g. a columnar
    processed cache, which carries no per-sensor metadata). `synced` is True when a majority
    of profiles report `time_synced` (NTP-locked sensor clocks are comparable across sensors).
    """
    with h5py.File(path, "r") as f:
        if f.attrs.get("layout") == "columnar":
            return None, None
        names = sorted(n for n in f.keys() if n.startswith("profile_"))[start:end]
        ips = [str(f[n].attrs.get("source_ip", "?")) for n in names]
        synced_flags = [bool(f[n].attrs.get("time_synced", False)) for n in names]
    synced = bool(synced_flags) and (sum(synced_flags) >= len(synced_flags) / 2)
    return ips, synced


def _placement_seconds(profiles: list[profileData], source_ips: list[str] | None,
                       synced: bool, time_source: str = "auto") -> tuple[np.ndarray, str]:
    """Per-profile capture time (s) relative to the earliest profile, plus the clock label.

    Each profile keeps its own absolute time (no monotonic guard), so interleaved sensors sit
    at their true instants. Clock choice (`time_source="auto"`): the sensor clock for a single
    sensor or time-synced sensors; the common PC `arrivalTime` for multiple *unsynced* sensors
    (their sensor clocks share no timeline). "sensor"/"arrival" force the choice.
    """
    sensor_t = np.array([p.sensorTime if p.sensorTime is not None else np.nan for p in profiles], float)
    arrival_t = np.array([p.arrivalTime if p.arrivalTime is not None else np.nan for p in profiles], float)
    n_sensors = len(set(source_ips)) if source_ips else 1

    if time_source == "sensor":
        t, label = sensor_t, "sensor clock (forced)"
    elif time_source == "arrival":
        t, label = arrival_t, "PC arrival clock (forced)"
    elif n_sensors > 1 and not synced:
        t, label = arrival_t, "PC arrival clock (multi-sensor, unsynced)"
    else:
        t, label = sensor_t, "sensor clock"

    if not np.any(np.isfinite(t)):  # chosen clock absent -> fall back to the other, then index
        other = arrival_t if t is sensor_t else sensor_t
        t = other if np.any(np.isfinite(other)) else np.arange(len(profiles), dtype=float)
        label += " -> fallback"
    idx = np.arange(len(t))
    good = np.isfinite(t)
    if not good.all():  # interpolate timing for any profile missing a timestamp
        t = np.interp(idx, idx[good], t[good])
    return t - t.min(), label


def _build_cloud(profiles: list[profileData], elapsed: np.ndarray, point_step: int,
                 sensor_idx: list[int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Stack profiles into one `(P, 3)` array: (across, height, along@1 m/s), all in mm.

    The Z column is the along-track position *at 1 m/s* (= elapsed seconds x 1000 mm); the
    viewer scales it by the chosen speed at display time, so the cloud is built only once.
    Returns `(points, per_point_sensor)` — the second is None unless `sensor_idx` is given.
    """
    across, height, along, sidx = [], [], [], []
    for i, (p, e) in enumerate(zip(profiles, elapsed)):
        if p.x.shape[0] == 0:
            continue
        x = p.x[::point_step] * PROFILE_UNITS_TO_MM
        z = p.z[::point_step] * PROFILE_UNITS_TO_MM
        across.append(x)
        height.append(z)
        along.append(np.full(x.shape[0], e * M_PER_S_TO_MM_PER_S))
        if sensor_idx is not None:
            sidx.append(np.full(x.shape[0], sensor_idx[i], dtype=np.int32))
    if not across:
        return np.empty((0, 3)), (np.empty(0, dtype=np.int32) if sensor_idx is not None else None)
    pts = np.column_stack([np.concatenate(across), np.concatenate(height), np.concatenate(along)])
    return pts, (np.concatenate(sidx) if sensor_idx is not None else None)


def view_raw_profiles(path: str = DEFAULT_PATH, speed: float = SPEED_DEFAULT,
                      profile_step: int = 1, point_step: int = 1,
                      start: int = 0, end: int | None = None,
                      use_slider: bool = True, colour_by: str = "auto",
                      time_source: str = "auto", cmap: str = "viridis",
                      point_size: int = 3, show: bool = True) -> pv.Plotter:
    """Show every raw profile of `path` as a 3D cloud, spaced by a hypothetical belt speed.

    Args:
        path: raw acquisition HDF5 (``profile_NNNNNN`` layout).
        speed: initial hypothetical rollerband speed in m/s (the slider's start value, or the
            fixed value when `use_slider=False`).
        profile_step / point_step: draw every Nth profile / point (subsampling for large files).
        start / end: profile index range to load (defaults to the whole file).
        use_slider: add the belt-speed slider; if False, the speed is fixed at `speed`.
        colour_by: "sensor" (per source IP, with legend), "height", or "auto" (sensor when the
            file has >1 sensor, else height).
        time_source: "auto" (sensor clock if usable, else PC arrival), or force "sensor"/"arrival".
        cmap: colour map for the height colouring.
        point_size: rendered point size in pixels.
        show: open the interactive window (False for headless use / tests).

    Returns the PyVista Plotter (so a caller can inspect actors headlessly).
    """
    n_total = count_profiles(path)
    end = n_total if end is None else min(end, n_total)
    profiles = load_profiles(path, start, end)
    if not profiles:
        raise ValueError(f"no profiles loaded from {path}")
    source_ips, synced = _read_sensor_meta(path, start, end)

    elapsed, clock_label = _placement_seconds(profiles, source_ips, bool(synced), time_source)

    # Respect the point budget: thin profiles (not points, to keep each cross-section intact).
    lengths = np.array([p.x.shape[0] for p in profiles])
    drawn = (int(lengths.sum()) / max(1, profile_step)) / max(1, point_step)
    if drawn > MAX_POINTS:
        profile_step *= int(np.ceil(drawn / MAX_POINTS))
        logger.warning("large file: drawing every %d-th profile to stay under %d points",
                       profile_step, MAX_POINTS)

    sub = profiles[::profile_step]
    sub_elapsed = elapsed[::profile_step]
    sub_ips = source_ips[::profile_step] if source_ips is not None else None

    # Decide colouring. "auto" -> by sensor when the file has more than one.
    unique_ips = sorted(set(sub_ips)) if sub_ips is not None else []
    by_sensor = (colour_by == "sensor") or (colour_by == "auto" and len(unique_ips) > 1)

    ip_to_idx = {ip: k for k, ip in enumerate(unique_ips)}
    sensor_idx = [ip_to_idx[ip] for ip in sub_ips] if (by_sensor and sub_ips is not None) else None
    pts, per_point_idx = _build_cloud(sub, sub_elapsed, point_step, sensor_idx)
    logger.info("drawing %d points from %d/%d profiles | clock: %s | sensors: %s",
                pts.shape[0], len(sub), len(profiles), clock_label,
                {SENSOR_LABELS.get(ip, ip): sub_ips.count(ip) for ip in unique_ips} if sub_ips else "1")

    cloud = pv.PolyData(pts)

    if not pv.OFF_SCREEN:
        pv.global_theme.notebook = False  # native interactive window (not a static inline image)
    pl = pv.Plotter()
    if by_sensor and per_point_idx is not None:
        palette = [SENSOR_COLOURS.get(ip, _FALLBACK_COLOURS[k % len(_FALLBACK_COLOURS)])
                   for k, ip in enumerate(unique_ips)]
        rgb = (np.array(palette)[per_point_idx] * 255).astype(np.uint8)
        actor = pl.add_points(cloud, scalars=rgb, rgb=True, point_size=point_size,
                              render_points_as_spheres=False)
        pl.add_legend(labels=[[f"{SENSOR_LABELS.get(ip, ip)} ({ip.rsplit('.', 1)[-1]})", palette[k]]
                              for k, ip in enumerate(unique_ips)], bcolor="white", size=(0.18, 0.14))
    else:
        cloud["height (mm)"] = pts[:, 1]
        actor = pl.add_points(cloud, scalars="height (mm)", cmap=cmap, point_size=point_size,
                              render_points_as_spheres=False)
    pl.add_axes(xlabel="across", ylabel="height", zlabel="along")

    span = float(sub_elapsed.max() - sub_elapsed.min()) if len(sub_elapsed) else 0.0
    dt = np.diff(np.sort(sub_elapsed))
    mean_dt = float(dt[dt > 0].mean()) if np.any(dt > 0) else 0.0

    def apply_speed(v: float) -> float:
        """Scale only the along-track (Z) axis to the chosen speed; update the readout."""
        v = _snap_speed(v)
        actor.SetScale(1.0, 1.0, v)  # Z is baked at 1 m/s -> scaling by v gives mm at speed v
        length_m = v * span
        gap_mm = v * mean_dt * M_PER_S_TO_MM_PER_S
        pl.add_text(f"belt speed {v:.2f} m/s   |   print length {length_m:.2f} m   |   "
                    f"mean spacing {gap_mm:.2f} mm   |   {len(sub)} profiles   |   {clock_label}",
                    name="info", position="upper_edge", font_size=10)
        return v

    speed = _snap_speed(speed)
    if use_slider:
        holder: dict = {}  # holds the widget so the callback can snap the handle after creation

        def on_slide(value: float) -> None:
            v = apply_speed(value)
            widget = holder.get("widget")
            if widget is not None:  # snap the visible handle to the discrete step
                widget.GetSliderRepresentation().SetValue(v)
            pl.render()

        holder["widget"] = pl.add_slider_widget(
            on_slide, rng=[SPEED_MIN, SPEED_MAX], value=speed,
            title=f"rollerband speed (m/s, {SPEED_STEP:g} steps)",
            pointa=(0.30, 0.92), pointb=(0.72, 0.92), style="modern", fmt="%.2f")
    apply_speed(speed)  # set the initial scale + readout

    if show:
        pl.show()
    return pl


def main() -> None:
    ap = argparse.ArgumentParser(description="View raw LIDAR profiles stacked by a hypothetical belt speed.")
    ap.add_argument("path", nargs="?", default=DEFAULT_PATH, help="raw acquisition HDF5 file")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT, help="initial belt speed (m/s)")
    ap.add_argument("--profile-step", type=int, default=1, help="draw every Nth profile")
    ap.add_argument("--point-step", type=int, default=1, help="draw every Nth point")
    ap.add_argument("--start", type=int, default=0, help="first profile index to load")
    ap.add_argument("--end", type=int, default=None, help="stop before this profile index")
    ap.add_argument("--colour-by", choices=("auto", "sensor", "height"), default="auto",
                    help="colour points by sensor, by height, or auto")
    ap.add_argument("--time-source", choices=("auto", "sensor", "arrival"), default="auto",
                    help="clock used for along-track placement")
    ap.add_argument("--no-slider", action="store_true", help="fix the speed at --speed (no slider)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    view_raw_profiles(args.path, speed=args.speed, profile_step=args.profile_step,
                      point_step=args.point_step, start=args.start, end=args.end,
                      colour_by=args.colour_by, time_source=args.time_source,
                      use_slider=not args.no_slider)


if __name__ == "__main__":
    main()
