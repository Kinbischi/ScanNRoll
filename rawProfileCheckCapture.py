"""Capture QA check for a raw multi-sensor acquisition (SIDE TOOL — off the pipeline).

Reports, per sensor (by source IP): profile count, NTP time-sync status, rate, and dropped
frames. Drops are counted on each sensor's OWN clock (the sensor timestamp when it is synced,
else the PC arrival clock) by looking for whole-period gaps — this is immune to the PC arrival
jitter that made an earlier arrival-clock version over-count phantom drops.

`summarize()` evaluates in-memory arrays and is reused by ``rawProfileUdpCapturing`` for its
automatic warm-up pre-check; `check_capture()` is the standalone CLI that reads a file.

The verdict is driven by the SYNC gate (reliable). Completeness is reported accurately and only
fails the verdict when it is low enough (< COMPLETE_MIN_PCT) to indicate a real throughput
problem, not a few scattered drops.

Run:
    python rawProfileCheckCapture.py udp_profiles.h5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np

# Sensor identity by source IP (matches rawProfileViewer). left/right is a mounting assumption.
SENSOR_LABELS = {"192.168.0.101": "center", "192.168.0.102": "left", "192.168.0.103": "right"}

COMPLETE_MIN_PCT = 95.0  # completeness below this fails the verdict (a real throughput problem, not jitter)


def _missed_frames(times: np.ndarray) -> tuple[int, int, float]:
    """Count dropped frames in one sensor's timestamp series via whole-period gaps.

    Returns (missed, expected, rate_hz). The period is the median inter-profile spacing; a gap of
    ~k periods implies k-1 missed frames. Robust to jitter (rounding) and needs no nominal rate.
    """
    tc = np.sort(times[np.isfinite(times)])
    d = np.diff(tc)
    if d.size == 0:
        return 0, len(tc), 0.0
    period = float(np.median(d))
    if period <= 0:
        return 0, len(tc), 0.0
    missed = int(np.round(np.clip(d / period, 1, None) - 1).sum())
    return missed, len(tc) + missed, 1.0 / period


def summarize(ip: np.ndarray, at: np.ndarray, syn: np.ndarray,
              ts: np.ndarray | None = None, label: str = "capture") -> bool:
    """Print a per-sensor sync/completeness report from aligned arrays; return True if it passes.

    `ip`/`at`/`syn` are per-profile source IP, PC arrival time (s), and time_synced flag; `ts` is the
    sensor-clock time (s) when available (preferred for synced sensors). Passing = every sensor is
    time-synced AND completeness (over the synced sensors) >= COMPLETE_MIN_PCT.
    """
    if len(ip) == 0:
        print(f"{label}: no profiles.")
        return False
    sensors = sorted(set(ip))
    dur = float(np.nanmax(at) - np.nanmin(at)) if len(at) > 1 else 0.0

    print(f"\n=== {label} ===")
    print(f"{len(ip)} profiles | {len(sensors)} sensors | {dur:.1f} s | {len(ip) / dur if dur else 0:.0f}/s overall\n")

    all_synced = True
    synced_missed = synced_expected = 0
    print(f"{'sensor':<16}{'count':>8}{'rate':>9}{'synced':>9}{'drops':>16}")
    for s in sensors:
        m = ip == s
        synced_s = bool(syn[m].all())
        all_synced &= synced_s
        # Count drops on the sensor's own clock when synced (jitter-free), else the arrival clock (approx).
        clock = ts[m] if (ts is not None and synced_s) else at[m]
        missed, expected, rate = _missed_frames(clock)
        if synced_s:
            synced_missed += missed
            synced_expected += expected
        pct = 100 * missed / expected if expected else 0.0
        approx = "" if synced_s else "~"
        label_s = f"{SENSOR_LABELS.get(s, '?')} ({s.rsplit('.', 1)[-1]})"
        synced_str = "yes" if synced_s else f"NO {int(syn[m].sum())}/{int(m.sum())}"
        print(f"{label_s:<16}{int(m.sum()):>8}{rate:>7.0f}Hz{synced_str:>9}{approx + f'{missed} ({pct:.2f}%)':>16}")

    complete_pct = 100 * (1 - synced_missed / synced_expected) if synced_expected else float("nan")

    print()
    if all_synced:
        print("[OK]   all sensors time-synced")
    else:
        bad = [SENSOR_LABELS.get(s, s) for s in sensors if not bool(syn[ip == s].all())]
        print(f"[FAIL] not time-synced: {', '.join(bad)}  -> fix NTP (re-check sensor config / NetTime)")

    if not np.isnan(complete_pct):
        ok_complete = complete_pct >= COMPLETE_MIN_PCT
        note = "" if ok_complete else "  -> real frame loss; lower the PLC rate or thread the capture"
        scope = "" if all_synced else " (synced sensors only)"
        print(f"[{'OK' if ok_complete else 'WARN'}] completeness {complete_pct:.2f}%{scope}{note}")
    else:
        ok_complete = False
        print("[WARN] completeness not measurable (no synced sensor)")

    passed = all_synced and ok_complete
    print(f"\n{'==> GO' if passed else '==> NOT READY'}\n")
    return passed


def check_capture(path: str) -> bool:
    """Read a raw acquisition HDF5 and print its sync/completeness report; return True if it passes."""
    with h5py.File(path, "r") as f:
        names = [n for n in f.keys() if n.startswith("profile_")]
        if not names:
            print(f"{path}: no profiles found.")
            return False
        ip = np.array([str(f[n].attrs.get("source_ip", "?")) for n in names])
        at = np.array([float(f[n].attrs.get("arrival_time", np.nan)) for n in names])
        syn = np.array([bool(f[n].attrs.get("time_synced", False)) for n in names])
        ts = np.array([float(f[n].attrs.get("timestamp_sec", 0)) + float(f[n].attrs.get("timestamp_usec", 0)) / 1e6
                       for n in names])
    return summarize(ip, at, syn, ts, label=path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture QA: per-sensor sync + completeness of a raw capture.")
    ap.add_argument("path", nargs="?", default="udp_profiles.h5", help="raw acquisition HDF5 file")
    args = ap.parse_args()
    check_capture(args.path)


if __name__ == "__main__":
    main()
