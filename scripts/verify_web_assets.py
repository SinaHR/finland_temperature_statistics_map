#!/usr/bin/env python3
"""Verify step-5 web assets against the window histograms.

Check 1 (exactness): random (variable, ref-year, week) .bin files must equal
the corresponding day-of-year slice of the step-3 window files byte-for-byte.

Check 2 (meta sanity): cell count, WGS84 centers inside Finland's bounding
box, week/doy mapping, and bin spec consistency.

Check 3 (end-to-end climatology): reading only meta.json + .bin files the way
the frontend would (nearest cell by WGS84 coordinates, nearest week to the
date), the Helsinki July P(Tmax >= 25 degC) numbers must reproduce the known
values (small tolerance: the weekly snap shifts the window by 3 days).

Also reports the gzipped transfer size of a sample file.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climgrid_common import N_BINS, VARIABLES
from build_window_histograms import (
    BASELINE_REF_YEAR, FIRST_REF_YEAR, LAST_REF_YEAR, window_path,
)
from export_web_assets import N_WEEKS, WEB_DIR, WEEK_DOYS, hist_bin_path

RNG = np.random.default_rng(11)


def read_bin(variable: str, ref_year: int, week: int, n_cells: int) -> np.ndarray:
    raw = hist_bin_path(variable, ref_year, week).read_bytes()
    return np.frombuffer(raw, dtype="<u2").reshape(n_cells, N_BINS)


def check_exactness(n_cells: int) -> None:
    print("Check 1: .bin files vs window-file slices")
    cases = [("tday", BASELINE_REF_YEAR, 0), ("tmin", LAST_REF_YEAR, N_WEEKS - 1)]
    for _ in range(6):
        cases.append((
            list(VARIABLES)[RNG.integers(3)],
            int(RNG.integers(FIRST_REF_YEAR, LAST_REF_YEAR + 1)),
            int(RNG.integers(N_WEEKS)),
        ))
    for variable, ref_year, week in cases:
        got = read_bin(variable, ref_year, week, n_cells)
        with np.load(window_path(variable, ref_year)) as z:
            expected = z["hist"][WEEK_DOYS[week]]
        assert np.array_equal(got, expected), \
            f"MISMATCH {variable} R{ref_year} w{week}"
        print(f"  ok: {variable} R{ref_year} w{week:02d} (doy {WEEK_DOYS[week]})")
    print(f"  {len(cases)}/{len(cases)} exact matches\n")


def check_meta(meta: dict) -> None:
    print("Check 2: meta.json sanity")
    n = meta["n_cells"]
    assert n == 3911, f"unexpected cell count {n}"
    lat = np.asarray(meta["cells"]["lat_wgs84"])
    lon = np.asarray(meta["cells"]["lon_wgs84"])
    assert lat.min() > 59.0 and lat.max() < 70.4, "latitude outside Finland"
    assert lon.min() > 19.0 and lon.max() < 32.0, "longitude outside Finland"
    assert meta["weeks"]["doys_0based"] == WEEK_DOYS
    assert meta["bins"]["n_bins"] == N_BINS
    assert meta["baseline_ref_year"] == BASELINE_REF_YEAR
    print(f"  ok: {n} cells, lat {lat.min():.2f}..{lat.max():.2f}, "
          f"lon {lon.min():.2f}..{lon.max():.2f}\n")


def check_climatology(meta: dict) -> None:
    print("Check 3: end-to-end climatology through the web format")
    lat = np.asarray(meta["cells"]["lat_wgs84"])
    lon = np.asarray(meta["cells"]["lon_wgs84"])
    # frontend-style lookup: nearest cell to Helsinki, nearest week to Jul 15
    hel = int(np.argmin((lat - 60.17) ** 2 + (np.cos(np.radians(60.17)) * (lon - 24.94)) ** 2))
    jul15 = 195
    week = int(np.argmin(np.abs(np.asarray(WEEK_DOYS) - jul15)))
    n = meta["n_cells"]

    def p_at_least(h, x):
        return h[int((x + 45) / 0.5):].sum() / h.sum()

    h0 = read_bin("tmax", BASELINE_REF_YEAR, week, n)[hel].astype(np.int64)
    h1 = read_bin("tmax", LAST_REF_YEAR, week, n)[hel].astype(np.int64)
    p0, p1 = p_at_least(h0, 25.0), p_at_least(h1, 25.0)
    print(f"  Helsinki, week {week} (doy {WEEK_DOYS[week]}, Jul 15 snapped): "
          f"P(Tmax >= 25) baseline {p0:.3f}, R2025 {p1:.3f}, ratio {p1/p0:.2f}x")
    assert abs(p0 - 0.074) < 0.02 and abs(p1 - 0.163) < 0.02, \
        "web-format numbers stray too far from verified values"
    assert h0.sum() == 930 and h1.sum() == 930

    sample = hist_bin_path("tmax", BASELINE_REF_YEAR, week)
    gz = len(gzip.compress(sample.read_bytes(), 6))
    print(f"  sample transfer size: {sample.stat().st_size/1e6:.2f} MB raw, "
          f"{gz/1e3:.0f} KB gzipped\n")


if __name__ == "__main__":
    meta = json.loads((WEB_DIR / "meta.json").read_text())
    check_exactness(meta["n_cells"])
    check_meta(meta)
    check_climatology(meta)
    print("all web-asset verification checks passed")
