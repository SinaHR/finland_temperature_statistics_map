#!/usr/bin/env python3
"""Assemble 30-year window histograms from per-year layers (pipeline step 3).

For each reference year R in 2000-2025, each land cell, each day-of-year, and
each variable, the window histogram is the sum of the 30 per-year histograms
of years R-29 .. R (settled convention; R = 2000 -> 1971-2000). Per-year
histograms are the bincounts of the 31-day (+/-15 d) calendar windows, which
wrap across year boundaries: windows of year Y near Jan 1 / Dec 31 include
days of Y-1 / Y+1. Only the very end of the series truncates: for R = 2025,
day-of-year windows near Dec 31 lack the early-Jan-2026 days (histogram totals
915-930 instead of 930). Nothing is recomputed from raw data — everything
sums from the step-2 layers.

Output: data/derived/windows/<var>_R<year>.npz, key "hist", uint16 array of
shape (365, n_land, 170) — day-of-year (0-based, Feb 29 dropped) x land cell
x temperature bin. The land-cell order (row, col) is recorded once in
windows_meta.json.

Idempotent: existing outputs with a metadata entry are skipped (--force
rebuilds).

Usage:
    python scripts/build_window_histograms.py                  # everything
    python scripts/build_window_histograms.py --variables tmax --ref-years 2000 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climgrid_common import (
    DAYS_PER_YEAR, FIRST_YEAR, HALF_WINDOW, LAST_YEAR, MISSING, N_BINS,
    PROJECT_ROOT, VARIABLES, load_layer,
)

WINDOWS_DIR = PROJECT_ROOT / "data" / "derived" / "windows"
META_PATH = WINDOWS_DIR / "windows_meta.json"

FIRST_REF_YEAR = 2000
LAST_REF_YEAR = 2025
WINDOW_YEARS = 30
BASELINE_REF_YEAR = 2000

N_YEARS = LAST_YEAR - FIRST_YEAR + 1
N_DAYS_TOTAL = N_YEARS * DAYS_PER_YEAR


def window_path(variable: str, ref_year: int) -> Path:
    return WINDOWS_DIR / f"{variable}_R{ref_year}.npz"


def compute_land_mask(sample_layer: np.ndarray) -> np.ndarray:
    """Static land mask (n_lat, n_lon). Layer QA already established that
    exactly the same cells have data on every day of every file."""
    return sample_layer[0] != MISSING


def load_series(variable: str, land_flat: np.ndarray) -> np.ndarray:
    """All years of one variable as a flat day series over land cells:
    uint8 (N_DAYS_TOTAL, n_land); day t = (year-FIRST_YEAR)*365 + doy."""
    n_land = int(land_flat.sum())
    series = np.empty((N_DAYS_TOTAL, n_land), dtype=np.uint8)
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        layer = load_layer(variable, year).reshape(DAYS_PER_YEAR, -1)
        t0 = (year - FIRST_YEAR) * DAYS_PER_YEAR
        series[t0:t0 + DAYS_PER_YEAR] = layer[:, land_flat]
    if (series == MISSING).any():
        raise AssertionError(f"{variable}: missing values inside land mask")
    return series


def window_positions(ref_year: int, doy: int) -> np.ndarray:
    """Flat series positions of the 30x31-day pooled window for (R, doy);
    positions past the end of the series (R=2025, late-Dec doys) drop out."""
    years = np.arange(ref_year - WINDOW_YEARS + 1, ref_year + 1)
    offsets = np.arange(-HALF_WINDOW, HALF_WINDOW + 1)
    pos = ((years - FIRST_YEAR) * DAYS_PER_YEAR)[:, None] + doy + offsets[None, :]
    pos = pos.ravel()
    return pos[(pos >= 0) & (pos < N_DAYS_TOTAL)]


def build_one(variable: str, ref_year: int, series: np.ndarray) -> dict:
    """Build one (variable, ref_year) window file. Returns QA stats."""
    n_land = series.shape[1]
    cell_base = (np.arange(n_land, dtype=np.int64) * N_BINS)[None, :]
    hist = np.empty((DAYS_PER_YEAR, n_land, N_BINS), dtype=np.uint16)
    for doy in range(DAYS_PER_YEAR):
        pos = window_positions(ref_year, doy)
        combined = series[pos].astype(np.int64) + cell_base
        counts = np.bincount(combined.ravel(), minlength=n_land * N_BINS)
        if counts.max() > np.iinfo(np.uint16).max:
            raise AssertionError("bin count overflows uint16")
        hist[doy] = counts.reshape(n_land, N_BINS).astype(np.uint16)

    totals = hist.sum(axis=2, dtype=np.int64)
    stats = {
        "total_min": int(totals.min()),
        "total_max": int(totals.max()),
        "n_underfull_day_cells": int((totals < WINDOW_YEARS * 31).sum()),
        "peak_bin_count": int(hist.max()),
    }

    out = window_path(variable, ref_year)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, hist=hist)
    tmp.replace(out)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variables", nargs="+", choices=sorted(VARIABLES),
                        default=sorted(VARIABLES))
    parser.add_argument("--ref-years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        default=[FIRST_REF_YEAR, LAST_REF_YEAR])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    first, last = args.ref_years
    if not (FIRST_REF_YEAR <= first <= last <= LAST_REF_YEAR):
        parser.error(f"ref years must be within {FIRST_REF_YEAR}-{LAST_REF_YEAR}")

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    else:
        sample = load_layer("tday", FIRST_YEAR)
        land = compute_land_mask(sample)
        iy, ix = np.nonzero(land)
        meta = {
            "description": "30-year window histograms summed from per-year layers",
            "window_convention": (
                f"reference year R pools years R-{WINDOW_YEARS-1}..R, each with a "
                f"+/-{HALF_WINDOW}-day calendar window that wraps across year "
                f"boundaries (spilling into R-{WINDOW_YEARS} and R+1); baseline "
                f"R={BASELINE_REF_YEAR} = 1971-2000; series ends {LAST_YEAR}-12-31, "
                f"so R={LAST_YEAR} windows near Dec 31 are truncated (>=915 of 930)"),
            "array_layout": "hist[doy_0based_feb29_dropped, land_cell, temp_bin], uint16",
            "baseline_ref_year": BASELINE_REF_YEAR,
            "ref_years": list(range(FIRST_REF_YEAR, LAST_REF_YEAR + 1)),
            "land_cells_row_col": np.stack([iy, ix], axis=1).tolist(),
            "windows": {},
        }

    land_flat = np.zeros(116 * 68, dtype=bool)
    idx = np.array(meta["land_cells_row_col"])
    land_flat[idx[:, 0] * 68 + idx[:, 1]] = True

    for variable in args.variables:
        todo = [r for r in range(first, last + 1)
                if args.force or not (window_path(variable, r).exists()
                                      and f"{variable}_R{r}" in meta["windows"])]
        if not todo:
            print(f"{variable}: all requested windows already built")
            continue
        print(f"{variable}: loading {N_YEARS} layers ...")
        series = load_series(variable, land_flat)
        for r in todo:
            t0 = time.monotonic()
            stats = build_one(variable, r, series)
            meta["windows"][f"{variable}_R{r}"] = stats
            tmp = META_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            tmp.replace(META_PATH)
            size_mb = window_path(variable, r).stat().st_size / 1e6
            print(f"  {variable}_R{r}: built in {time.monotonic()-t0:.1f}s, "
                  f"{size_mb:.0f} MB, totals {stats['total_min']}-{stats['total_max']}, "
                  f"peak bin {stats['peak_bin_count']}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
