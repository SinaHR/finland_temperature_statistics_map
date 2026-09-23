#!/usr/bin/env python3
"""Verify step-2 layers against raw data and known climatology.

Check 1 (exactness): for random (variable, year, doy, cell) combinations —
including year-boundary wrap cases and the 1961/2025 series edges — compute
the 31-day-window histogram two independent ways:
  a) from the per-year layers (bincount of the window slice), and
  b) directly from the raw NetCDF files (drop Feb 29, wrap into adjacent
     years, bin the temperatures),
and require identical counts.

Check 2 (climatology sanity): empirical exceedance probabilities for known
cells/dates must be plausible and move in the expected direction between the
1971-2000 baseline and the most recent 1996-2025 window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climgrid_common import (
    DAYS_PER_YEAR, FIRST_YEAR, HALF_WINDOW, LAST_YEAR, MISSING, N_BINS,
    RAW_DIR, VARIABLES, load_layer, open_climgrid, temp_to_bin,
)

RNG = np.random.default_rng(42)


def raw_year_values(variable: str, year: int) -> np.ndarray | None:
    """Raw daily values for one year, Feb 29 dropped: (365, n_lat, n_lon)."""
    path = RAW_DIR / variable / f"{variable}_{year}.nc"
    if not path.exists():
        return None
    with open_climgrid(path) as ds:
        v = ds[VARIABLES[variable]].values
        t = ds["Time"].values.astype("M8[D]")
    months = (t.astype("M8[M]") - t.astype("M8[Y]")).astype(int) + 1
    days = (t.astype("M8[D]") - t.astype("M8[M]")).astype(int) + 1
    return v[~((months == 2) & (days == 29))]


def window_positions(year: int, doy: int) -> list[tuple[int, int]]:
    """(year, doy) pairs of the 31-day window, wrapping across year ends;
    positions outside the available series (before FIRST_YEAR / after
    LAST_YEAR) are dropped."""
    out = []
    for off in range(-HALF_WINDOW, HALF_WINDOW + 1):
        y, d = year, doy + off
        if d < 0:
            y, d = y - 1, d + DAYS_PER_YEAR
        elif d >= DAYS_PER_YEAR:
            y, d = y + 1, d - DAYS_PER_YEAR
        if FIRST_YEAR <= y <= LAST_YEAR:
            out.append((y, d))
    return out


def hist_from_layers(layers: dict, variable: str, year: int, doy: int,
                     iy: int, ix: int) -> np.ndarray:
    counts = np.zeros(N_BINS, dtype=np.int64)
    for y, d in window_positions(year, doy):
        if (variable, y) not in layers:
            layers[(variable, y)] = load_layer(variable, y)
        b = layers[(variable, y)][d, iy, ix]
        if b != MISSING:
            counts[b] += 1
    return counts


def hist_from_raw(raw_cache: dict, variable: str, year: int, doy: int,
                  iy: int, ix: int) -> np.ndarray:
    counts = np.zeros(N_BINS, dtype=np.int64)
    for y, d in window_positions(year, doy):
        if (variable, y) not in raw_cache:
            raw_cache[(variable, y)] = raw_year_values(variable, y)
        v = raw_cache[(variable, y)][d, iy, ix]
        if np.isfinite(v):
            b = temp_to_bin(np.array([v]))[0]
            counts[b] += 1
    return counts


def check_exactness() -> None:
    print("Check 1: layer-derived vs raw-derived window histograms")
    layers, raw_cache = {}, {}
    # deliberate edge cases + random draws
    cases = [
        ("tday", 1961, 2), ("tday", 2025, 362), ("tmin", 1999, 20),
        ("tmax", 2000, 195), ("tday", 1971, 364), ("tmax", 1972, 0),
    ]
    for _ in range(9):
        cases.append((
            list(VARIABLES)[RNG.integers(3)],
            int(RNG.integers(FIRST_YEAR, LAST_YEAR + 1)),
            int(RNG.integers(DAYS_PER_YEAR)),
        ))
    # a fixed known-land cell (Helsinki area) + random cells checked lazily
    n_ok = 0
    for variable, year, doy in cases:
        # pick a random land cell from the layer
        lay = load_layer(variable, year)
        land = np.argwhere(lay[doy] != MISSING)
        iy, ix = land[RNG.integers(len(land))]
        h_lay = hist_from_layers(layers, variable, year, doy, iy, ix)
        h_raw = hist_from_raw(raw_cache, variable, year, doy, iy, ix)
        assert h_lay.sum() > 0, f"empty window {variable} {year} doy={doy}"
        if not np.array_equal(h_lay, h_raw):
            raise AssertionError(
                f"MISMATCH {variable} year={year} doy={doy} cell=({iy},{ix})")
        n_ok += 1
        expected = len(window_positions(year, doy))
        print(f"  ok: {variable} year={year} doy={doy:3d} cell=({iy:3d},{ix:2d}) "
              f"n={h_lay.sum()} (window days available: {expected})")
    print(f"  {n_ok}/{len(cases)} exact matches\n")


def window_30y_hist(layers: dict, variable: str, ref_year: int, doy: int,
                    iy: int, ix: int) -> np.ndarray:
    counts = np.zeros(N_BINS, dtype=np.int64)
    for y in range(ref_year - 29, ref_year + 1):
        counts += hist_from_layers(layers, variable, y, doy, iy, ix)
    return counts


def nearest_cell(meta_grid: dict, e: float, n: float) -> tuple[int, int]:
    lon = np.asarray(meta_grid["lon_m"])
    lat = np.asarray(meta_grid["lat_m"])
    return int(np.abs(lat - n).argmin()), int(np.abs(lon - e).argmin())


def check_climatology() -> None:
    import json
    print("Check 2: climatology sanity (1971-2000 baseline vs 1996-2025)")
    meta = json.loads((Path(RAW_DIR).parent.parent / "derived" / "layers"
                       / "layers_meta.json").read_text())
    layers = {}
    hel = nearest_cell(meta["grid"], 385000, 6672000)   # Helsinki
    sod = nearest_cell(meta["grid"], 487000, 7480000)   # Sodankylä
    jul15, jan15 = 195, 14                              # 0-based doy

    def p_at_least(counts, x):
        i = int((x + 45) / 0.5)
        return counts[i:].sum() / counts.sum()

    def p_below(counts, x):
        i = int((x + 45) / 0.5)
        return counts[:i].sum() / counts.sum()

    h_base = window_30y_hist(layers, "tmax", 2000, jul15, *hel)
    h_new = window_30y_hist(layers, "tmax", 2025, jul15, *hel)
    p0, p1 = p_at_least(h_base, 25.0), p_at_least(h_new, 25.0)
    print(f"  Helsinki Jul 15, P(Tmax >= 25 C): 1971-2000 = {p0:.3f} "
          f"(n={h_base.sum()}), 1996-2025 = {p1:.3f} (n={h_new.sum()}), "
          f"ratio {p1/p0:.2f}x")
    assert 0.005 < p0 < 0.25 and p1 > p0, "implausible Helsinki summer result"

    h_base = window_30y_hist(layers, "tday", 2000, jan15, *sod)
    h_new = window_30y_hist(layers, "tday", 2025, jan15, *sod)
    q0, q1 = p_below(h_base, -20.0), p_below(h_new, -20.0)
    print(f"  Sodankylä Jan 15, P(Tday < -20 C): 1971-2000 = {q0:.3f} "
          f"(n={h_base.sum()}), 1996-2025 = {q1:.3f} (n={h_new.sum()}), "
          f"ratio {q1/q0:.2f}x")
    assert 0.02 < q0 < 0.6 and q1 < q0, "implausible Sodankylä winter result"
    print("  both checks plausible and in the expected direction\n")


if __name__ == "__main__":
    check_exactness()
    check_climatology()
    print("all verification checks passed")
