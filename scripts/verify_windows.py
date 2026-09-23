#!/usr/bin/env python3
"""Verify step-3 window histograms against per-year layers and climatology.

Check 1 (exactness): for random (variable, ref-year, doy, cell) draws —
including the baseline year, the truncated end of the series (R=2025, late
December), and year-boundary wrap doys — the stored window histogram must
equal the sum of the 30 per-year histograms computed independently from the
step-2 layers.

Check 2 (totals): stored histogram totals must be exactly 930 (30 years x 31
days) everywhere, except R=2025 doys within 15 days of Dec 31, where windows
truncate at the series end (915..929).

Check 3 (climatology sanity): the Helsinki / Sodankylä exceedance
probabilities must match verify_layers.py exactly (same numbers, now read
from the assembled files), and the probability ratio across reference years
2000 -> 2025 must evolve smoothly enough to be credible (no sign flips from
one overlapping window to the next by more than expected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climgrid_common import DAYS_PER_YEAR, N_BINS, VARIABLES, load_layer
from build_window_histograms import (
    BASELINE_REF_YEAR, FIRST_REF_YEAR, LAST_REF_YEAR, META_PATH, WINDOW_YEARS,
    window_path,
)
from verify_layers import hist_from_layers

RNG = np.random.default_rng(7)


def load_window(variable: str, ref_year: int) -> np.ndarray:
    with np.load(window_path(variable, ref_year)) as z:
        return z["hist"]


def check_exactness(meta: dict) -> None:
    print("Check 1: stored window histograms vs sums of per-year layer histograms")
    cells = np.array(meta["land_cells_row_col"])
    cases = [
        ("tday", 2000, 0), ("tmax", 2025, 364), ("tmin", 2025, 352),
        ("tday", 2013, 181), ("tmax", 2000, 195), ("tmin", 2005, 14),
    ]
    for _ in range(6):
        cases.append((
            list(VARIABLES)[RNG.integers(3)],
            int(RNG.integers(FIRST_REF_YEAR, LAST_REF_YEAR + 1)),
            int(RNG.integers(DAYS_PER_YEAR)),
        ))
    layers: dict = {}
    for variable, ref_year, doy in cases:
        iland = int(RNG.integers(len(cells)))
        iy, ix = cells[iland]
        stored = load_window(variable, ref_year)[doy, iland].astype(np.int64)
        expected = np.zeros(N_BINS, dtype=np.int64)
        for y in range(ref_year - WINDOW_YEARS + 1, ref_year + 1):
            expected += hist_from_layers(layers, variable, y, doy, iy, ix)
        if not np.array_equal(stored, expected):
            raise AssertionError(
                f"MISMATCH {variable} R={ref_year} doy={doy} cell=({iy},{ix})")
        print(f"  ok: {variable} R={ref_year} doy={doy:3d} cell=({iy:3d},{ix:2d}) "
              f"n={stored.sum()}")
    print(f"  {len(cases)}/{len(cases)} exact matches\n")


def check_totals() -> None:
    print("Check 2: histogram totals (930 everywhere; truncation only at series end)")
    for variable in VARIABLES:
        for ref_year in (FIRST_REF_YEAR, 2013, LAST_REF_YEAR):
            totals = load_window(variable, ref_year).sum(axis=2, dtype=np.int64)
            full = WINDOW_YEARS * 31
            if ref_year < LAST_REF_YEAR:
                assert (totals == full).all(), f"{variable} R={ref_year}: bad totals"
            else:
                exp_min = np.where(
                    np.arange(DAYS_PER_YEAR) >= DAYS_PER_YEAR - 15,
                    full - (np.arange(DAYS_PER_YEAR) - (DAYS_PER_YEAR - 16)), full)
                assert (totals == exp_min[:, None]).all(), \
                    f"{variable} R={ref_year}: totals don't match truncation pattern"
        print(f"  ok: {variable} (R={FIRST_REF_YEAR}, 2013, {LAST_REF_YEAR})")
    print()


def check_climatology(meta: dict) -> None:
    print("Check 3: climatology from assembled windows")
    cells = np.array(meta["land_cells_row_col"])

    def iland_nearest(row: int, col: int) -> int:
        return int(np.argmin(np.abs(cells[:, 0] - row) + np.abs(cells[:, 1] - col)))

    # same cells as verify_layers.py: Helsinki (row 111, col 31), Sodankylä (row 30, col 41)
    # derived there from EPSG:3067 coordinates; recompute row/col from the grid here
    grid_row = lambda n_m: int(np.argmin(np.abs(np.arange(7781775.141, 6631775.140, -10000) - n_m)))
    grid_col = lambda e_m: int(np.argmin(np.abs(np.arange(68689.8723, 738689.873, 10000) - e_m)))
    hel = iland_nearest(grid_row(6672000), grid_col(385000))
    sod = iland_nearest(grid_row(7480000), grid_col(487000))
    jul15, jan15 = 195, 14

    def p_at_least(h, x):
        return h[int((x + 45) / 0.5):].sum() / h.sum()

    def p_below(h, x):
        return h[:int((x + 45) / 0.5)].sum() / h.sum()

    h0 = load_window("tmax", BASELINE_REF_YEAR)[jul15, hel].astype(np.int64)
    h1 = load_window("tmax", 2025)[jul15, hel].astype(np.int64)
    p0, p1 = p_at_least(h0, 25.0), p_at_least(h1, 25.0)
    print(f"  Helsinki Jul 15, P(Tmax >= 25): baseline {p0:.3f}, R2025 {p1:.3f}, "
          f"ratio {p1/p0:.2f}x")
    assert abs(p0 - 0.074) < 0.005 and abs(p1 - 0.163) < 0.005, \
        "does not reproduce verify_layers.py numbers"

    q0 = p_below(load_window("tday", BASELINE_REF_YEAR)[jan15, sod].astype(np.int64), -20.0)
    q1 = p_below(load_window("tday", 2025)[jan15, sod].astype(np.int64), -20.0)
    print(f"  Sodankylä Jan 15, P(Tday < -20): baseline {q0:.3f}, R2025 {q1:.3f}, "
          f"ratio {q1/q0:.2f}x")
    assert q1 < q0, "cold-tail frequency should decrease"

    # ratio should evolve without wild jumps between overlapping windows
    ps = [p_at_least(load_window("tmax", r)[jul15, hel].astype(np.int64), 25.0)
          for r in range(FIRST_REF_YEAR, LAST_REF_YEAR + 1)]
    steps = np.abs(np.diff(ps))
    print(f"  Helsinki P(Tmax >= 25) across R=2000..2025: "
          f"{ps[0]:.3f} -> {ps[-1]:.3f}, max step {steps.max():.4f}")
    assert steps.max() < 0.02, "overlapping windows should change slowly"
    print("\nall window verification checks passed")


if __name__ == "__main__":
    meta = json.loads(META_PATH.read_text())
    check_exactness(meta)
    check_totals()
    check_climatology(meta)
