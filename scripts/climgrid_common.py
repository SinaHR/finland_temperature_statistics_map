"""Shared constants and helpers for the ClimGrid preprocessing pipeline.

Settled methodology (see README.md "Methodology" — do not change without asking):
  - histogram bins: 0.5 degC width spanning -45 .. +40 degC -> 170 bins;
  - Feb 29 dropped entirely; the calendar has 365 days (doy 0..364, 0-based);
  - +/-15-day calendar windows that wrap across the year boundary;
  - 30-year climate windows (R-29 .. R), baseline R = 2000 (1971-2000).

Per-year layers (step 2 output) store each daily value as its bin index
(uint8), shape (365, n_lat, n_lon), missing/sea = 255. The per-year histogram
of any (cell, doy, year) is the bincount of the layer values in the 31-day
window around (doy, year); 30-year window histograms are sums of those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "climgrid"
LAYERS_DIR = PROJECT_ROOT / "data" / "derived" / "layers"

# variable key -> data variable name inside the NetCDF files
VARIABLES = {"tday": "Tday", "tmax": "Tmax", "tmin": "Tmin"}

FIRST_YEAR = 1961
LAST_YEAR = 2025

# --- histogram binning (settled) ---
BIN_START = -45.0  # degC, left edge of bin 0
BIN_STOP = 40.0    # degC, right edge of last bin
BIN_WIDTH = 0.5    # degC
N_BINS = 170       # (BIN_STOP - BIN_START) / BIN_WIDTH
MISSING = np.uint8(255)  # sentinel for NaN / sea cells in layer files

DAYS_PER_YEAR = 365   # Feb 29 dropped
HALF_WINDOW = 15      # +/- days -> 31-day calendar window


def bin_edges() -> np.ndarray:
    """All 171 bin edges; bin i covers [edges[i], edges[i+1])."""
    return BIN_START + BIN_WIDTH * np.arange(N_BINS + 1)


def temp_to_bin(values: np.ndarray) -> np.ndarray:
    """Map temperatures (degC) to uint8 bin indices; NaN -> MISSING.

    Values outside [BIN_START, BIN_STOP] are clamped into the end bins, which
    keeps totals and within-range exceedance probabilities exact.
    """
    v = values.astype(np.float64)
    nan = np.isnan(v)
    idx = np.floor((np.where(nan, BIN_START, v) - BIN_START) / BIN_WIDTH)
    idx = np.clip(idx, 0, N_BINS - 1).astype(np.uint8)
    return np.where(nan, MISSING, idx)


def open_climgrid(path: Path) -> xr.Dataset:
    """Open a raw ClimGrid NetCDF with normalized coordinate names.

    Files up to 2024 use (Time, Lat, Lon); 2025+ files use
    (time, northing, easting) with an explicit CRS variable. The grid values
    are identical; normalize to the older names.
    """
    ds = xr.open_dataset(path)
    rename = {old: new for old, new in
              (("time", "Time"), ("northing", "Lat"), ("easting", "Lon"))
              if old in ds.coords}
    return ds.rename(rename) if rename else ds


def layer_path(variable: str, year: int) -> Path:
    return LAYERS_DIR / variable / f"{variable}_{year}_bins.npz"


def load_layer(variable: str, year: int) -> np.ndarray:
    """Load one per-year layer: uint8 (DAYS_PER_YEAR, n_lat, n_lon)."""
    with np.load(layer_path(variable, year)) as z:
        return z["bins"]
