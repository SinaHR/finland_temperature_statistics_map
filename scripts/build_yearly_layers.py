#!/usr/bin/env python3
"""Build per-year binned layers from raw ClimGrid NetCDF files (pipeline step 2).

For each (variable, year), reads the raw daily grid, drops Feb 29, validates
full contiguous calendar coverage and grid consistency, maps every daily value
to its 0.5 degC histogram bin (see climgrid_common), and writes a compressed
uint8 layer of shape (365, n_lat, n_lon) with 255 = missing/sea.

These layers are the "per-year histogram" building blocks: the histogram of
any (cell, doy, year) is the bincount of the 31-day window slice around
(doy, year), and 30-year window histograms (step 3) are sums over years.
Storing bin indices instead of materialized per-doy histograms is lossless
and ~200x smaller.

Idempotent: existing layers with a matching metadata entry are skipped
(--force rebuilds). Metadata, per-layer QA stats, and the source file sha256
(from the download manifest) go to data/derived/layers/layers_meta.json.

Usage:
    python scripts/build_yearly_layers.py                   # everything
    python scripts/build_yearly_layers.py --variables tmin --years 1999 1999
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climgrid_common import (
    BIN_START, BIN_STOP, BIN_WIDTH, DAYS_PER_YEAR, FIRST_YEAR, LAST_YEAR,
    LAYERS_DIR, MISSING, N_BINS, RAW_DIR, VARIABLES, layer_path, open_climgrid,
    temp_to_bin,
)

META_PATH = LAYERS_DIR / "layers_meta.json"


def month_day(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized (month, day-of-month) from datetime64 array."""
    months = (times.astype("M8[M]") - times.astype("M8[Y]")).astype(int) + 1
    days = (times.astype("M8[D]") - times.astype("M8[M]")).astype(int) + 1
    return months, days


def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {
        "bin_spec": {
            "start_degC": BIN_START, "stop_degC": BIN_STOP,
            "width_degC": BIN_WIDTH, "n_bins": N_BINS,
            "missing_sentinel": int(MISSING),
            "note": "bin i covers [start + i*width, start + (i+1)*width); "
                    "out-of-range values clamped into end bins",
        },
        "calendar": "365-day, Feb 29 dropped, doy 0-based from Jan 1",
        "grid": None,
        "layers": {},
    }


def save_meta(meta: dict) -> None:
    tmp = META_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    tmp.replace(META_PATH)


def build_one(variable: str, year: int, meta: dict, manifest: dict) -> str:
    key = f"{variable}_{year}"
    out_path = layer_path(variable, year)
    raw_path = RAW_DIR / variable / f"{variable}_{year}.nc"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw file missing: {raw_path} (run download_climgrid.py)")

    with open_climgrid(raw_path) as ds:
        da = ds[VARIABLES[variable]]
        times = ds["Time"].values.astype("M8[D]")

        # validate: contiguous daily coverage of the full calendar year
        if str(times[0]) != f"{year}-01-01" or str(times[-1]) != f"{year}-12-31":
            raise ValueError(f"{key}: time axis does not span the full year")
        if not np.all(np.diff(times).astype(int) == 1):
            raise ValueError(f"{key}: time axis is not contiguous daily")

        # validate: grid identical across all files
        grid = {
            "lon_m": np.round(ds["Lon"].values, 4).tolist(),
            "lat_m": np.round(ds["Lat"].values, 4).tolist(),
            "crs": ds.attrs.get("crs", ""),
        }
        if meta["grid"] is None:
            meta["grid"] = grid
        elif grid["lon_m"] != meta["grid"]["lon_m"] or grid["lat_m"] != meta["grid"]["lat_m"]:
            raise ValueError(f"{key}: grid differs from reference grid")

        # drop Feb 29
        months, days = month_day(times)
        keep = ~((months == 2) & (days == 29))
        values = da.values[keep]
        if values.shape[0] != DAYS_PER_YEAR:
            raise ValueError(f"{key}: {values.shape[0]} days after dropping Feb 29")

    finite = np.isfinite(values)
    bins = temp_to_bin(values)

    # QA stats
    n_finite_per_day = finite.sum(axis=(1, 2))
    stats = {
        "source_sha256": manifest.get(key, {}).get("sha256"),
        "n_days": int(values.shape[0]),
        "cells_with_data_min_per_day": int(n_finite_per_day.min()),
        "cells_with_data_max_per_day": int(n_finite_per_day.max()),
        "clamped_below_range": int((values[finite] < BIN_START).sum()),
        "clamped_above_range": int((values[finite] > BIN_STOP).sum()),
        "value_min_degC": float(np.nanmin(values)),
        "value_max_degC": float(np.nanmax(values)),
    }

    # self-check on a random sample: bin index must contain the (clamped) value
    rng = np.random.default_rng(year)
    flat_idx = rng.choice(np.flatnonzero(finite.ravel()), size=200)
    v = values.ravel()[flat_idx].astype(np.float64)
    b = bins.ravel()[flat_idx].astype(np.float64)
    v_clamped = np.clip(v, BIN_START, np.nextafter(BIN_STOP, -np.inf))
    lo = BIN_START + b * BIN_WIDTH
    if not np.all((v_clamped >= lo) & (v_clamped < lo + BIN_WIDTH)):
        raise AssertionError(f"{key}: bin self-check failed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, bins=bins)
    tmp.replace(out_path)

    meta["layers"][key] = stats
    return "built"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variables", nargs="+", choices=sorted(VARIABLES),
                        default=sorted(VARIABLES))
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        default=[FIRST_YEAR, LAST_YEAR])
    parser.add_argument("--force", action="store_true",
                        help="rebuild layers even if they already exist")
    args = parser.parse_args()

    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()

    first, last = args.years
    tasks = [(v, y) for v in args.variables for y in range(first, last + 1)]
    print(f"{len(tasks)} layers: {args.variables}, {first}-{last} -> {LAYERS_DIR}")

    counts = {"built": 0, "skipped": 0}
    for i, (variable, year) in enumerate(tasks, 1):
        key = f"{variable}_{year}"
        if not args.force and layer_path(variable, year).exists() and key in meta["layers"]:
            counts["skipped"] += 1
            continue
        status = build_one(variable, year, meta, manifest)
        counts[status] += 1
        if i % 10 == 0 or i == len(tasks):
            save_meta(meta)
            print(f"[{i}/{len(tasks)}] {key}: {status}")

    save_meta(meta)

    clamped_lo = sum(s["clamped_below_range"] for s in meta["layers"].values())
    clamped_hi = sum(s["clamped_above_range"] for s in meta["layers"].values())
    print(f"\ndone: {counts['built']} built, {counts['skipped']} skipped")
    print(f"clamped values across all layers: {clamped_lo} below {BIN_START} degC, "
          f"{clamped_hi} above {BIN_STOP} degC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
