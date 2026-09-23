#!/usr/bin/env python3
"""Export static web assets from window histograms (pipeline step 5).

Produces the files the frontend fetches directly (no backend):

data/web/meta.json
    Grid geometry (land cells with EPSG:3067 and WGS84 centers), bin spec,
    week -> day-of-year mapping, reference years, provenance and license.

data/web/hist/<var>/R<year>/w<week>.bin
    One file per (variable, reference year, week): the 30-year window
    histograms of all land cells at that week's representative day-of-year.
    Raw little-endian uint16, shape (n_land, 170), C order — cell index
    matches meta.json's land-cell list, bin i covers
    [-45 + 0.5*i, -45 + 0.5*(i+1)) degC. ~1.3 MB raw, ~200 KB gzipped.

Day-of-year reduction (part of the settled methodology): queries snap to the nearest
of 52 weekly representative doys (3, 10, ..., 360; max snap distance 4 days).
The +/-15-day pooling makes neighboring-doy windows nearly identical, so
snapping changes at most 4 of 31 pooled days per window year.

The map layer and the cell detail view are served by the same files: a map
query for (variable, R, week) fetches that file plus the baseline
(variable, R=2000, week); clicking a cell reads its row from data already
fetched. Threshold and direction changes need no fetch at all.

Idempotent: existing .bin files are skipped unless --force; meta.json is
always (re)written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climgrid_common import (
    BIN_START, BIN_STOP, BIN_WIDTH, DAYS_PER_YEAR, N_BINS, PROJECT_ROOT,
    VARIABLES,
)
from build_window_histograms import (
    BASELINE_REF_YEAR, FIRST_REF_YEAR, LAST_REF_YEAR, META_PATH as WINDOWS_META,
    window_path,
)

WEB_DIR = PROJECT_ROOT / "data" / "web"
N_WEEKS = 52
WEEK_DOYS = [k * 7 + 3 for k in range(N_WEEKS)]  # 0-based doys 3, 10, ..., 360


def hist_bin_path(variable: str, ref_year: int, week: int) -> Path:
    return WEB_DIR / "hist" / variable / f"R{ref_year}" / f"w{week:02d}.bin"


def write_meta(windows_meta: dict) -> None:
    layers_meta = json.loads(
        (PROJECT_ROOT / "data" / "derived" / "layers" / "layers_meta.json").read_text())
    lon_m = np.asarray(layers_meta["grid"]["lon_m"])   # cell-center easting
    lat_m = np.asarray(layers_meta["grid"]["lat_m"])   # cell-center northing
    cells = np.asarray(windows_meta["land_cells_row_col"])
    east = lon_m[cells[:, 1]]
    north = lat_m[cells[:, 0]]
    to_wgs84 = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)
    wlon, wlat = to_wgs84.transform(east, north)

    meta = {
        "provenance": {
            "dataset": "FMI ClimGrid daily gridded observations, ~10 km, Finland",
            "source": "Finnish Meteorological Institute via Paituli (CSC), "
                      "https://www.nic.funet.fi/index/geodata/ilmatiede/",
            "license": "CC BY 4.0, (c) Finnish Meteorological Institute",
            "reference": "Aalto et al. (2016), doi:10.1002/2015JD024651",
            "years": "1961-2025",
            "download_manifest": "data/raw/climgrid/manifest.json (sha256 per file)",
        },
        "crs": "EPSG:3067 (ETRS89 / TM35FIN); wgs84 centers provided for convenience",
        "cell_size_m": 10000,
        "n_cells": int(len(cells)),
        "cells": {
            "row_col": cells.tolist(),
            "east_m": np.round(east, 1).tolist(),
            "north_m": np.round(north, 1).tolist(),
            "lon_wgs84": np.round(wlon, 5).tolist(),
            "lat_wgs84": np.round(wlat, 5).tolist(),
        },
        "bins": {
            "start_degC": BIN_START, "stop_degC": BIN_STOP,
            "width_degC": BIN_WIDTH, "n_bins": N_BINS,
            "note": "bin i covers [start+i*width, start+(i+1)*width); "
                    "out-of-range daily values were clamped into end bins",
        },
        "weeks": {
            "doys_0based": WEEK_DOYS,
            "note": "365-day calendar, Feb 29 dropped; queries snap to nearest "
                    "weekly doy (max 4 days; windows pool +/-15 days)",
        },
        "ref_years": list(range(FIRST_REF_YEAR, LAST_REF_YEAR + 1)),
        "baseline_ref_year": BASELINE_REF_YEAR,
        "window_convention": windows_meta["window_convention"],
        "hist_files": {
            "path": "hist/<var>/R<year>/w<week>.bin",
            "variables": sorted(VARIABLES),
            "format": "raw little-endian uint16, shape (n_cells, n_bins), C order",
            "note": "probabilities = tail sums / total; totals are 930 except "
                    "R=2025 late-December truncation (>=915); a histogram row of "
                    "all zeros cannot occur on land cells",
        },
    }
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    (WEB_DIR / "meta.json").write_text(json.dumps(meta) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variables", nargs="+", choices=sorted(VARIABLES),
                        default=sorted(VARIABLES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    windows_meta = json.loads(WINDOWS_META.read_text())
    write_meta(windows_meta)
    print(f"meta.json written ({WEB_DIR / 'meta.json'})")

    doys = np.asarray(WEEK_DOYS)
    n_written = n_skipped = 0
    for variable in args.variables:
        for ref_year in range(FIRST_REF_YEAR, LAST_REF_YEAR + 1):
            targets = [hist_bin_path(variable, ref_year, w) for w in range(N_WEEKS)]
            if not args.force and all(t.exists() for t in targets):
                n_skipped += N_WEEKS
                continue
            with np.load(window_path(variable, ref_year)) as z:
                weekly = z["hist"][doys]          # (N_WEEKS, n_land, N_BINS)
            for week, target in enumerate(targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".bin.tmp")
                tmp.write_bytes(np.ascontiguousarray(
                    weekly[week], dtype="<u2").tobytes())
                tmp.replace(target)
                n_written += 1
            print(f"  {variable} R{ref_year}: {N_WEEKS} weekly files")
    print(f"done: {n_written} written, {n_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
