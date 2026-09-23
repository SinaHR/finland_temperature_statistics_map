#!/usr/bin/env python3
"""Download FMI ClimGrid daily temperature grids for Finland.

Source: FMI (Finnish Meteorological Institute) daily gridded observational
dataset ("ClimGrid", ~10 km, EPSG:3067), 1961 onward. License CC BY 4.0.
Distributed via the Paituli spatial data service (CSC), mirrored on the
funet.fi file server as one NetCDF file per variable per year:

    https://www.nic.funet.fi/index/geodata/ilmatiede/
        10km_daily_mean_temperature/netcdf/tday_<year>.nc
        10km_daily_maximum_temperature/netcdf/tmax_<year>.nc
        10km_daily_minimum_temperature/netcdf/tmin_<year>.nc

Dataset reference: Aalto et al. (2016), "New gridded daily climatology of
Finland: Permutation-based uncertainty estimates and temporal trends in
climate", JGR Atmospheres, doi:10.1002/2015JD024651.

The script is idempotent and restartable:
  - files are downloaded to a .part file and renamed only when complete;
  - existing files whose size matches the remote Content-Length are skipped;
  - every completed download is recorded in data/raw/climgrid/manifest.json
    with URL, size, sha256, remote Last-Modified, and download time, so the
    exact dataset version used is pinned and auditable.

Usage:
    python scripts/download_climgrid.py                 # everything, 1961-2025
    python scripts/download_climgrid.py --variables tday --years 2020 2025
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://www.nic.funet.fi/index/geodata/ilmatiede"

# variable name (as used in file names and NetCDF) -> directory on the server
VARIABLES = {
    "tday": "10km_daily_mean_temperature",   # daily mean temperature
    "tmax": "10km_daily_maximum_temperature",  # daily maximum temperature
    "tmin": "10km_daily_minimum_temperature",  # daily minimum temperature
}

FIRST_YEAR = 1961
LAST_YEAR = 2025  # latest complete year available on the server

CHUNK_SIZE = 1 << 20  # 1 MiB
MAX_RETRIES = 4
RETRY_BACKOFF_S = 5.0
TIMEOUT_S = 60


def file_url(variable: str, year: int) -> str:
    return f"{BASE_URL}/{VARIABLES[variable]}/netcdf/{variable}_{year}.nc"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def download_one(
    session: requests.Session, variable: str, year: int, out_dir: Path, manifest: dict
) -> str:
    """Download one file if needed. Returns 'downloaded', 'skipped', or 'failed'."""
    url = file_url(variable, year)
    dest = out_dir / variable / f"{variable}_{year}.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = f"{variable}_{year}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            head = session.head(url, timeout=TIMEOUT_S)
            head.raise_for_status()
            remote_size = int(head.headers["Content-Length"])
            remote_mtime = head.headers.get("Last-Modified", "")

            if dest.exists() and dest.stat().st_size == remote_size:
                if key not in manifest:  # backfill manifest for pre-existing files
                    manifest[key] = {
                        "url": url,
                        "size": remote_size,
                        "sha256": sha256_of(dest),
                        "last_modified": remote_mtime,
                        "downloaded_at": None,
                    }
                return "skipped"

            part = dest.with_suffix(".nc.part")
            with session.get(url, stream=True, timeout=TIMEOUT_S) as resp:
                resp.raise_for_status()
                with part.open("wb") as f:
                    for chunk in resp.iter_content(CHUNK_SIZE):
                        f.write(chunk)

            if part.stat().st_size != remote_size:
                raise IOError(
                    f"size mismatch: got {part.stat().st_size}, expected {remote_size}"
                )
            part.replace(dest)

            manifest[key] = {
                "url": url,
                "size": remote_size,
                "sha256": sha256_of(dest),
                "last_modified": remote_mtime,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            return "downloaded"

        except (requests.RequestException, IOError) as exc:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed for {key}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)

    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=sorted(VARIABLES),
        default=sorted(VARIABLES),
        help="variables to download (default: all)",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        default=[FIRST_YEAR, LAST_YEAR],
        help=f"inclusive year range (default: {FIRST_YEAR} {LAST_YEAR})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "climgrid",
        help="output directory (default: data/raw/climgrid)",
    )
    args = parser.parse_args()

    first, last = args.years
    if not (FIRST_YEAR <= first <= last <= LAST_YEAR):
        parser.error(f"years must satisfy {FIRST_YEAR} <= FIRST <= LAST <= {LAST_YEAR}")

    out_dir: Path = args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    tasks = [(v, y) for v in args.variables for y in range(first, last + 1)]
    print(f"{len(tasks)} files: {args.variables}, {first}-{last} -> {out_dir}")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failed: list[str] = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "finland-climate-map/0.1 (data download script)"
        for i, (variable, year) in enumerate(tasks, 1):
            status = download_one(session, variable, year, out_dir, manifest)
            counts[status] += 1
            if status == "failed":
                failed.append(f"{variable}_{year}")
            if status != "skipped" or i % 25 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {variable}_{year}: {status}")
            if i % 10 == 0 or i == len(tasks):
                save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)
    print(f"\ndone: {counts['downloaded']} downloaded, "
          f"{counts['skipped']} skipped, {counts['failed']} failed")
    if failed:
        print("failed files: " + ", ".join(failed))
        print("re-run the script to retry; completed files are skipped.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
