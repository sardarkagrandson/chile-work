#!/usr/bin/env python3
"""Step 2: barred subsample.

Two independent bar indicators are attached to the star-forming dwarfs:
  (a) the SGA-based bar catalogue provided by V. Cuomo (bar_length, bar_ellipticity),
      joined with SGA-2020 for the disc PA / axis ratio / D26;
  (b) Galaxy Zoo DESI vote fractions (bar_strong/weak/no).
A galaxy is kept as "barred" if it passes (a) with bar_ellipticity above the
threshold OR (b) with the GZ cuts; the flags are kept so the two can be
compared (agreement between them is itself a useful preliminary result).

    python scripts/02_crossmatch_bars.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import numpy as np

from bartw.bar_catalog import crossmatch_desi, join_sga, read_bar_catalog, read_sga
from bartw.config import data_dir, load_config
from bartw.geometry import inclination_from_axis_ratio
from bartw.morphology import crossmatch, read_gz_desi, select_barred
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--bars", default=str(data_dir("raw") / "desi_legacy_bars_sga.fits"))
ap.add_argument("--sga", default=str(data_dir("raw") / "SGA-2020.fits"))
ap.add_argument("--gz", default=str(data_dir("raw") / "gz_desi_deep_learning_catalog_friendly.parquet"))
ap.add_argument("--bar-ellip-min", type=float, default=0.3)
ap.add_argument("--bar-length-unit", default=None, help="arcmin|arcsec once confirmed with the authors")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)

sf = pd.read_parquet(data_dir("interim") / "01_sf_dwarfs.parquet")
print(f"star-forming dwarfs: {len(sf)}")

# (a) SGA bar catalogue
bars = read_bar_catalog(args.bars, args.bar_length_unit)
try:
    sga = read_sga(args.sga)
    bars = join_sga(bars, sga)
    print(f"bar catalogue: {len(bars)} rows, SGA geometry joined for {bars['SGA_PA'].notna().sum()}")
except FileNotFoundError:
    print("SGA-2020.fits not found: disc geometry will come from the isophotal step only")
df = crossmatch_desi(sf, bars, radius_arcsec=3.0)
df["BARRED_SGA"] = df["IN_BAR_CATALOG"] & (df["BAR_ELLIP"] >= args.bar_ellip_min)
print(f"  in bar catalogue: {int(df['IN_BAR_CATALOG'].sum())};  bar_ellip >= {args.bar_ellip_min}: {int(df['BARRED_SGA'].sum())}")
if "SGA_BA" in df:
    df["INCL_SGA_DEG"] = inclination_from_axis_ratio(df["SGA_BA"], cfg["geometry"]["q0"])

# (b) Galaxy Zoo DESI
try:
    gz = read_gz_desi(args.gz)
    df = crossmatch(df, gz, cfg["galaxy_zoo_desi"]["match_radius_arcsec"])
    barred_gz, steps = select_barred(df, cfg["galaxy_zoo_desi"])
    for k, v in steps.items():
        print(f"  GZ: {k:25s} {v:>7d}")
    df["BARRED_GZ"] = df.index.isin(barred_gz.index)
    df["BAR_CLASS_GZ"] = np.nan
    df.loc[barred_gz.index, "BAR_CLASS_GZ"] = barred_gz["BAR_CLASS"]
except FileNotFoundError:
    print("GZ DESI catalogue not found: skipping indicator (b)")
    df["BARRED_GZ"] = False

df["BARRED"] = df["BARRED_SGA"] | df["BARRED_GZ"]
both = df["BARRED_SGA"] & df["BARRED_GZ"]
print(f"barred (either): {int(df['BARRED'].sum())}   both indicators: {int(both.sum())}")
out = data_dir("interim") / "02_barred_candidates.parquet"
df[df["BARRED"]].reset_index(drop=True).to_parquet(out)
df.to_parquet(data_dir("interim") / "02_all_with_bar_flags.parquet")
print(f"wrote {out}")
