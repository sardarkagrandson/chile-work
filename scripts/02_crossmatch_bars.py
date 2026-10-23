#!/usr/bin/env python3
"""Step 2: barred subsample and MUSE-query candidate list.

Bar indicators attached to the star-forming dwarfs:
  (a) the SGA-based DESI Legacy Imaging bar catalogue (V. Cuomo): joined on
      SGA_ID when the dwarf is an SGA galaxy (797 of 14,786), else by position;
  (b) Galaxy Zoo DESI vote fractions, if the catalogue is available locally.
Then the TW geometric pre-selection (inclination from SGA/Tractor b/a; the
bar PA is not known yet, so only the inclination is applied here) and a
priority tier for the MUSE archive query.

    python scripts/02_crossmatch_bars.py [--bar-ellip-min 0.3]

Writes data/interim/02_barred_candidates.parquet and the small, git-tracked
products/barred_dwarf_candidates.csv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import numpy as np
import pandas as pd

from bartw.bar_catalog import crossmatch_desi, read_bar_catalog
from bartw.config import ROOT, data_dir, load_config
from bartw.morphology import crossmatch, read_gz_desi, select_barred

ap = argparse.ArgumentParser()
ap.add_argument("--bars", default=str(data_dir("raw") / "desi_legacy_bars_sga.fits"))
ap.add_argument("--gz", default=str(data_dir("raw") / "gz_desi_deep_learning_catalog_friendly.parquet"))
ap.add_argument("--bar-ellip-min", type=float, default=0.3)
ap.add_argument("--bar-above-disc-min", type=float, default=0.1)
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
g = cfg["geometry"]

sf = pd.read_parquet(data_dir("interim") / "01_sf_dwarfs.parquet")
print(f"star-forming dwarfs: {len(sf)}")

# (a) bar catalogue: SGA_ID join first, positional match for the rest
bars = read_bar_catalog(args.bars)
by_id = sf.merge(bars.drop(columns=["GALAXY"]), left_on="sga_id", right_on="SGA_ID", how="left")
df = crossmatch_desi(sf, bars, radius_arcsec=3.0)
for col in ["BAR_LENGTH_RAW", "BAR_ELLIP", "SGA_RA", "SGA_DEC"]:
    df[col] = np.where(by_id["SGA_ID"].notna(), by_id[col], df[col])
df["IN_BAR_CATALOG"] = by_id["SGA_ID"].notna() | df["IN_BAR_CATALOG"]
df["BAR_MATCH"] = np.where(by_id["SGA_ID"].notna(), "SGA_ID", np.where(df["IN_BAR_CATALOG"], "position", ""))
# The catalogue records a bar *measurement* for every entry, not a detection:
# require the bar to be (i) elongated enough and (ii) more elongated than the
# inclined disc, otherwise the "bar" fit may just trace the disc isophotes.
df["BAR_MINUS_DISC_ELLIP"] = df["BAR_ELLIP"] - (1.0 - df["DISC_BA"])
df["BAR_STRONG_ENOUGH"] = df["IN_BAR_CATALOG"] & (df["BAR_ELLIP"] >= args.bar_ellip_min)
df["BAR_ABOVE_DISC"] = df["IN_BAR_CATALOG"] & (df["BAR_MINUS_DISC_ELLIP"] >= args.bar_above_disc_min)
df["BAR_IMPLAUSIBLE"] = df["IN_BAR_CATALOG"] & (df["BAR_ELLIP"] > 0.7)
df["BARRED_SGA"] = df["BAR_STRONG_ENOUGH"] & df["BAR_ABOVE_DISC"] & ~df["BAR_IMPLAUSIBLE"]
print(f"  in bar catalogue: {int(df['IN_BAR_CATALOG'].sum())}  (SGA dwarfs in sample: {int(sf['sga_id'].notna().sum())})")
print(f"  bar_ellip >= {args.bar_ellip_min}: {int(df['BAR_STRONG_ENOUGH'].sum())};  bar more elongated than disc by >= {args.bar_above_disc_min}: "
      f"{int(df['BAR_ABOVE_DISC'].sum())};  bar_ellip > 0.7 (implausible): {int(df['BAR_IMPLAUSIBLE'].sum())}")
print(f"  BARRED_SGA (all three): {int(df['BARRED_SGA'].sum())}")

# (b) Galaxy Zoo DESI (optional)
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
    print("  GZ DESI catalogue not found locally: indicator (b) skipped")
    df["BARRED_GZ"] = False

df["BARRED"] = df["BARRED_SGA"] | df["BARRED_GZ"]
cand = df[df["BARRED"]].copy()
print(f"barred (either indicator): {len(cand)}")

# TW pre-selection on inclination only (bar PA comes from step 4)
cand["INCL_OK"] = (cand["INCL_DEG"] >= g["incl_min_deg"]) & (cand["INCL_DEG"] <= g["incl_max_deg"])
print(f"  inclination {g['incl_min_deg']}-{g['incl_max_deg']} deg: {int(cand['INCL_OK'].sum())}")
# priority for the MUSE query: resolution tier, bar strength, inclination
cand["PRIORITY"] = np.select(
    [cand["INCL_OK"] & (cand["Z_TIER"].str.startswith("A")) & (cand["BAR_ELLIP"] >= 0.4),
     cand["INCL_OK"] & (cand["Z_TIER"].str.startswith("A")),
     cand["INCL_OK"] & (cand["Z_TIER"].str.startswith("B"))],
    [1, 2, 3], 4)
print("  priority:", cand["PRIORITY"].value_counts().sort_index().to_dict())
cand = cand.sort_values(["PRIORITY", "BAR_ELLIP"], ascending=[True, False]).reset_index(drop=True)
cand.to_parquet(data_dir("interim") / "02_barred_candidates.parquet")
df.to_parquet(data_dir("interim") / "02_all_with_bar_flags.parquet")

small = ["TARGETID", "sga_galaxy", "pgc", "RA", "DEC", "Z", "luminosity_distance_mpc", "kpc_per_arcsec",
         "logmstar_h0678", "SFR_Msun_yr", "r_mag_dered", "sma_sb25", "DISC_PA", "DISC_BA", "DISC_PA_SOURCE",
         "INCL_DEG", "BAR_ELLIP", "BAR_MINUS_DISC_ELLIP", "BAR_LENGTH_RAW", "BAR_MATCH", "BARRED_GZ", "n_companions", "isolated",
         "Z_TIER", "INCL_OK", "PRIORITY"]
prod = ROOT / "products"; prod.mkdir(exist_ok=True)
cand[[c for c in small if c in cand]].round(4).to_csv(prod / "barred_dwarf_candidates.csv", index=False)
print(f"wrote products/barred_dwarf_candidates.csv ({len(cand)} rows) and data/interim/02_barred_candidates.parquet")
