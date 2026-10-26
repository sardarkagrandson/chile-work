#!/usr/bin/env python3
"""Route 2: barred dwarfs with MUSE data, independent of star formation.

Cross-match the whole SGA-based bar catalogue (89,780 galaxies) with the
cached ESO MUSE product lists (from step 3 / notebook section 2), then keep
the dwarfs. Star formation is not required for the TW method, and the deep
archival MUSE cubes of dwarfs are mostly quiescent cluster members (Virgo,
Fornax), i.e. the population of the three existing dwarf TW measurements.

Dwarf cut without DESI spectroscopy: absolute magnitude from the SGA r-band
isophotal magnitude and the LEDA redshift (SGA-2020.fits, Z_LEDA) or the
DESI redshift when the galaxy is in the DESI dwarf table. M_r > -18 is
roughly log M*/Msun < 9.3 for a red dwarf (M/L ~ 1.5). Galaxies without any
redshift are kept in a separate list for a NED look-up.

    python scripts/02b_bar_catalog_muse.py [--mr-faint -18] [--bar-ellip-min 0.3]

Writes products/sga_barred_muse_candidates.csv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import numpy as np
import pandas as pd

from bartw.bar_catalog import read_bar_catalog, read_sga, join_sga
from bartw.config import ROOT, data_dir, load_config
from bartw.geometry import inclination_from_axis_ratio
from bartw.muse import crossmatch_local, summarize_coverage

ap = argparse.ArgumentParser()
ap.add_argument("--bars", default=str(data_dir("raw") / "desi_legacy_bars_sga.fits"))
ap.add_argument("--sga", default=str(data_dir("raw") / "SGA-2020.fits"))
ap.add_argument("--mr-faint", type=float, default=-18.0, help="keep M_r fainter than this (dwarfs)")
ap.add_argument("--bar-ellip-min", type=float, default=0.3)
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
H0 = cfg["cosmology"]["H0"]

bars = read_bar_catalog(args.bars)
print(f"bar catalogue: {len(bars)} galaxies")
try:
    bars = join_sga(bars, read_sga(args.sga))
    print(f"  SGA-2020 joined: PA/BA for {bars['SGA_PA'].notna().sum()}, Z_LEDA for {bars['SGA_Z_LEDA'].notna().sum()}")
except FileNotFoundError:
    print("  SGA-2020.fits not found: no LEDA redshifts / disc geometry (see scripts/download_data.sh)")
    for c in ("SGA_PA", "SGA_BA", "SGA_Z_LEDA", "SGA_D26"):
        bars[c] = np.nan

# DESI redshifts for the galaxies that are in the paper's dwarf table
z = bars["SGA_Z_LEDA"].to_numpy(float)
z_src = np.where(np.isfinite(z) & (z > 0), "LEDA", "")
paper = data_dir("interim") / "01_sf_dwarfs.parquet"
if paper.exists():
    sf = pd.read_parquet(paper)[["sga_id", "Z"]].dropna()
    m = bars.merge(sf, left_on="SGA_ID", right_on="sga_id", how="left")
    use = m["Z"].notna().to_numpy()
    z = np.where(use, m["Z"].to_numpy(float), z)
    z_src = np.where(use, "DESI", z_src)
bars["Z_BEST"], bars["Z_SOURCE"] = z, z_src
ok_z = np.isfinite(z) & (z > 0)
dist = np.where(ok_z, 299792.458 * z / H0, np.nan)                      # Mpc, Hubble flow
bars["DIST_MPC"] = dist
bars["M_R"] = bars["R_MAG_SB25"] - 5 * np.log10(dist * 1e6 / 10)
bars.loc[bars["R_MAG_SB25"] <= 0, "M_R"] = np.nan

# MUSE coverage from the cached product lists
cube_cache = data_dir("raw") / "eso_muse_all_cube.parquet"
raw_cache = data_dir("raw") / "eso_muse_all_raw.parquet"
if not cube_cache.exists():
    raise SystemExit("cached MUSE lists not found; run scripts/03_query_muse.py or notebook section 2 first")
r = cfg["muse"]["search_radius_arcmin"]
hits = [crossmatch_local(bars["SGA_RA"], bars["SGA_DEC"], pd.read_parquet(cube_cache), "cube", r)]
if raw_cache.exists():
    hits.append(crossmatch_local(bars["SGA_RA"], bars["SGA_DEC"], pd.read_parquet(raw_cache), "raw", r))
hits = pd.concat(hits, ignore_index=True)
sample = bars.rename(columns={"SGA_RA": "RA", "SGA_DEC": "DEC"}).reset_index(drop=True)
cov = summarize_coverage(sample, hits, cfg["muse"])
has = cov[(cov["N_MUSE_CUBES"] > 0) | cov["MUSE_RAW_ONLY"]].copy()
print(f"bar-catalogue galaxies with MUSE data: {len(has)} (cubes: {int((has['N_MUSE_CUBES'] > 0).sum())})")

has["INCL_DEG"] = inclination_from_axis_ratio(has["SGA_BA"], cfg["geometry"]["q0"])
has["BAR_ABOVE_DISC"] = has["BAR_ELLIP"] - (1 - has["SGA_BA"])
strong = has["BAR_ELLIP"] >= args.bar_ellip_min
dwarf = has["M_R"] > args.mr_faint
noz = ~np.isfinite(has["M_R"])
print(f"  bar_ellip >= {args.bar_ellip_min}: {int(strong.sum())};  of which dwarfs (M_r > {args.mr_faint}): {int((strong & dwarf).sum())};"
      f"  without redshift: {int((strong & noz).sum())}")
out = has[strong & (dwarf | noz)].copy()
out["CLASS"] = np.where(dwarf[strong & (dwarf | noz)], "dwarf", "no redshift")
out["RESOLVED"] = out["Z_BEST"] < cfg["desi"]["z_max"]
out = out.sort_values(["CLASS", "RESOLVED", "BAR_ELLIP"], ascending=[True, False, False])
cols = ["SGA_ID", "GALAXY", "RA", "DEC", "Z_BEST", "Z_SOURCE", "DIST_MPC", "R_MAG_SB25", "M_R", "SGA_D26", "SGA_PA",
        "SGA_BA", "INCL_DEG", "BAR_ELLIP", "BAR_ABOVE_DISC", "N_MUSE_CUBES", "MUSE_EXPTIME_S", "MUSE_PUBLIC",
        "MUSE_RAW_ONLY", "MUSE_PROG_IDS", "CLASS", "RESOLVED"]
prod = ROOT / "products"; prod.mkdir(exist_ok=True)
out[[c for c in cols if c in out]].round(4).to_csv(prod / "sga_barred_muse_candidates.csv", index=False)
print(f"wrote products/sga_barred_muse_candidates.csv ({len(out)} rows)")
print(out[[c for c in ("GALAXY", "Z_BEST", "M_R", "INCL_DEG", "BAR_ELLIP", "MUSE_EXPTIME_S", "CLASS") if c in out]].head(25).to_string())
