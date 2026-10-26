#!/usr/bin/env python3
"""Route 2 with a DESI parent sample: bar catalogue + MUSE coverage for ANY
galaxy table (CSV, FITS or parquet) with RA, Dec, redshift and (optionally)
stellar mass -- e.g. the velocity-limited sample 200 < cz < 10,300 km/s with
no mass or magnitude cut. No star-formation cut is applied: the TW method
needs stellar kinematics only.

    python scripts/02c_any_sample_bars_muse.py TABLE \
        --ra-col target_ra --dec-col target_dec --z-col z --mass-col logmstar [--mass-h 1.0] \
        [--sga-col sga_id] [--logm-max 9.0] [--bar-ellip-min 0.3] [--name all_dwarfs]

Column names are case-insensitive. If --mass-col is omitted or the column is
missing, no mass cut is applied and LOGMSTAR is NaN. --mass-h is the h the
mass column assumes (FastSpecFit LOGMSTAR: 1.0; the paper table's
logmstar_h0678: 0.678); masses are rescaled to the config H0.

Writes products/<name>_barred_muse_candidates.csv and
data/interim/<name>_with_bar_flags.parquet.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import numpy as np
import pandas as pd

from bartw.bar_catalog import crossmatch_desi, read_bar_catalog
from bartw.config import ROOT, data_dir, load_config
from bartw.geometry import inclination_from_axis_ratio
from bartw.muse import crossmatch_local, summarize_coverage


def read_any(path):
    path = str(path)
    if path.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.io import fits
        from astropy.table import Table
        with fits.open(path, memmap=True) as h:
            t = Table(h[1].data)
        for c in t.colnames:                      # drop multidimensional columns, decode bytes
            if len(t[c].shape) > 1:
                t.remove_column(c)
        df = t.to_pandas()
        for c in df.columns:
            if df[c].dtype == object and isinstance(df[c].iloc[0], bytes):
                df[c] = df[c].str.decode("utf-8").str.strip()
        return df
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def find_col(df, name):
    if name is None:
        return None
    m = {c.lower(): c for c in df.columns}
    if name.lower() not in m:
        raise SystemExit(f"column {name!r} not in table; available: {list(df.columns)[:40]} ...")
    return m[name.lower()]


ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("--ra-col", default="target_ra"); ap.add_argument("--dec-col", default="target_dec")
ap.add_argument("--z-col", default="z"); ap.add_argument("--id-col", default="targetid")
ap.add_argument("--mass-col", default="logmstar"); ap.add_argument("--mass-h", type=float, default=1.0)
ap.add_argument("--sga-col", default=None, help="SGA_ID column if the table has one (exact join, else positional 3\")")
ap.add_argument("--logm-max", type=float, default=9.0); ap.add_argument("--logm-min", type=float, default=6.0)
ap.add_argument("--bar-ellip-min", type=float, default=0.3)
ap.add_argument("--bar-above-disc-min", type=float, default=0.1)
ap.add_argument("--name", default="all_dwarfs")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)

df = read_any(args.table)
print(f"input table: {len(df)} rows, {df.shape[1]} columns")
ra, dec, zc = find_col(df, args.ra_col), find_col(df, args.dec_col), find_col(df, args.z_col)
idc = find_col(df, args.id_col) if args.id_col.lower() in {c.lower() for c in df.columns} else None
mc = find_col(df, args.mass_col) if args.mass_col and args.mass_col.lower() in {c.lower() for c in df.columns} else None
s = pd.DataFrame({"RA": df[ra].astype(float), "DEC": df[dec].astype(float), "Z": df[zc].astype(float)})
s["TARGETID"] = df[idc] if idc else np.arange(len(df))
s["LOGMSTAR"] = df[mc].astype(float) + 2 * np.log10(args.mass_h * 100 / cfg["cosmology"]["H0"]) if mc else np.nan
for c in ("shape_e1", "shape_e2", "shape_r", "sga_id", "ba", "pa", "sma_sb25", "d25_adopted_arcsec", "logmstar_h0678", "SFR_Msun_yr"):
    col = {x.lower(): x for x in df.columns}.get(c.lower())
    if col:
        s[c] = df[col].to_numpy()
ok = np.isfinite(s.RA) & np.isfinite(s.DEC) & np.isfinite(s.Z) & (s.Z > 0)
s = s[ok].reset_index(drop=True)
s["CZ_KMS"] = 299792.458 * s.Z
s["KPC_PER_ARCSEC"] = s.Z * 299792.458 / cfg["cosmology"]["H0"] * 1000 / 206264.8
print(f"  with valid RA/Dec/z: {len(s)};  cz range {s.CZ_KMS.min():.0f}-{s.CZ_KMS.max():.0f} km/s")

# disc geometry from Tractor shapes when available (convention verified against SGA)
if "shape_e1" in s:
    e = np.hypot(s.shape_e1.astype(float), s.shape_e2.astype(float))
    s["TR_BA"] = (1 - e) / (1 + e)
    s["TR_PA"] = np.degrees(0.5 * np.arctan2(s.shape_e2.astype(float), s.shape_e1.astype(float))) % 180
    s["DISC_BA"] = s["ba"].astype(float).where(s["ba"].notna(), s.TR_BA) if "ba" in s else s.TR_BA
    s["DISC_PA"] = s["pa"].astype(float).where(s["pa"].notna(), s.TR_PA) if "pa" in s else s.TR_PA
    s["INCL_DEG"] = inclination_from_axis_ratio(s.DISC_BA, cfg["geometry"]["q0"])

# dwarf cut (mass if available)
if mc:
    dwarf = (s.LOGMSTAR >= args.logm_min) & (s.LOGMSTAR <= args.logm_max)
    print(f"  dwarfs ({args.logm_min} <= log M*(H0={cfg['cosmology']['H0']:.0f}) <= {args.logm_max}): {int(dwarf.sum())}")
else:
    dwarf = np.ones(len(s), bool)
    print("  no mass column: no dwarf cut applied")
s["IS_DWARF"] = dwarf
s["Z_TIER"] = np.select([s.Z < cfg["desi"]["z_max"], s.Z < 0.02], ["A", "B"], "C")

# bar catalogue
bars = read_bar_catalog(data_dir("raw") / "desi_legacy_bars_sga.fits")
sga_col = find_col(df, args.sga_col) if args.sga_col else ("sga_id" if "sga_id" in s else None)
d = crossmatch_desi(s, bars, radius_arcsec=3.0)
if sga_col:
    key = pd.to_numeric(s[sga_col] if sga_col in s else df.loc[ok, sga_col].reset_index(drop=True), errors="coerce")
    by_id = pd.DataFrame({"SGA_ID": key}).merge(bars, on="SGA_ID", how="left")
    for col in ["BAR_LENGTH_RAW", "BAR_ELLIP", "SGA_RA", "SGA_DEC"]:
        d[col] = np.where(by_id["BAR_ELLIP"].notna(), by_id[col], d[col])
    d["IN_BAR_CATALOG"] = by_id["BAR_ELLIP"].notna() | d["IN_BAR_CATALOG"]
d["BAR_MINUS_DISC_ELLIP"] = d["BAR_ELLIP"] - (1 - d["DISC_BA"]) if "DISC_BA" in d else np.nan
d["BARRED_SGA"] = d.IN_BAR_CATALOG & (d.BAR_ELLIP >= args.bar_ellip_min) & (d.BAR_ELLIP <= 0.7) & \
                  ((d.BAR_MINUS_DISC_ELLIP >= args.bar_above_disc_min) | d.BAR_MINUS_DISC_ELLIP.isna())
print(f"  in bar catalogue: {int(d.IN_BAR_CATALOG.sum())} (dwarfs: {int((d.IN_BAR_CATALOG & d.IS_DWARF).sum())});"
      f"  barred (ellip >= {args.bar_ellip_min}, above disc): {int(d.BARRED_SGA.sum())} (dwarfs: {int((d.BARRED_SGA & d.IS_DWARF).sum())})")

# MUSE coverage from the cached lists
cube_cache, raw_cache = data_dir("raw") / "eso_muse_all_cube.parquet", data_dir("raw") / "eso_muse_all_raw.parquet"
if not cube_cache.exists():
    raise SystemExit("cached MUSE lists not found; run notebook section 2 or scripts/03_query_muse.py first")
r = cfg["muse"]["search_radius_arcmin"]
hits = [crossmatch_local(d.RA, d.DEC, pd.read_parquet(cube_cache), "cube", r)]
if raw_cache.exists():
    hits.append(crossmatch_local(d.RA, d.DEC, pd.read_parquet(raw_cache), "raw", r))
cov = summarize_coverage(d, pd.concat(hits, ignore_index=True), cfg["muse"])
has = (cov.N_MUSE_CUBES > 0) | cov.MUSE_RAW_ONLY
print(f"  with MUSE data: all {int(has.sum())} | dwarfs {int((has & cov.IS_DWARF).sum())} | "
      f"barred dwarfs {int((has & cov.IS_DWARF & cov.BARRED_SGA).sum())} | barred dwarfs at z<{cfg['desi']['z_max']}: "
      f"{int((has & cov.IS_DWARF & cov.BARRED_SGA & (cov.Z_TIER == 'A')).sum())}")
cov.to_parquet(data_dir("interim") / f"{args.name}_with_bar_flags.parquet")

out = cov[has & cov.IS_DWARF].copy()
out["PRIORITY"] = np.select([out.BARRED_SGA & (out.Z_TIER == "A"), out.BARRED_SGA, out.IN_BAR_CATALOG], [1, 2, 3], 4)
out = out.sort_values(["PRIORITY", "MUSE_EXPTIME_S"], ascending=[True, False])
cols = ["TARGETID", "RA", "DEC", "Z", "CZ_KMS", "LOGMSTAR", "KPC_PER_ARCSEC", "Z_TIER", "DISC_PA", "DISC_BA", "INCL_DEG",
        "IN_BAR_CATALOG", "BAR_ELLIP", "BAR_MINUS_DISC_ELLIP", "BARRED_SGA", "N_MUSE_CUBES", "MUSE_EXPTIME_S",
        "MUSE_PUBLIC", "MUSE_RAW_ONLY", "MUSE_PROG_IDS", "PRIORITY"]
prod = ROOT / "products"; prod.mkdir(exist_ok=True)
path = prod / f"{args.name}_barred_muse_candidates.csv"
out[[c for c in cols if c in out]].round(4).to_csv(path, index=False)
print(f"wrote {path.relative_to(ROOT)} ({len(out)} MUSE-covered dwarfs; priority 1: {int((out.PRIORITY == 1).sum())}, "
      f"2: {int((out.PRIORITY == 2).sum())}, 3 (in bar catalogue, weak): {int((out.PRIORITY == 3).sum())})")
