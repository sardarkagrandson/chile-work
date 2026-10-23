#!/usr/bin/env python3
"""Step 4: disc and bar geometry from Legacy Surveys cutouts, TW suitability.

For every galaxy with MUSE coverage: fetch an r-band cutout, fit isophotes,
measure disc PA/ellipticity (outer isophotes) and bar PA/length (ellipticity
peak), compute inclination and |PA_bar - PA_disc|, flag TW-suitable targets.

    python scripts/04_measure_geometry.py [--input data/products/sample_with_muse.csv] [--all]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import numpy as np
import pandas as pd

from bartw.config import data_dir, load_config
from bartw.geometry import inclination_from_axis_ratio, kpc_per_arcsec, tw_suitable
from bartw.photometry import bar_geometry, disc_geometry, fetch_cutout, isophote_profiles

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=str(data_dir("products") / "sample_with_muse.csv"))
ap.add_argument("--all", action="store_true", help="measure every candidate, not only MUSE-covered ones")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
g = cfg["geometry"]

df = pd.read_csv(args.input)
if not args.all:
    df = df[(df["N_MUSE_CUBES"] > 0) | df["MUSE_RAW_ONLY"]].reset_index(drop=True)
print(f"measuring geometry for {len(df)} galaxies")

rows = []
for i, r in df.iterrows():
    rec = dict(TARGETID=r["TARGETID"])
    try:
        img, hdr = fetch_cutout(r["RA"], r["DEC"], g["cutout_layer"], g["cutout_pixscale_arcsec"], g["cutout_size_pix"])
        prof = isophote_profiles(img)
        pix = g["cutout_pixscale_arcsec"]
        # disc: outer third of the fitted range; bar: inner part
        smax = np.nanmax(prof["sma"])
        pa_d, pa_d_err, eps_d, eps_d_err = disc_geometry(prof, 0.6 * smax, 0.95 * smax)
        bar = bar_geometry(prof, pa_d, eps_d)
        rec.update(PA_DISC=pa_d, PA_DISC_ERR=pa_d_err, EPS_DISC=eps_d,
                   INCL_DEG=float(inclination_from_axis_ratio(1 - eps_d, g["q0"])),
                   PA_BAR=bar["pa_bar"], A_BAR_ARCSEC=bar["a_bar"] * pix if np.isfinite(bar["a_bar"]) else np.nan,
                   A_BAR_ERR_ARCSEC=bar["a_bar_err"] * pix if np.isfinite(bar.get("a_bar_err", np.nan)) else np.nan,
                   EPS_BAR=bar["eps_bar"])
    except Exception as exc:
        print(f"  {r['TARGETID']}: failed ({exc})")
        rec.update(PA_DISC=np.nan, PA_BAR=np.nan, INCL_DEG=np.nan, A_BAR_ARCSEC=np.nan)
    rows.append(rec)
geo = pd.DataFrame(rows)
ok, dpa, why = tw_suitable(geo["INCL_DEG"], geo["PA_BAR"], geo["PA_DISC"], g)
geo["DPA_DEG"], geo["TW_SUITABLE"], geo["TW_REASON"] = dpa, ok, why
out = df.merge(geo, on="TARGETID")
out["KPC_PER_ARCSEC"] = kpc_per_arcsec(z=out["Z"], H0=cfg["cosmology"]["H0"])
out["A_BAR_KPC"] = out["A_BAR_ARCSEC"] * out["KPC_PER_ARCSEC"]
print(f"TW-suitable: {int(out['TW_SUITABLE'].sum())} / {len(out)}")
print(out["TW_REASON"].value_counts().to_string())
path = data_dir("products") / "sample_geometry.csv"
out.to_csv(path, index=False)
print(f"wrote {path}")
