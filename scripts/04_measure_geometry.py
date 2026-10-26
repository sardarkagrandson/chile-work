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
from bartw.photometry import cutout_size_for, fetch_cutout, measure_geometry

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
        r25 = float(r["sma_sb25"]) if np.isfinite(r.get("sma_sb25", np.nan)) and r.get("sma_sb25", 0) > 0 \
            else float(r["d25_adopted_arcsec"]) / 2 if np.isfinite(r.get("d25_adopted_arcsec", np.nan)) else 15.0
        pix = g["cutout_pixscale_arcsec"]
        img, hdr = fetch_cutout(r["RA"], r["DEC"], g["cutout_layer"], pix, cutout_size_for(r25, pix))
        res = measure_geometry(img, pix, r25)
        rec.update(R25_ARCSEC=r25, PA_DISC=res["pa_disc"], PA_DISC_ERR=res["pa_disc_err"], EPS_DISC=res["eps_disc"],
                   INCL_DEG=float(inclination_from_axis_ratio(1 - res["eps_disc"], g["q0"])),
                   PA_BAR=res["pa_bar"], A_BAR_ARCSEC=res["a_bar"], A_BAR_ERR_ARCSEC=res["a_bar_err"],
                   EPS_BAR=res["eps_bar"], GEOM_STATUS=res["status"])
    except Exception as exc:
        print(f"  {r['TARGETID']}: failed ({exc})")
        rec.update(PA_DISC=np.nan, PA_DISC_ERR=np.nan, PA_BAR=np.nan, INCL_DEG=np.nan, A_BAR_ARCSEC=np.nan, GEOM_STATUS="failed")
    rows.append(rec)
geo = pd.DataFrame(rows)
ok, dpa, why = tw_suitable(geo["INCL_DEG"], geo["PA_BAR"], geo["PA_DISC"], g)
geo["DPA_DEG"] = dpa
geo["TW_SUITABLE"] = ok & (geo["PA_DISC_ERR"] <= 5.0)
geo["TW_REASON"] = np.where(ok & (geo["PA_DISC_ERR"] > 5.0), "disc PA error > 5 deg", why)
out = df.merge(geo, on="TARGETID")
out["KPC_PER_ARCSEC"] = kpc_per_arcsec(z=out["Z"], H0=cfg["cosmology"]["H0"])
out["A_BAR_KPC"] = out["A_BAR_ARCSEC"] * out["KPC_PER_ARCSEC"]
print(f"TW-suitable: {int(out['TW_SUITABLE'].sum())} / {len(out)}")
print(out["TW_REASON"].value_counts().to_string())
path = data_dir("products") / "sample_geometry.csv"
out.to_csv(path, index=False)
print(f"wrote {path}")
