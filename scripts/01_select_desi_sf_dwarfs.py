#!/usr/bin/env python3
"""Step 1: the star-forming dwarf parent sample.

Default: start from the DESI DR1 star-forming dwarf sample of Thomas et al.
(``data/raw/DESI_dwarfs_paper_sample.csv``, 14,786 galaxies, already BPT
star-forming, log M* <= 9.0 at h = 0.678, z < 0.0333). Only the redshift
window for spatial resolution is applied here (tiers, not a hard cut, are
recorded in ``Z_TIER``).

Alternative (``--from-fastspecfit``): rebuild a sample from the FastSpecFit
Iron VAC with the cuts in config/selection.yaml (useful to extend to higher
mass or to quiescent dwarfs).

    python scripts/01_select_desi_sf_dwarfs.py
    python scripts/01_select_desi_sf_dwarfs.py --from-fastspecfit --fastspec data/raw/fastspec-iron-main-bright.fits
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse
import json

import numpy as np

from bartw.config import data_dir, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--paper", default=str(data_dir("raw") / "DESI_dwarfs_paper_sample.csv"))
ap.add_argument("--from-fastspecfit", action="store_true")
ap.add_argument("--fastspec", default=str(data_dir("raw") / "fastspec-iron-main-bright.fits"))
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
c = cfg["desi"]

if args.from_fastspecfit:
    from bartw.desi import read_fastspec, select_star_forming_dwarfs
    print(f"reading {args.fastspec} ...")
    sel, steps = select_star_forming_dwarfs(read_fastspec(args.fastspec), cfg)
else:
    from bartw.paper_sample import read_paper_sample, standardize, tractor_geometry
    df = standardize(tractor_geometry(read_paper_sample(args.paper), cfg["geometry"]["q0"]))
    steps = {"paper sample (SF dwarfs, log M*<=9.0, z<0.0333)": len(df)}
    df["Z_TIER"] = np.select([df["Z"] < c["z_max"], df["Z"] < 0.02], ["A (z<%.3f)" % c["z_max"], "B (z<0.02)"], "C")
    for t, n in df["Z_TIER"].value_counts().sort_index().items():
        steps[f"tier {t}"] = int(n)
    steps["with SGA geometry"] = int((df["DISC_PA_SOURCE"] == "SGA").sum())
    sel = df

for k, v in steps.items():
    print(f"  {k:50s} {v:>7d}")
out = data_dir("interim") / "01_sf_dwarfs.parquet"
sel.to_parquet(out)
json.dump(steps, open(data_dir("interim") / "01_steps.json", "w"), indent=1)
print(f"wrote {out}  ({len(sel)} galaxies)")
