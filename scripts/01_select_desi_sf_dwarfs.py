#!/usr/bin/env python3
"""Step 1: star-forming dwarf galaxies from the DESI DR1 FastSpecFit VAC.

    python scripts/01_select_desi_sf_dwarfs.py [--fastspec data/raw/fastspec-iron-main-bright.fits]

Writes data/interim/01_sf_dwarfs.parquet and prints the selection funnel.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse
import json

from bartw.config import data_dir, load_config
from bartw.desi import read_fastspec, select_star_forming_dwarfs

ap = argparse.ArgumentParser()
ap.add_argument("--fastspec", default=str(data_dir("raw") / "fastspec-iron-main-bright.fits"))
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
print(f"reading {args.fastspec} ...")
df = read_fastspec(args.fastspec)
sel, steps = select_star_forming_dwarfs(df, cfg)
for k, v in steps.items():
    print(f"  {k:25s} {v:>9d}")
out = data_dir("interim") / "01_sf_dwarfs.parquet"
sel.to_parquet(out)
json.dump(steps, open(data_dir("interim") / "01_steps.json", "w"), indent=1)
print(f"wrote {out}  ({len(sel)} galaxies)")
