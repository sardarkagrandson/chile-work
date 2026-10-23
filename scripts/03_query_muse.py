#!/usr/bin/env python3
"""Step 3: which barred candidates have MUSE data in the ESO archive?

    python scripts/03_query_muse.py [--input data/interim/02_barred_candidates.parquet]

Writes data/interim/03_muse_hits.parquet (one row per archive product) and
data/products/sample_with_muse.csv (one row per galaxy with coverage summary).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import pandas as pd

from bartw.config import data_dir, load_config
from bartw.muse import print_schema, query_muse, summarize_coverage

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=str(data_dir("interim") / "02_barred_candidates.parquet"))
ap.add_argument("--schema", action="store_true", help="print ivoa.ObsCore and dbo.raw columns and exit")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)["muse"]

if args.schema:
    print_schema(cfg["tap_url"], "ivoa.ObsCore")
    print_schema(cfg["tap_url"], "dbo.raw")
    raise SystemExit

sample = pd.read_parquet(args.input)
print(f"querying ESO archive for {len(sample)} galaxies ...")
hits = query_muse(sample["RA"].to_numpy(), sample["DEC"].to_numpy(), cfg)
hits.to_parquet(data_dir("interim") / "03_muse_hits.parquet")
summary = summarize_coverage(sample, hits, cfg)
n_cube = int((summary["N_MUSE_CUBES"] > 0).sum())
print(f"  with reduced MUSE cubes: {n_cube}  (public: {int(summary['MUSE_PUBLIC'].sum())}, "
      f"deep enough: {int((summary['MUSE_DEEP_ENOUGH'] & (summary['N_MUSE_CUBES']>0)).sum())})")
print(f"  raw MUSE data only:      {int(summary['MUSE_RAW_ONLY'].sum())}")
out = data_dir("products") / "sample_with_muse.csv"
summary.to_csv(out, index=False)
print(f"wrote {out}")
