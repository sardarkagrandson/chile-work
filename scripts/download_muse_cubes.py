#!/usr/bin/env python3
"""Download ESO Phase-3 MUSE cubes for chosen galaxies.

    python scripts/download_muse_cubes.py --targetid 39628274099294133 [--targetid ...] [--all-ready]
    python scripts/download_muse_cubes.py --dp-id ADP.2019-02-25T15:33:18.321

Looks the dp_ids up in data/interim/03_muse_hits.parquet (notebook 01) or in
the cached cube list matched around the galaxy position, then fetches
https://dataportal.eso.org/dataPortal/file/<dp_id> into data/raw/cubes/.
Public cubes download anonymously; for proprietary ones set ESO_USER and
ESO_PASSWORD in the environment (astroquery.eso login) or download by hand.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse
import os

import numpy as np
import pandas as pd
import requests

from bartw.config import ROOT, data_dir, load_config
from bartw.muse import crossmatch_local

ap = argparse.ArgumentParser()
ap.add_argument("--targetid", type=int, action="append", default=[])
ap.add_argument("--dp-id", action="append", default=[])
ap.add_argument("--all-ready", action="store_true", help="every galaxy in products/*_tw_ready.csv")
ap.add_argument("--table", default=None, help="CSV with TARGETID, RA, DEC (default: products/*_tw_ready.csv, *_barred_muse_candidates.csv)")
ap.add_argument("--min-exptime", type=float, default=600.0)
args = ap.parse_args()
cfg = load_config()
out_dir = data_dir("raw") / "cubes"; out_dir.mkdir(parents=True, exist_ok=True)

# positions of the requested galaxies
tables = [args.table] if args.table else sorted(str(p) for p in (ROOT / "products").glob("*_tw_ready.csv")) + \
         sorted(str(p) for p in (ROOT / "products").glob("*_candidates.csv"))
gal = pd.concat([pd.read_csv(t) for t in tables], ignore_index=True).drop_duplicates("TARGETID") if tables else pd.DataFrame()
want = set(args.targetid)
if args.all_ready:
    for t in (ROOT / "products").glob("*_tw_ready.csv"):
        want |= set(pd.read_csv(t).TARGETID)
sel = gal[gal.TARGETID.isin(want)] if len(gal) else pd.DataFrame()
if len(want) and len(sel) < len(want):
    print("TARGETIDs not found in the product tables:", sorted(want - set(sel.TARGETID)))

dp_ids = list(args.dp_id)
if len(sel):
    cubes = pd.read_parquet(data_dir("raw") / "eso_muse_all_cube.parquet")
    hits = crossmatch_local(sel.RA.to_numpy(), sel.DEC.to_numpy(), cubes, "cube", cfg["muse"]["search_radius_arcmin"])
    hits = hits[hits.t_exptime.fillna(0) >= args.min_exptime]
    for i, grp in hits.groupby("idx"):
        tid = int(sel.TARGETID.iloc[i])
        print(f"{tid}: {len(grp)} cube(s)")
        for _, h in grp.sort_values("t_exptime", ascending=False).iterrows():
            print(f"   {h.dp_id}  {h.t_exptime:.0f} s  {h.get('proposal_id', '')}  released {str(h.get('obs_release_date', ''))[:10]}")
            dp_ids.append(str(h.dp_id))

if not dp_ids:
    raise SystemExit("nothing to download")

session = requests.Session()
user, pwd = os.environ.get("ESO_USER"), os.environ.get("ESO_PASSWORD")
if user and pwd:
    try:
        from astroquery.eso import Eso
        eso = Eso(); eso.login(user, password=pwd)
        session = eso._session
        print("logged in to ESO as", user)
    except Exception as exc:
        print("ESO login failed, trying anonymous:", exc)

for dp in dict.fromkeys(dp_ids):
    dest = out_dir / f"{dp}.fits"
    if dest.exists() and dest.stat().st_size > 1e6:
        print("have", dest.name); continue
    url = f"https://dataportal.eso.org/dataPortal/file/{dp}"
    print("downloading", url)
    with session.get(url, stream=True, timeout=120) as r:
        if r.status_code == 401:
            print(f"   {dp}: proprietary (401). Set ESO_USER/ESO_PASSWORD or download from archive.eso.org"); continue
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    print(f"   -> {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")
print("cubes in", out_dir)
