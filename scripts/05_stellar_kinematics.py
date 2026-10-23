#!/usr/bin/env python3
"""Step 5: stellar kinematic maps from a MUSE cube (Voronoi + pPXF).

    python scripts/05_stellar_kinematics.py CUBE.fits --z 0.004 --templates spectra_emiles_9.0.npz \
        --out data/products/kin_<name>.fits [--target-sn 40]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

from bartw.kinematics import fit_bins, read_muse_cube, save_maps, voronoi_bins, white_light

ap = argparse.ArgumentParser()
ap.add_argument("cube")
ap.add_argument("--z", type=float, required=True)
ap.add_argument("--templates", required=True, help="pPXF SPS .npz file (e.g. E-MILES)")
ap.add_argument("--out", required=True)
ap.add_argument("--target-sn", type=float, default=40.0)
ap.add_argument("--snr-min", type=float, default=1.0)
args = ap.parse_args()

cube = read_muse_cube(args.cube)
sig, noise = white_light(cube)
bins = voronoi_bins(sig, noise, args.target_sn, args.snr_min)
print(f"{bins.max() + 1} Voronoi bins")
maps = fit_bins(cube, bins, args.z, args.templates)
save_maps(args.out, maps, cube.header)
print(f"wrote {args.out}")
