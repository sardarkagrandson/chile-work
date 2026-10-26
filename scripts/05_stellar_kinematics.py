#!/usr/bin/env python3
"""Step 5: stellar kinematic maps from a MUSE cube (Voronoi + pPXF).

    python scripts/05_stellar_kinematics.py CUBE.fits --z 0.004 --templates data/raw/templates/spectra_emiles_9.0.npz \
        --out data/products/kin_NAME.fits [--target-sn 40] [--snr-min 1] [--lam-range 4750 7000]

Takes ~1 s per Voronoi bin; a 1'x1' dwarf at S/N 40 gives a few hundred to a
couple of thousand bins. Only the 4750-7000 A range is read to save memory.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse
import time

import numpy as np

from bartw.kinematics import fit_bins, read_muse_cube, save_maps, voronoi_bins, white_light

ap = argparse.ArgumentParser()
ap.add_argument("cube")
ap.add_argument("--z", type=float, required=True)
ap.add_argument("--templates", required=True, help="pPXF SPS .npz file (e.g. E-MILES)")
ap.add_argument("--out", required=True)
ap.add_argument("--target-sn", type=float, default=40.0)
ap.add_argument("--snr-min", type=float, default=1.0)
ap.add_argument("--lam-range", type=float, nargs=2, default=(4750.0, 7000.0))
ap.add_argument("--degree", type=int, default=12)
args = ap.parse_args()

t0 = time.time()
cube = read_muse_cube(args.cube, lam_range=(args.lam_range[0] - 50, args.lam_range[1] + 50))
print(f"cube {cube.data.shape} read in {time.time() - t0:.0f} s; {cube.lam[0]:.0f}-{cube.lam[-1]:.0f} A, pixscale {cube.pixscale:.3f}\"")
sig, noise = white_light(cube)
bins = voronoi_bins(sig, noise, args.target_sn, args.snr_min)
print(f"{bins.max() + 1} Voronoi bins from {(bins >= 0).sum()} spaxels (S/N >= {args.snr_min})")
maps = fit_bins(cube, bins, args.z, args.templates, lam_range=tuple(args.lam_range), degree=args.degree)
ok = np.isfinite(maps["v"])
print(f"fitted {ok.sum()}/{bins.max() + 1} bins in {(time.time() - t0) / 60:.1f} min; "
      f"V median {np.nanmedian(maps['v']):.0f} km/s (cz = {299792.458 * args.z:.0f}), "
      f"sigma median {np.nanmedian(maps['sigma']):.0f} km/s, median V error {np.nanmedian(maps['v_err']):.1f} km/s")
save_maps(args.out, maps, cube.header)
print(f"wrote {args.out}  (FLUX, V, V_ERR, SIGMA, SIGMA_ERR, BINNUM)")
