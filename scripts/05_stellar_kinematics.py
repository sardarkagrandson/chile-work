#!/usr/bin/env python3
"""Step 5: stellar kinematic maps from a MUSE cube (Voronoi + pPXF).

    python scripts/05_stellar_kinematics.py CUBE.fits --z 0.004 --templates data/raw/templates/spectra_emiles_9.0.npz \
        --out data/products/kin_NAME.fits [--target-sn 40] [--snr-min 1] [--lam-range 4750 5950]

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
ap.add_argument("--z", type=float, help="redshift (default: from products/ via --targetid)")
ap.add_argument("--templates", required=True, help="pPXF SPS .npz file (e.g. E-MILES)")
ap.add_argument("--out", required=True)
ap.add_argument("--target-sn", type=float, default=40.0)
ap.add_argument("--snr-min", type=float, default=1.0)
ap.add_argument("--lam-range", type=float, nargs=2, default=(4750.0, 5950.0), help="blue range by default: the red is full of OH sky residuals")
ap.add_argument("--degree", type=int, default=12)
ap.add_argument("--targetid", type=int, help="look up the nucleus position in products/ and read only a region around it")
ap.add_argument("--ra", type=float); ap.add_argument("--dec", type=float)
ap.add_argument("--half-size", type=float, default=45.0, help="half side of the region read around the nucleus [arcsec]")
args = ap.parse_args()
center = None
if args.targetid is not None:
    from bartw.lookup import lookup_target
    g = lookup_target(args.targetid); args.ra, args.dec = g["RA"], g["DEC"]
    if args.z is None and "Z" in g:
        args.z = g["Z"]; print(f"z = {args.z:.5f} from products/")
    if "A_BAR_ARCSEC" in g:
        args.half_size = max(args.half_size, 1.6 * g["A_BAR_ARCSEC"] + 10)
if args.z is None:
    raise SystemExit("--z is required (or --targetid with a redshift in products/)")
if args.ra is not None and args.dec is not None:
    center = (args.ra, args.dec)
    print(f"reading a {2 * args.half_size:.0f}\" region around RA={args.ra:.5f} Dec={args.dec:.5f}")

t0 = time.time()
cube = read_muse_cube(args.cube, lam_range=(args.lam_range[0] - 50, args.lam_range[1] + 50),
                      center=center, half_size_arcsec=args.half_size if center else None)
print(f"cube {cube.data.shape} read in {time.time() - t0:.0f} s; {cube.lam[0]:.0f}-{cube.lam[-1]:.0f} A, pixscale {cube.pixscale:.3f}\"")
sig, noise = white_light(cube)
bins = voronoi_bins(sig, noise, args.target_sn, args.snr_min)
print(f"{bins.max() + 1} Voronoi bins from {(bins >= 0).sum()} spaxels (S/N >= {args.snr_min})")
import pathlib as _pl
_pl.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
maps = fit_bins(cube, bins, args.z, args.templates, lam_range=tuple(args.lam_range), degree=args.degree,
                checkpoint=(args.out, cube.header))
ok = np.isfinite(maps["v"])
bright = ok & (maps["bin_flux"] >= np.nanpercentile(maps["bin_flux"][ok], 70)) if ok.any() else ok
print(f"fitted {ok.sum()}/{bins.max() + 1} bins in {(time.time() - t0) / 60:.1f} min; "
      f"V median {np.nanmedian(maps['v']):.0f} km/s (cz = {299792.458 * args.z:.0f}), "
      f"sigma median {np.nanmedian(maps['sigma']):.0f} km/s; median V error {np.nanmedian(maps['v_err']):.1f} km/s "
      f"(brightest 30% of bins: {np.nanmedian(maps['v_err'][bright]):.1f} km/s)")
save_maps(args.out, maps, cube.header)
print(f"wrote {args.out}  (FLUX, V, V_ERR, SIGMA, SIGMA_ERR, BINNUM)")
