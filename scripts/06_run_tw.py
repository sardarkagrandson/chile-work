#!/usr/bin/env python3
"""Step 6: Tremaine-Weinberg measurement on stellar kinematic maps.

    python scripts/06_run_tw.py KIN.fits --pa 35 --pa-err 3 --incl 50 --bar-pa 60 --a-bar 9 \
        --distance 16.5 [--x0 150 --y0 150] [--vc 80 --vc-err 8 --a-bar-err 1] --name NGC0000

KIN.fits is the output of 05_stellar_kinematics.py (FLUX, V, V_ERR HDUs).
Angles in degrees (PA north through east), lengths in arcsec, distance in Mpc.
Produces a TW diagram + slit overlay in figures/ and appends to
data/products/tw_results.csv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from bartw.config import ROOT, data_dir, load_config
from bartw.geometry import deproject_bar_length, pa_difference
from bartw.plotting import slit_overlay, tw_diagram
from bartw.tw import default_slit_centers, measure_pattern_speed, rotation_rate

ap = argparse.ArgumentParser()
ap.add_argument("kin")
ap.add_argument("--name", required=True)
ap.add_argument("--pa", type=float, required=True); ap.add_argument("--pa-err", type=float, default=None)
ap.add_argument("--incl", type=float, required=True)
ap.add_argument("--bar-pa", type=float, required=True)
ap.add_argument("--a-bar", type=float, required=True, help="sky-plane bar semi-major axis [arcsec]")
ap.add_argument("--a-bar-err", type=float, default=None)
ap.add_argument("--distance", type=float, required=True)
ap.add_argument("--x0", type=float, default=None); ap.add_argument("--y0", type=float, default=None)
ap.add_argument("--pixscale", type=float, default=0.2)
ap.add_argument("--vc", type=float, default=None); ap.add_argument("--vc-err", type=float, default=None)
ap.add_argument("--east-right", action="store_true")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)["tw"]

with fits.open(args.kin) as h:
    flux, vel = h["FLUX"].data.astype(float), h["V"].data.astype(float)
    verr = h["V_ERR"].data.astype(float) if "V_ERR" in h else None
ny, nx = flux.shape
x0 = (nx - 1) / 2 if args.x0 is None else args.x0
y0 = (ny - 1) / 2 if args.y0 is None else args.y0
pa_err = cfg["pa_error_deg"] if args.pa_err is None else args.pa_err
dpa = float(pa_difference(args.bar_pa, args.pa))
yc = default_slit_centers(args.a_bar, dpa, cfg["n_slits"], cfg["slit_width_arcsec"], cfg["slit_ymax_factor"])
xl = cfg["slit_half_length_factor"] * args.a_bar

res = measure_pattern_speed(flux, vel, args.pixscale, x0, y0, args.pa, args.incl, yc, cfg["slit_width_arcsec"],
                            x_half_length_arcsec=xl, vel_err=verr, distance_mpc=args.distance,
                            pa_err_deg=pa_err, n_mc=cfg["n_mc"], rng=0, east_left=not args.east_right)
print(res.summary())

row = dict(name=args.name, pa=args.pa, pa_err=pa_err, incl=args.incl, bar_pa=args.bar_pa, dpa=dpa,
           a_bar_sky_arcsec=args.a_bar, omega_sini=res.omega_sini, omega_sini_err=res.omega_sini_err,
           omega_p=res.omega_p, omega_p_err=res.omega_p_err, v_sys=res.v_sys)
if args.vc is not None:
    a_kpc = deproject_bar_length(args.a_bar, args.bar_pa, args.pa, args.incl) * res.scale_kpc_arcsec
    a_err = (args.a_bar_err or 0.1 * args.a_bar) * res.scale_kpc_arcsec
    rr = rotation_rate(res.omega_p, res.omega_p_err, args.vc, args.vc_err or 0.1 * args.vc, a_kpc, a_err, rng=0,
                       slow_boundary=cfg["fast_slow_boundary"])
    row.update(a_bar_kpc=a_kpc, vc=args.vc, **{k: v for k, v in rr.items()})
    print(f"a_bar = {a_kpc:.2f} kpc, R_cr = {rr['R_cr']:.2f} (+{rr['R_cr_err_hi']:.2f}/-{rr['R_cr_err_lo']:.2f}) kpc, "
          f"R = {rr['R']:.2f} (+{rr['R_err_hi']:.2f}/-{rr['R_err_lo']:.2f})  -> {rr['regime']} "
          f"(p_slow={rr['p_slow']:.2f}, p_fast={rr['p_fast']:.2f}, p_uf={rr['p_ultrafast']:.2f})")

figdir = ROOT / "figures"; figdir.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
slit_overlay(flux, vel, res, cfg["slit_width_arcsec"], args.pixscale, not args.east_right, axes[:2], xl)
tw_diagram(res, axes[2], title=args.name)
fig.tight_layout(); fig.savefig(figdir / f"tw_{args.name}.png", dpi=150)

out = data_dir("products") / "tw_results.csv"
new = not out.exists()
with open(out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(f"appended to {out}; figure figures/tw_{args.name}.png")
