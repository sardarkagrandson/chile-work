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
ap.add_argument("--targetid", type=int, help="fill PA, inclination, bar PA/length and the nucleus pixel from products/ and the map WCS")
ap.add_argument("--pa", type=float); ap.add_argument("--pa-err", type=float, default=None)
ap.add_argument("--incl", type=float)
ap.add_argument("--bar-pa", type=float)
ap.add_argument("--a-bar", type=float, help="sky-plane bar semi-major axis [arcsec]")
ap.add_argument("--a-bar-err", type=float, default=None)
ap.add_argument("--distance", type=float, help="Mpc (default: from the catalogue redshift)")
ap.add_argument("--x0", type=float, default=None); ap.add_argument("--y0", type=float, default=None)
ap.add_argument("--pixscale", type=float, default=0.2)
ap.add_argument("--vc", type=float, default=None); ap.add_argument("--vc-err", type=float, default=None)
ap.add_argument("--east-right", action="store_true")
ap.add_argument("--no-kin-pa", action="store_true", help="use the photometric PA instead of fitting the kinematic PA")
ap.add_argument("--kin-radius", type=float, default=None, help="radius [arcsec] for the kinematic PA fit (default 2 a_bar)")
ap.add_argument("--max-verr", type=float, default=None, help="drop spaxels with V error above this [km/s] (default: 4x the median of the bright bins)")
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)["tw"]

with fits.open(args.kin) as h:
    flux, vel = h["FLUX"].data.astype(float), h["V"].data.astype(float)
    verr = h["V_ERR"].data.astype(float) if "V_ERR" in h else None
    hdr = h["V"].header
ny, nx = flux.shape
if args.targetid is not None:
    from astropy.wcs import WCS
    from bartw.lookup import lookup_target
    g = lookup_target(args.targetid)
    defaults = dict(pa="PA_DISC", pa_err="PA_DISC_ERR", incl="INCL_GEOM", bar_pa="PA_BAR", a_bar="A_BAR_ARCSEC", a_bar_err="A_BAR_ERR_ARCSEC")
    for a, k in defaults.items():
        if getattr(args, a) is None and k in g:
            setattr(args, a, g[k])
    if args.incl is None and "INCL_DEG" in g:
        args.incl = g["INCL_DEG"]
    if args.distance is None and "KPC_PER_ARCSEC" in g:
        args.distance = g["KPC_PER_ARCSEC"] * 206.2648        # Mpc, Hubble flow from the DESI redshift
    if args.x0 is None or args.y0 is None:
        try:
            x0w, y0w = WCS(hdr).celestial.all_world2pix(g["RA"], g["DEC"], 0)
            args.x0, args.y0 = float(x0w), float(y0w)
        except Exception as exc:
            print("could not place the nucleus from the WCS:", exc)
    print(f"TARGETID {args.targetid}: PA={args.pa}, PA_err={args.pa_err}, incl={args.incl}, bar PA={args.bar_pa}, "
          f"a_bar={args.a_bar}\", D={args.distance:.1f} Mpc, nucleus pixel ({args.x0:.1f}, {args.y0:.1f})")
for a in ("pa", "incl", "bar_pa", "a_bar", "distance"):
    if getattr(args, a) is None:
        raise SystemExit(f"--{a.replace('_', '-')} is required (or give --targetid with a measured geometry)")
x0 = (nx - 1) / 2 if args.x0 is None else args.x0
y0 = (ny - 1) / 2 if args.y0 is None else args.y0
pa_err = cfg["pa_error_deg"] if args.pa_err is None else args.pa_err
pa_phot, pa_phot_err = args.pa, pa_err
pa_kin = pa_kin_err = np.nan
# spaxels with unreliable velocities (failed/unstable fits): keep the mask small and report it
mask = np.isfinite(flux) & np.isfinite(vel) & (flux > 0)
if verr is not None:
    bright = mask & (flux >= np.nanpercentile(flux[mask], 70))
    vmax = args.max_verr if args.max_verr is not None else 4.0 * np.nanmedian(verr[bright])
    bad = mask & ~(verr <= vmax)
    print(f"velocity-error mask: {bad.sum()} of {mask.sum()} spaxels have V error > {vmax:.0f} km/s "
          f"({100 * bad.sum() / mask.sum():.1f}%) and are excluded")
    mask &= ~bad
if not args.no_kin_pa:
    from bartw.kinpa import fit_kinematic_pa
    rk = args.kin_radius if args.kin_radius is not None else 2.0 * args.a_bar
    kp = fit_kinematic_pa(np.where(mask, vel, np.nan), x0, y0, east_left=not args.east_right,
                          radius_pix=rk / args.pixscale, v_err=verr)
    pa_kin, pa_kin_err = kp["pa_kin"], max(kp["pa_kin_err"], 1.0)
    print(f"kinematic PA = {pa_kin:.1f} +/- {pa_kin_err:.1f} deg (within {rk:.0f}\", {kp['n_pix']} spaxels, V_sys {kp['v_sys']:.1f}); "
          f"photometric PA = {pa_phot:.1f} +/- {pa_phot_err:.1f} deg; difference {pa_difference(pa_kin, pa_phot):.1f} deg; "
          f"receding side at PA {kp['receding_pa']:.0f}")
    args.pa, pa_err = pa_kin, pa_kin_err
dpa = float(pa_difference(args.bar_pa, args.pa))
print(f"using disc PA {args.pa:.1f} +/- {pa_err:.1f}; bar-disc angle {dpa:.1f} deg")
yc = default_slit_centers(args.a_bar, dpa, cfg["n_slits"], cfg["slit_width_arcsec"], cfg["slit_ymax_factor"])
xl = cfg["slit_half_length_factor"] * args.a_bar

res = measure_pattern_speed(flux, vel, args.pixscale, x0, y0, args.pa, args.incl, yc, cfg["slit_width_arcsec"],
                            x_half_length_arcsec=xl, vel_err=verr, mask=mask, distance_mpc=args.distance,
                            pa_err_deg=pa_err, n_mc=cfg["n_mc"], rng=0, east_left=not args.east_right)
print(res.summary())
omega_abs = abs(res.omega_p)

row = dict(name=args.name, pa=args.pa, pa_err=pa_err, pa_phot=pa_phot, pa_phot_err=pa_phot_err, pa_kin=pa_kin, pa_kin_err=pa_kin_err,
           incl=args.incl, bar_pa=args.bar_pa, dpa=dpa, a_bar_sky_arcsec=args.a_bar, distance_mpc=args.distance,
           omega_sini=res.omega_sini, omega_sini_err=res.omega_sini_err, omega_p=omega_abs, omega_p_err=res.omega_p_err,
           omega_p_err_fit=res.omega_p_err_fit, omega_p_err_pa=res.omega_p_err_pa, prograde=res.prograde, v_sys=res.v_sys)
if args.vc is not None:
    a_kpc = deproject_bar_length(args.a_bar, args.bar_pa, args.pa, args.incl) * res.scale_kpc_arcsec
    a_err = (args.a_bar_err or 0.1 * args.a_bar) * res.scale_kpc_arcsec
    rr = rotation_rate(omega_abs, res.omega_p_err, args.vc, args.vc_err or 0.1 * args.vc, a_kpc, a_err, rng=0,
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
