#!/usr/bin/env python3
"""Step 7: Halpha gas kinematics from the same MUSE cube: velocity field, gas
kinematic PA (inside and outside the bar), rotation curve and circular velocity.

    python scripts/07_gas_kinematics.py CUBE.fits --targetid ID --out data/products/gas_NAME.fits
          [--incl 55] [--pa 33] [--snr-min 3]

Compares the gas kinematic PA with the stellar one (if --stars KIN.fits is given)
and the photometric PA, and writes the rotation curve to products/<name>_rotcurve.csv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from bartw.config import ROOT, load_config
from bartw.gaskin import circular_velocity, line_moments, rotation_curve
from bartw.geometry import pa_difference
from bartw.kinematics import read_muse_cube, sky_to_pixel
from bartw.kinpa import fit_kinematic_pa
from bartw.lookup import lookup_target

ap = argparse.ArgumentParser()
ap.add_argument("cube"); ap.add_argument("--targetid", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--name", default=None)
ap.add_argument("--incl", type=float, default=None, help="inclination for the rotation curve (default: geometry table)")
ap.add_argument("--pa", type=float, default=None, help="PA for the rotation curve (default: gas kinematic PA, outer annulus)")
ap.add_argument("--snr-min", type=float, default=3.0)
ap.add_argument("--half-size", type=float, default=60.0)
ap.add_argument("--stars", default=None, help="stellar kinematics file (kin_NAME.fits) for the PA comparison")
args = ap.parse_args()
cfg = load_config()
g = lookup_target(args.targetid)
name = args.name or f"T{args.targetid}"
z, a_bar = g["Z"], g.get("A_BAR_ARCSEC", 15.0)
incl = args.incl if args.incl is not None else g.get("INCL_GEOM", g.get("INCL_DEG", 50.0))
lc = 6562.8 * (1 + z)
cube = read_muse_cube(args.cube, lam_range=(lc - 60, lc + 60), center=(g["RA"], g["DEC"]), half_size_arcsec=max(args.half_size, 1.6 * a_bar + 10))
x0, y0 = sky_to_pixel(cube, g["RA"], g["DEC"])
pix = cube.pixscale
east_left = cube.header.get("CD1_1", -1) < 0
m = line_moments(cube, z, snr_min=args.snr_min)
print(f"Halpha detected (S/N >= {args.snr_min}) in {m['good'].sum()} of {np.isfinite(m['snr']).sum()} spaxels; "
      f"median V {np.nanmedian(m['vel']):.1f} km/s (cz = {299792.458 * z:.0f})")

r_in, r_out = 1.2 * a_bar, min(3.0 * a_bar, 0.95 * args.half_size)
kp_out = fit_kinematic_pa(m["vel"], x0, y0, east_left=east_left, radius_pix=r_out / pix, rmin_pix=r_in / pix, v_err=m["vel_err"])
kp_in = fit_kinematic_pa(m["vel"], x0, y0, east_left=east_left, radius_pix=r_in / pix, v_err=m["vel_err"])
kp_all = fit_kinematic_pa(m["vel"], x0, y0, east_left=east_left, radius_pix=r_out / pix, v_err=m["vel_err"])
print(f"gas kinematic PA: annulus {r_in:.0f}-{r_out:.0f}\": {kp_out['pa_kin']:.1f} +/- {kp_out['pa_kin_err']:.1f} "
      f"({kp_out['n_pix']} spx, amplitude {kp_out['v_amp']:.1f} km/s, {'detected' if kp_out['detected'] else 'NOT detected'}); "
      f"bar region: {kp_in['pa_kin']:.1f} +/- {kp_in['pa_kin_err']:.1f}; all: {kp_all['pa_kin']:.1f}; V_sys {kp_out['v_sys']:.1f}")
if "PA_DISC" in g:
    print(f"photometric PA {g['PA_DISC']:.1f} +/- {g.get('PA_DISC_ERR', np.nan):.1f}: gas - phot = {pa_difference(kp_out['pa_kin'], g['PA_DISC']):.1f} deg")
if args.stars:
    with fits.open(args.stars) as h:
        vs, ves = h["V"].data.astype(float), h["V_ERR"].data.astype(float)
        from astropy.wcs import WCS
        xs, ys = (float(v) for v in WCS(h["V"].header).celestial.all_world2pix(g["RA"], g["DEC"], 0))
    ks = fit_kinematic_pa(vs, xs, ys, east_left=east_left, radius_pix=r_out / pix, rmin_pix=r_in / pix, v_err=ves)
    print(f"stellar kinematic PA (same annulus): {ks['pa_kin']:.1f} +/- {ks['pa_kin_err']:.1f}; gas - stars = {pa_difference(kp_out['pa_kin'], ks['pa_kin']):.1f} deg; "
          f"gas V_sys - stellar V_sys = {kp_out['v_sys'] - ks['v_sys']:.1f} km/s")

pa_rc = args.pa if args.pa is not None else (kp_out["pa_kin"] if kp_out["detected"] else kp_all["pa_kin"])
rc = rotation_curve(m["vel"], x0, y0, pa_rc, incl, kp_out["v_sys"], pix, east_left=east_left, weights=m["flux"], v_err=m["vel_err"],
                    r_max_arcsec=0.95 * args.half_size)
vc, vc_err, nr = circular_velocity(rc, r_min_arcsec=1.2 * a_bar)
print(f"rotation curve along PA {pa_rc:.1f}, i = {incl:.1f}: V_c beyond 1.2 a_bar ({1.2 * a_bar:.0f}\") = {vc:.1f} +/- {vc_err:.1f} km/s from {nr} rings"
      f"  [i uncertain: V_c scales as 1/sin i]")
import pandas as pd
prod = ROOT / "products"; prod.mkdir(exist_ok=True)
pd.DataFrame(dict(r_arcsec=rc["r"], v_rot=rc["v"], v_err=rc["v_err"], n=rc["n"])).to_csv(prod / f"{name}_rotcurve.csv", index=False)
pd.DataFrame([dict(TARGETID=args.targetid, name=name, pa_gas_annulus=kp_out["pa_kin"], pa_gas_err=kp_out["pa_kin_err"], pa_gas_bar=kp_in["pa_kin"],
                   pa_gas_all=kp_all["pa_kin"], v_sys_gas=kp_out["v_sys"], detected=kp_out["detected"], incl_used=incl,
                   vc=vc, vc_err=vc_err, n_rings=nr)]).to_csv(prod / f"{name}_gaskin.csv", index=False)

hdr = fits.Header(); hdr.update(cube.wcs.to_header())
fits.HDUList([fits.PrimaryHDU(header=hdr), fits.ImageHDU(m["flux"].astype(np.float32), name="HA_FLUX", header=hdr),
              fits.ImageHDU(m["vel"].astype(np.float32), name="V", header=hdr), fits.ImageHDU(m["vel_err"].astype(np.float32), name="V_ERR", header=hdr),
              fits.ImageHDU(m["snr"].astype(np.float32), name="SNR", header=hdr)]).writeto(args.out, overwrite=True)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
with np.errstate(all="ignore"):
    ax[0].imshow(np.log10(m["flux"]), origin="lower", cmap="gray_r"); ax[0].set_title("log Halpha flux")
vr = np.nanpercentile(np.abs(m["vel"] - kp_out["v_sys"]), 95)
im = ax[1].imshow(m["vel"] - kp_out["v_sys"], origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr); plt.colorbar(im, ax=ax[1], fraction=0.046)
for pa_, col, lab in ((kp_out["pa_kin"], "k", "gas PA (annulus)"), (g.get("PA_DISC", np.nan), "C3", "photometric PA"), (g.get("PA_BAR", np.nan), "C2", "bar PA")):
    if np.isfinite(pa_):
        L = r_out / pix; sgn = -1 if east_left else 1
        ax[1].plot([x0 - L * sgn * np.sin(np.radians(pa_)), x0 + L * sgn * np.sin(np.radians(pa_))],
                   [y0 - L * np.cos(np.radians(pa_)), y0 + L * np.cos(np.radians(pa_))], color=col, lw=1.2, label=lab)
ax[1].legend(fontsize=7, loc="lower right"); ax[1].set_title(f"Halpha V - {kp_out['v_sys']:.0f}")
ax[2].errorbar(rc["r"], rc["v"], rc["v_err"], fmt="o-", ms=3); ax[2].axvline(a_bar, color="C2", ls="--", label="a_bar")
ax[2].axhline(vc, color="C3", ls=":", label=f"V_c = {vc:.0f} km/s"); ax[2].set_xlabel("R [arcsec]"); ax[2].set_ylabel("V_rot [km/s]"); ax[2].legend()
for a in ax[:2]:
    a.set_xticks([]); a.set_yticks([])
out = ROOT / "figures" / f"gas_{name}.png"; fig.tight_layout(); fig.savefig(out, dpi=120); print("figure:", out)
print(f"wrote {args.out}, products/{name}_rotcurve.csv, products/{name}_gaskin.csv")
