#!/usr/bin/env python3
"""Check a MUSE cube before fitting: field coverage of the bar, nucleus pixel position.

    python scripts/inspect_cube.py CUBE.fits --ra 150.1 --dec 2.3 [--pa-bar 98 --a-bar 19 --eps-bar 0.6]
                                   [--pa-disc 80 --r-disc 60 --eps-disc 0.4] [--out figures/cube_NAME.png]

Prints the nucleus pixel coordinates (x0, y0 for scripts/06_run_tw.py), the
offset of the field centre from the nucleus, and whether the bar and 1.5 bar
lengths along the major axis fall inside the field.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from bartw.kinematics import read_muse_cube, sky_to_pixel, white_light

ap = argparse.ArgumentParser()
ap.add_argument("cube")
ap.add_argument("--targetid", type=int, help="look up RA/Dec and geometry in products/ (alternative to --ra/--dec)")
ap.add_argument("--ra", type=float); ap.add_argument("--dec", type=float)
ap.add_argument("--pa-bar", type=float); ap.add_argument("--a-bar", type=float, help="arcsec"); ap.add_argument("--eps-bar", type=float, default=0.6)
ap.add_argument("--pa-disc", type=float); ap.add_argument("--r-disc", type=float, help="arcsec"); ap.add_argument("--eps-disc", type=float, default=0.3)
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.targetid is not None:
    from bartw.lookup import lookup_target
    g = lookup_target(args.targetid)
    args.ra, args.dec = g["RA"], g["DEC"]
    for a, k in (("pa_bar", "PA_BAR"), ("a_bar", "A_BAR_ARCSEC"), ("eps_bar", "EPS_BAR"), ("pa_disc", "PA_DISC"), ("eps_disc", "EPS_DISC")):
        if getattr(args, a) is None or k in ("EPS_BAR", "EPS_DISC") and k in g:
            setattr(args, a, g.get(k, getattr(args, a)))
    if args.r_disc is None and "A_BAR_ARCSEC" in g:
        args.r_disc = 3.0 * g["A_BAR_ARCSEC"]
    print(f"TARGETID {args.targetid}: RA={args.ra:.5f} Dec={args.dec:.5f}; geometry from products/: "
          f"PA_disc={args.pa_disc}, PA_bar={args.pa_bar}, a_bar={args.a_bar}")
if args.ra is None or args.dec is None:
    raise SystemExit("give --targetid or --ra/--dec")

half = 1.5 * (args.a_bar or 20.0) + 20.0
cube = read_muse_cube(args.cube, lam_range=(4800.0, 5500.0), center=(args.ra, args.dec), half_size_arcsec=max(half, 45.0))
sig, noise = white_light(cube)
ny, nx = sig.shape
x0, y0 = sky_to_pixel(cube, args.ra, args.dec)
pix = cube.pixscale
inside = 0 <= x0 < nx and 0 <= y0 < ny
cx, cy = (nx - 1) / 2, (ny - 1) / 2
print(f"region read around the nucleus: {nx}x{ny} pix = {nx * pix:.0f}\" x {ny * pix:.0f}\", pixscale {pix:.3f}\" "
      f"(full cube {cube.header.get('NAXIS1')}x{cube.header.get('NAXIS2')} after cut)")
print(f"nucleus at pixel x0={x0:.1f}, y0={y0:.1f}  ({'inside' if inside else 'OUTSIDE'} the field); "
      f"offset from field centre {np.hypot(x0 - cx, y0 - cy) * pix:.1f}\"")
snr_nuc = sig[int(round(y0)), int(round(x0))] / noise[int(round(y0)), int(round(x0))] if inside else np.nan
print(f"S/N per spaxel at the nucleus (white light): {snr_nuc:.1f}")
# east-left check: RA should increase towards smaller x
east_left = cube.header.get("CD1_1", -1) < 0
print("orientation: east left (standard)" if east_left else "orientation: EAST RIGHT -> use --east-right in 06_run_tw.py")

def covered(pa_deg, length_arcsec):
    """fraction of the segment +-length along PA (from the nucleus) inside the field with finite data"""
    t = np.linspace(-1, 1, 101)
    dx = -np.sin(np.radians(pa_deg)) * (1 if east_left else -1)   # east component in pixel x
    dy = np.cos(np.radians(pa_deg))
    xs = x0 + t * length_arcsec / pix * dx
    ys = y0 + t * length_arcsec / pix * dy
    ok = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    fin = np.zeros_like(ok)
    fin[ok] = np.isfinite(sig[ys[ok].astype(int), xs[ok].astype(int)])
    return fin.mean()

if args.pa_bar is not None and args.a_bar is not None:
    print(f"bar (a={args.a_bar:.1f}\", PA {args.pa_bar:.0f}): {100 * covered(args.pa_bar, args.a_bar):.0f}% inside; "
          f"1.5 a_bar baseline: {100 * covered(args.pa_bar, 1.5 * args.a_bar):.0f}% inside")
if args.pa_disc is not None:
    print(f"disc major axis to {1.5 * (args.a_bar or 10):.0f}\": {100 * covered(args.pa_disc, 1.5 * (args.a_bar or 10)):.0f}% inside")

fig, ax = plt.subplots(figsize=(6.5, 6))
with np.errstate(all="ignore"):
    img = np.arcsinh(sig / np.nanmedian(noise))
ax.imshow(img, origin="lower", cmap="gray_r")
ax.plot(x0, y0, "r+", ms=14, mew=2, label="nucleus (catalogue)")
sign = 1 if east_left else -1
if args.pa_bar is not None and args.a_bar is not None:
    ax.add_patch(Ellipse((x0, y0), 2 * args.a_bar / pix, 2 * args.a_bar / pix * (1 - args.eps_bar),
                         angle=90 + sign * args.pa_bar, fill=False, ec="C2", lw=1.5, label="bar"))
if args.pa_disc is not None and args.r_disc is not None:
    ax.add_patch(Ellipse((x0, y0), 2 * args.r_disc / pix, 2 * args.r_disc / pix * (1 - args.eps_disc),
                         angle=90 + sign * args.pa_disc, fill=False, ec="C3", ls="--", lw=1.2, label="disc"))
ax.legend(loc="upper right", fontsize=8); ax.set_title(pathlib.Path(args.cube).name, fontsize=9)
out = args.out or (pathlib.Path(__file__).resolve().parents[1] / "figures" / f"cube_{pathlib.Path(args.cube).stem}.png")
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("figure:", out)
