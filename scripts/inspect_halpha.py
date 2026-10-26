#!/usr/bin/env python3
"""Look at the data behind the velocities: binned spectra around Halpha (and
[OIII]5007, [SII]) at several positions, with the stellar and gas velocities
marked, plus absorption-free gas velocity maps from [OIII] and [SII] compared
with Halpha and with the stars.

    python scripts/inspect_halpha.py CUBE.fits --targetid ID --stars data/products/kin_NAME.fits --name NAME
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from bartw.config import ROOT
from bartw.gaskin import HALPHA, NII_A, NII_B, LINES, fit_halpha_bins, fit_line_bins
from bartw.kinematics import read_muse_cube, sky_to_pixel, stack_bins
from bartw.kinpa import fit_kinematic_pa
from bartw.lookup import lookup_target

C = 299792.458
ap = argparse.ArgumentParser()
ap.add_argument("cube"); ap.add_argument("--targetid", type=int, required=True); ap.add_argument("--stars", required=True)
ap.add_argument("--name", default="gal"); ap.add_argument("--offsets", type=float, default=15.0, help="arcsec from the nucleus for the sample positions")
args = ap.parse_args()
g = lookup_target(args.targetid); z = g["Z"]; a_bar = g.get("A_BAR_ARCSEC", 15.0)

with fits.open(args.stars) as h:
    khdr = h["V"].header; vs_map = h["V"].data.astype(float); bins = h["BINNUM"].data.astype(int); ves_map = h["V_ERR"].data.astype(float)
kw = WCS(khdr).celestial; ny, nx = vs_map.shape
ra_c, dec_c = (float(v) for v in kw.all_pix2world((nx - 1) / 2, (ny - 1) / 2, 0))
lam_lo, lam_hi = 4800.0, 6800.0
cube = read_muse_cube(args.cube, lam_range=(lam_lo, lam_hi), center=(ra_c, dec_c), half_size_arcsec=(nx - 1) / 2 * 0.2)
assert cube.data.shape[1:] == vs_map.shape, (cube.data.shape, vs_map.shape)
x0, y0 = sky_to_pixel(cube, g["RA"], g["DEC"]); pix = cube.pixscale
east_left = cube.header.get("CD1_1", -1) < 0
nb = bins.max() + 1
vstar = np.full(nb, np.nan); okb = bins >= 0; vstar[bins[okb]] = vs_map[okb]

# --- absorption-free gas lines
print("fitting [OIII]5007, [SII] and Halpha in the stellar bins ...")
o3 = fit_line_bins(cube, bins, z, "oiii", progress=False)
s2 = fit_line_bins(cube, bins, z, "sii", progress=False)
ha = fit_halpha_bins(cube, bins, z, v_star=vstar, progress=False)
for nm, r in (("[OIII]", o3), ("[SII]", s2), ("Halpha", ha)):
    ok = np.isfinite(r["v"])
    d = r["v"] - vstar
    print(f"{nm:7s}: {ok.sum():5d} bins; median V {np.nanmedian(r['v']):.1f}; median (gas - stars) {np.nanmedian(d):.1f} km/s, "
          f"scatter (MAD) {1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d))):.1f} km/s")
d_o3_ha = o3["v"] - ha["v"]
print(f"[OIII] - Halpha: median {np.nanmedian(d_o3_ha):.1f} km/s, MAD {1.4826 * np.nanmedian(np.abs(d_o3_ha - np.nanmedian(d_o3_ha))):.1f} km/s")
r_in, r_out = 1.2 * a_bar, 0.95 * 0.5 * min(ny, nx) * pix
for nm, vm, vem in (("stars", vs_map, ves_map), ("[OIII]", o3["v_map"], o3["v_err_map"]), ("[SII]", s2["v_map"], s2["v_err_map"]), ("Halpha", ha["v_map"], ha["v_err_map"])):
    kp = fit_kinematic_pa(vm, x0, y0, east_left=east_left, radius_pix=r_out / pix, rmin_pix=r_in / pix, v_err=vem)
    print(f"kinematic PA ({nm:7s}, annulus {r_in:.0f}-{r_out:.0f}\"): {kp['pa_kin']:6.1f} +/- {kp['pa_kin_err']:.1f}; amplitude {kp['v_amp']:5.1f} km/s; "
          f"V_sys {kp['v_sys']:.1f}; {'detected' if kp['detected'] else 'not detected'}")

# --- spectra at sample positions
def bin_at(dx_arcsec, dy_arcsec):
    sgn = -1 if east_left else 1
    xi, yi = int(round(x0 + sgn * dx_arcsec / pix)), int(round(y0 + dy_arcsec / pix))
    return bins[yi, xi] if 0 <= xi < nx and 0 <= yi < ny else -1

d = args.offsets
pos = {"nucleus": bin_at(0, 0), f"N {d:.0f}\"": bin_at(0, d), f"S {d:.0f}\"": bin_at(0, -d), f"E {d:.0f}\"": bin_at(d, 0), f"W {d:.0f}\"": bin_at(-d, 0),
       f"N {2*d:.0f}\"": bin_at(0, 2 * d), f"S {2*d:.0f}\"": bin_at(0, -2 * d)}
lam_ha, sp_ha, _ = stack_bins(cube, bins, (6520 * (1 + z), 6610 * (1 + z)))
lam_o3, sp_o3, _ = stack_bins(cube, bins, (4980 * (1 + z), 5035 * (1 + z)))
fig, axes = plt.subplots(len(pos), 2, figsize=(14, 2.6 * len(pos)))
for row, (lab, b) in enumerate(pos.items()):
    if b < 0:
        continue
    for col, (lam_, sp_, lines, title) in enumerate(((lam_ha, sp_ha, [(HALPHA, "Ha"), (NII_A, "[NII]"), (NII_B, "[NII]")], "Halpha region"),
                                                     (lam_o3, sp_o3, [(LINES["oiii"], "[OIII]")], "[OIII]5007"))):
        ax = axes[row, col]; ax.plot(lam_, sp_[:, b], "k-", lw=0.8)
        for l0, nm in lines:
            ax.axvline(l0 * (1 + vstar[b] / C), color="C0", ls="--", lw=1, label="stellar V" if nm in ("Ha", "[OIII]") else None)
            vg = ha["v"][b] if col == 0 else o3["v"][b]
            if np.isfinite(vg):
                ax.axvline(l0 * (1 + vg / C), color="C3", ls=":", lw=1.2, label="gas V" if nm in ("Ha", "[OIII]") else None)
        ax.set_title(f"{lab}  bin {b}: V* = {vstar[b]:.0f}, V(Ha) = {ha['v'][b]:.0f}, V([OIII]) = {o3['v'][b]:.0f} km/s" if col == 0 else title, fontsize=9)
        if row == 0:
            ax.legend(fontsize=8)
axes[-1, 0].set_xlabel("wavelength [A]"); axes[-1, 1].set_xlabel("wavelength [A]")
out = ROOT / "figures" / f"spectra_{args.name}.png"; fig.tight_layout(); fig.savefig(out, dpi=110); print("figure:", out)

# --- velocity maps side by side
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
vsys = np.nanmedian(vs_map); vr = 60
for a, (nm, vm) in zip(ax, (("stars", vs_map), ("Halpha", ha["v_map"]), ("[OIII]", o3["v_map"]), ("[SII]", s2["v_map"]))):
    im = a.imshow(vm - vsys, origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr); a.set_title(f"{nm}  (V - {vsys:.0f})"); a.set_xticks([]); a.set_yticks([])
    a.plot(x0, y0, "k+", ms=10)
plt.colorbar(im, ax=ax, fraction=0.02, label="km/s")
out2 = ROOT / "figures" / f"velmaps_{args.name}.png"; fig.savefig(out2, dpi=110); print("figure:", out2)
