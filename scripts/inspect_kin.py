#!/usr/bin/env python3
"""Diagnostic figure for a kinematics file: flux, V, V error, sigma, chi2 maps and
V error vs flux. Use it to judge whether the central fits are trustworthy before TW.

    python scripts/inspect_kin.py data/products/kin_NAME.fits [--vsys 1050] [--out figures/kin_NAME.png]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

ap = argparse.ArgumentParser()
ap.add_argument("kin"); ap.add_argument("--vsys", type=float, default=None); ap.add_argument("--out", default=None)
ap.add_argument("--vrange", type=float, default=None, help="half range of the V colour scale [km/s]")
args = ap.parse_args()
h = fits.open(args.kin)
flux, v, ve = h["FLUX"].data, h["V"].data, h["V_ERR"].data
sig = h["SIGMA"].data; chi2 = h["CHI2"].data if "CHI2" in h else None
vef = h["V_ERR_FORMAL"].data if "V_ERR_FORMAL" in h else None
vsys = args.vsys if args.vsys is not None else np.nanmedian(v)
ok = np.isfinite(flux) & np.isfinite(v)
bright = ok & (flux >= np.nanpercentile(flux[ok], 70))
print(f"{ok.sum()} spaxels with velocities; V_sys (median) {np.nanmedian(v):.1f}")
print(f"V error: median all {np.nanmedian(ve[ok]):.1f}, bright 30% {np.nanmedian(ve[bright]):.1f} km/s"
      + (f"; formal (no chi2 scaling) bright 30% {np.nanmedian(vef[bright]):.1f} km/s" if vef is not None else ""))
if chi2 is not None:
    print(f"chi2_red: median all {np.nanmedian(chi2[ok]):.2f}, bright 30% {np.nanmedian(chi2[bright]):.2f}")
print(f"sigma: median bright 30% {np.nanmedian(sig[bright]):.1f} km/s; fraction of bins with sigma < 5 km/s: {np.mean(sig[ok] < 5):.2f}")
vr = args.vrange or np.nanpercentile(np.abs(v[bright] - vsys), 95)
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
with np.errstate(all="ignore"):
    im = ax[0, 0].imshow(np.log10(flux), origin="lower", cmap="gray_r"); plt.colorbar(im, ax=ax[0, 0], fraction=0.046); ax[0, 0].set_title("log flux")
im = ax[0, 1].imshow(v - vsys, origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr); plt.colorbar(im, ax=ax[0, 1], fraction=0.046); ax[0, 1].set_title(f"V - {vsys:.0f} km/s")
im = ax[0, 2].imshow(ve, origin="lower", cmap="viridis", vmin=0, vmax=np.nanpercentile(ve[ok], 90)); plt.colorbar(im, ax=ax[0, 2], fraction=0.046); ax[0, 2].set_title("V error (chi2-scaled)")
im = ax[1, 0].imshow(sig, origin="lower", cmap="magma", vmin=0, vmax=np.nanpercentile(sig[ok], 95)); plt.colorbar(im, ax=ax[1, 0], fraction=0.046); ax[1, 0].set_title("sigma")
if chi2 is not None:
    im = ax[1, 1].imshow(chi2, origin="lower", cmap="cividis", vmin=0, vmax=np.nanpercentile(chi2[ok], 95)); plt.colorbar(im, ax=ax[1, 1], fraction=0.046); ax[1, 1].set_title("chi2_red")
ax[1, 2].scatter(np.log10(flux[ok]), ve[ok], s=2, alpha=0.3, label="chi2-scaled")
if vef is not None:
    ax[1, 2].scatter(np.log10(flux[ok]), vef[ok], s=2, alpha=0.3, label="formal")
ax[1, 2].set_xlabel("log flux"); ax[1, 2].set_ylabel("V error [km/s]"); ax[1, 2].set_yscale("log"); ax[1, 2].legend()
for a in ax.ravel()[:5]:
    a.set_xticks([]); a.set_yticks([])
out = args.out or str(pathlib.Path(__file__).resolve().parents[1] / "figures" / f"kin_{pathlib.Path(args.kin).stem}.png")
fig.tight_layout(); fig.savefig(out, dpi=120); print("figure:", out)
