"""Diagnostic figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tw import sky_to_disc_axes


def tw_diagram(res, ax=None, title=None):
    """<V> vs <X> with the ODR fit."""
    ax = ax or plt.gca()
    I = res.integrals
    ax.errorbar(I.x_mean, I.v_mean - res.v_sys, xerr=I.x_err, yerr=I.v_err, fmt="o", color="C0", ms=5)
    xx = np.linspace(np.nanmin(I.x_mean) * 1.2, np.nanmax(I.x_mean) * 1.2, 10)
    ax.plot(xx, res.omega_sini * xx, "-", color="C3",
            label=rf"$\Omega_p\sin i$ = {res.omega_sini:.2f} $\pm$ {res.omega_sini_err:.2f} km s$^{{-1}}$ arcsec$^{{-1}}$")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\langle X\rangle$ [arcsec]")
    ax.set_ylabel(r"$\langle V\rangle - V_{\rm sys}$ [km s$^{-1}$]")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def slit_overlay(flux, vel, res, slit_width, pixscale, east_left=True, axes=None, x_half_length=None):
    """Flux and velocity maps with the pseudo-slits drawn."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    ny, nx = flux.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    I = res.integrals
    for ax, img, cmap, lab in ((axes[0], np.log10(flux), "gray_r", "log flux"),
                               (axes[1], vel - res.v_sys, "RdBu_r", r"$V - V_{\rm sys}$")):
        im = ax.imshow(img, origin="lower", cmap=cmap)
        plt.colorbar(im, ax=ax, label=lab, fraction=0.046)
        ax.set_xticks([]); ax.set_yticks([])
    # slit edges
    x0 = y0 = None
    X, Y = sky_to_disc_axes(xx, yy, (nx - 1) / 2, (ny - 1) / 2, res.pa_deg, east_left)
    X, Y = X * pixscale, Y * pixscale
    for ax in axes:
        for yc in I.y_center:
            sel = np.abs(Y - yc) <= slit_width / 2
            if x_half_length is not None:
                sel &= np.abs(X) <= x_half_length
            ax.contour(sel.astype(float), levels=[0.5], colors="k", linewidths=0.5)
    return axes
