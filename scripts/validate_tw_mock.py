#!/usr/bin/env python3
"""Validate the TW implementation on synthetic barred discs and make
figures/tw_mock_validation.png (also used in docs/03_tw_method.md)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run without pip install
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bartw.config import ROOT
from bartw.geometry import pa_difference
from bartw.mock import MockParams, make_mock
from bartw.plotting import slit_overlay, tw_diagram
from bartw.tw import default_slit_centers, measure_pattern_speed

p = MockParams(psf_fwhm_arcsec=0.8, flux_noise_frac=0.05, vel_noise_kms=5.0)
flux, vel, verr, t = make_mock(p)
dpa = pa_difference(t["bar_pa_sky"], p.pa_deg)
yc = default_slit_centers(t["a_bar_sky_arcsec"], dpa, 9, 1.0)
xl = 1.5 * t["a_bar_sky_arcsec"]
res = measure_pattern_speed(flux, vel, p.pixscale, t["x0"], t["y0"], p.pa_deg, p.incl_deg, yc, 1.0,
                            x_half_length_arcsec=xl, vel_err=verr, distance_mpc=p.distance_mpc,
                            pa_err_deg=3.0, n_mc=300, rng=0)
print("input  Omega_p =", p.omega_p, "km/s/kpc")
print("output", res.summary())

# PA sensitivity curve (deterministic)
dpas = np.linspace(-6, 6, 25)
om = []
for d in dpas:
    r = measure_pattern_speed(flux, vel, p.pixscale, t["x0"], t["y0"], p.pa_deg + d, p.incl_deg, yc, 1.0,
                              x_half_length_arcsec=xl, vel_err=verr, distance_mpc=p.distance_mpc)
    om.append(r.omega_p)

fig = plt.figure(figsize=(16, 4.6))
axes = [fig.add_subplot(1, 4, k) for k in (1, 2, 3, 4)]
slit_overlay(flux, vel, res, 1.0, p.pixscale, True, axes[:2], xl)
axes[0].set_title("mock: disc + rigid Ferrers bar", fontsize=10)
tw_diagram(res, axes[2], title=rf"input $\Omega_p$={p.omega_p} km/s/kpc, recovered {res.omega_p:.1f}$\pm${res.omega_p_err:.1f}")
axes[3].plot(dpas, om, "o-", ms=3)
axes[3].axhline(p.omega_p, color="C3", ls="--", label="truth")
axes[3].axvspan(-3, 3, color="0.9", label=r"$\pm$3$^\circ$ (typical PA error)")
axes[3].set_xlabel(r"disc PA error [deg]"); axes[3].set_ylabel(r"recovered $\Omega_p$ [km/s/kpc]")
axes[3].set_title("sensitivity to the disc PA"); axes[3].legend(fontsize=8)
fig.tight_layout()
out = ROOT / "figures" / "tw_mock_validation.png"
fig.savefig(out, dpi=130)
print("wrote", out)
