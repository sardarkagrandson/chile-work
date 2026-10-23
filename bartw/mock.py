"""Synthetic barred discs with a known pattern speed, to validate :mod:`bartw.tw`.

Construction
------------
The TW relation follows from the continuity equation for a tracer whose
density pattern rotates rigidly. The TW integrals are linear in Sigma and in
Sigma*V, so a sum of components that *each* satisfy continuity satisfies the
relation exactly. We therefore build:

* an axisymmetric exponential disc with an arbitrary rotation curve V(r)
  (axisymmetric + purely azimuthal flow => divergence free; it contributes
  exactly zero to both TW integrals for symmetric slits), and
* a Ferrers-like bar whose material rotates as a solid body with Omega_p
  (a rigidly rotating pattern => satisfies continuity with d/dt = -Omega_p d/dphi).

The observed luminosity-weighted velocity is
V = (Sigma_d V_d + Sigma_b V_b) / (Sigma_d + Sigma_b),
which is exactly what an IFU measures for a single stellar population.
Recovering Omega_p sin i from such a mock is therefore a clean test of the
pseudo-slit integration, slope fitting, coordinate conventions and error
propagation, independent of any dynamical model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class MockParams:
    nx: int = 300
    ny: int = 300
    pixscale: float = 0.2            # arcsec/pix (MUSE)
    distance_mpc: float = 16.5       # Virgo
    pa_deg: float = 35.0             # disc major axis PA (N through E)
    incl_deg: float = 50.0
    h_disc_arcsec: float = 10.0      # disc scale length
    vflat: float = 80.0              # km/s (dwarf)
    r_turn_arcsec: float = 6.0
    a_bar_arcsec: float = 9.0        # bar semi-major axis (disc plane)
    b_bar_arcsec: float = 3.5
    bar_phi_deg: float = 40.0        # bar angle in the disc plane from major axis
    bar_contrast: float = 4.0        # Sigma_b(0) / Sigma_d(0); dwarf bars are weak, 1.5 is a hard case
    ferrers_n: float = 1.0
    omega_p: float = 30.0            # km/s/kpc  (fast bar for these numbers)
    v_sys: float = 1200.0
    psf_fwhm_arcsec: float = 0.0
    flux_noise_frac: float = 0.0
    vel_noise_kms: float = 0.0
    east_left: bool = True
    seed: int = 1


def kpc_per_arcsec(distance_mpc):
    return distance_mpc * 1000.0 / 206264.806


def make_mock(p: MockParams):
    """Return flux, V_los, V_err maps (2-D), plus a dict with ground truth."""
    rng = np.random.default_rng(p.seed)
    yy, xx = np.mgrid[0:p.ny, 0:p.nx].astype(float)
    x0, y0 = (p.nx - 1) / 2.0, (p.ny - 1) / 2.0
    pa = np.radians(p.pa_deg)
    dx = (xx - x0) * (-1.0 if p.east_left else 1.0) * p.pixscale
    dy = (yy - y0) * p.pixscale
    # sky coordinates along major (X) and minor (Y) axes
    X = dx * np.sin(pa) + dy * np.cos(pa)
    Y = -dx * np.cos(pa) + dy * np.sin(pa)
    inc = np.radians(p.incl_deg)
    # disc-plane coordinates
    xd, yd = X, Y / np.cos(inc)
    r = np.hypot(xd, yd)

    # axisymmetric disc with circular rotation
    sig_d = np.exp(-r / p.h_disc_arcsec)
    vrot = p.vflat * np.tanh(r / p.r_turn_arcsec)
    with np.errstate(invalid="ignore", divide="ignore"):
        vd_y = np.where(r > 0, vrot * xd / r, 0.0)     # y-component of circular flow

    # rigidly rotating Ferrers bar: v = Omega_p (-y, x)
    scale = kpc_per_arcsec(p.distance_mpc)
    omega_arcsec = p.omega_p * scale                    # km/s/arcsec
    phi = np.radians(p.bar_phi_deg)
    xb = xd * np.cos(phi) + yd * np.sin(phi)
    yb = -xd * np.sin(phi) + yd * np.cos(phi)
    m2 = (xb / p.a_bar_arcsec) ** 2 + (yb / p.b_bar_arcsec) ** 2
    sig_b = p.bar_contrast * np.clip(1.0 - m2, 0.0, None) ** p.ferrers_n
    vb_y = omega_arcsec * xd                            # solid body, y-component

    # line of sight: V_los = v_y sin i  (tilt about the major axis)
    sig = sig_d + sig_b
    vlos = p.v_sys + np.sin(inc) * (sig_d * vd_y + sig_b * vb_y) / sig

    if p.psf_fwhm_arcsec > 0:
        s = p.psf_fwhm_arcsec / 2.3548 / p.pixscale
        mom0 = gaussian_filter(sig, s)
        mom1 = gaussian_filter(sig * vlos, s)
        sig, vlos = mom0, mom1 / mom0

    verr = np.full_like(vlos, max(p.vel_noise_kms, 1e-3))
    if p.flux_noise_frac > 0:
        sig = sig * (1 + p.flux_noise_frac * rng.standard_normal(sig.shape))
        sig = np.clip(sig, 1e-6, None)
    if p.vel_noise_kms > 0:
        vlos = vlos + p.vel_noise_kms * rng.standard_normal(vlos.shape)

    # bar PA on the sky (for slit placement, as a real analysis would measure)
    bar_pa_sky = p.pa_deg + np.degrees(np.arctan2(np.sin(phi) * np.cos(inc), np.cos(phi)))
    a_bar_sky = p.a_bar_arcsec * np.sqrt(np.cos(phi) ** 2 + (np.sin(phi) * np.cos(inc)) ** 2)
    truth = dict(x0=x0, y0=y0, omega_p=p.omega_p, omega_sini_arcsec=omega_arcsec * np.sin(inc),
                 scale_kpc_arcsec=scale, bar_pa_sky=bar_pa_sky, a_bar_sky_arcsec=a_bar_sky,
                 v_sys=p.v_sys)
    return sig, vlos, verr, truth
