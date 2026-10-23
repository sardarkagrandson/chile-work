"""Projection geometry and Tremaine-Weinberg applicability tests."""
from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM

ARCSEC_PER_RAD = 206264.806


def inclination_from_axis_ratio(q, q0=0.3):
    """Inclination (deg) from apparent axis ratio q=b/a for an oblate disc of
    intrinsic thickness q0 (Hubble 1926): cos^2 i = (q^2 - q0^2)/(1 - q0^2).

    Dwarfs are thicker than giant discs; q0 = 0.3-0.4 is appropriate
    (e.g. Sanchez-Janssen+2010), vs 0.2 for massive spirals.
    Returns 90 deg where q <= q0.
    """
    q = np.asarray(q, dtype=float)
    cos2 = np.clip((q**2 - q0**2) / (1.0 - q0**2), 0.0, 1.0)
    return np.degrees(np.arccos(np.sqrt(cos2)))


def axis_ratio_from_inclination(incl_deg, q0=0.3):
    i = np.radians(incl_deg)
    return np.sqrt(np.cos(i) ** 2 * (1 - q0**2) + q0**2)


def pa_difference(pa1_deg, pa2_deg):
    """Smallest difference between two position angles (deg, in [0, 90])."""
    d = np.abs((np.asarray(pa1_deg) - np.asarray(pa2_deg) + 90.0) % 180.0 - 90.0)
    return d


def deproject_bar_pa(pa_bar_deg, pa_disc_deg, incl_deg):
    """Bar position angle in the disc plane, relative to the disc major axis."""
    dpa = np.radians(np.asarray(pa_bar_deg) - np.asarray(pa_disc_deg))
    return np.degrees(np.arctan2(np.sin(dpa) / np.cos(np.radians(incl_deg)), np.cos(dpa)))


def deproject_bar_length(a_bar_sky, pa_bar_deg, pa_disc_deg, incl_deg):
    """Deprojected bar semi-major axis (Gadotti+2007 eq. 1)."""
    dpa = np.radians(np.asarray(pa_bar_deg) - np.asarray(pa_disc_deg))
    ci = np.cos(np.radians(incl_deg))
    return a_bar_sky * np.sqrt(np.cos(dpa) ** 2 + (np.sin(dpa) / ci) ** 2)


def kpc_per_arcsec(z=None, distance_mpc=None, H0=70.0, Om0=0.3):
    """Physical scale. Prefer a redshift-independent distance when available
    (at z < 0.01 peculiar velocities dominate the Hubble flow)."""
    if distance_mpc is None:
        if z is None:
            raise ValueError("need z or distance_mpc")
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        distance_mpc = cosmo.angular_diameter_distance(z).value
    return np.asarray(distance_mpc) * 1000.0 / ARCSEC_PER_RAD


def tw_suitable(incl_deg, pa_bar_deg, pa_disc_deg, cfg_geom):
    """Boolean mask + reason strings for the TW geometric criteria
    (Garma-Oehmichen+2022; Geron+2023; Cuomo+2024): intermediate inclination
    and a bar that is aligned with neither the major nor the minor axis."""
    incl = np.atleast_1d(incl_deg).astype(float)
    dpa = pa_difference(pa_bar_deg, pa_disc_deg)
    ok_i = (incl >= cfg_geom["incl_min_deg"]) & (incl <= cfg_geom["incl_max_deg"])
    ok_pa = (dpa >= cfg_geom["dpa_min_deg"]) & (dpa <= cfg_geom["dpa_max_deg"])
    reasons = np.where(~ok_i, "inclination", np.where(~ok_pa, "bar_pa", "ok"))
    return ok_i & ok_pa, dpa, reasons
