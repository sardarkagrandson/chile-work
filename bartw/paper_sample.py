"""The star-forming dwarf sample of Robin Thomas' DESI DR1 paper
(``DESI_dwarfs.csv``, 14,786 galaxies, 318 columns).

This table *is* step 1 of the pipeline: DESI DR1 (main + SV + special),
star forming on both the [NII] and [SII] BPT diagrams with S/N > 3 in all six
lines, log M*/Msun <= 9.0 (FastSpecFit LOGMSTAR rescaled to h = 0.678,
column ``logmstar_h0678``), z < 0.0333, one spectrum per galaxy
(``retain_closest_spectrum``). It already carries what the later steps need:

* ``galaxy_center_ra/dec``   -- adopted galaxy centre (Tractor or SGA)
* ``shape_e1, shape_e2``     -- Tractor ellipticity components -> disc b/a and PA
* ``sga_id, pa, ba, sma_sb25, d26`` -- SGA-2020 geometry for the 797 SGA members
* ``kpc_per_arcsec, luminosity_distance_mpc`` -- physical scale
* ``SFR_Msun_yr, OH_mean, dn4000`` -- star formation / metallicity / age
* ``P_gg, nearest_sep_kpc, Q_0.5_300, n_companions, isolated`` -- environment

Tractor convention check (done on the 601 SGA members with e > 0.2):
PA = 0.5 * atan2(e2, e1) in degrees agrees with the SGA PA (north through
east) to a median 1.5 deg; b/a = (1-e)/(1+e) agrees with SGA b/a to 0.05.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import inclination_from_axis_ratio

KEEP = ["targetid", "survey", "program", "z", "zwarn", "spectype", "galaxy_center_ra",
        "galaxy_center_dec", "target_ra", "target_dec", "sga_id", "sga_galaxy", "pgc", "morphtype",
        "shape_r", "shape_e1", "shape_e2", "sersic", "pa", "ba", "sma_sb25", "sma_sb26", "d26",
        "r_mag_sb25", "r_mag_dered", "g_minus_r_dered", "m_r", "logmstar", "logmstar_h0678",
        "log_mstar_bell", "kpc_per_arcsec", "luminosity_distance_mpc", "d25_adopted_arcsec",
        "d25_adopted_kpc", "d25_adopted_method", "SFR_Msun_yr", "L_Halpha_ergs", "halpha_ew",
        "halpha_snr", "OH_mean", "OH_std", "dn4000", "velocity_kms", "P_gg", "nearest_sep_kpc",
        "Q_0.5_300", "n_companions", "isolated", "BPT_COMBINED_CLASS", "INCLUDE_IN_SF_ANALYSIS"]


def read_paper_sample(path, columns=KEEP):
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in set(columns))
    return df


def tractor_geometry(df, q0=0.3):
    """Disc ellipticity, b/a, PA (deg, N through E) and inclination from the
    Tractor shape components. Note the Tractor shape is the *total* light
    model (single Sersic/exp), so for strongly barred dwarfs it is biased
    towards the bar PA; step 4 refines it with outer isophotes."""
    e = np.hypot(df["shape_e1"], df["shape_e2"])
    out = df.copy()
    out["TR_ELLIP"] = e
    out["TR_BA"] = (1 - e) / (1 + e)
    out["TR_PA"] = np.degrees(0.5 * np.arctan2(df["shape_e2"], df["shape_e1"])) % 180.0
    out["TR_INCL_DEG"] = inclination_from_axis_ratio(out["TR_BA"], q0)
    # prefer SGA geometry when present (outer isophotes, less bar-biased)
    has_sga = df["ba"].notna() & df["pa"].notna()
    out["DISC_BA"] = np.where(has_sga, df["ba"], out["TR_BA"])
    out["DISC_PA"] = np.where(has_sga, df["pa"], out["TR_PA"])
    out["DISC_PA_SOURCE"] = np.where(has_sga, "SGA", "Tractor")
    out["INCL_DEG"] = inclination_from_axis_ratio(out["DISC_BA"], q0)
    return out


def standardize(df):
    """Upper-case the handful of columns the generic pipeline modules expect
    (RA, DEC, Z, TARGETID, LOGMSTAR_H70, SFR_H70) without dropping the rest."""
    out = df.copy()
    out["TARGETID"] = out["targetid"]
    out["RA"] = out["galaxy_center_ra"]
    out["DEC"] = out["galaxy_center_dec"]
    out["Z"] = out["z"]
    # paper masses are for h = 0.678; shift to the config H0 later if needed
    out["LOGMSTAR_H70"] = out["logmstar_h0678"] + 2 * np.log10(67.8 / 70.0)
    out["SFR_H70"] = out["SFR_Msun_yr"]
    return out
