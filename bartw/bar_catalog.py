"""The "Properties of bars for galaxies from DESI Legacy Imaging Surveys"
catalogue (89,780 SGA-2020 galaxies; provided by V. Cuomo, Oct 2026).

Columns: SGA_ID, GALAXY, RA, DEC, G/R/Z_MAG_SB25, bar_length, bar_ellipticity.

Caveats found on inspection (2026-10-23):
* ``bar_length`` has no unit in the header. For NGC 4314, NGC 3351 and
  IC 3167 the values (0.61, 0.33, 0.15) are about half the literature bar
  semi-major axes in arcmin (~1.1, ~0.85, ~0.25), so it is either a different
  estimator (e.g. radius of maximum ellipticity) in arcmin, or normalised by
  an SGA radius. **Confirm the unit and definition with the catalogue authors
  before using it as a_bar in the rotation rate.** The loader keeps the raw
  value in ``BAR_LENGTH_RAW`` and converts to arcsec only if a unit is given.
* No bar position angle and no disc geometry are included. Both come from
  joining SGA-2020 (``PA``, ``BA``, ``D26``) on ``SGA_ID`` and, for the bar PA,
  from our own isophotal analysis (:mod:`bartw.photometry`).
* ``bar_ellipticity`` is a bar-strength proxy. Values < ~0.2 are probably
  not real bars (compare IC 3167: 0.07, a known weak lopsided bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

SGA_COLS = ["SGA_ID", "GALAXY", "RA", "DEC", "PA", "BA", "D26", "D25_LEDA", "PA_LEDA",
            "BA_LEDA", "MORPHTYPE", "Z_LEDA", "SB_D25_LEDA", "R_MAG_SB26"]


def read_bar_catalog(path, bar_length_unit=None):
    with fits.open(path) as h:
        df = Table(h[1].data).to_pandas()
    df["GALAXY"] = df["GALAXY"].astype(str).str.strip()
    df = df.rename(columns={"RA": "SGA_RA", "DEC": "SGA_DEC", "bar_length": "BAR_LENGTH_RAW",
                            "bar_ellipticity": "BAR_ELLIP"})
    df["BAR_LENGTH_ARCSEC"] = np.nan
    if bar_length_unit == "arcmin":
        df["BAR_LENGTH_ARCSEC"] = df["BAR_LENGTH_RAW"] * 60.0
    elif bar_length_unit == "arcsec":
        df["BAR_LENGTH_ARCSEC"] = df["BAR_LENGTH_RAW"]
    return df


def read_sga(path, columns=SGA_COLS):
    """SGA-2020 main table (SGA-2020.fits, HDU 'ELLIPSE' or 1)."""
    with fits.open(path, memmap=True) as h:
        hdu = h["ELLIPSE"] if "ELLIPSE" in h else h[1]
        cols = [c for c in columns if c in hdu.columns.names]
        df = Table(hdu.data)[cols].to_pandas()
    for c in ("GALAXY", "MORPHTYPE"):
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    return df.add_prefix("SGA_").rename(columns={"SGA_SGA_ID": "SGA_ID"})


def join_sga(bars, sga):
    return bars.merge(sga, on="SGA_ID", how="left", suffixes=("", "_sga"))


def crossmatch_desi(df_desi, bars, radius_arcsec=3.0):
    """Attach bar-catalogue rows to the DESI sample by position. SGA centres
    and DESI BGS fibre positions for SGA galaxies agree to ~1"; 3" allows
    for off-centre fibre placement in large galaxies."""
    c1 = SkyCoord(df_desi["RA"].to_numpy() * u.deg, df_desi["DEC"].to_numpy() * u.deg)
    c2 = SkyCoord(bars["SGA_RA"].to_numpy() * u.deg, bars["SGA_DEC"].to_numpy() * u.deg)
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    ok = sep.arcsec <= radius_arcsec
    b = bars.iloc[idx].reset_index(drop=True)
    b.loc[~ok, :] = np.nan
    out = pd.concat([df_desi.reset_index(drop=True), b], axis=1)
    out["BAR_SEP_ARCSEC"] = np.where(ok, sep.arcsec, np.nan)
    out["IN_BAR_CATALOG"] = ok
    return out
