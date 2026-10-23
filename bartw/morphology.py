"""Barred galaxies from Galaxy Zoo DESI (Walmsley et al. 2023, MNRAS 526, 4768).

The catalogue (Zenodo 8360385, ``gz_desi_deep_learning_catalog_friendly``)
gives, for 8.7M Legacy Surveys galaxies with r < 19, the predicted vote
fraction for each answer of the decision tree. The bar question is only asked
of featured, not-edge-on galaxies, so the recommended selection is:

    featured-or-disk >= 0.27,  not edge-on >= 0.68,  bar_no < 0.5

(Masters+2011/2021 and the DESI bar papers of 2025-26 use the same cuts).
Vote fractions are *not* probabilities of a bar being present; they are the
expected volunteer vote fraction. The strong/weak split follows
strong/(strong+weak) > 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

GZ_COLS = {
    "id": "dr8_id",
    "ra": "ra",
    "dec": "dec",
    "featured": "smooth-or-featured_featured-or-disk_fraction",
    "edge_on": "disk-edge-on_yes_fraction",
    "bar_strong": "bar_strong_fraction",
    "bar_weak": "bar_weak_fraction",
    "bar_no": "bar_no_fraction",
    "spiral": "has-spiral-arms_yes_fraction",
    "merging_none": "merging_none_fraction",
}


def read_gz_desi(path, extra_cols=()):
    """Read only the needed columns of the (large) GZ DESI catalogue.
    Accepts .parquet or .csv(.gz)."""
    cols = list(GZ_COLS.values()) + list(extra_cols)
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in cols)
    return df


def crossmatch(df_desi, df_gz, radius_arcsec=1.5):
    """Nearest-neighbour sky match; returns df_desi with GZ columns appended
    (NaN where no match within ``radius_arcsec``)."""
    c1 = SkyCoord(df_desi["RA"].to_numpy() * u.deg, df_desi["DEC"].to_numpy() * u.deg)
    c2 = SkyCoord(df_gz[GZ_COLS["ra"]].to_numpy() * u.deg, df_gz[GZ_COLS["dec"]].to_numpy() * u.deg)
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    ok = sep.arcsec <= radius_arcsec
    gz = df_gz.iloc[idx].reset_index(drop=True).add_prefix("GZ_")
    gz.loc[~ok, :] = np.nan
    out = pd.concat([df_desi.reset_index(drop=True), gz], axis=1)
    out["GZ_SEP_ARCSEC"] = np.where(ok, sep.arcsec, np.nan)
    return out


def select_barred(df, cfg_gz):
    """Apply the GZ DESI disc + bar cuts. Returns (barred DataFrame, counts)."""
    g = {k: f"GZ_{v}" for k, v in GZ_COLS.items()}
    steps = {"matched to GZ DESI": int(np.isfinite(df["GZ_SEP_ARCSEC"]).sum())}
    disc = (df[g["featured"]] >= cfg_gz["featured_min"]) & \
           ((1.0 - df[g["edge_on"]]) >= cfg_gz["not_edge_on_min"])
    steps["featured, not edge-on"] = int(disc.sum())
    bar = disc & (df[g["bar_no"]] < cfg_gz["bar_no_max"]) & \
          ((df[g["bar_strong"]] + df[g["bar_weak"]]) >= cfg_gz["bar_strong_plus_weak_min"])
    steps["barred"] = int(bar.sum())
    out = df[bar].copy()
    ratio = out[g["bar_strong"]] / (out[g["bar_strong"]] + out[g["bar_weak"]])
    out["BAR_CLASS"] = np.where(ratio > cfg_gz["strong_bar_ratio"], "strong", "weak")
    out["GZ_BAR_FRACTION"] = out[g["bar_strong"]] + out[g["bar_weak"]]
    return out.reset_index(drop=True), steps
