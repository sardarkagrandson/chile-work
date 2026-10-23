"""Star-forming dwarf galaxies from the DESI DR1 FastSpecFit ("Iron") VAC.

Files: ``fastspec-iron-main-bright.fits`` (BGS). Depending on the FastSpecFit
version the stellar-population quantities (LOGMSTAR, SFR, DN4000) live in the
``FASTSPEC`` HDU (v2.x) or in a separate ``SPECPHOT`` HDU (v3.x); the loader
below finds columns by name across HDUs so both layouts work.

LOGMSTAR and SFR in the VAC assume h = 1. We convert to the configured H0:
log M(h) = LOGMSTAR + 2 log10(100/H0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

META_COLS = ["TARGETID", "RA", "DEC", "SURVEY", "PROGRAM", "HEALPIX", "Z", "ZWARN",
             "DELTACHI2", "SPECTYPE", "PHOTSYS", "BGS_TARGET"]
PHYS_COLS = ["LOGMSTAR", "SFR", "DN4000", "VDISP", "VDISP_IVAR", "ABSMAG01_SDSS_R",
             "ABSMAG01_SDSS_G"]
LINES = ["HALPHA", "HBETA", "NII_6584", "OIII_5007", "SII_6716", "SII_6731"]
LINE_COLS = [f"{l}_{s}" for l in LINES for s in ("FLUX", "FLUX_IVAR")] + \
            ["HALPHA_EW", "HALPHA_EW_IVAR", "HALPHA_SIGMA", "NARROW_SIGMA"]


def read_fastspec(path, columns=None):
    """Read a FastSpecFit catalogue into a DataFrame, gathering the requested
    columns from whichever HDU holds them. Rows are aligned across HDUs."""
    want = list(dict.fromkeys(columns or (META_COLS + PHYS_COLS + LINE_COLS)))
    frames = []
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            if not hasattr(hdu, "columns"):
                continue
            names = set(hdu.columns.names)
            cols = [c for c in want if c in names and c not in {col for f in frames for col in f.columns}]
            if cols:
                t = Table(hdu.data)[cols]
                frames.append(t.to_pandas())
    if not frames:
        raise ValueError(f"none of the requested columns found in {path}")
    df = pd.concat(frames, axis=1)
    for c in ("SURVEY", "PROGRAM", "SPECTYPE", "PHOTSYS"):
        if c in df and df[c].dtype == object:
            df[c] = df[c].str.decode("utf-8", errors="ignore") if isinstance(df[c].iloc[0], bytes) else df[c]
            df[c] = df[c].str.strip()
    missing = sorted(set(want) - set(df.columns))
    if missing:
        print(f"[desi] columns not found (skipped): {missing}")
    return df


def line_snr(df, line):
    f, iv = df[f"{line}_FLUX"].to_numpy(), df[f"{line}_FLUX_IVAR"].to_numpy()
    with np.errstate(invalid="ignore"):
        return np.where(iv > 0, f * np.sqrt(iv), 0.0)


def bpt_classification(df, snr_min=3.0, ew_min=3.0):
    """Return an array of labels: 'SF', 'composite', 'AGN', 'SF_WHAN' (SF by
    WHAN when the [OIII]/Hbeta axis is unavailable), 'retired', 'unclassified'.

    [NII]-BPT: Kauffmann+03 (SF below) and Kewley+01 (AGN above) curves.
    WHAN (Cid Fernandes+11): SF if log([NII]/Ha) < -0.4 and EW(Ha) > 3 A.
    """
    snr = {l: line_snr(df, l) for l in ("HALPHA", "HBETA", "NII_6584", "OIII_5007")}
    with np.errstate(divide="ignore", invalid="ignore"):
        n2ha = np.log10(df["NII_6584_FLUX"] / df["HALPHA_FLUX"])
        o3hb = np.log10(df["OIII_5007_FLUX"] / df["HBETA_FLUX"])
    ew = df["HALPHA_EW"].to_numpy() if "HALPHA_EW" in df else np.full(len(df), np.nan)

    kauff = 0.61 / (n2ha - 0.05) + 1.3
    kewley = 0.61 / (n2ha - 0.47) + 1.19
    all4 = (snr["HALPHA"] > snr_min) & (snr["HBETA"] > snr_min) & \
           (snr["NII_6584"] > snr_min) & (snr["OIII_5007"] > snr_min)
    sf = all4 & ((o3hb < kauff) & (n2ha < 0.05))
    agn = all4 & ((o3hb > kewley) | (n2ha > 0.47))
    comp = all4 & ~sf & ~agn
    two = (snr["HALPHA"] > snr_min) & (snr["NII_6584"] > snr_min) & ~all4
    whan_sf = two & (n2ha < -0.4) & (ew > ew_min)
    retired = (snr["HALPHA"] > snr_min) & (ew <= ew_min)

    label = np.full(len(df), "unclassified", dtype=object)
    label[retired] = "retired"
    label[whan_sf] = "SF_WHAN"
    label[comp] = "composite"
    label[agn] = "AGN"
    label[sf] = "SF"
    return label, n2ha, o3hb


def select_star_forming_dwarfs(df, cfg):
    """Apply the redshift-quality, redshift, dwarf-mass and star-forming cuts
    from ``cfg['desi']``. Returns (selected DataFrame, dict of step counts)."""
    c = cfg["desi"]
    h_corr = 2.0 * np.log10(100.0 / cfg["cosmology"]["H0"])
    df = df.copy()
    df["LOGMSTAR_H70"] = df["LOGMSTAR"] + h_corr
    df["SFR_H70"] = df["SFR"] * 10**h_corr
    with np.errstate(divide="ignore", invalid="ignore"):
        df["LOG_SSFR"] = np.log10(df["SFR_H70"]) - df["LOGMSTAR_H70"]
    df["BPT_CLASS"], df["LOG_N2HA"], df["LOG_O3HB"] = bpt_classification(
        df, c["line_snr_min"], c["halpha_ew_min"])

    steps = {"input": len(df)}
    m = np.ones(len(df), bool)
    if "SURVEY" in df:
        m &= (df["SURVEY"] == c["survey"]) & (df["PROGRAM"] == c["program"])
        steps["survey/program"] = int(m.sum())
    m &= (df["ZWARN"] <= c["zwarn_max"]) & (df["DELTACHI2"] >= c["deltachi2_min"])
    if "SPECTYPE" in df:
        m &= df["SPECTYPE"] == c["spectype"]
    steps["good redshift"] = int(m.sum())
    m &= (df["Z"] > c["z_min"]) & (df["Z"] < c["z_max"])
    steps["redshift window"] = int(m.sum())
    m &= (df["LOGMSTAR_H70"] >= c["logmstar_min"]) & (df["LOGMSTAR_H70"] <= c["logmstar_max"])
    steps["dwarf mass"] = int(m.sum())
    m &= np.isin(df["BPT_CLASS"], ["SF", "SF_WHAN"]) & (df["LOG_SSFR"] > c["log_ssfr_min"])
    steps["star forming"] = int(m.sum())
    # one spectrum per object (repeat observations share TARGETID)
    out = df[m].sort_values("DELTACHI2", ascending=False).drop_duplicates("TARGETID")
    steps["unique TARGETID"] = len(out)
    return out.reset_index(drop=True), steps
