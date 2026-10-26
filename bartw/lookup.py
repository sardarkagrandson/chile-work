"""Look up a galaxy (by TARGETID) in the product tables: position and, if
measured, disc/bar geometry. Used by the cube scripts so the user never has to
copy coordinates by hand."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .config import ROOT

GEOM_COLS = ["PA_DISC", "PA_DISC_ERR", "EPS_DISC", "INCL_GEOM", "INCL_DEG", "PA_BAR", "A_BAR_ARCSEC",
             "A_BAR_ERR_ARCSEC", "EPS_BAR", "DPA_DEG", "KPC_PER_ARCSEC", "Z", "LOGMSTAR"]


def lookup_target(targetid):
    """Return a dict with RA, DEC, Z and any geometry columns found for this TARGETID,
    searching products/*tw_ready*.csv, *sweep_geometry*.csv, *geometry*.csv, *candidates*.csv."""
    out = {}
    patterns = ["*tw_ready*.csv", "*sweep_geometry*.csv", "*geometry*.csv", "*candidates*.csv", "*muse*.csv"]
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / "products" / pat))):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "TARGETID" not in df:
                continue
            row = df[df.TARGETID == targetid]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            for c in ["RA", "DEC"] + GEOM_COLS:
                if c in row and c not in out and np.isfinite(pd.to_numeric(row[c], errors="coerce")):
                    out[c] = float(row[c])
    if "RA" not in out:
        raise KeyError(f"TARGETID {targetid} not found in products/ tables")
    if "EPS_DISC" not in out and "INCL_GEOM" in out:
        from .geometry import axis_ratio_from_inclination
        out["EPS_DISC"] = 1 - float(axis_ratio_from_inclination(out["INCL_GEOM"], 0.3))
    return out
