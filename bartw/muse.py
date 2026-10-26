"""Cross-match the sample with the ESO archive for MUSE observations.

Uses the ESO TAP service (``https://archive.eso.org/tap_obs``) through pyvo.

* ``ivoa.ObsCore``  -- reduced (Phase 3) data products: pipeline-processed
  MUSE cubes (``dataproduct_type='cube'``). Immediately usable.
* ``dbo.raw``       -- raw frames. Lets us find targets observed but not yet
  released as Phase 3 cubes (reduce ourselves with the ESO MUSE pipeline), and
  proprietary data that becomes public within the project timeline.

Column names were taken from the ESO TAP documentation; run
``print_schema()`` once to confirm them against the live ``TAP_SCHEMA``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

OBSCORE_QUERY = """
SELECT target_name, s_ra, s_dec, s_fov, t_exptime, t_min, em_min, em_max,
       obs_release_date, proposal_id, obs_collection, dp_id, dataproduct_type,
       instrument_name, obs_id, access_url
FROM ivoa.ObsCore
WHERE instrument_name = '{instrument}'
  AND dataproduct_type = 'cube'
  AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1
"""

RAW_QUERY = """
SELECT object, ra, dec, exposure, date_obs, prog_id, dp_id, dp_cat, dp_tech,
       ins_mode, release_date
FROM dbo.raw
WHERE instrument = '{instrument}'
  AND dp_cat = 'SCIENCE'
  AND CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1
"""


ALL_CUBES_QUERY = """
SELECT target_name, s_ra, s_dec, s_fov, t_exptime, t_min, em_min, em_max,
       obs_release_date, proposal_id, obs_collection, dp_id, dataproduct_type,
       instrument_name, obs_id, access_url
FROM ivoa.ObsCore
WHERE instrument_name = '{instrument}' AND dataproduct_type = 'cube'
"""

ALL_RAW_QUERY = """
SELECT object, ra, dec, exposure, date_obs, prog_id, dp_id, dp_cat, dp_tech,
       ins_mode, release_date
FROM dbo.raw
WHERE instrument = '{instrument}' AND dp_cat = 'SCIENCE'
"""


def fetch_all_muse(cfg_muse, kind="cube", maxrec=2_000_000, cache_dir=None):
    """Download the *whole* list of MUSE Phase-3 cubes (kind='cube') or raw
    science frames (kind='raw') once, so that any number of targets can be
    cross-matched locally. Tens of thousands of rows for cubes, a few
    hundred thousand for raw frames. Cached as parquet in ``cache_dir``.
    Falls back to an asynchronous TAP job if the synchronous query is
    rejected for size."""
    import pathlib
    if cache_dir is not None:
        cache = pathlib.Path(cache_dir) / f"eso_muse_all_{kind}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            print(f"[muse] {len(df)} {kind} rows from cache {cache.name}")
            return df
    svc = _service(cfg_muse["tap_url"])
    q = (ALL_CUBES_QUERY if kind == "cube" else ALL_RAW_QUERY).format(instrument=cfg_muse["instrument"])
    try:
        res = svc.search(q, maxrec=maxrec)
    except Exception as exc:
        print(f"[muse] sync query failed ({exc}); trying an async job")
        job = svc.submit_job(q, maxrec=maxrec)
        job.run().wait()
        res = job.fetch_result()
    df = res.to_table().to_pandas()
    print(f"[muse] fetched {len(df)} MUSE {kind} rows from the ESO archive")
    if cache_dir is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df


def crossmatch_local(ra, dec, products, kind, radius_arcmin):
    """Match targets to a downloaded product table (from fetch_all_muse) on the
    sky. Returns the same per-hit layout as query_muse (idx, kind, ...)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    if len(products) == 0:
        return pd.DataFrame(columns=["idx", "kind"])
    rc, dc = ("s_ra", "s_dec") if kind == "cube" else ("ra", "dec")
    ok = np.isfinite(products[rc].to_numpy(float)) & np.isfinite(products[dc].to_numpy(float))
    prod = products[ok].reset_index(drop=True)
    c_t = SkyCoord(np.atleast_1d(ra) * u.deg, np.atleast_1d(dec) * u.deg)
    c_p = SkyCoord(prod[rc].to_numpy(float) * u.deg, prod[dc].to_numpy(float) * u.deg)
    i_t, i_p, sep, _ = c_p.search_around_sky(c_t, radius_arcmin * u.arcmin)
    hits = prod.iloc[i_p].reset_index(drop=True)
    hits.insert(0, "idx", i_t)
    hits.insert(1, "kind", kind)
    hits["sep_arcsec"] = sep.arcsec
    return hits


def muse_coverage(ra, dec, cfg_muse, cache_dir=None, raw=None):
    """Preferred entry point: download the MUSE product lists once and match
    all targets locally. Returns the per-hit table (like query_muse)."""
    include_raw = cfg_muse.get("include_raw", True) if raw is None else raw
    r = cfg_muse["search_radius_arcmin"]
    parts = [crossmatch_local(ra, dec, fetch_all_muse(cfg_muse, "cube", cache_dir=cache_dir), "cube", r)]
    if include_raw:
        parts.append(crossmatch_local(ra, dec, fetch_all_muse(cfg_muse, "raw", cache_dir=cache_dir), "raw", r))
    out = pd.concat(parts, ignore_index=True)
    out.attrs["n_failed_queries"] = 0
    return out


def _service(tap_url):
    import pyvo
    return pyvo.dal.TAPService(tap_url)


def print_schema(tap_url, table="ivoa.ObsCore"):
    svc = _service(tap_url)
    res = svc.search(f"SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
                     f"WHERE table_name = '{table}'")
    print(res.to_table())


def query_muse(ra, dec, cfg_muse, raw=None, sleep=0.3):
    """Cone-search the ESO archive around each (ra, dec) [deg].

    Returns a DataFrame with one row per MUSE product/frame and an
    ``idx`` column pointing back to the input row.
    """
    svc = _service(cfg_muse["tap_url"])
    r_deg = cfg_muse["search_radius_arcmin"] / 60.0
    include_raw = cfg_muse.get("include_raw", True) if raw is None else raw
    rows = []
    n_fail = 0
    for i, (r, d) in enumerate(zip(np.atleast_1d(ra), np.atleast_1d(dec))):
        for kind, q in (("cube", OBSCORE_QUERY), ("raw", RAW_QUERY if include_raw else None)):
            if q is None:
                continue
            try:
                res = svc.search(q.format(instrument=cfg_muse["instrument"], ra=r, dec=d, r_deg=r_deg))
                df = res.to_table().to_pandas()
            except Exception as exc:  # network / ADQL errors: record and continue
                n_fail += 1
                if n_fail <= 3:
                    print(f"[muse] query failed for idx={i} ({kind}): {exc}")
                continue
            if len(df):
                df.insert(0, "idx", i)
                df.insert(1, "kind", kind)
                rows.append(df)
            time.sleep(sleep)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["idx", "kind"])
    out.attrs["n_failed_queries"] = n_fail
    if n_fail:
        print(f"[muse] {n_fail} queries failed (network or ADQL); coverage is incomplete")
    return out


def summarize_coverage(sample, hits, cfg_muse):
    """Per-galaxy summary: number of cubes, total exposure, deepest cube,
    public/proprietary status, raw-only availability."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    out = sample.copy()
    out["N_MUSE_CUBES"] = 0
    out["MUSE_EXPTIME_S"] = 0.0
    out["MUSE_PUBLIC"] = False
    out["MUSE_RAW_ONLY"] = False
    out["MUSE_DP_IDS"] = ""
    out["MUSE_PROG_IDS"] = ""
    for i, grp in hits.groupby("idx"):
        cubes = grp[grp["kind"] == "cube"]
        raws = grp[grp["kind"] == "raw"]
        out.loc[i, "N_MUSE_CUBES"] = len(cubes)
        if len(cubes):
            out.loc[i, "MUSE_EXPTIME_S"] = float(cubes["t_exptime"].fillna(0).sum())
            rel = pd.to_datetime(cubes["obs_release_date"], errors="coerce", utc=True).dt.tz_localize(None)
            out.loc[i, "MUSE_PUBLIC"] = bool((rel <= now).any())
            out.loc[i, "MUSE_DP_IDS"] = ";".join(cubes["dp_id"].astype(str))
            out.loc[i, "MUSE_PROG_IDS"] = ";".join(sorted(set(cubes["proposal_id"].astype(str))))
        elif len(raws):
            out.loc[i, "MUSE_RAW_ONLY"] = True
            out.loc[i, "MUSE_EXPTIME_S"] = float(raws["exposure"].fillna(0).sum())
            out.loc[i, "MUSE_PROG_IDS"] = ";".join(sorted(set(raws["prog_id"].astype(str))))
    out["MUSE_DEEP_ENOUGH"] = out["MUSE_EXPTIME_S"] >= cfg_muse["min_exptime_s"]
    return out
