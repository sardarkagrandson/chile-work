"""Local cross-match of targets against a downloaded MUSE product table."""
import numpy as np
import pandas as pd

from bartw.muse import crossmatch_local, summarize_coverage


def test_crossmatch_local_and_summary():
    prods = pd.DataFrame({
        "s_ra": [10.0, 10.0 + 20 / 3600 / np.cos(np.radians(5)), 50.0],
        "s_dec": [5.0, 5.0, -20.0],
        "t_exptime": [2400.0, 1200.0, 3600.0],
        "obs_release_date": ["2020-01-01T00:00:00Z", "2099-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
        "dp_id": ["A", "B", "C"], "proposal_id": ["P1", "P2", "P3"],
    })
    hits = crossmatch_local([10.0, 30.0], [5.0, 5.0], prods, "cube", radius_arcmin=0.5)
    assert sorted(hits.idx) == [0, 0]            # two cubes near target 0, none near target 1
    assert set(hits.dp_id) == {"A", "B"}
    sample = pd.DataFrame({"RA": [10.0, 30.0], "DEC": [5.0, 5.0]})
    cfg = dict(min_exptime_s=1800)
    cov = summarize_coverage(sample, hits, cfg)
    assert cov.N_MUSE_CUBES.tolist() == [2, 0]
    assert cov.MUSE_EXPTIME_S.iloc[0] == 3600.0
    assert bool(cov.MUSE_PUBLIC.iloc[0]) and bool(cov.MUSE_DEEP_ENOUGH.iloc[0])
    assert cov.MUSE_DP_IDS.iloc[0] in ("A;B", "B;A")
