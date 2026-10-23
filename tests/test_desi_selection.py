"""Selection logic on a tiny synthetic FastSpecFit-like table."""
import numpy as np
import pandas as pd

from bartw.config import load_config
from bartw.desi import bpt_classification, select_star_forming_dwarfs


def _fake(n=6):
    df = pd.DataFrame({
        "TARGETID": np.arange(n), "RA": np.linspace(10, 11, n), "DEC": np.zeros(n),
        "SURVEY": ["main"] * n, "PROGRAM": ["bright"] * n, "SPECTYPE": ["GALAXY"] * n,
        "Z": [0.005, 0.008, 0.010, 0.05, 0.009, 0.007], "ZWARN": [0, 0, 0, 0, 4, 0],
        "DELTACHI2": [100.0] * n,
        # h=1 masses; +0.31 after conversion
        "LOGMSTAR": [8.5, 8.0, 9.6, 8.5, 8.5, 8.5], "SFR": [0.05, 0.02, 0.5, 0.05, 0.05, 1e-5],
        "HALPHA_FLUX": [100.0] * n, "HALPHA_FLUX_IVAR": [1.0] * n,
        "HBETA_FLUX": [30.0] * n, "HBETA_FLUX_IVAR": [1.0] * n,
        "NII_6584_FLUX": [10.0, 10.0, 10.0, 10.0, 10.0, 200.0], "NII_6584_FLUX_IVAR": [1.0] * n,
        "OIII_5007_FLUX": [60.0] * n, "OIII_5007_FLUX_IVAR": [1.0] * n,
        "HALPHA_EW": [30.0] * n, "HALPHA_EW_IVAR": [1.0] * n,
    })
    return df


def test_bpt_labels():
    df = _fake()
    lab, n2, o3 = bpt_classification(df)
    assert list(lab[:5]) == ["SF"] * 5
    assert lab[5] == "AGN"


def test_selection_steps():
    cfg = load_config()
    out, steps = select_star_forming_dwarfs(_fake(), cfg)
    # row0 ok, row1 ok, row2 too massive (9.6+0.31), row3 z too high, row4 bad ZWARN, row5 AGN
    assert sorted(out["TARGETID"]) == [0, 1]
    assert steps["input"] == 6 and steps["unique TARGETID"] == 2
    assert out["LOGMSTAR_H70"].iloc[0] == pytest.approx(8.5 + 2 * np.log10(100 / 70))


import pytest  # noqa: E402
