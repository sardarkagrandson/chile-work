import numpy as np
import pytest

from bartw.geometry import (axis_ratio_from_inclination, deproject_bar_length,
                            inclination_from_axis_ratio, kpc_per_arcsec, pa_difference, tw_suitable)


def test_inclination_roundtrip():
    for i in (20.0, 45.0, 70.0):
        q = axis_ratio_from_inclination(i, q0=0.3)
        assert inclination_from_axis_ratio(q, q0=0.3) == pytest.approx(i, abs=1e-6)
    assert inclination_from_axis_ratio(0.25, q0=0.3) == pytest.approx(90.0)
    assert inclination_from_axis_ratio(1.0, q0=0.3) == pytest.approx(0.0)


def test_pa_difference_wraps():
    assert pa_difference(175.0, 5.0) == pytest.approx(10.0)
    assert pa_difference(10.0, 100.0) == pytest.approx(90.0)
    assert pa_difference(30.0, 30.0) == 0.0


def test_deproject_bar_length():
    # bar along the major axis: no correction; along the minor axis: 1/cos i
    assert deproject_bar_length(10.0, 40.0, 40.0, 60.0) == pytest.approx(10.0)
    assert deproject_bar_length(10.0, 130.0, 40.0, 60.0) == pytest.approx(20.0)


def test_scale():
    assert kpc_per_arcsec(distance_mpc=16.5) == pytest.approx(0.080, abs=0.001)
    assert kpc_per_arcsec(z=0.01, H0=70) == pytest.approx(0.205, abs=0.01)


def test_tw_suitable():
    cfg = dict(incl_min_deg=25, incl_max_deg=70, dpa_min_deg=10, dpa_max_deg=80)
    ok, dpa, why = tw_suitable([45, 15, 45, 45], [60, 60, 25, 115], [20, 20, 20, 20], cfg)
    assert list(ok) == [True, False, False, False]
    assert list(why) == ["ok", "inclination", "bar_pa", "bar_pa"]
