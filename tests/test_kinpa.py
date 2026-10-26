import numpy as np
import pytest

from bartw.kinpa import fit_kinematic_pa
from bartw.mock import MockParams, make_mock


@pytest.mark.parametrize("pa,incl,east_left", [(35.0, 50.0, True), (120.0, 40.0, True), (170.0, 60.0, False)])
def test_kinematic_pa_recovers_disc_pa(pa, incl, east_left):
    p = MockParams(pa_deg=pa, incl_deg=incl, east_left=east_left, vel_noise_kms=5.0, bar_contrast=0.0, nx=200, ny=200)
    flux, vel, verr, t = make_mock(p)
    r = fit_kinematic_pa(vel, t["x0"], t["y0"], east_left=east_left, radius_pix=80, v_err=verr)
    d = abs(((r["pa_kin"] - pa + 90) % 180) - 90)
    assert d < 2.0, (r["pa_kin"], pa)
    assert r["pa_kin_err"] < 3.0
    assert abs(r["v_sys"] - p.v_sys) < 2.0
    # mock: V_los = +v_y sin i, receding side at positive X i.e. towards the PA direction
    assert abs(((r["receding_pa"] - pa + 180) % 360) - 180) < 5.0


def test_kinematic_pa_with_bar_still_close():
    p = MockParams(pa_deg=35.0, incl_deg=50.0, bar_phi_deg=40.0, vel_noise_kms=5.0)
    flux, vel, verr, t = make_mock(p)
    r = fit_kinematic_pa(vel, t["x0"], t["y0"], radius_pix=100, v_err=verr)
    assert abs(((r["pa_kin"] - 35.0 + 90) % 180) - 90) < 5.0


def test_annulus_fit_and_no_detection():
    p = MockParams(pa_deg=35.0, incl_deg=50.0, bar_phi_deg=40.0, vel_noise_kms=5.0)
    flux, vel, verr, t = make_mock(p)
    a = t["a_bar_sky_arcsec"] / p.pixscale
    r = fit_kinematic_pa(vel, t["x0"], t["y0"], radius_pix=3 * a, rmin_pix=1.2 * a, v_err=verr)
    assert abs(((r["pa_kin"] - 35.0 + 90) % 180) - 90) < 3.0 and r["detected"]
    # no rotation: pure noise field -> not detected, PA error inflated
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 10, vel.shape)
    r0 = fit_kinematic_pa(noise + 500, t["x0"], t["y0"], radius_pix=3 * a, v_err=np.full(vel.shape, 10.0))
    assert not r0["detected"] and r0["pa_kin_err"] >= 45
