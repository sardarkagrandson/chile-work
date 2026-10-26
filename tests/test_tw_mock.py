"""The TW implementation must recover the input pattern speed of a synthetic
rigidly rotating bar (see bartw.mock for why this is an exact test)."""
import numpy as np
import pytest

from bartw.geometry import pa_difference
from bartw.mock import MockParams, make_mock
from bartw.tw import default_slit_centers, measure_pattern_speed, rotation_rate, fit_slope


def _run(p, n_mc=0, pa_err=0.0):
    flux, vel, verr, t = make_mock(p)
    dpa = pa_difference(t["bar_pa_sky"], p.pa_deg)
    yc = default_slit_centers(t["a_bar_sky_arcsec"], dpa, 9, 1.0)
    return measure_pattern_speed(
        flux, vel, p.pixscale, t["x0"], t["y0"], p.pa_deg, p.incl_deg, yc, 1.0,
        x_half_length_arcsec=1.5 * t["a_bar_sky_arcsec"], vel_err=verr,
        distance_mpc=p.distance_mpc, pa_err_deg=pa_err, n_mc=n_mc, rng=0,
        east_left=p.east_left), t


@pytest.mark.parametrize("kw", [
    dict(),
    dict(pa_deg=120.0, incl_deg=40.0, bar_phi_deg=-55.0),
    dict(pa_deg=170.0, incl_deg=65.0, bar_phi_deg=30.0, omega_p=12.0),
    dict(east_left=False),
    dict(psf_fwhm_arcsec=0.8),
])
def test_recovers_omega_p_noiseless(kw):
    p = MockParams(**kw)
    res, t = _run(p)
    assert res.omega_p == pytest.approx(p.omega_p, rel=0.05)
    assert res.omega_sini == pytest.approx(t["omega_sini_arcsec"], rel=0.05)
    assert res.v_sys == pytest.approx(p.v_sys, abs=1.0)


def test_recovers_omega_p_with_noise():
    p = MockParams(psf_fwhm_arcsec=0.8, flux_noise_frac=0.05, vel_noise_kms=5.0)
    res, _ = _run(p, n_mc=50, pa_err=3.0)
    assert abs(res.omega_p - p.omega_p) < 2.5 * res.omega_p_err
    assert res.omega_p_err_pa > res.omega_p_err_fit  # PA dominates, as in the literature


def test_wrong_pa_biases_result():
    """A 5 deg PA error must move the answer noticeably: this is the known
    dominant systematic and the reason for the PA Monte Carlo."""
    p = MockParams()
    flux, vel, verr, t = make_mock(p)
    dpa = pa_difference(t["bar_pa_sky"], p.pa_deg)
    yc = default_slit_centers(t["a_bar_sky_arcsec"], dpa, 9, 1.0)
    good = measure_pattern_speed(flux, vel, p.pixscale, t["x0"], t["y0"], p.pa_deg, p.incl_deg, yc, 1.0,
                                 x_half_length_arcsec=1.5 * t["a_bar_sky_arcsec"], distance_mpc=p.distance_mpc)
    bad = measure_pattern_speed(flux, vel, p.pixscale, t["x0"], t["y0"], p.pa_deg + 5, p.incl_deg, yc, 1.0,
                                x_half_length_arcsec=1.5 * t["a_bar_sky_arcsec"], distance_mpc=p.distance_mpc)
    assert abs(bad.omega_p - p.omega_p) > 3 * abs(good.omega_p - p.omega_p)


def test_fit_slope_recovers_line():
    rng = np.random.default_rng(1)
    x = np.linspace(-2, 2, 9)
    v = 1.7 * x + 1000 + rng.normal(0, 0.1, x.size)
    f = fit_slope(x, v, np.full(x.size, 0.05), np.full(x.size, 0.1))
    assert f["slope"] == pytest.approx(1.7, abs=0.1)
    assert f["v_sys"] == pytest.approx(1000, abs=0.3)


def test_rotation_rate_regimes():
    r = rotation_rate(omega_p=30.0, omega_p_err=3.0, vc=80.0, vc_err=5.0, a_bar_kpc=1.0, a_bar_err_kpc=0.1, rng=0)
    assert r["R"] == pytest.approx(80 / 30, rel=0.05)
    assert r["regime"] == "slow"
    r = rotation_rate(omega_p=30.0, omega_p_err=1.0, vc=80.0, vc_err=2.0, a_bar_kpc=2.3, a_bar_err_kpc=0.05, rng=0)
    assert r["regime"] == "fast"
    assert abs(r["p_fast"] + r["p_slow"] + r["p_ultrafast"] - 1) < 1e-9


def test_prograde_flag_on_mock():
    p = MockParams()
    res, t = _run(p)
    assert res.rotation_sign != 0
    assert res.prograde is True
