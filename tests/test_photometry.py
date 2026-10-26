"""Disc/bar geometry recovery on a synthetic Legacy-Surveys-like image."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from bartw.geometry import pa_difference
from bartw.mock import MockParams, make_mock
from bartw.photometry import measure_geometry


def _image(pa=35.0, incl=50.0, bar_phi=40.0, seed=3):
    # 0.262"/pix like Legacy Surveys; disc scale length 6", bar 9" x 3.5"
    p = MockParams(nx=240, ny=240, pixscale=0.262, pa_deg=pa, incl_deg=incl, bar_phi_deg=bar_phi,
                   h_disc_arcsec=6.0, a_bar_arcsec=9.0, b_bar_arcsec=3.0, bar_contrast=2.5, seed=seed)
    flux, _, _, t = make_mock(p)
    rng = np.random.default_rng(seed)
    img = gaussian_filter(flux, 1.3 / 2.355 / p.pixscale)
    img = 50.0 * img / img.max()
    img += rng.normal(0, 0.3, img.shape) + 2.0            # sky + noise
    # a few stars
    for _ in range(6):
        x, y = rng.integers(10, 230, 2)
        yy, xx = np.mgrid[0:240, 0:240]
        img += 20 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2**2))
    return img, p, t


@pytest.mark.parametrize("pa,incl,bar_phi", [(35.0, 50.0, 40.0), (120.0, 40.0, -55.0), (80.0, 60.0, 30.0)])
def test_recovers_disc_and_bar(pa, incl, bar_phi):
    img, p, t = _image(pa, incl, bar_phi)
    r25 = 3.0 * p.h_disc_arcsec                               # ~ R25 for an exponential dwarf disc
    res = measure_geometry(img, p.pixscale, r25)
    assert res["status"] == "ok", res["status"]
    assert pa_difference(res["pa_disc"], pa) < 6.0
    q_true = np.sqrt(np.cos(np.radians(incl)) ** 2)           # mock disc is infinitely thin
    assert abs((1 - res["eps_disc"]) - q_true) < 0.08
    assert pa_difference(res["pa_bar"], t["bar_pa_sky"]) < 10.0
    assert 0.6 * t["a_bar_sky_arcsec"] < res["a_bar"] < 1.4 * t["a_bar_sky_arcsec"]


def test_no_bar_in_pure_disc():
    img, p, t = _image()
    p2 = MockParams(nx=240, ny=240, pixscale=0.262, h_disc_arcsec=6.0, bar_contrast=0.0)
    flux, _, _, _ = make_mock(p2)
    rng = np.random.default_rng(0)
    img = 50 * gaussian_filter(flux, 2.0) / flux.max() + rng.normal(0, 0.3, flux.shape) + 2.0
    res = measure_geometry(img, p2.pixscale, 18.0)
    assert np.isnan(res["a_bar"])
    assert "bar" in res["status"] or res["status"] == "ok" and np.isnan(res["pa_bar"])


def test_auto_size_mode_recovers_bar():
    """Without a catalogue R25 the size comes from the image itself."""
    img, p, t = _image()
    res = measure_geometry(img, p.pixscale, None)
    assert res["size_mode"].startswith("auto")
    assert res["status"] == "ok", res["status"]
    assert pa_difference(res["pa_disc"], p.pa_deg) < 6.0
    assert pa_difference(res["pa_bar"], t["bar_pa_sky"]) < 10.0
    assert 0.6 * t["a_bar_sky_arcsec"] < res["a_bar"] < 1.4 * t["a_bar_sky_arcsec"]
    assert not res["edge_limited"]
