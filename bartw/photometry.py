"""Disc and bar geometry from Legacy Surveys imaging.

We need, for each candidate: the disc position angle and ellipticity (from the
outer isophotes), and the bar position angle and semi-major axis (from the
ellipticity peak / PA plateau of the isophotes; Aguerri+2009, Cuomo+2019).
Legacy Surveys DR9/DR10 cutouts (0.262"/pix, grz) are fetched from the
viewer cutout service; the r band is used.
"""
from __future__ import annotations

import io

import numpy as np
import requests
from astropy.io import fits

CUTOUT_URL = ("https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}"
              "&layer={layer}&pixscale={pixscale}&size={size}&bands={band}")


def fetch_cutout(ra, dec, layer="ls-dr9", pixscale=0.262, size=512, band="r", timeout=60):
    """Return (image, header) of a Legacy Surveys cutout."""
    url = CUTOUT_URL.format(ra=ra, dec=dec, layer=layer, pixscale=pixscale, size=size, band=band)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with fits.open(io.BytesIO(r.content)) as hdul:
        img = np.squeeze(hdul[0].data).astype(float)
        hdr = hdul[0].header
    return img, hdr


def isophote_profiles(img, x0=None, y0=None, sma0=5.0, maxsma=None, step=0.1):
    """Ellipse-fit the isophotes (photutils) and return a table with sma,
    ellipticity, PA (deg, N through E assuming east-left image), intensity."""
    from photutils.isophote import Ellipse, EllipseGeometry

    ny, nx = img.shape
    x0 = (nx - 1) / 2 if x0 is None else x0
    y0 = (ny - 1) / 2 if y0 is None else y0
    maxsma = maxsma or 0.45 * min(nx, ny)
    geom = EllipseGeometry(x0=x0, y0=y0, sma=sma0, eps=0.2, pa=np.radians(45.0))
    ell = Ellipse(img, geom)
    iso = ell.fit_image(sma0=sma0, minsma=1.0, maxsma=maxsma, step=step, linear=False)
    t = iso.to_table()
    # photutils PA: counter-clockwise from +x; convert to astronomical PA
    # (from +y=north through -x=east) for an east-left image: PA = theta - 90.
    pa_astro = (np.degrees(t["pa"].value if hasattr(t["pa"], "value") else t["pa"]) - 90.0) % 180.0
    return dict(sma=np.asarray(t["sma"]), eps=np.asarray(t["ellipticity"]),
                pa=np.asarray(pa_astro), intens=np.asarray(t["intens"]),
                eps_err=np.asarray(t["ellipticity_err"]), pa_err=np.degrees(np.asarray(t["pa_err"])))


def disc_geometry(prof, sma_min, sma_max):
    """Disc PA and ellipticity: median of the isophotes between sma_min and
    sma_max (chosen where eps and PA are flat and S/N is still good).
    Returns pa, pa_err, eps, eps_err."""
    s = (prof["sma"] >= sma_min) & (prof["sma"] <= sma_max) & np.isfinite(prof["eps"])
    if s.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    pa = prof["pa"][s]
    # circular median for PA
    ang = np.radians(2 * pa)
    pa_med = (np.degrees(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))) / 2) % 180
    pa_err = np.std(((pa - pa_med + 90) % 180) - 90)
    return pa_med, max(pa_err, 1.0), float(np.median(prof["eps"][s])), float(np.std(prof["eps"][s]))


def bar_geometry(prof, pa_disc, eps_disc, sma_min=2.0, eps_drop=0.1, pa_tol=10.0):
    """Bar PA and length from the ellipticity profile.

    a_bar is taken as the mean of two standard estimators (Aguerri+2009):
    (1) sma where eps drops by ``eps_drop`` from its peak, and
    (2) sma where PA departs by more than ``pa_tol`` from the PA at the peak.
    The PA at the peak is the bar PA. Returns dict with pa_bar, a_bar,
    a_bar_err, eps_bar and the two estimates; NaN if no bar signature.
    """
    sma, eps, pa = prof["sma"], prof["eps"], prof["pa"]
    ok = (sma >= sma_min) & np.isfinite(eps)
    if ok.sum() < 5:
        return dict(pa_bar=np.nan, a_bar=np.nan, a_bar_err=np.nan, eps_bar=np.nan)
    # restrict to the inner half of the fitted range for the peak search
    inner = ok & (sma <= 0.5 * sma[ok].max())
    k = np.argmax(np.where(inner, eps, -np.inf))
    eps_max, pa_bar, a_peak = eps[k], pa[k], sma[k]
    if eps_max - eps_disc < 0.1:  # bar must be more elongated than the disc
        return dict(pa_bar=np.nan, a_bar=np.nan, a_bar_err=np.nan, eps_bar=eps_max)
    beyond = np.where(ok & (sma > a_peak))[0]
    a1 = a2 = np.nan
    for j in beyond:
        if np.isnan(a1) and eps[j] <= eps_max - eps_drop:
            a1 = sma[j]
        if np.isnan(a2) and abs(((pa[j] - pa_bar + 90) % 180) - 90) > pa_tol:
            a2 = sma[j]
        if not np.isnan(a1) and not np.isnan(a2):
            break
    ests = np.array([a1, a2])
    ests = ests[np.isfinite(ests)]
    if ests.size == 0:
        return dict(pa_bar=pa_bar, a_bar=np.nan, a_bar_err=np.nan, eps_bar=eps_max)
    return dict(pa_bar=float(pa_bar), a_bar=float(ests.mean()),
                a_bar_err=float(max(abs(np.diff(ests)).max() / 2 if ests.size > 1 else 0.0, a_peak * 0.1)),
                eps_bar=float(eps_max), a_bar_eps=a1, a_bar_pa=a2, a_peak=float(a_peak))
