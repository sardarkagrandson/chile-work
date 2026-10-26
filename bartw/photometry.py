"""Disc and bar geometry from Legacy Surveys imaging.

We need, for each candidate: the disc position angle and ellipticity (from the
outer isophotes), and the bar position angle and semi-major axis (from the
ellipticity peak / PA plateau of the isophotes; Aguerri+2009, Cuomo+2019).
Legacy Surveys DR9/DR10 cutouts (0.262"/pix, grz) are fetched from the
viewer cutout service; the r band is used.

Dwarfs are small (R25 ~ 10-30") and clumpy, so the recipe is tuned for them:

* the cutout and the fitting range are scaled to the galaxy's R25 (from the
  SGA or the Tractor model in the paper table), not a fixed 512 pixels;
* the ellipse centre is fixed at the catalogue centre (free centres drift
  onto HII regions);
* isophotes with a failed fit (photutils stop_code >= 3) or below the sky
  noise are discarded instead of being carried as frozen values;
* the disc is measured between 0.5 R25 and R25, the bar is searched inside
  0.6 R25 and must be resolved (a_peak > 2 PSF FWHM) and must keep a stable
  PA over at least three consecutive isophotes.
"""
from __future__ import annotations

import io

import numpy as np
import requests
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

CUTOUT_URL = ("https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}"
              "&layer={layer}&pixscale={pixscale}&size={size}&bands={band}")

NAN_RESULT = dict(pa_disc=np.nan, pa_disc_err=np.nan, eps_disc=np.nan, eps_disc_err=np.nan,
                  pa_bar=np.nan, a_bar=np.nan, a_bar_err=np.nan, eps_bar=np.nan, a_peak=np.nan,
                  a_bar_eps=np.nan, a_bar_pa=np.nan, n_disc_iso=0, status="")


def fetch_cutout(ra, dec, layer="ls-dr9", pixscale=0.262, size=512, band="r", timeout=60):
    """Return (image, header) of a Legacy Surveys cutout. ``size`` in pixels (<= 3000)."""
    size = int(min(max(size, 64), 3000))
    url = CUTOUT_URL.format(ra=ra, dec=dec, layer=layer, pixscale=pixscale, size=size, band=band)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with fits.open(io.BytesIO(r.content)) as hdul:
        img = np.squeeze(hdul[0].data).astype(float)
        hdr = hdul[0].header
    return img, hdr


def cutout_size_for(r25_arcsec, pixscale=0.262, factor=4.0, min_pix=160, max_pix=1200):
    """Cutout side (pixels) covering ``factor`` x R25 on a side (>= 2 R25 of sky
    beyond the galaxy for the noise estimate)."""
    if not np.isfinite(r25_arcsec) or r25_arcsec <= 0:
        return 512
    return int(np.clip(factor * r25_arcsec / pixscale, min_pix, max_pix))


def sky_noise(img, border_frac=0.15):
    """Sigma-clipped sky level and per-pixel noise from the image border."""
    ny, nx = img.shape
    b = int(border_frac * min(nx, ny))
    edge = np.concatenate([img[:b].ravel(), img[-b:].ravel(), img[:, :b].ravel(), img[:, -b:].ravel()])
    edge = edge[np.isfinite(edge)]
    mean, med, std = sigma_clipped_stats(edge, sigma=3.0)
    return med, std


def isophote_profiles(img, x0=None, y0=None, sma0=None, maxsma=None, step=0.1, fix_center=True,
                      min_snr=1.0, eps0=0.2, pa0_deg=45.0):
    """Ellipse-fit the isophotes (photutils) with the centre fixed and return
    a dict of arrays: sma [pix], eps, pa [deg, N through E for an east-left
    image], intens, errors, stop_code. Failed isophotes (stop_code >= 3) and
    isophotes below ``min_snr`` x sky noise are removed. Empty arrays are
    returned when nothing usable was fitted."""
    from photutils.isophote import Ellipse, EllipseGeometry

    ny, nx = img.shape
    x0 = (nx - 1) / 2 if x0 is None else x0
    y0 = (ny - 1) / 2 if y0 is None else y0
    maxsma = maxsma or 0.45 * min(nx, ny)
    sma0 = sma0 or max(3.0, 0.1 * maxsma)
    sky, noise = sky_noise(img)
    data = img - sky
    # photutils PA is measured from +x counter-clockwise; astronomical PA (N through E,
    # east-left image) = theta - 90 deg.
    empty = dict(sma=np.array([]), eps=np.array([]), pa=np.array([]), intens=np.array([]),
                 eps_err=np.array([]), pa_err=np.array([]), stop_code=np.array([]), sky=sky, noise=noise)
    # sigma-clip deviant pixels along each ellipse (foreground stars, HII knots);
    # without this a single star on the first isophote aborts the whole fit.
    # If the first isophote still fails, retry from other starting radii.
    iso = None
    for s0 in [sma0, 0.5 * sma0, 2.0 * sma0, 4.0]:
        geom = EllipseGeometry(x0=x0, y0=y0, sma=s0, eps=eps0, pa=np.radians((pa0_deg + 90.0) % 180.0),
                               fix_center=fix_center)
        ell = Ellipse(data, geometry=geom)
        try:
            iso = ell.fit_image(sma0=s0, minsma=1.0, maxsma=maxsma, step=step, linear=False,
                                fflag=0.5, maxgerr=0.5, sclip=2.5, nclip=3)
        except Exception:
            iso = None
        if iso is not None and len(iso) > 0:
            break
    if iso is None or len(iso) == 0:
        return empty
    t = iso.to_table()
    pa_val = t["pa"].value if hasattr(t["pa"], "value") else np.asarray(t["pa"])
    pa_err = t["pa_err"].value if hasattr(t["pa_err"], "value") else np.asarray(t["pa_err"])
    stop = np.asarray(t["stop_code"]) if "stop_code" in t.colnames else np.zeros(len(t))
    intens = np.asarray(t["intens"], float)
    good = (stop <= 2) & np.isfinite(np.asarray(t["ellipticity"], float)) & (intens > min_snr * noise)
    # pa may come as Quantity in degrees (newer photutils) or radians (older)
    pa_deg = np.asarray(pa_val, float)
    if np.nanmax(np.abs(pa_deg)) <= 2 * np.pi + 1e-6:
        pa_deg = np.degrees(pa_deg)
        pa_err = np.degrees(np.asarray(pa_err, float))
    pa_astro = (pa_deg - 90.0) % 180.0
    return dict(sma=np.asarray(t["sma"], float)[good], eps=np.asarray(t["ellipticity"], float)[good],
                pa=pa_astro[good], intens=intens[good],
                eps_err=np.asarray(t["ellipticity_err"], float)[good], pa_err=np.asarray(pa_err, float)[good],
                stop_code=stop[good], sky=sky, noise=noise)


def _circ_median_pa(pa):
    ang = np.radians(2 * np.asarray(pa))
    return (np.degrees(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))) / 2) % 180


def disc_geometry(prof, sma_min, sma_max):
    """Disc PA and ellipticity: medians over the isophotes with sma_min <= sma <= sma_max.
    Returns pa, pa_err, eps, eps_err, n_isophotes."""
    s = (prof["sma"] >= sma_min) & (prof["sma"] <= sma_max) & np.isfinite(prof["eps"])
    if s.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, int(s.sum())
    pa = prof["pa"][s]
    pa_med = _circ_median_pa(pa)
    pa_err = np.std(((pa - pa_med + 90) % 180) - 90)
    return pa_med, max(pa_err, 1.0), float(np.median(prof["eps"][s])), float(np.std(prof["eps"][s])), int(s.sum())


def bar_geometry(prof, pa_disc, eps_disc, sma_min_pix=2.0, sma_max_pix=None, eps_drop=0.1,
                 pa_tol=10.0, min_eps_excess=0.1, min_a_peak_pix=6.0, n_stable=3):
    """Bar PA and length from the ellipticity profile (Aguerri+2009 estimators).

    The bar is the ellipticity peak inside ``sma_max_pix`` that (i) exceeds the
    disc ellipticity by ``min_eps_excess``, (ii) is resolved (a_peak >=
    ``min_a_peak_pix``) and (iii) has a PA stable within ``pa_tol`` over at
    least ``n_stable`` consecutive isophotes around the peak. a_bar is the mean
    of the ellipticity-drop and PA-change radii beyond the peak; half their
    difference (at least 10% of a_peak) is the error.
    """
    out = dict(pa_bar=np.nan, a_bar=np.nan, a_bar_err=np.nan, eps_bar=np.nan, a_peak=np.nan,
               a_bar_eps=np.nan, a_bar_pa=np.nan, status="")
    sma, eps, pa = prof["sma"], prof["eps"], prof["pa"]
    if sma.size < 5:
        out["status"] = "too few isophotes"
        return out
    sma_max_pix = sma_max_pix or 0.5 * sma.max()
    inner = (sma >= sma_min_pix) & (sma <= sma_max_pix) & np.isfinite(eps)
    if inner.sum() < 3:
        out["status"] = "no inner isophotes"
        return out
    k = int(np.argmax(np.where(inner, eps, -np.inf)))
    eps_max, pa_bar, a_peak = eps[k], pa[k], sma[k]
    out.update(eps_bar=float(eps_max), a_peak=float(a_peak))
    if not np.isfinite(eps_disc) or eps_max - eps_disc < min_eps_excess:
        out["status"] = "bar not more elongated than disc"
        return out
    if a_peak < min_a_peak_pix:
        out["status"] = "ellipticity peak unresolved"
        return out
    dpa = np.abs(((pa - pa_bar + 90) % 180) - 90)
    stable = dpa <= pa_tol
    # require n_stable consecutive stable isophotes containing the peak
    lo = k
    while lo - 1 >= 0 and stable[lo - 1]:
        lo -= 1
    hi = k
    while hi + 1 < sma.size and stable[hi + 1]:
        hi += 1
    if hi - lo + 1 < n_stable:
        out["status"] = "bar PA not stable"
        return out
    a1 = a2 = np.nan
    for j in range(k + 1, sma.size):
        if np.isnan(a1) and eps[j] <= eps_max - eps_drop:
            a1 = sma[j]
        if np.isnan(a2) and dpa[j] > pa_tol:
            a2 = sma[j]
        if not np.isnan(a1) and not np.isnan(a2):
            break
    ests = np.array([a1, a2])
    ests = ests[np.isfinite(ests)]
    if ests.size == 0:
        out.update(pa_bar=float(pa_bar), status="bar end not reached")
        return out
    err = 0.5 * float(np.ptp(ests)) if ests.size > 1 else 0.0
    out.update(pa_bar=float(pa_bar), a_bar=float(ests.mean()), a_bar_err=float(max(err, 0.1 * a_peak)),
               a_bar_eps=a1, a_bar_pa=a2, status="ok")
    return out


def measure_geometry(img, pixscale, r25_arcsec, psf_fwhm_arcsec=1.3, x0=None, y0=None):
    """Full disc + bar geometry from an image (sky-subtracted internally).

    ``r25_arcsec`` sets the fitting range (to 1.5 R25), the disc window
    (0.5-1.0 R25) and the bar search window (< 0.6 R25). Returns a dict with
    the profile (``prof``) and the results in arcsec / degrees."""
    res = dict(NAN_RESULT)
    if not np.isfinite(r25_arcsec) or r25_arcsec <= 0:
        r25_arcsec = 0.15 * min(img.shape) * pixscale
    r25 = r25_arcsec / pixscale
    prof = isophote_profiles(img, x0, y0, sma0=max(3.0, 0.15 * r25), maxsma=1.5 * r25)
    res["prof"] = prof
    res["r25_pix"] = r25
    if prof["sma"].size < 5:
        res["status"] = "isophote fit failed"
        return res
    pa_d, pa_d_err, eps_d, eps_d_err, n_d = disc_geometry(prof, 0.5 * r25, 1.0 * r25)
    res.update(pa_disc=pa_d, pa_disc_err=pa_d_err, eps_disc=eps_d, eps_disc_err=eps_d_err, n_disc_iso=n_d)
    if n_d < 3:
        res["status"] = "disc not measured (too few isophotes in 0.5-1 R25)"
        return res
    bar = bar_geometry(prof, pa_d, eps_d, sma_min_pix=1.5 * psf_fwhm_arcsec / pixscale,
                       sma_max_pix=0.6 * r25, min_a_peak_pix=2.0 * psf_fwhm_arcsec / pixscale)
    for k in ("a_bar", "a_bar_err", "a_peak", "a_bar_eps", "a_bar_pa"):
        bar[k] = bar[k] * pixscale if np.isfinite(bar[k]) else np.nan
    res.update(bar)
    return res


def plot_geometry(img, res, pixscale, title="", axes=None):
    """Image + ellipticity and PA profiles with the disc and bar measurements."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse as MplEllipse
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    prof = res["prof"]
    ny, nx = img.shape
    sky = prof.get("sky", 0.0)
    noise = prof.get("noise", 1.0)
    axes[0].imshow(np.arcsinh((img - sky) / max(noise, 1e-9)), origin="lower", cmap="gray_r", vmin=-1, vmax=8)
    axes[0].set_title(title)
    c = ((nx - 1) / 2, (ny - 1) / 2)
    if np.isfinite(res["eps_disc"]):
        a = res["r25_pix"]
        axes[0].add_patch(MplEllipse(c, 2 * a, 2 * a * (1 - res["eps_disc"]), angle=90 + res["pa_disc"],
                                     fill=False, ec="C3", ls="--", lw=1))
    if np.isfinite(res["a_bar"]):
        a = res["a_bar"] / pixscale
        axes[0].add_patch(MplEllipse(c, 2 * a, 2 * a * (1 - res["eps_bar"]), angle=90 + res["pa_bar"],
                                     fill=False, ec="C2", lw=1))
    s = prof["sma"] * pixscale
    axes[1].errorbar(s, prof["eps"], prof["eps_err"], fmt=".-", ms=3, lw=0.8)
    axes[1].axhline(res["eps_disc"], color="C3", ls="--", label="disc")
    if np.isfinite(res["a_bar"]):
        axes[1].axvline(res["a_bar"], color="C2", label="a_bar")
    axes[1].set_xlabel("sma [arcsec]"); axes[1].set_ylabel("ellipticity"); axes[1].legend(fontsize=8)
    axes[2].errorbar(s, prof["pa"], np.clip(prof["pa_err"], 0, 90), fmt=".-", ms=3, lw=0.8)
    axes[2].axhline(res["pa_disc"], color="C3", ls="--")
    if np.isfinite(res["pa_bar"]):
        axes[2].axhline(res["pa_bar"], color="C2")
    axes[2].set_xlabel("sma [arcsec]"); axes[2].set_ylabel("PA [deg]"); axes[2].set_ylim(0, 180)
    for ax in axes[1:]:
        ax.axvspan(0.5 * res["r25_pix"] * pixscale, res["r25_pix"] * pixscale, color="0.92", zorder=0)
    return axes
