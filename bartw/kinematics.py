"""Stellar kinematic maps from a MUSE cube: Voronoi binning + pPXF.

This is the standard recipe used in Cuomo+2019/2024 for TW on MUSE data:

1. white-light (or a line-free window) image -> S/N per spaxel
2. Voronoi binning (Cappellari & Copin 2003) to a target S/N (~ 40 for
   reliable V and sigma in dwarfs)
3. pPXF (Cappellari 2017, 2023) on each bin with E-MILES SSP templates,
   masking gas emission lines, 2 moments (V, sigma) -- enough for TW.

Requires ``ppxf`` and ``vorbin`` (pip) and an SPS template file in pPXF
``.npz`` format, e.g. ``spectra_emiles_9.0.npz`` from
https://github.com/micappe/ppxf_data (set ``templates`` accordingly).

The output maps (flux, V, V_err, sigma, sigma_err, bin number) are what
:mod:`bartw.tw` consumes. Note that TW must use the *same* tracer for the
photometric and kinematic integrals, so the flux map here is the stellar
continuum flux integrated over the fitted wavelength range (not an
emission-line or broad-band image), and the bar/disc photometry should be
consistent with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits

C_KMS = 299792.458


@dataclass
class Cube:
    data: np.ndarray      # (nlam, ny, nx)
    var: np.ndarray | None
    lam: np.ndarray       # Angstrom, air
    header: fits.Header
    pixscale: float       # arcsec/pix


def read_muse_cube(path):
    """Read a MUSE DATACUBE_FINAL (ESO Phase 3 format: DATA and STAT extensions)."""
    with fits.open(path, memmap=True) as h:
        hdu = h["DATA"] if "DATA" in h else h[1]
        data = np.asarray(hdu.data, dtype=np.float32)
        hdr = hdu.header
        var = np.asarray(h["STAT"].data, dtype=np.float32) if "STAT" in h else None
    n = data.shape[0]
    lam = hdr["CRVAL3"] + (np.arange(n) + 1 - hdr["CRPIX3"]) * hdr.get("CD3_3", hdr.get("CDELT3"))
    pixscale = abs(hdr.get("CD2_2", hdr.get("CDELT2", 0.2 / 3600))) * 3600.0
    return Cube(data, var, lam, hdr, pixscale)


def white_light(cube, lam_range=(4800.0, 5500.0)):
    """Signal and noise images over a wavelength window (per spaxel)."""
    s = (cube.lam >= lam_range[0]) & (cube.lam <= lam_range[1])
    signal = np.nanmedian(cube.data[s], axis=0)
    if cube.var is not None:
        noise = np.sqrt(np.nanmedian(cube.var[s], axis=0))
    else:  # fall back to a robust estimate from the spectral residuals
        noise = 1.4826 * np.nanmedian(np.abs(np.diff(cube.data[s], axis=0)), axis=0) / np.sqrt(2)
    return signal, noise


def voronoi_bins(signal, noise, target_sn=40.0, snr_min=1.0, mask=None):
    """Voronoi-bin spaxels above ``snr_min``. Returns bin_num map (-1 unbinned)."""
    from vorbin.voronoi_2d_binning import voronoi_2d_binning

    ny, nx = signal.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    good = np.isfinite(signal) & np.isfinite(noise) & (noise > 0) & (signal / noise >= snr_min)
    if mask is not None:
        good &= mask
    bin_num, *_ = voronoi_2d_binning(xx[good], yy[good], signal[good], noise[good],
                                     target_sn, plot=False, quiet=True)
    out = np.full(signal.shape, -1, dtype=int)
    out[good] = bin_num
    return out


def fit_bins(cube, bin_num, z, templates, lam_range=(4750.0, 7000.0), fwhm_gal=2.65,
             moments=2, degree=12, mask_lines=True, quiet=True):
    """pPXF fit of each Voronoi bin. Returns dict of per-bin arrays and maps.

    ``templates``: path to a pPXF SPS .npz file. ``fwhm_gal``: MUSE LSF (~2.65 A
    at 5000 A; fine for TW since only V matters).
    """
    from ppxf import ppxf_util as util
    from ppxf.ppxf import ppxf
    from ppxf.sps_util import sps_lib

    s = (cube.lam >= lam_range[0]) & (cube.lam <= lam_range[1])
    lam = cube.lam[s]
    nb = bin_num.max() + 1
    # stack spectra per bin
    flat = cube.data[s].reshape(s.sum(), -1)
    varf = cube.var[s].reshape(s.sum(), -1) if cube.var is not None else None
    bins = bin_num.ravel()
    spectra = np.zeros((s.sum(), nb))
    variances = np.zeros_like(spectra)
    for b in range(nb):
        m = bins == b
        spectra[:, b] = np.nansum(flat[:, m], axis=1)
        variances[:, b] = np.nansum(varf[:, m], axis=1) if varf is not None else np.nan

    # log-rebin one spectrum to get velscale
    spec0, ln_lam, velscale = util.log_rebin([lam[0], lam[-1]], spectra[:, 0])
    sps = sps_lib(templates, velscale, fwhm_gal, norm_range=[5070, 5950])
    lam_gal = np.exp(ln_lam)
    goodpix = util.determine_goodpixels(ln_lam, sps.lam_range_temp, z) if mask_lines else None
    vel0 = C_KMS * np.log(1 + z)
    start = [vel0, 50.0]

    V = np.full(nb, np.nan); dV = V.copy(); S = V.copy(); dS = V.copy(); F = V.copy(); chi2 = V.copy()
    for b in range(nb):
        gal, _, _ = util.log_rebin([lam[0], lam[-1]], spectra[:, b], velscale=velscale)
        if np.isfinite(variances[:, b]).all():
            noise, _, _ = util.log_rebin([lam[0], lam[-1]], variances[:, b], velscale=velscale)
            noise = np.sqrt(np.clip(noise, 1e-12, None))
        else:
            noise = np.full_like(gal, np.nanstd(gal) * 0.1)
        norm = np.nanmedian(gal)
        if not np.isfinite(norm) or norm <= 0:
            continue
        pp = ppxf(sps.templates, gal / norm, noise / norm, velscale, start, moments=moments,
                  degree=degree, lam=lam_gal, lam_temp=sps.lam_temp, goodpixels=goodpix, quiet=quiet)
        V[b], S[b] = pp.sol[0], pp.sol[1]
        err = pp.error * np.sqrt(pp.chi2)
        dV[b], dS[b] = err[0], err[1]
        F[b] = norm
        chi2[b] = pp.chi2

    def to_map(arr):
        out = np.full(bin_num.shape, np.nan)
        ok = bin_num >= 0
        out[ok] = arr[bin_num[ok]]
        return out

    # per-spaxel flux (each spaxel keeps its own continuum flux; bins only share V)
    flux_map = np.nanmedian(cube.data[s], axis=0)
    return dict(v=V, v_err=dV, sigma=S, sigma_err=dS, bin_flux=F, chi2=chi2,
                v_map=to_map(V), v_err_map=to_map(dV), sigma_map=to_map(S), flux_map=flux_map,
                bin_num=bin_num, velscale=velscale)


def save_maps(path, maps, header=None):
    hdus = [fits.PrimaryHDU(header=header)]
    for k in ("flux_map", "v_map", "v_err_map", "sigma_map"):
        hdus.append(fits.ImageHDU(maps[k].astype(np.float32), name=k.replace("_map", "").upper()))
    hdus.append(fits.ImageHDU(maps["bin_num"].astype(np.int32), name="BINNUM"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
