"""Stellar kinematic maps from a MUSE cube: Voronoi binning + pPXF.

Standard recipe (Cuomo+2019, 2024 for TW on MUSE data):

1. white-light image in a line-free window -> S/N per spaxel
2. Voronoi binning (Cappellari & Copin 2003) to a target S/N (~40 for
   reliable V and sigma in dwarfs)
3. pPXF (Cappellari 2017, 2023) on each bin with E-MILES SSP templates,
   masking gas emission lines, 2 moments (V, sigma) -- enough for TW.

Requires ``ppxf`` and ``vorbin`` (pip) and an SPS template file in pPXF
``.npz`` format, e.g. ``spectra_emiles_9.0.npz`` from
https://github.com/micappe/ppxf_data.

The output maps (flux, V, V_err, sigma, sigma_err, bin number) are what
:mod:`bartw.tw` consumes. TW must use the *same* tracer for the photometric
and kinematic integrals, so the flux map is the stellar continuum integrated
over the fitted wavelength range (per spaxel), not an emission-line image.

MUSE specifics handled here: DATA/STAT extensions, NaNs outside the field
and in the AO sodium notch (5780-5970 A), the ~2.6 A LSF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

C_KMS = 299792.458

# Observed-frame wavelengths (A) of the sky emission lines that leave residuals
# in MUSE cubes ([OI] 5577, 6300, 6364; NaD; the OH bands). Residuals differ
# between exposures of a mosaic and bias velocities if left in the fit.
SKY_LINES = [5577.34, 5889.95, 5895.92, 6300.30, 6363.78,
             6498.7, 6533.0, 6544.0, 6553.6, 6562.2, 6568.8, 6577.3, 6596.6, 6604.1, 6627.6, 6634.0, 6645.0,
             6664.0, 6683.0, 6826.0, 6829.6, 6841.0, 6863.9, 6871.0, 6889.3, 6900.8, 6912.6, 6923.2, 6939.5, 6948.9,
             6978.0, 7003.9]


def sky_mask(lam, half_width=3.0, lines=SKY_LINES):
    """Boolean array, True where ``lam`` is within half_width of a sky line."""
    lam = np.asarray(lam, float)
    m = np.zeros(lam.shape, bool)
    for l0 in lines:
        m |= np.abs(lam - l0) <= half_width
    return m


@dataclass
class Cube:
    data: np.ndarray      # (nlam, ny, nx)
    var: "np.ndarray | None"
    lam: np.ndarray       # Angstrom (air, as delivered by the MUSE pipeline)
    header: fits.Header
    pixscale: float       # arcsec/pix
    wcs: "WCS | None" = None


def read_muse_cube(path, lam_range=None, center=None, half_size_arcsec=None):
    """Read a MUSE DATACUBE_FINAL (ESO Phase 3: DATA and STAT extensions).

    ``lam_range`` (Angstrom) restricts the wavelength axis at read time;
    ``center=(ra, dec)`` with ``half_size_arcsec`` reads only a square
    spatial region around that position (essential for mosaics: a 14 GB
    AMUSING mosaic becomes ~1 GB). The returned header/WCS are updated so
    that pixel coordinates refer to the cutout."""
    with fits.open(path, memmap=True) as h:
        hdu = h["DATA"] if "DATA" in h else h[1]
        hdr = hdu.header.copy()
        n = hdr["NAXIS3"]
        step = hdr.get("CD3_3", hdr.get("CDELT3"))
        lam = hdr["CRVAL3"] + (np.arange(n) + 1 - hdr["CRPIX3"]) * step
        s = slice(None)
        if lam_range is not None:
            idx = np.where((lam >= lam_range[0]) & (lam <= lam_range[1]))[0]
            s = slice(idx[0], idx[-1] + 1)
        pixscale = abs(hdr.get("CD2_2", hdr.get("CDELT2", 0.2 / 3600))) * 3600.0
        try:
            wcs = WCS(hdr).celestial
        except Exception:
            wcs = None
        ys, xs = slice(None), slice(None)
        if center is not None and half_size_arcsec is not None:
            if wcs is None:
                raise ValueError("cube has no celestial WCS; cannot cut out around a sky position")
            x0, y0 = (float(v) for v in wcs.all_world2pix(center[0], center[1], 0))
            hp = int(round(half_size_arcsec / pixscale))
            ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
            xa, xb = max(int(round(x0)) - hp, 0), min(int(round(x0)) + hp + 1, nx)
            ya, yb = max(int(round(y0)) - hp, 0), min(int(round(y0)) + hp + 1, ny)
            if xb <= xa or yb <= ya:
                raise ValueError("requested centre lies outside the cube")
            ys, xs = slice(ya, yb), slice(xa, xb)
            hdr["CRPIX1"] = hdr["CRPIX1"] - xa
            hdr["CRPIX2"] = hdr["CRPIX2"] - ya
            hdr["NAXIS1"], hdr["NAXIS2"] = xb - xa, yb - ya
            wcs = WCS(hdr).celestial
        data = np.asarray(hdu.data[s, ys, xs], dtype=np.float32)
        var = np.asarray(h["STAT"].data[s, ys, xs], dtype=np.float32) if "STAT" in h else None
        lam = lam[s]
    return Cube(data, var, lam, hdr, pixscale, wcs)


def sky_to_pixel(cube, ra, dec):
    """Pixel (x, y) of a sky position, from the cube WCS."""
    if cube.wcs is None:
        raise ValueError("cube has no celestial WCS")
    x, y = cube.wcs.all_world2pix(ra, dec, 0)
    return float(x), float(y)


def white_light(cube, lam_range=(4800.0, 5500.0)):
    """Signal and noise images (per spaxel) over a line-free window."""
    s = (cube.lam >= lam_range[0]) & (cube.lam <= lam_range[1])
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        signal = np.nanmedian(cube.data[s], axis=0)
        if cube.var is not None:
            noise = np.sqrt(np.nanmedian(cube.var[s], axis=0))
        else:
            noise = 1.4826 * np.nanmedian(np.abs(np.diff(cube.data[s], axis=0)), axis=0) / np.sqrt(2)
    return signal, noise


def voronoi_bins(signal, noise, target_sn=40.0, snr_min=1.0, mask=None):
    """Voronoi-bin spaxels above ``snr_min``. Returns bin_num map (-1 unbinned)."""
    from vorbin.voronoi_2d_binning import voronoi_2d_binning

    ny, nx = signal.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    good = np.isfinite(signal) & np.isfinite(noise) & (noise > 0) & (signal / noise >= snr_min) & (signal > 0)
    if mask is not None:
        good &= mask
    if good.sum() < 10:
        raise ValueError("fewer than 10 spaxels above the S/N threshold")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_num, *_ = voronoi_2d_binning(xx[good], yy[good], signal[good], noise[good],
                                         target_sn, pixelsize=1.0, plot=False, quiet=True)
    out = np.full(signal.shape, -1, dtype=int)
    out[good] = bin_num
    return out


def stack_bins(cube, bin_num, lam_range):
    """Sum spectra (and variances) per Voronoi bin. Returns lam, spectra (nlam, nbin), var."""
    s = (cube.lam >= lam_range[0]) & (cube.lam <= lam_range[1])
    lam = cube.lam[s]
    nb = int(bin_num.max()) + 1
    bins = bin_num.ravel()
    flat = cube.data[s].reshape(s.sum(), -1)
    varf = cube.var[s].reshape(s.sum(), -1) if cube.var is not None else None
    spectra = np.zeros((s.sum(), nb), dtype=np.float64)
    variances = np.zeros_like(spectra)
    order = np.argsort(bins)
    sb = bins[order]
    edges = np.searchsorted(sb, np.arange(nb + 1))
    with np.errstate(all="ignore"):
        for b in range(nb):
            cols = order[edges[b]:edges[b + 1]]
            spectra[:, b] = np.nansum(flat[:, cols], axis=1)
            variances[:, b] = np.nansum(varf[:, cols], axis=1) if varf is not None else np.nan
            # wavelengths where every spaxel was NaN (AO notch, gaps) -> NaN, not 0
            allnan = np.all(~np.isfinite(flat[:, cols]), axis=1)
            spectra[allnan, b] = np.nan
    return lam, spectra, variances


def load_templates(templates, velscale, fwhm_gal=2.65, norm_range=(5070.0, 5950.0)):
    """SPS templates for pPXF: a path to a pPXF .npz library (via sps_lib) or a
    pre-built (templates_2d, lam_temp) tuple (used by the tests)."""
    if isinstance(templates, (tuple, list)):
        from ppxf import ppxf_util as util
        temp, lam_lin = np.asarray(templates[0], float), np.asarray(templates[1], float)
        temp = temp.reshape(temp.shape[0], -1)
        # pPXF needs the templates log-rebinned on the galaxy's velocity scale (sps_lib does this itself)
        cols = []
        for k in range(temp.shape[1]):
            t, ln_lam_t, _ = util.log_rebin([lam_lin[0], lam_lin[-1]], temp[:, k], velscale=velscale)
            cols.append(t / np.median(t))
        return np.column_stack(cols), np.exp(ln_lam_t)
    from ppxf.sps_util import sps_lib
    sps = sps_lib(str(templates), velscale, fwhm_gal=fwhm_gal, norm_range=list(norm_range))
    return sps.templates.reshape(sps.templates.shape[0], -1), sps.lam_temp


def fit_bins(cube, bin_num, z, templates, lam_range=(4750.0, 5950.0), fwhm_gal=2.65,
             moments=2, degree=12, mask_lines=True, mask_sky=True, quiet=True, progress=True,
             checkpoint=None, checkpoint_every=300):
    """pPXF fit of each Voronoi bin. Returns dict of per-bin arrays and maps.

    ``templates``: pPXF SPS .npz path, or (templates_2d, lam_temp).
    ``fwhm_gal``: instrumental LSF FWHM in Angstrom (MUSE ~2.65 A at 5000 A;
    only V matters for TW so this is not critical).
    ``lam_range``: default 4750-5950 A (Hbeta, Mgb, Fe lines; the only sky
    residuals are [OI]5577 and NaD, both masked). The red part of MUSE is
    full of OH residuals that differ between mosaic exposures and bias V.
    ``mask_sky``: drop +-3 A around the sky lines in ``SKY_LINES``.
    ``checkpoint``: (path, header) -> the maps are written there every
    ``checkpoint_every`` bins and at the end, so a crash loses little.
    """
    from ppxf import ppxf_util as util
    from ppxf.ppxf import ppxf

    lam, spectra, variances = stack_bins(cube, bin_num, lam_range)
    nb = spectra.shape[1]
    lam_range_gal = [lam[0], lam[-1]]
    finite = np.isfinite(spectra)
    # log-rebin ALL bins at once on one shared grid (avoids one-pixel length
    # mismatches between bins); masked wavelengths are carried as a weight map
    gal_all, ln_lam, velscale = util.log_rebin(lam_range_gal, np.where(finite, spectra, 0.0))
    bad_all, _, _ = util.log_rebin(lam_range_gal, (~finite).astype(float))
    var_fill = np.where(np.isfinite(variances) & (variances > 0), variances, np.nan)
    med_var = np.nanmedian(var_fill, axis=0)
    var_fill = np.where(np.isfinite(var_fill), var_fill, med_var[None, :])
    have_var = np.isfinite(med_var) & (med_var > 0)
    var_all, _, _ = util.log_rebin(lam_range_gal, np.where(np.isfinite(var_fill), var_fill, 0.0))
    lam_gal = np.exp(ln_lam)
    temp, lam_temp = load_templates(templates, velscale, fwhm_gal)
    lam_range_temp = [lam_temp[0], lam_temp[-1]]
    good_lines = util.determine_goodpixels(ln_lam, lam_range_temp, z) if mask_lines else np.arange(ln_lam.size)
    if mask_sky:
        good_lines = np.intersect1d(good_lines, np.where(~sky_mask(lam_gal))[0])
    vel0 = C_KMS * np.log(1 + z)
    start = [vel0, 60.0]

    V = np.full(nb, np.nan); dV = V.copy(); S = V.copy(); dS = V.copy(); F = V.copy(); chi2 = V.copy()
    dV0 = V.copy()

    def to_map(arr):
        out = np.full(bin_num.shape, np.nan)
        ok = bin_num >= 0
        out[ok] = arr[bin_num[ok]]
        return out

    s = (cube.lam >= lam_range[0]) & (cube.lam <= lam_range[1])
    with np.errstate(all="ignore"):
        flux_map = np.nanmedian(cube.data[s], axis=0)      # per-spaxel continuum flux (TW photometric tracer)
    flux_map[bin_num < 0] = np.nan

    def result():
        return dict(v=V, v_err=dV, v_err_formal=dV0, sigma=S, sigma_err=dS, bin_flux=F, chi2=chi2,
                    v_map=to_map(V), v_err_map=to_map(dV), v_err_formal_map=to_map(dV0), sigma_map=to_map(S),
                    sigma_err_map=to_map(dS), chi2_map=to_map(chi2),
                    flux_map=flux_map, bin_num=bin_num, velscale=velscale, v_sys_guess=vel0)

    n_fail = 0
    for b in range(nb):
        if finite[:, b].sum() < 0.5 * finite.shape[0]:
            continue
        gal = gal_all[:, b].copy()
        if have_var[b]:
            noise = np.sqrt(np.clip(var_all[:, b], 1e-12, None))
        else:
            noise = np.full_like(gal, max(np.nanstd(np.diff(gal)) / np.sqrt(2), 1e-6))
        norm = np.nanmedian(gal[gal > 0]) if np.any(gal > 0) else np.nan
        if not np.isfinite(norm) or norm <= 0:
            continue
        gal, noise = gal / norm, noise / norm
        usable = (bad_all[:, b] < 0.01) & np.isfinite(gal) & np.isfinite(noise) & (noise > 0)
        goodpix = np.intersect1d(good_lines, np.where(usable)[0])
        if goodpix.size < 100:
            continue
        gal[~usable] = 0.0
        noise[~usable] = 1e3
        try:
            pp = ppxf(temp, gal, noise, velscale, start, moments=moments, degree=degree,
                      lam=lam_gal, lam_temp=lam_temp, goodpixels=goodpix, quiet=quiet)
        except Exception as exc:
            n_fail += 1
            if progress and n_fail <= 5:
                print(f"[ppxf] bin {b} failed: {exc}")
            continue
        V[b], S[b] = pp.sol[0], pp.sol[1]
        err = pp.error * np.sqrt(max(pp.chi2, 1.0))
        dV[b], dS[b] = err[0], err[1]
        dV0[b] = pp.error[0]
        F[b] = norm
        chi2[b] = pp.chi2
        if progress and (b + 1) % 50 == 0:
            print(f"[ppxf] {b + 1}/{nb} bins")
        if checkpoint is not None and (b + 1) % checkpoint_every == 0:
            save_maps(checkpoint[0], result(), checkpoint[1])
    if n_fail:
        print(f"[ppxf] {n_fail} bins failed")
    return result()


def save_maps(path, maps, header=None):
    """Write FLUX, V, V_ERR, SIGMA, SIGMA_ERR, BINNUM image extensions (+ celestial WCS)."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    if header is not None:
        try:
            hdr.update(WCS(header).celestial.to_header())
        except Exception:
            pass
        for k in ("OBJECT", "DATE-OBS", "EXPTIME", "HIERARCH ESO OBS PROG ID"):
            if k in header:
                hdr[k if not k.startswith("HIERARCH") else "PROGID"] = header[k]
    hdus = [fits.PrimaryHDU(header=hdr)]
    for k, name in (("flux_map", "FLUX"), ("v_map", "V"), ("v_err_map", "V_ERR"), ("v_err_formal_map", "V_ERR_FORMAL"),
                    ("sigma_map", "SIGMA"), ("sigma_err_map", "SIGMA_ERR"), ("chi2_map", "CHI2")):
        if k in maps:
            hdus.append(fits.ImageHDU(maps[k].astype(np.float32), name=name, header=hdr))
    hdus.append(fits.ImageHDU(maps["bin_num"].astype(np.int32), name="BINNUM", header=hdr))
    fits.HDUList(hdus).writeto(path, overwrite=True)
