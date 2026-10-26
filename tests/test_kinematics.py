"""Voronoi + pPXF pipeline on a synthetic MUSE-like cube with a known velocity field."""
import numpy as np
import pytest
from astropy.io import fits

from bartw.kinematics import fit_bins, read_muse_cube, stack_bins, voronoi_bins, white_light

C = 299792.458


def _template(lam):
    """Continuum with a few Gaussian absorption lines (rest frame)."""
    spec = np.ones_like(lam)
    for lc, depth, w in ((4861, 0.5, 3.0), (5175, 0.6, 4.0), (5270, 0.3, 3.0), (5893, 0.5, 3.0), (6563, 0.4, 3.0)):
        spec -= depth * np.exp(-0.5 * ((lam - lc) / w) ** 2)
    return spec


def _make_cube(path, z=0.004, vmax=60.0, nx=30, ny=30, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.arange(4750.0, 7001.0, 1.25)
    tlam = np.arange(4600.0, 7200.0, 0.5)
    temp = _template(tlam)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - 14.5, yy - 14.5)
    vel = C * z + vmax * np.tanh(r / 6.0) * (xx - 14.5) / np.maximum(r, 1e-3)   # rotation about the y axis
    flux = 50.0 * np.exp(-r / 8.0)
    data = np.empty((lam.size, ny, nx), np.float32)
    for j in range(ny):
        for i in range(nx):
            shifted = np.interp(lam, tlam * (1 + vel[j, i] / C), temp)
            data[:, j, i] = flux[j, i] * shifted
    noise = 1.0
    data += rng.normal(0, noise, data.shape).astype(np.float32)
    var = np.full_like(data, noise**2)
    data[:, :3, :] = np.nan; var[:, :3, :] = np.nan                      # outside the field
    notch = (lam > 5780) & (lam < 5970); data[notch] = np.nan; var[notch] = np.nan   # AO notch
    hdr = fits.Header()
    hdr["CRVAL3"], hdr["CRPIX3"], hdr["CD3_3"] = lam[0], 1.0, 1.25
    hdr["CD2_2"] = 0.2 / 3600; hdr["CD1_1"] = -0.2 / 3600
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"; hdr["CRVAL1"], hdr["CRVAL2"] = 150.0, 2.0
    hdr["CRPIX1"], hdr["CRPIX2"] = 15.5, 15.5
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header=hdr, name="DATA"),
                  fits.ImageHDU(var, header=hdr, name="STAT")]).writeto(path, overwrite=True)
    return vel, flux, (temp, tlam)


def test_cube_pipeline_recovers_velocity_field(tmp_path):
    path = tmp_path / "cube.fits"
    vel_true, flux_true, templates = _make_cube(path)
    cube = read_muse_cube(path)
    assert cube.data.shape[0] == cube.lam.size and cube.var is not None
    sig, noise = white_light(cube)
    bins = voronoi_bins(sig, noise, target_sn=40.0, snr_min=1.0)
    assert bins.max() >= 10 and (bins[:3] == -1).all()
    lam, spectra, var = stack_bins(cube, bins, (4750.0, 7000.0))
    assert np.isnan(spectra[(lam > 5800) & (lam < 5950)]).all()      # notch kept as NaN
    maps = fit_bins(cube, bins, 0.004, templates, quiet=True, progress=False)
    ok = np.isfinite(maps["v_map"])
    assert ok.sum() > 0.8 * (bins >= 0).sum()
    resid = (maps["v_map"] - vel_true)[ok]
    # Voronoi bins average the field; require the map to track the input to ~velscale/2
    assert np.nanmedian(np.abs(resid)) < 20.0
    # rotation sense: receding side has larger V
    assert np.nanmean(maps["v_map"][:, 20:]) > np.nanmean(maps["v_map"][:, :10])


def test_spatial_cutout_updates_wcs(tmp_path):
    path = tmp_path / "cube.fits"
    _make_cube(path)
    full = read_muse_cube(path, lam_range=(4800, 5500))
    ra, dec = full.wcs.all_pix2world(20.0, 18.0, 0)
    cut = read_muse_cube(path, lam_range=(4800, 5500), center=(float(ra), float(dec)), half_size_arcsec=1.0)
    assert cut.data.shape[1:] == (11, 11)
    x, y = cut.wcs.all_world2pix(ra, dec, 0)
    assert abs(x - 5) < 1e-6 and abs(y - 5) < 1e-6
    np.testing.assert_allclose(cut.data[:, 5, 5], full.data[:, 18, 20])


def test_gas_moments_and_rotation_curve(tmp_path):
    """Add a Halpha line that follows the velocity field; recover V and the rotation curve."""
    from bartw.gaskin import circular_velocity, line_moments, rotation_curve
    from bartw.kinpa import fit_kinematic_pa
    path = tmp_path / "cube.fits"
    vel_true, flux, templates = _make_cube(path, z=0.004, vmax=60.0)
    with fits.open(path, mode="update") as h:
        data = h["DATA"].data
        lam = 4750.0 + np.arange(data.shape[0]) * 1.25
        ny, nx = data.shape[1:]
        for j in range(ny):
            for i in range(nx):
                if np.isfinite(vel_true[j, i]) and j >= 3:
                    lc = 6562.8 * (1 + vel_true[j, i] / C)
                    data[:, j, i] += 30.0 * np.exp(-0.5 * ((lam - lc) / 2.0) ** 2)
        h.flush()
    cube = read_muse_cube(path)
    m = line_moments(cube, 0.004)
    ok = m["good"]
    assert ok.sum() > 0.8 * (np.isfinite(vel_true).sum() - 3 * nx)
    assert np.nanmedian(np.abs(m["vel"] - vel_true)[ok]) < 10.0   # moment centroid on 57 km/s pixels, stellar absorption underneath
    kp = fit_kinematic_pa(m["vel"], 14.5, 14.5, east_left=True, radius_pix=12, v_err=m["vel_err"])
    # mock rotation is about the y axis: V depends on x -> major axis along x = east-west -> PA 90
    assert abs(((kp["pa_kin"] - 90) + 90) % 180 - 90) < 5.0
    # the mock field has no inclination factor; with an assumed i the recovered V_c is vmax / sin i
    rc = rotation_curve(m["vel"], 14.5, 14.5, kp["pa_kin"], 60.0, C * 0.004, 0.2, east_left=True, dr_arcsec=0.4)
    vc, vc_err, n = circular_velocity(rc, r_min_arcsec=1.5)
    assert 45 < vc * np.sin(np.radians(60.0)) < 65 and n >= 2


def test_fit_halpha_bins_recovers_gas_velocity(tmp_path):
    """Emission on top of stellar absorption: the binned profile fit must not be biased."""
    from bartw.gaskin import fit_halpha_bins
    path = tmp_path / "cube.fits"
    vel_true, flux, templates = _make_cube(path, z=0.004, vmax=60.0)
    with fits.open(path, mode="update") as h:
        data = h["DATA"].data
        lam = 4750.0 + np.arange(data.shape[0]) * 1.25
        ny, nx = data.shape[1:]
        for j in range(3, ny):
            for i in range(nx):
                lc = 6562.8 * (1 + (vel_true[j, i] + 15.0) / C)      # gas offset by +15 km/s from the stars
                data[:, j, i] += 6.0 * np.exp(-0.5 * ((lam - lc) / 2.0) ** 2)   # weak emission (EW ~ absorption EW)
        h.flush()
    cube = read_muse_cube(path)
    sig, noise = white_light(cube)
    bins = voronoi_bins(sig, noise, target_sn=30.0)
    # per-bin stellar velocity = flux-weighted true velocity
    nb = bins.max() + 1
    vstar = np.array([np.average(vel_true[bins == b], weights=flux[bins == b]) for b in range(nb)])
    g = fit_halpha_bins(cube, bins, 0.004, v_star=vstar, progress=False, sig_abs_grid=(3.0, 5.0, 8.0))  # mock absorption is 3 A wide
    ok = np.isfinite(g["v"])
    assert ok.sum() > 0.7 * nb
    # recovers the +15 km/s gas offset to better than half its value even when the emission is as weak as the absorption
    assert abs(np.nanmedian((g["v"] - vstar)[ok]) - 15.0) < 8.0
    assert np.nanmedian(np.abs(g["v"] - vstar - 15.0)[ok]) < 10.0
