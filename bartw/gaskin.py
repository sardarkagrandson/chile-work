"""Ionised-gas (Halpha) kinematics from a MUSE cube, per spaxel.

Fast moment analysis, enough for a kinematic PA and a rotation curve:
for every spaxel the continuum is estimated from two side windows and the
flux-weighted centroid of the line within +-``half_window`` Angstrom of the
expected wavelength gives the velocity. [NII]6584 lies 21 A redward, so a
12 A half window at z < 0.01 is safe. S/N from the peak over the side-band
scatter. Returns flux, velocity, velocity error (centroid error) and S/N maps.

Gas and stars need not share a line of nodes (the gas can be warped or
disturbed), but when they agree the disc PA is settled and the gas rotation
curve, corrected for inclination, is the circular velocity for the TW
rotation rate (asymmetric drift is negligible for the cold gas).
"""
from __future__ import annotations

import numpy as np

C_KMS = 299792.458
HALPHA = 6562.80  # air


def line_moments(cube, z, lam0=HALPHA, half_window=12.0, side=(20.0, 40.0), snr_min=3.0):
    """Flux, velocity, velocity error and S/N maps of an emission line."""
    lc = lam0 * (1 + z)
    lam = cube.lam
    win = (lam >= lc - half_window) & (lam <= lc + half_window)
    sb = ((lam >= lc - side[1]) & (lam <= lc - side[0])) | ((lam >= lc + side[0]) & (lam <= lc + side[1]))
    if win.sum() < 5 or sb.sum() < 5:
        raise ValueError("line window outside the cube wavelength range")
    with np.errstate(all="ignore"):
        cont = np.nanmedian(cube.data[sb], axis=0)
        noise = 1.4826 * np.nanmedian(np.abs(cube.data[sb] - cont[None]), axis=0)
        spec = cube.data[win] - cont[None]
        spec = np.where(np.isfinite(spec), spec, 0.0)
        pos = np.clip(spec, 0, None)
        flux = pos.sum(axis=0) * np.median(np.diff(lam))
        peak = spec.max(axis=0)
        snr = peak / np.where(noise > 0, noise, np.nan)
        lw = lam[win][:, None, None]
        cen = (pos * lw).sum(axis=0) / np.where(pos.sum(axis=0) > 0, pos.sum(axis=0), np.nan)
        width = np.sqrt(np.clip((pos * (lw - cen[None]) ** 2).sum(axis=0) / np.where(pos.sum(axis=0) > 0, pos.sum(axis=0), np.nan), 0, None))
        vel = C_KMS * (cen / lam0 - 1.0)
        # centroid error ~ sigma_lambda / (S/N * sqrt(n_eff))
        verr = C_KMS * width / lam0 / np.maximum(snr, 1e-3) / np.sqrt(np.maximum(width / np.median(np.diff(lam)), 1.0))
    good = np.isfinite(vel) & (snr >= snr_min)
    vel = np.where(good, vel, np.nan)
    verr = np.where(good, verr, np.nan)
    return dict(flux=flux, vel=vel, vel_err=verr, snr=snr, good=good)


def rotation_curve(vel, x0, y0, pa_deg, incl_deg, v_sys, pixscale, east_left=True, wedge_deg=30.0,
                   dr_arcsec=1.0, r_max_arcsec=None, weights=None, v_err=None):
    """Rotation curve from a velocity field along the major axis.

    V_rot(R) = (V - V_sys) / (cos(phi) sin i), averaged in radial rings over
    the pixels within ``wedge_deg`` of the major axis on both sides (receding
    side taken positive). Returns R (arcsec), V_rot, error, N per ring.
    """
    ny, nx = vel.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    pa = np.radians(pa_deg)
    sgn = -1.0 if east_left else 1.0
    dx, dy = (xx - x0) * sgn, yy - y0
    X = dx * np.sin(pa) + dy * np.cos(pa)
    Y = -dx * np.cos(pa) + dy * np.sin(pa)
    ci, si = np.cos(np.radians(incl_deg)), np.sin(np.radians(incl_deg))
    xd, yd = X, Y / ci                              # disc plane
    R = np.hypot(xd, yd) * pixscale
    phi = np.arctan2(yd, xd)                        # azimuth in the disc plane from the major axis
    cosphi = np.cos(phi)
    ok = np.isfinite(vel) & (np.abs(np.degrees(np.arcsin(np.clip(np.abs(np.sin(phi)), 0, 1)))) <= wedge_deg) & (np.abs(cosphi) > 0.1)
    with np.errstate(all="ignore"):
        vrot = (vel - v_sys) / (cosphi * si)
    # sign: receding side (mean V > V_sys at X>0) positive
    side = np.nanmean(vel[ok & (X > 0)]) - np.nanmean(vel[ok & (X < 0)])
    if np.isfinite(side) and side < 0:
        vrot = -vrot
    if ok.sum() < 5:
        return dict(r=np.array([]), v=np.array([]), v_err=np.array([]), n=np.array([], int))
    r_max = r_max_arcsec or np.nanmax(R[ok])
    edges = np.arange(0, r_max + dr_arcsec, dr_arcsec)
    w = np.ones_like(vel) if weights is None else np.where(np.isfinite(weights), weights, 0.0)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = ok & (R >= a) & (R < b) & np.isfinite(vrot)
        if s.sum() < 5:
            continue
        ww = w[s]
        if v_err is not None:
            ww = ww / np.where(np.isfinite(v_err[s]) & (v_err[s] > 0), v_err[s] ** 2, np.nan)
            ww = np.where(np.isfinite(ww), ww, 0.0)
        if ww.sum() <= 0:
            continue
        v = np.average(vrot[s], weights=ww)
        e = np.sqrt(np.average((vrot[s] - v) ** 2, weights=ww) / max(s.sum() - 1, 1))
        out.append((0.5 * (a + b), v, e, int(s.sum())))
    arr = np.array(out) if out else np.zeros((0, 4))
    return dict(r=arr[:, 0], v=arr[:, 1], v_err=arr[:, 2], n=arr[:, 3].astype(int))


def circular_velocity(rc, r_min_arcsec, r_max_arcsec=None):
    """Weighted mean of the rotation curve beyond r_min (outside the bar) ->
    (V_c, error, n_rings). Error includes the ring-to-ring scatter."""
    s = (rc["r"] >= r_min_arcsec)
    if r_max_arcsec is not None:
        s &= rc["r"] <= r_max_arcsec
    if s.sum() == 0:
        return np.nan, np.nan, 0
    v, e = rc["v"][s], np.where(rc["v_err"][s] > 0, rc["v_err"][s], np.nanmedian(rc["v_err"][s]))
    w = 1 / e**2
    vc = np.sum(w * v) / np.sum(w)
    err = np.sqrt(1 / np.sum(w) + (np.std(v) ** 2 / max(s.sum(), 1)))
    return float(vc), float(max(err, 0.05 * abs(vc))), int(s.sum())


# --------------------------------------------------------------------------- #
# proper line fitting on the stellar Voronoi bins
# --------------------------------------------------------------------------- #
NII_A, NII_B = 6548.05, 6583.45


def _gauss(lam, lc, sig):
    return np.exp(-0.5 * ((lam - lc) / sig) ** 2)


def fit_halpha_bins(cube, bin_num, z, v_star=None, half_window=45.0, min_snr=3.0, progress=True,
                    v_grid_step=4.0, v_half_range=500.0, sig_em_grid=(1.3, 1.8, 2.5, 3.5), sig_abs_grid=(5.0, 8.0, 12.0)):
    """Fit Halpha + [NII]6548,6584 emission plus a broad Halpha absorption on the
    spectra summed in Voronoi bins, by a grid search in velocity with a linear
    least-squares solve for the amplitudes at each grid point (fast and never
    fails to converge).

    Model: c0 + c1 (l - l0) + A_Ha G(l; l_Ha(v), s_em) + A_N2 [G(l; l_6584(v), s_em) + G(l; l_6548(v), s_em)/3]
           - A_abs G(l; l_Ha(v_abs), s_abs)
    The absorption centre is the per-bin stellar velocity ``v_star`` when given
    (else cz); s_em and s_abs run over small grids. Velocity error from the
    delta chi^2 = 1 width of the chi^2(v) curve, scaled by sqrt(chi2_red).
    """
    from .kinematics import stack_bins
    lc0 = HALPHA * (1 + z)
    lam, spectra, variances = stack_bins(cube, bin_num, (lc0 - half_window, lc0 + half_window))
    nb = spectra.shape[1]
    v0 = C_KMS * z
    vgrid = np.arange(v0 - v_half_range, v0 + v_half_range + v_grid_step, v_grid_step)
    V = np.full(nb, np.nan); dV = V.copy(); S = V.copy(); F = V.copy(); SNR = V.copy(); EWabs = V.copy()
    x = lam - lc0
    for b in range(nb):
        y = spectra[:, b]
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        yy = y[ok]; xx = x[ok]; lam_ok = lam[ok]
        var = variances[ok, b]
        err = np.sqrt(var) if np.isfinite(var).all() and (var > 0).all() else np.full_like(yy, max(np.std(np.diff(yy)) / np.sqrt(2), 1e-6))
        w = 1.0 / err
        vabs = v_star[b] if (v_star is not None and np.isfinite(v_star[b])) else v0
        best = (np.inf, None)
        for s_abs in sig_abs_grid:
            labs = HALPHA * (1 + vabs / C_KMS)
            g_abs = -_gauss(lam_ok, labs, s_abs)
            for s_em in sig_em_grid:
                # design matrices for all velocities at once: (nv, npix, 5)
                lha = HALPHA * (1 + vgrid / C_KMS)[:, None]
                l1 = NII_A * (1 + vgrid / C_KMS)[:, None]
                l2 = NII_B * (1 + vgrid / C_KMS)[:, None]
                g_ha = _gauss(lam_ok[None, :], lha, s_em)
                g_n2 = _gauss(lam_ok[None, :], l2, s_em) + _gauss(lam_ok[None, :], l1, s_em) / 3.0
                nv = vgrid.size
                A = np.empty((nv, yy.size, 5))
                A[:, :, 0] = 1.0; A[:, :, 1] = xx[None, :]; A[:, :, 2] = g_ha; A[:, :, 3] = g_n2; A[:, :, 4] = g_abs[None, :]
                Aw = A * w[None, :, None]
                yw = (yy * w)[None, :]
                # normal equations, batched
                AtA = np.einsum("vpi,vpj->vij", Aw, Aw)
                Aty = np.einsum("vpi,vp->vi", Aw, yw)
                try:
                    coef = np.linalg.solve(AtA, Aty[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    continue
                resid = yw - np.einsum("vpi,vi->vp", Aw, coef)
                chi2 = np.sum(resid**2, axis=1)
                k = int(np.argmin(chi2))
                if chi2[k] < best[0]:
                    best = (chi2[k], (k, coef[k], chi2, s_em, s_abs, AtA[k]))
        if best[1] is None:
            continue
        k, coef, chi2, s_em, s_abs, AtA_k = best[1]
        dof = max(yy.size - 6, 1)
        chi2r = best[0] / dof
        # amplitude error from the normal matrix
        try:
            cov = np.linalg.inv(AtA_k) * max(chi2r, 1.0)
            a_err = np.sqrt(max(cov[2, 2], 0))
        except np.linalg.LinAlgError:
            a_err = np.inf
        a_ha = coef[2]
        if a_ha <= 0 or a_ha / a_err < min_snr:
            continue
        # velocity error: parabola through the chi2 minimum (delta chi2 = 1), scaled by chi2_red
        km, kp = max(k - 1, 0), min(k + 1, vgrid.size - 1)
        curv = (chi2[km] - 2 * chi2[k] + chi2[kp]) / v_grid_step**2 if kp > km else np.nan
        verr = np.sqrt(2.0 / curv) if np.isfinite(curv) and curv > 0 else v_grid_step
        vbest = vgrid[k]
        if kp > km and curv > 0:
            vbest = vgrid[k] + 0.5 * v_grid_step * (chi2[km] - chi2[kp]) / (chi2[km] - 2 * chi2[k] + chi2[kp])
        V[b], dV[b] = vbest, verr * np.sqrt(max(chi2r, 1.0))
        S[b] = C_KMS * np.sqrt(max(s_em**2 - (2.6 / 2.355) ** 2, 0.0)) / HALPHA
        F[b] = a_ha * s_em * np.sqrt(2 * np.pi)
        SNR[b] = a_ha / a_err
        EWabs[b] = coef[4] * s_abs * np.sqrt(2 * np.pi) / max(coef[0], 1e-9)
        if progress and (b + 1) % 500 == 0:
            print(f"[gas] {b + 1}/{nb} bins")

    def to_map(arr):
        out = np.full(bin_num.shape, np.nan)
        okb = bin_num >= 0
        out[okb] = arr[bin_num[okb]]
        return out

    return dict(v=V, v_err=dV, sigma=S, flux=F, snr=SNR, ew_abs=EWabs,
                v_map=to_map(V), v_err_map=to_map(dV), sigma_map=to_map(S), flux_map=to_map(F), snr_map=to_map(SNR))
