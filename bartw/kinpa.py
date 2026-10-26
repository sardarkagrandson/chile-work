"""Kinematic position angle of a velocity field (Krajnovic et al. 2006 method).

For a trial PA the velocity field is folded into the bi-(anti)symmetric
version it would have if PA were the line of nodes:
V_sym(x, y) = [V(x, y) - V(-x, y) + V(x, -y) - V(-x, -y)] / 4 in the frame
where x runs along the major axis. The PA minimizing chi^2 between V and
V_sym is the kinematic PA; the error is the range where chi^2 rises by
1 sigma (scaled so that the best fit has chi^2_red = 1), with a floor set by
the grid step. Also returns the receding-side direction, which the TW slope
sign is compared against.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _fold(v, x0, y0, pa_deg, east_left=True, radius_pix=None):
    ny, nx = v.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    pa = np.radians(pa_deg)
    sgn = -1.0 if east_left else 1.0
    dx, dy = (xx - x0) * sgn, yy - y0          # dx positive towards east
    X = dx * np.sin(pa) + dy * np.cos(pa)      # along the major axis
    Y = -dx * np.cos(pa) + dy * np.sin(pa)
    good = np.isfinite(v)
    if radius_pix is not None:
        good &= np.hypot(X, Y) <= radius_pix
    filled = np.where(np.isfinite(v), v, 0.0)
    wgt = np.isfinite(v).astype(float)

    def sample(Xs, Ys):
        # back to pixel coordinates
        dxs = Xs * np.sin(pa) - Ys * np.cos(pa)
        dys = Xs * np.cos(pa) + Ys * np.sin(pa)
        px, py = x0 + dxs * sgn, y0 + dys
        val = map_coordinates(filled, [py, px], order=1, mode="constant", cval=0.0)
        w = map_coordinates(wgt, [py, px], order=1, mode="constant", cval=0.0)
        return np.where(w > 0.99, val / np.maximum(w, 1e-9), np.nan)

    v1, v2, v3 = sample(-X, Y), sample(X, -Y), sample(-X, -Y)
    vsym = (v - v1 + v2 - v3) / 4.0
    ok = good & np.isfinite(vsym)
    return vsym, ok, X


def fit_kinematic_pa(v, x0, y0, east_left=True, radius_pix=None, v_err=None, step=1.0, v_sys=None):
    """Return dict(pa_kin, pa_kin_err, v_sys, receding_pa, chi2_curve) with PA in degrees
    (north through east, 0-180). ``receding_pa`` is the PA (0-360) of the side with
    V > V_sys along the kinematic major axis."""
    v = np.asarray(v, float)
    if v_sys is None:
        v_sys = np.nanmedian(v)
    vv = v - v_sys
    pas = np.arange(0.0, 180.0, step)
    chi2 = np.full(pas.size, np.nan)
    for i, pa in enumerate(pas):
        vsym, ok, _ = _fold(vv, x0, y0, pa, east_left, radius_pix)
        if ok.sum() < 20:
            continue
        res = (vv - vsym)[ok]
        if v_err is not None:
            e = np.asarray(v_err, float)[ok]
            e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e))
            chi2[i] = np.sum((res / e) ** 2)
        else:
            chi2[i] = np.sum(res**2)
    k = int(np.nanargmin(chi2))
    pa_best = pas[k]
    # parabolic refinement around the minimum
    km, kp = (k - 1) % pas.size, (k + 1) % pas.size
    y0_, y1, y2 = chi2[km], chi2[k], chi2[kp]
    if np.isfinite([y0_, y1, y2]).all() and (y0_ - 2 * y1 + y2) > 0:
        pa_best = pas[k] + 0.5 * step * (y0_ - y2) / (y0_ - 2 * y1 + y2)
    # 1-sigma error: delta chi2 = 1 after scaling to chi2_red = 1 at the minimum
    _, ok, _ = _fold(vv, x0, y0, pa_best, east_left, radius_pix)
    dof = max(int(ok.sum()) - 1, 1)
    scale = chi2[k] / dof
    thresh = chi2[k] + max(scale, 1e-12) * 3.0        # 3 sigma range -> /3 below (robust to the discrete grid)
    inside = chi2 <= thresh
    # contiguous range around k on the circle
    width = 0
    for d in range(1, pas.size // 2):
        if inside[(k + d) % pas.size]:
            width = d
        else:
            break
    width_m = 0
    for d in range(1, pas.size // 2):
        if inside[(k - d) % pas.size]:
            width_m = d
        else:
            break
    err = max(0.5 * (width + width_m) * step / 3.0, 0.5 * step)
    # receding side: sign of the mean velocity for X > 0 along the best PA
    _, ok, X = _fold(vv, x0, y0, pa_best, east_left, radius_pix)
    pos = ok & (X > 0)
    receding_pa = pa_best % 360 if np.nanmean(vv[pos]) > 0 else (pa_best + 180) % 360
    return dict(pa_kin=float(pa_best % 180), pa_kin_err=float(err), v_sys=float(v_sys),
                receding_pa=float(receding_pa), pas=pas, chi2=chi2, n_pix=int(ok.sum()))
