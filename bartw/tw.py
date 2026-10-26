"""Tremaine-Weinberg (1984) bar pattern speed measurement.

For a tracer obeying the continuity equation in a disc hosting a single
rigidly rotating pattern, for every pseudo-slit parallel to the disc major
axis at projected offset Y:

    Omega_p sin(i) * <X> = <V> - V_sys ,

    <X> = int Sigma X dX / int Sigma dX ,   <V> = int Sigma V_los dX / int Sigma dX

The slope of <V> versus <X> across slits gives Omega_p sin i. The dominant
systematic is the disc position angle (Debattista 2003; Garma-Oehmichen+2020),
so the error budget here is a Monte Carlo over PA plus the fit error.

Coordinate convention
---------------------
Images are assumed in the standard astronomical orientation (north up, east
left, i.e. x increasing to the *west* in pixel space, so RA increases with
decreasing x). Position angles are measured from north (+y) through east (-x).
If your cube is stored with east to the right, pass ``east_left=False``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from .geometry import kpc_per_arcsec


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
def sky_to_disc_axes(x, y, x0, y0, pa_deg, east_left=True):
    """Rotate pixel coordinates so that X runs along the disc major axis
    (positive towards the PA direction) and Y along the minor axis."""
    pa = np.radians(pa_deg)
    dx = (x - x0) * (-1.0 if east_left else 1.0)   # dx positive towards east
    dy = y - y0                                      # dy positive towards north
    X = dx * np.sin(pa) + dy * np.cos(pa)
    Y = -dx * np.cos(pa) + dy * np.sin(pa)
    return X, Y


# --------------------------------------------------------------------------- #
# integrals
# --------------------------------------------------------------------------- #
@dataclass
class SlitIntegrals:
    y_center: np.ndarray        # slit offsets (same units as X, Y)
    x_mean: np.ndarray          # <X>
    v_mean: np.ndarray          # <V>
    x_err: np.ndarray
    v_err: np.ndarray
    npix: np.ndarray
    flux_sum: np.ndarray


def slit_integrals(flux, vel, X, Y, y_centers, width, x_half_length=None,
                   vel_err=None, mask=None):
    """Photometric and kinematic TW integrals in pseudo-slits.

    Parameters
    ----------
    flux, vel : 2-D arrays
        Tracer surface brightness (same tracer as the kinematics!) and
        line-of-sight velocity. Voronoi-binned maps are fine: each spaxel
        carries its bin velocity and its own flux.
    X, Y : 2-D arrays from :func:`sky_to_disc_axes`.
    y_centers : slit centres along the minor axis; ``width`` the slit width.
    x_half_length : optional truncation of the slits along X (must enclose the
        whole bar; the axisymmetric disc cancels for symmetric limits).
    vel_err : optional velocity error map, propagated into v_err.
    mask : boolean array, True = use pixel.
    """
    flux = np.asarray(flux, float)
    vel = np.asarray(vel, float)
    good = np.isfinite(flux) & np.isfinite(vel) & (flux > 0)
    if mask is not None:
        good &= np.asarray(mask, bool)
    if x_half_length is not None:
        good &= np.abs(X) <= x_half_length
    if vel_err is None:
        vel_err = np.zeros_like(vel)

    out = {k: [] for k in ("x_mean", "v_mean", "x_err", "v_err", "npix", "flux_sum")}
    for yc in np.atleast_1d(y_centers):
        sel = good & (np.abs(Y - yc) <= width / 2.0)
        n = int(sel.sum())
        if n < 5:
            for k in out:
                out[k].append(np.nan if k != "npix" else n)
            continue
        f = flux[sel]
        fsum = f.sum()
        xm = np.sum(f * X[sel]) / fsum
        vm = np.sum(f * vel[sel]) / fsum
        # statistical errors: velocity errors propagate linearly; <X> error
        # from the flux-weighted scatter (pixel-noise-limited estimate)
        verr = np.sqrt(np.sum((f * vel_err[sel]) ** 2)) / fsum
        xerr = np.sqrt(np.sum((f * (X[sel] - xm)) ** 2)) / fsum / np.sqrt(n)
        out["x_mean"].append(xm)
        out["v_mean"].append(vm)
        out["x_err"].append(xerr)
        out["v_err"].append(verr)
        out["npix"].append(n)
        out["flux_sum"].append(fsum)
    return SlitIntegrals(np.atleast_1d(y_centers), **{k: np.asarray(v, float) for k, v in out.items()})


# --------------------------------------------------------------------------- #
# slope fit
# --------------------------------------------------------------------------- #
def fit_slope(x, v, x_err=None, v_err=None, n_iter=20):
    """Straight-line fit v = slope*x + v_sys with errors on both axes.

    Uses the effective-variance method (iteratively reweighted least squares
    with weights 1/(sigma_v^2 + slope^2 sigma_x^2); York 1966 / Orear 1982),
    which is what scipy.odr converges to for a straight line. Falls back to
    ordinary least squares when errors are missing or zero.
    """
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    ok = np.isfinite(x) & np.isfinite(v)
    x, v = x[ok], v[ok]
    if x.size < 3:
        return dict(slope=np.nan, slope_err=np.nan, v_sys=np.nan, v_sys_err=np.nan, n=x.size)
    slope, vsys = np.polyfit(x, v, 1)
    have_err = (x_err is not None and v_err is not None)
    if have_err:
        sx, sv = np.asarray(x_err, float)[ok], np.asarray(v_err, float)[ok]
        have_err = np.all(np.isfinite(sx)) and np.all(np.isfinite(sv)) and np.all((sx > 0) | (sv > 0))
    if not have_err:
        (slope, vsys), cov = np.polyfit(x, v, 1, cov=True)
        s_err, v_err_fit = np.sqrt(np.diag(cov))
        return dict(slope=float(slope), slope_err=float(s_err), v_sys=float(vsys),
                    v_sys_err=float(v_err_fit), n=int(x.size))
    for _ in range(n_iter):
        w = 1.0 / (sv**2 + slope**2 * sx**2)
        W = w.sum()
        xm, vm = np.sum(w * x) / W, np.sum(w * v) / W
        new_slope = np.sum(w * (x - xm) * (v - vm)) / np.sum(w * (x - xm) ** 2)
        vsys = vm - new_slope * xm
        if abs(new_slope - slope) < 1e-10 * max(abs(slope), 1e-30):
            slope = new_slope
            break
        slope = new_slope
    w = 1.0 / (sv**2 + slope**2 * sx**2)
    W = w.sum()
    xm = np.sum(w * x) / W
    sxx = np.sum(w * (x - xm) ** 2)
    # scale errors by reduced chi^2 when > 1 (accounts for scatter beyond the quoted errors)
    chi2 = np.sum(w * (v - slope * x - vsys) ** 2) / max(x.size - 2, 1)
    f = np.sqrt(max(chi2, 1.0))
    s_err = f / np.sqrt(sxx)
    v_err_fit = f * np.sqrt(1.0 / W + xm**2 / sxx)
    return dict(slope=float(slope), slope_err=float(s_err), v_sys=float(vsys),
                v_sys_err=float(v_err_fit), n=int(x.size))


# --------------------------------------------------------------------------- #
# full measurement
# --------------------------------------------------------------------------- #
@dataclass
class TWResult:
    omega_sini: float               # km/s/arcsec (slope of <V> vs <X>)
    omega_sini_err: float
    omega_p: float                  # km/s/kpc
    omega_p_err: float
    omega_p_err_fit: float          # fit-only component
    omega_p_err_pa: float           # PA Monte Carlo component (16-84 / 2)
    v_sys: float
    pa_deg: float
    incl_deg: float
    scale_kpc_arcsec: float
    integrals: SlitIntegrals
    pa_mc: np.ndarray = field(default_factory=lambda: np.array([]))
    omega_mc: np.ndarray = field(default_factory=lambda: np.array([]))
    rotation_sign: float = 0.0      # sign of dV/dX of the disc along the major axis (+1: receding at +X)
    prograde: "bool | None" = None  # TW slope has the same sign as the disc rotation (must be True)

    def summary(self):
        tag = "" if self.prograde is None else (" [prograde]" if self.prograde else " [RETROGRADE: unphysical, check PA/centre]")
        return (f"Omega_p sin i = {self.omega_sini:.3f} +/- {self.omega_sini_err:.3f} km/s/arcsec ; "
                f"|Omega_p| = {abs(self.omega_p):.1f} +/- {self.omega_p_err:.1f} km/s/kpc "
                f"(fit {self.omega_p_err_fit:.1f}, PA {self.omega_p_err_pa:.1f}); "
                f"V_sys = {self.v_sys:.1f} km/s{tag}")


def default_slit_centers(a_bar_arcsec, dpa_deg, n_slits, width, ymax_factor=0.9):
    """Slit offsets that all cross the bar: |Y| <= ymax_factor * a_bar*|sin dPA|
    where dPA is the (sky) angle between bar and disc major axis."""
    ymax = ymax_factor * a_bar_arcsec * abs(np.sin(np.radians(dpa_deg)))
    ymax = max(ymax, width)  # at least one slit on each side
    return np.linspace(-ymax, ymax, n_slits)


def measure_pattern_speed(flux, vel, pixscale, x0, y0, pa_deg, incl_deg,
                          slit_centers_arcsec, slit_width_arcsec,
                          x_half_length_arcsec=None, vel_err=None, mask=None,
                          distance_mpc=None, z=None, pa_err_deg=0.0, n_mc=0,
                          rng=None, east_left=True, H0=70.0):
    """Run the TW measurement on 2-D flux and velocity maps.

    All angular inputs in arcsec; ``pixscale`` in arcsec/pixel. Physical
    scale from ``distance_mpc`` (preferred) or ``z``.
    """
    ny, nx = flux.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    scale = float(kpc_per_arcsec(z=z, distance_mpc=distance_mpc, H0=H0))
    sini = np.sin(np.radians(incl_deg))

    def run(pa):
        X, Y = sky_to_disc_axes(xx, yy, x0, y0, pa, east_left)
        X, Y = X * pixscale, Y * pixscale
        ints = slit_integrals(flux, vel, X, Y, slit_centers_arcsec, slit_width_arcsec,
                              x_half_length_arcsec, vel_err, mask)
        fit = fit_slope(ints.x_mean, ints.v_mean, ints.x_err, ints.v_err)
        return ints, fit

    ints, fit = run(pa_deg)
    omega_sini = fit["slope"]
    omega_p = omega_sini / sini / scale
    err_fit = fit["slope_err"] / sini / scale

    pa_mc = omega_mc = np.array([])
    err_pa = 0.0
    if pa_err_deg > 0 and n_mc > 0:
        rng = np.random.default_rng(rng)
        pa_mc = rng.normal(pa_deg, pa_err_deg, n_mc)
        omega_mc = np.array([run(p)[1]["slope"] for p in pa_mc]) / sini / scale
        lo, hi = np.nanpercentile(omega_mc, [16, 84])
        err_pa = 0.5 * (hi - lo)

    err_tot = float(np.hypot(err_fit, err_pa))
    # disc rotation sense along the major axis (flux-weighted mean V on each side of the minor axis)
    X, Y = sky_to_disc_axes(xx, yy, x0, y0, pa_deg, east_left)
    X, Y = X * pixscale, Y * pixscale
    okm = np.isfinite(flux) & np.isfinite(vel) & (flux > 0)
    if mask is not None:
        okm &= np.asarray(mask, bool)
    band = okm & (np.abs(Y) <= max(2 * slit_width_arcsec, 2.0))
    xl = x_half_length_arcsec or np.nanmax(np.abs(X[band])) if band.any() else 0
    plus, minus = band & (X > 0.3 * xl), band & (X < -0.3 * xl)
    rot = 0.0
    if plus.sum() > 5 and minus.sum() > 5:
        vp = np.average(vel[plus], weights=flux[plus]); vm = np.average(vel[minus], weights=flux[minus])
        rot = float(np.sign(vp - vm))
    prograde = None if rot == 0 else bool(np.sign(omega_sini) == rot)
    return TWResult(rotation_sign=rot, prograde=prograde, omega_sini=omega_sini, omega_sini_err=float(np.hypot(fit["slope_err"], err_pa * sini * scale)),
                    omega_p=omega_p, omega_p_err=err_tot, omega_p_err_fit=err_fit,
                    omega_p_err_pa=err_pa, v_sys=fit["v_sys"], pa_deg=pa_deg,
                    incl_deg=incl_deg, scale_kpc_arcsec=scale, integrals=ints,
                    pa_mc=pa_mc, omega_mc=omega_mc)


# --------------------------------------------------------------------------- #
# circular velocity and rotation rate
# --------------------------------------------------------------------------- #
def asymmetric_drift_vc(v_phi, sigma_R, R, h_R, h_sigma=None, dlnV_dlnR=0.0):
    """Circular velocity from the stellar azimuthal velocity via the asymmetric
    drift equation (Binney & Tremaine 2008, eq. 4.228) for an exponential disc
    with exponentially declining sigma_R (scale h_sigma; default 2 h_R):

        Vc^2 = Vphi^2 + sigma_R^2 [ R/h_R + 2 R/h_sigma - 1 + sigma_phi^2/sigma_R^2 ]

    with sigma_phi^2/sigma_R^2 = 0.5 (1 + dlnV/dlnR) from the epicyclic
    approximation. This is the approach used in e.g. Aguerri+2015 and Cuomo+2019
    for TW samples; it is approximate for dwarfs (pressure support can be
    significant), so prefer gas rotation (Halpha) when a gas map is available.
    """
    if h_sigma is None:
        h_sigma = 2.0 * h_R
    ratio2 = 0.5 * (1.0 + dlnV_dlnR)
    return np.sqrt(v_phi**2 + sigma_R**2 * (R / h_R + 2.0 * R / h_sigma - 1.0 + ratio2))


def rotation_rate(omega_p, omega_p_err, vc, vc_err, a_bar_kpc, a_bar_err_kpc,
                  n_mc=20000, rng=None, slow_boundary=1.4):
    """Corotation radius R_cr = Vc/Omega_p and rotation rate R = R_cr/a_bar with
    Monte Carlo errors and the probabilities of the ultrafast (R<1),
    fast (1<=R<=1.4) and slow (R>1.4) regimes."""
    rng = np.random.default_rng(rng)
    om = rng.normal(omega_p, omega_p_err, n_mc)
    v = rng.normal(vc, vc_err, n_mc)
    a = rng.normal(a_bar_kpc, a_bar_err_kpc, n_mc)
    ok = (om > 0) & (a > 0) & (v > 0)
    rcr = v[ok] / om[ok]
    R = rcr / a[ok]
    p16, p50, p84 = np.percentile(R, [16, 50, 84])
    r16, r50, r84 = np.percentile(rcr, [16, 50, 84])
    return dict(
        R_cr=r50, R_cr_err_lo=r50 - r16, R_cr_err_hi=r84 - r50,
        R=p50, R_err_lo=p50 - p16, R_err_hi=p84 - p50,
        p_ultrafast=float(np.mean(R < 1.0)),
        p_fast=float(np.mean((R >= 1.0) & (R <= slow_boundary))),
        p_slow=float(np.mean(R > slow_boundary)),
        regime=("slow" if p50 > slow_boundary else "fast" if p50 >= 1.0 else "ultrafast"),
    )
