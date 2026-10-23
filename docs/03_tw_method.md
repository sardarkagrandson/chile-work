# The Tremaine-Weinberg method as implemented in `bartw/tw.py`

## Theory

For a tracer that satisfies the continuity equation and whose surface density
pattern rotates rigidly with Omega_p in a flat disc (Tremaine & Weinberg
1984), integrating along pseudo-slits parallel to the disc major axis at
offset Y gives

    Omega_p sin i <X> = <V> - V_sys,
    <X> = int Sigma X dX / int Sigma dX,   <V> = int Sigma V_los dX / int Sigma dX .

Several slits at different Y give points (<X>, <V>) on a straight line whose
slope is Omega_p sin i and whose intercept is V_sys. The method needs:

1. the same tracer for Sigma and V_los (stellar continuum flux and stellar
   velocity from the same spectra; Halpha would need the gas to satisfy
   continuity, which it does not);
2. a single, steady pattern (no strong spiral arms or secondary bars inside
   the integration region; lopsidedness breaks the assumption, cf. IC 3167);
3. a well-determined disc PA (the slits must be parallel to the line of
   nodes), intermediate inclination, and bar PA away from both axes;
4. slits extending to where the non-axisymmetric density vanishes
   (>= 1.5 a_bar on both sides, symmetric about the centre).

## Implementation

* `sky_to_disc_axes`: rotates pixel coordinates to the (X, Y) frame of the
  disc major/minor axes (PA from north through east, east-left images;
  `east_left=False` for flipped cubes).
* `slit_integrals`: flux-weighted <X> and <V> per slit, with error on <V>
  from the velocity error map and on <X> from the flux-weighted scatter.
  Voronoi-binned maps are handled naturally (each spaxel keeps its own flux
  and carries its bin velocity).
* `default_slit_centers`: n odd slits with |Y| <= 0.9 a_bar |sin dPA| so
  that every slit crosses the bar; width >= PSF FWHM.
* `fit_slope`: straight line with errors on both axes (effective-variance
  iterative weighting, equivalent to ODR for a line), errors scaled by
  sqrt(chi2_red) when > 1.
* `measure_pattern_speed`: converts the slope to Omega_p in km/s/kpc using
  sin i and the physical scale (prefer a redshift-independent distance), and
  runs a Monte Carlo over the disc PA (Gaussian, sigma = `pa_error_deg`,
  default 3 deg), repeating the full slit analysis each time. The PA term is
  added in quadrature to the fit error and both are reported separately.
* `asymmetric_drift_vc`: V_c from stellar V_phi and sigma_R for an
  exponential disc (Binney & Tremaine 2008 eq. 4.228 with the epicyclic
  closure), as used in Aguerri+2015 / Cuomo+2019. For dwarfs this correction
  is large; the Halpha rotation curve from the same MUSE cube (star-forming
  targets) gives an independent V_c, and we will quote both.
* `rotation_rate`: R_cr = V_c / Omega_p, R = R_cr / a_bar (deprojected bar
  length) with a Monte Carlo over all three inputs, returning the median,
  16-84 % interval and the probabilities of the ultrafast (R < 1), fast
  (1-1.4) and slow (R > 1.4) regimes, following Cuomo+2022.

## Validation on synthetic discs (`bartw/mock.py`, `tests/test_tw_mock.py`)

The TW integrals are linear in Sigma and Sigma V, so a disc made of
components that each satisfy continuity satisfies the relation exactly:
an axisymmetric exponential disc with any rotation curve (contributes zero
to both integrals for symmetric slits) plus a Ferrers bar whose material
rotates as a solid body at Omega_p. The observed velocity is the
flux-weighted mean. This tests the slit integration, the fit, the coordinate
conventions and the error propagation with a known answer and no dynamical
modelling.

Results (input Omega_p = 30 km/s/kpc, D = 16.5 Mpc, i = 50 deg, PA = 35 deg,
bar at 40 deg in the disc plane, 9 slits of 1"):

| case | recovered Omega_p [km/s/kpc] | fit error | PA (3 deg) error |
|------|------------------------------|-----------|------------------|
| noiseless, strong bar | 30.1 | 0.1 | 9.7 |
| + PSF 0.8", 5% flux noise, 5 km/s velocity noise | 26.9 | 1.3 | 10.8 |
| noiseless, weak bar (contrast 1.5) | 30.3 | 0.3 | 19.7 |
| PA = 120, i = 40, bar at -55 deg | 29.9 | 0.1 | -- |
| flipped east/west convention | 30.1 | -- | -- |

All noiseless cases are recovered to better than 2%; the pytest suite requires
5%. The PA Monte Carlo dominates the error budget by an order of magnitude,
as found by Debattista (2003) and Garma-Oehmichen+(2020), and it doubles for
the weak bar because <X> shrinks while the spurious contribution of the
misaligned axisymmetric disc does not. For dwarfs this is *the* limiting
systematic, which is why the plan includes: kinematic PA from the stellar
velocity field (`pafit`, Krajnovic+2006) combined with the photometric PA,
deep outer isophotes from Legacy Surveys, and a selection that favours
strong bars (bar_ellipticity >= 0.3). See `figures/tw_mock_validation.png`
(right panel: recovered Omega_p versus imposed PA error).

## Quality flags to report per galaxy (following Geron+2023, Cuomo+2024)

* photometric vs kinematic PA difference (< 5 deg preferred);
* number of slits with |<X>| > 2 sigma_X; linearity of the TW diagram
  (chi2_red, NRMSE < 0.2);
* V_sys from the TW intercept vs V_sys from the velocity field (consistency
  check on the centre and PA);
* fraction of the slit area outside the field of view (should be 0);
* Omega_p / error > 3 for inclusion in the R statistics; otherwise report as
  a probability of regimes only.
