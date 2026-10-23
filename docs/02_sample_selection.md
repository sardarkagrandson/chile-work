# Sample selection: criteria and justification

Every threshold is in `config/selection.yaml`; the funnel counts are written
to `data/interim/01_steps.json` by step 1 and printed by steps 2-4.

## Step 1 -- star-forming dwarfs in DESI DR1 (`bartw/desi.py`)

Source: FastSpecFit "Iron" VAC, `fastspec-iron-main-bright.fits` (BGS).

| cut | value | why |
|-----|-------|-----|
| SURVEY/PROGRAM | main / bright | BGS is the only DESI programme that targets r < 19.5 galaxies; dwarfs at z < 0.05 are BGS Bright (and some BGS Faint) targets |
| ZWARN, DELTACHI2, SPECTYPE | 0, > 40, GALAXY | DESI DR1 "good redshift" definition for BGS |
| redshift | 0.002 < z < 0.012 | D <~ 50 Mpc (scale <= 0.25 kpc/arcsec). Dwarf bars are ~0.5-2 kpc (Cuomo+2024: 1.0-2.1 kpc); TW needs several slits across the bar at >= PSF spacing, so a 1 kpc bar must subtend >= 4-5". A "reach" sample to z = 0.02 is one config change. Lower limit removes objects whose redshift distances are meaningless; for all targets we later use redshift-independent distances (Cosmicflows-4) where available |
| stellar mass | 7.5 <= log M*/Msun <= 9.5 | FastSpecFit LOGMSTAR (Chabrier, h=1) + 0.31 for H0=70. 9.5 is the conventional "dwarf" ceiling and matches the LMC-mass regime; the three Virgo dwarfs with TW are at ~1e9 Msun. Below 10^7.5 bars are not expected/detectable |
| star forming | BPT (Kauffmann+03, S/N > 3 in 4 lines) or WHAN SF; log sSFR > -11 | matches Robin's DESI dwarf paper; removes AGN/LINER contamination of the Halpha kinematics; the WHAN fallback keeps low-surface-brightness dwarfs whose Hbeta or [OIII] fall below S/N 3 |
| unique object | highest DELTACHI2 per TARGETID | repeat observations |

Expected size: a few thousand galaxies (DR1 BGS has ~6.7 M galaxies; the
z < 0.012 dwarf tail is ~0.1%).

## Step 2 -- barred (`bartw/bar_catalog.py`, `bartw/morphology.py`)

Two independent indicators, both kept as flags:

**(a) DESI Legacy Imaging bar catalogue** (SGA-based, from V. Cuomo). Positional
match within 3" to the SGA centre. Flag `BARRED_SGA` if
`bar_ellipticity >= 0.3` (a standard "real bar" ellipticity threshold,
e.g. Aguerri+2009 use eps_bar >= 0.2-0.3 in the inner profile; IC 3167's weak
bar has 0.07 here, so the threshold removes weak/lopsided features which are
anyway poor TW targets). The join with SGA-2020 brings PA, BA, D26 for the
disc. See `docs/notes/bar_catalog_inspection.md` for the open questions on
the bar_length unit.

**(b) Galaxy Zoo DESI** (Walmsley+2023). Match within 1.5". Disc:
featured-or-disk >= 0.27 and not-edge-on >= 0.68; bar: bar_no < 0.5 (i.e.
strong + weak >= 0.5). These are the cuts of Masters+2011/2021 and of the
2025-26 DESI bar papers, so our bar fraction is directly comparable.
Strong/weak label from strong/(strong+weak) > 0.5. Caveat: GZ DESI is
limited to r < 19 and the deep-learning classifications are least reliable
for small, low-surface-brightness objects, i.e. our dwarfs; (a) and (b)
disagreeing is expected and worth quantifying.

`BARRED = BARRED_SGA or BARRED_GZ`; the "both" subset is the gold sample.

## Step 3 -- MUSE coverage (`bartw/muse.py`)

ESO TAP (`ivoa.ObsCore`, instrument MUSE, dataproduct_type cube) cone search
of 30" around each galaxy (MUSE WFM is 1'x1'; the bar and the disc beyond it
must be inside the field, so the pointing centre must be within ~30" of the
nucleus). Also `dbo.raw` science frames, to find observed-but-unreleased
targets. Minimum total exposure 1800 s (the Virgo dwarfs used ~1 h per
pointing to reach S/N ~ 40 per Voronoi bin at mu_r ~ 23 mag/arcsec^2).
Proprietary cubes are kept and flagged with their release date.

## Step 4 -- TW geometry (`bartw/photometry.py`, `bartw/geometry.py`)

From Legacy Surveys r-band cutouts (0.262"/pix): isophote fits ->
disc PA and ellipticity from the outer isophotes (60-95% of the fitted
range) -> inclination with q0 = 0.3 (dwarfs are thick; q0 = 0.2 would bias
i high by a few degrees). Bar PA at the ellipticity peak; a_bar from the
ellipticity-drop and PA-change estimators (Aguerri+2009), averaged, with
half their difference as error. Keep if

* 25 <= i <= 70 deg: below, sin i is small and <V> is tiny; above, the
  slits do not sample the bar and the thick disc violates the thin-disc
  assumption (Garma-Oehmichen+2022 show the ultrafast fraction halves when
  restricting to 30-60);
* 10 <= |PA_bar - PA_disc| <= 80 deg: bars aligned with either axis give
  <X> -> 0 or <V> -> 0 (Geron+2023 use > 10 deg from both axes);
* no strong lopsidedness, companions inside the field, or dust lanes
  (visual check on the cutout + the MUSE white-light image);
* bar fully inside the MUSE field, with disc coverage to >= 1.5 a_bar.

Final sample = TW-suitable barred star-forming dwarfs with usable MUSE data,
plus the ROSIE-IFU galaxy.
