# Project plan: bar pattern speeds of star-forming dwarf galaxies with DESI DR1 and MUSE

Robin Thomas & Virginia Cuomo (Universidad de La Serena) -- started October 2026

## 1. Science case

Bars are the main internal driver of secular evolution in disc galaxies. Their
dynamical state is set by the pattern speed Omega_p, usually expressed through
the rotation rate R = R_cr / a_bar (corotation radius over bar length):
R < 1 "ultrafast" (dynamically forbidden), 1 <= R <= 1.4 "fast", R > 1.4 "slow".
Dynamical friction against a dense dark-matter halo brakes bars, so theory
predicts slow bars wherever the central regions are dark-matter dominated
(Debattista & Sellwood 2000; Athanassoula 2003). Dwarf galaxies are exactly
that regime, yet essentially all Tremaine-Weinberg (TW) measurements in the
local Universe are for massive discs and find fast bars (Aguerri+2015;
Guo+2019; Garma-Oehmichen+2020, 2022; Geron+2023; Cuomo+2020 review).

Only three dwarfs have TW pattern speeds so far, all Virgo cluster members
observed with MUSE by this team: IC 3167 (R = 1.7 +0.5/-0.3, slow and
lopsided, interacting; Cuomo+2022), NGC 4483 and NGC 4516 (R ~ 1.75 for
NGC 4516; Cuomo+2024). Their bars are shorter, weaker and rotate more slowly
than those of massive galaxies, consistent with either a different bar
formation mechanism (tidal triggering) or with dark-matter braking, but
three objects, all in one cluster, cannot separate the two.

**Goal:** enlarge the sample of dwarf barred galaxies with measured bar
pattern speeds by an order of magnitude, spanning field and group
environments, using DESI DR1 spectroscopy to build a clean, homogeneous
parent sample of star-forming dwarfs and archival MUSE cubes (plus one new
ROSIE-IFU dataset) for the TW measurement.

Questions we can answer with ~10-30 dwarfs:

1. Are bars in dwarfs generically slow (R > 1.4), or is slowness tied to
   interaction/cluster infall as in IC 3167?
2. How do Omega_p, R_cr and R scale with stellar mass, gas fraction and bar
   strength when the dwarf regime is added to the MaNGA/CALIFA relations?
3. Does the environment (DESI gives spectroscopic neighbours and group
   membership for every target) correlate with the bar dynamical state?
4. Bar length and strength vs. mass: do the DESI Legacy bar catalogue
   measurements agree with detailed isophotal analysis in the dwarf regime?

## 2. Why DESI DR1 + MUSE

* DESI DR1 Bright Galaxy Survey gives ~6-7 million galaxy spectra to r < 19.5
  with uniform redshifts, FastSpecFit stellar masses, SFRs and emission-line
  fluxes. This is the parent sample of Robin's submitted dwarf paper, so the
  selection is already understood and the dwarf sample is characterised
  (mass, SFR, dynamics).
* The DESI Legacy Imaging bar catalogue (89,780 SGA galaxies with bar length
  and ellipticity; see `docs/notes/bar_catalog_inspection.md`) and Galaxy Zoo
  DESI (8.7 M galaxies with bar vote fractions; Walmsley+2023) provide two
  independent bar indicators over the same footprint.
* MUSE is the only IFU with the field (1'x1'), sampling (0.2") and depth to
  measure stellar kinematics in low-surface-brightness dwarf discs out to
  beyond the bar; the ESO archive now holds > 10 years of cubes (nearby
  galaxy programmes, Fornax 3D, PHANGS-MUSE, MAGPI, cluster dwarf programmes,
  our own Virgo programme). Many dwarfs were observed as fillers or
  companions and never analysed for bars.

## 3. Work packages

| WP | Content | Lead | Repo entry point |
|----|---------|------|------------------|
| 1 | Parent sample: DESI DR1 BGS star-forming dwarfs (quality, z, mass, BPT/WHAN) | Robin | `scripts/01_select_desi_sf_dwarfs.py` |
| 2 | Barred subsample: SGA bar catalogue + GZ DESI; agreement study | Robin | `scripts/02_crossmatch_bars.py` |
| 3 | Archival IFU coverage: ESO TAP query for MUSE cubes and raw frames; priority list | Robin | `scripts/03_query_muse.py` |
| 4 | Geometry: disc PA/inclination, bar PA/length from Legacy Surveys imaging; TW suitability | Robin + Virginia | `scripts/04_measure_geometry.py` |
| 5 | Stellar kinematics from MUSE cubes (Voronoi + pPXF); kinematic PA | Virginia (expertise) + Robin | `scripts/05_stellar_kinematics.py` |
| 6 | TW pattern speeds, V_c (asymmetric drift and Halpha), R_cr, R, error budget | both | `scripts/06_run_tw.py`, `bartw/tw.py` |
| 7 | ROSIE-IFU galaxy: reduction/handling of the new data, same TW pipeline | Virginia | same as 6 |
| 8 | Interpretation: R vs mass/environment/bar strength; comparison with MaNGA/CALIFA and the Virgo dwarfs; paper | both | `docs/` |

## 4. Selection summary (details and justification in `docs/02_sample_selection.md`)

1. Parent sample = the 14,786 star-forming dwarfs of the DESI DR1 paper
   (BPT star forming in [NII] and [SII], log M*/Msun <= 9.0, z < 0.0333).
2. Resolution tier A: z < 0.012 (D < ~50 Mpc, a 1 kpc bar spans >= 5 MUSE resolution elements): 1,782 galaxies.
3. Barred: bar catalogue bar_ellipticity >= 0.3 (131 of the 797 SGA dwarfs) or GZ DESI (featured>=0.27, not edge-on>=0.68, bar_no<0.5; to be run locally).
4. Inclination 25-70 deg: 125; priority 1-2 (tier A): 85, of which 52 at Dec < +25.
5. MUSE cube (or raw data) within 30" of the centre, t_exp >= 30 min.
6. TW geometry: 10 < |PA_bar - PA_disc| < 80 deg; undisturbed disc; bar inside the FoV (step 4, Legacy Surveys isophotes).

## 5. Timeline (weekly meetings)

| When | Milestone |
|------|-----------|
| Oct-Nov 2026 | WP1-3: parent sample, barred candidates, MUSE coverage table; fix bar-catalogue definitions; ROSIE galaxy added |
| Nov-Dec 2026 | WP4-5: geometry for all MUSE-covered candidates; kinematic maps for the first 3-5 cubes; TW on 1-2 test galaxies incl. a re-measurement of NGC 4516 as a benchmark against Cuomo+2024 |
| Jan-Feb 2027 | WP6: TW for the full MUSE sample; V_c from stars and gas; first R vs mass plot = **preliminary results for funding applications** |
| Mar-Apr 2027 | WP7-8: ROSIE galaxy; environment from DESI; comparison with the literature; draft paper |
| mid 2027 | Submit paper; proposal for new MUSE observations of the best TW-suitable dwarfs without archival data (the DESI selection yields a ranked target list for an ESO proposal) |

## 6. Risks and mitigations

* **Few dwarfs with MUSE data.** 85 priority candidates exist before the
  archive query; archival MUSE coverage of random nearby dwarfs is a few
  per cent, so expect a handful to ~10, plus the Virgo dwarfs and the ROSIE galaxy. Mitigation: include raw
  (unreduced/proprietary-expired) MUSE frames; include quiescent barred
  dwarfs as a secondary sample; use the ranked candidate list for a MUSE
  proposal (P-cycle 2027).
* **Weak bars -> small <X> -> PA-dominated errors.** The mock tests show a
  3 deg PA error gives ~30% error for a strong bar and ~65% for a weak one.
  Mitigation: kinematic PA (pafit) combined with photometric PA, deep outer
  isophotes, slit selection, and reporting probabilities of regimes rather
  than hard classifications (as in Cuomo+2022).
* **Pressure support in dwarfs.** Asymmetric drift corrections are large and
  model dependent. Mitigation: Halpha rotation from the same cube as an
  independent V_c for star-forming targets.
* **Bar catalogue units/definitions unknown.** Use it as a flag only; measure
  a_bar ourselves (WP4) until clarified.
* **Funding/visit.** FONDECYT postdoc closed to non-residents. The
  preliminary results (Feb 2027) target ANID international-collaboration
  calls, ESO Chile science visitor support, and Robin's institutional travel
  funds.

## 7. Data and code

All thresholds are in `config/selection.yaml`; all steps are scripts in
`scripts/`; the TW code is validated against synthetic discs (`tests/`,
`figures/tw_mock_validation.png`). Catalogues are large and are not in git
(`scripts/download_data.sh`, `docs/05_data_access.md`).
