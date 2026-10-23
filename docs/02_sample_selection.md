# Sample selection: criteria and justification

Every threshold is in `config/selection.yaml`; the funnel counts are written
to `data/interim/01_steps.json` by step 1 and printed by steps 2-4.

## Step 1 -- star-forming dwarfs in DESI DR1

**Default: the sample of Thomas et al. (DESI DR1 dwarf paper), `DESI_dwarfs.csv`.**
14,786 galaxies; DESI DR1 main + SV + special programmes; star forming on
both the [NII] and [SII] BPT diagrams with S/N > 3 in all six lines;
log M*/Msun <= 9.0 (FastSpecFit, h = 0.678); z < 0.0333; one spectrum per
galaxy. The table already carries Tractor shapes (b/a, PA), SGA-2020
geometry for its 797 SGA members, physical scale, SFR, metallicity and
environment (n_companions, nearest neighbour, tidal parameter Q, isolation
flag), so steps 2-4 and the interpretation (R vs environment) use the same
quantities as the paper. `scripts/01_select_desi_sf_dwarfs.py` only adds a
spatial-resolution tier:

| tier | cut | N | why |
|------|-----|---|-----|
| A | z < 0.012 (D <~ 50 Mpc, <= 0.25 kpc/") | 1,782 | a 1 kpc bar subtends >= 4-5" = >= 5 MUSE resolution elements; required for TW |
| B | 0.012 <= z < 0.02 | 3,235 | usable for the longest bars only |
| C | z >= 0.02 | 9,769 | not resolvable; kept for statistics (bar fraction vs mass/environment) |

The FastSpecFit route (`--from-fastspecfit`, cuts in `config/selection.yaml`)
is kept to extend the parent sample later, e.g. to 9.0 < log M* < 9.5 or to
quiescent dwarfs, with the same quality / BPT / WHAN logic.

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

**Result with indicator (a) alone (23 Oct 2026, `products/barred_dwarf_candidates.csv`):**

| step | N |
|------|---|
| star-forming dwarfs | 14,786 |
| of which SGA-2020 galaxies (the only ones the bar catalogue can contain) | 797 |
| in the bar catalogue (SGA_ID join + 3" positional) | 242 |
| bar_ellipticity >= 0.3 | 131 |
| bar more elongated than the inclined disc by >= 0.1 (the catalogue records a measurement for every entry, not a detection) | 77 |
| both, and bar_ellipticity <= 0.7 (9 implausible values removed) = `BARRED_SGA` | 66 |
| + 25 <= i <= 70 deg (SGA or Tractor b/a, q0 = 0.3) | 61 |
| priority 1: tier A and bar_ellipticity >= 0.4 | 34 |
| priority 2: tier A, 0.3 <= bar_ellipticity < 0.4 | 6 |
| priority 3: tier B | 16 |

The 66 is a lower bound: 94% of the dwarfs are not SGA galaxies (too small
for the atlas, D25 < 20"), and only GZ DESI (indicator b) can flag bars in
them. Several very high bar ellipticities (> 0.7) at low inclination are
suspicious in this regime and must be vetted on the Legacy Surveys images
in step 4 before any telescope time is spent on them.

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
