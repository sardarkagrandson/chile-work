# Inspection of the DESI Legacy Imaging bar catalogue (23 Oct 2026)

File: `Properties_of_bars_for_galaxies_from_DESI_legacy_imaging_surveys.fits`
(kept locally as `data/raw/desi_legacy_bars_sga.fits`; not in git).
Written with STIL/TOPCAT on 2025-01-08, table name "Joined".

| column | type | range | notes |
|---|---|---|---|
| SGA_ID | int | 2 .. 5,004,xxx | unique; 89,780 rows -> join key to SGA-2020 |
| GALAXY | str | | SGA names (PGC, NGC, SDSS J...) |
| RA, DEC | deg | full sky; 16,552 rows at Dec < -30 | SGA centres |
| G/R/Z_MAG_SB25 | mag | -1 (missing) .. ~21 | SGA isophotal magnitudes within SB 25 |
| bar_length | ? | 0.02 .. 3.36, median 0.41 | **unit not given** |
| bar_ellipticity | - | 0.0001 .. 0.93, median 0.31 | bar strength proxy |

Checks against known bars (literature bar semi-major axes):

| galaxy | bar_length | bar_ellip | literature a_bar | ratio |
|---|---|---|---|---|
| NGC 4314 | 0.612 | 0.535 | ~65-70" = 1.1' (Erwin 2005) | ~0.55 |
| NGC 3351 | 0.332 | 0.484 | ~50-55" = 0.85' | ~0.4 |
| IC 3167 | 0.149 | 0.071 | ~15" = 0.25' (Cuomo+2022) | ~0.6 |

So `bar_length` is **not** the bar semi-major axis in arcmin (values are a factor
~2 too small), unless it is a different estimator (radius of maximum
ellipticity, which underestimates the bar end) or is normalised by an SGA
radius. The low ellipticity of IC 3167 (0.07) for a known (weak, lopsided)
bar suggests the ellipticity column is close to the bar ellipticity measured
in the sky plane without deprojection.

=> Action item 3 in the kick-off notes: ask the catalogue authors for the
definition. Until then the pipeline uses the catalogue only as a *bar flag*
(`bar_ellipticity >= 0.3` by default) and measures a_bar and the bar PA
itself from Legacy Surveys cutouts (`bartw/photometry.py`).

Counts relevant to us (Dec > -20 as a crude DESI footprint; faint SB25 r mag
as a crude dwarf proxy):

| cut | N |
|---|---|
| Dec > -20 | 69,253 |
| + r_SB25 > 16, bar_ellip > 0.3 | 15,585 |
| + r_SB25 > 17, bar_ellip > 0.3 | 3,468 |
| + r_SB25 > 17, bar_ellip > 0.4 | 2,161 |

Most SGA galaxies with r_SB25 > 17 at z < 0.012 are dwarfs, so a few thousand
barred-dwarf candidates exist in the imaging; the DESI spectroscopy (BGS
Bright, r < 19.5) covers most of them. The MUSE cross-match will be the
limiting step (expect tens, not hundreds).

## Cross-match with the paper's star-forming dwarf sample (23 Oct 2026)

* 797 of the 14,786 dwarfs are SGA-2020 galaxies; 242 of those are in the
  bar catalogue (SGA_ID join; a 3" positional match finds the same objects
  plus one), i.e. the catalogue flags ~30% of SGA dwarfs as barred, a
  plausible dwarf bar fraction.
* Among the 242 matched dwarfs `bar_length` shows **no correlation** with
  R25 in arcsec (r = 0.00), with R25 in kpc (r = 0.08) or with distance
  (r = 0.05), and bar_length/R25 spans 0.3-3.6 if read in arcmin. For
  dwarfs the column therefore looks like noise or a quantity unrelated to
  the bar extent; it is not used anywhere in the pipeline.
* `bar_ellipticity` for the matched dwarfs: median 0.33 (16-84%: 0.16-0.55),
  with a tail to 0.9. Values > 0.7 in galaxies with i < 40 deg are not
  physical bars (an aligned companion, an edge-on background object or a
  failed fit) and need visual vetting.
* The Tractor shape convention was verified against SGA on 601 dwarfs:
  PA = 0.5 atan2(e2, e1) (north through east), median |dPA| = 1.5 deg;
  b/a = (1-e)/(1+e) agrees to 0.05.
