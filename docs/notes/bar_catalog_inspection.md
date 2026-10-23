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
