# Progress update sent to Virginia (29 Oct 2026)

## Pipeline (repo: sardarkagrandson/chile-work, branch claude/sharp-tesla-d9a3jx)

End to end: DESI DR1 sample -> bar catalogue / Galaxy Zoo cross-match -> ESO
archive search (bulk MUSE lists, local match) -> Legacy Surveys geometry
(disc PA/i, bar PA/length) -> MUSE cube download -> Voronoi + pPXF stellar
kinematics (4750-5950 A, sky masks) -> Halpha/[OIII]/[SII] gas kinematics
-> kinematic PA (annulus outside the bar) -> TW pseudo-slits with PA Monte
Carlo -> R_cr, R and regime probabilities. 33 tests; TW recovers synthetic
pattern speeds to 2 %.

## Sample funnel

| step | N |
|---|---|
| star-forming dwarfs (paper) | 14,786 |
| of which SGA / in bar catalogue / barred (ellip >= 0.3, above disc) | 797 / 242 / 66 |
| SF dwarfs at z < 0.012 with MUSE data | 21 (no usable bar) |
| all DESI galaxies 300 < cz < 15,000 km/s (no SF cut) | 222,110 |
| dwarfs (log M* <= 9.0) / with MUSE data / flagged barred | 126,726 / 479 / 9 |
| clear bars on inspection with MUSE cubes | 2: NGC 4487 (AMUSING), UGC 5288 |

## First measurements

* UGC 5288: no stellar rotation detected; TW slope 2.3 sigma -> upper limit. Not a TW target.
* NGC 4487: first stellar field was an artefact of OH sky residuals in the
  southern mosaic blocks (spurious N-S gradient at PA 33); diagnosed with the
  absorption-free [OIII]/[SII] fields (E-W line of nodes, small amplitude).
  Stellar fit redone on the blue range with sky masks; TW rerun pending.
  Inclination may be lower than the photometric 55 deg.

## Open questions for Virginia

1. Bar catalogue: construction, meaning of bar_length (uncorrelated with size for dwarfs), ellipticity > 0.85 entries.
2. Quiescent dwarfs: include them (TW needs stellar kinematics only)? Which MUSE programmes to prioritise?
3. ROSIE galaxy: name, data status.
4. MUSE proposal (P117) from the 40 barred SF dwarfs at z < 0.012 without archival data.
