# Data access

Nothing large is in git. `scripts/download_data.sh` fetches the public
catalogues into `data/raw/`. Approximate sizes in brackets.

| dataset | where | what we use |
|---|---|---|
| DESI DR1 FastSpecFit Iron VAC, `fastspec-iron-main-bright.fits` [~3 GB] | https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/ (check version dir, v3.0 expected); docs https://fastspecfit.readthedocs.io | TARGETID, RA, DEC, Z, ZWARN, DELTACHI2, SPECTYPE, LOGMSTAR, SFR, line fluxes/ivars, HALPHA_EW, VDISP. In v3 the stellar-population columns are in the SPECPHOT HDU; the loader handles both layouts |
| Galaxy Zoo DESI, `gz_desi_deep_learning_catalog_friendly.parquet` [~2 GB] | https://zenodo.org/records/8360385 | dr8_id, ra, dec, vote fractions for featured, edge-on, bar_strong/weak/no, spiral, merging |
| Siena Galaxy Atlas 2020, `SGA-2020.fits` [~1 GB] | https://portal.nersc.gov/project/cosmo/data/sga/2020/ | SGA_ID, PA, BA, D26, LEDA quantities |
| DESI Legacy Imaging bar catalogue [7 MB] | from V. Cuomo (not public); copy to `data/raw/desi_legacy_bars_sga.fits` | SGA_ID, bar_length, bar_ellipticity |
| Legacy Surveys cutouts | https://www.legacysurvey.org/viewer/fits-cutout?ra=..&dec=..&layer=ls-dr9&pixscale=0.262&size=512&bands=r | r-band images for isophote fitting (fetched on the fly by step 4) |
| ESO archive (MUSE) | TAP https://archive.eso.org/tap_obs ; tables `ivoa.ObsCore` (Phase 3 cubes) and `dbo.raw` (raw frames); run `scripts/03_query_muse.py --schema` to print the live column names | cube list per galaxy; download of cubes needs an ESO account for proprietary data; public cubes via the `access_url` |
| pPXF templates | E-MILES in pPXF npz format, e.g. `spectra_emiles_9.0.npz` from https://github.com/micappe/ppxf_data | step 5 |
| Distances | Cosmicflows-4 (Tully+2023) / NED-D for z < 0.01 targets | physical scale for Omega_p and a_bar |

Notes
* The cloud session used to set up this repo had no network access to these
  hosts, so steps 1-4 have not yet been run on real data; run them locally
  (a laptop is enough; step 1 needs ~8 GB RAM to read the BGS file, or use
  `columns=` in `read_fastspec` to reduce it).
* Column names in `bartw/muse.py` follow the ESO TAP documentation; verify
  once with `--schema` and adjust the two query strings if ESO renamed
  anything.
