# Bar pattern speeds of star-forming dwarf galaxies (DESI DR1 + MUSE)

Follow-up to Robin Thomas' DESI DR1 dwarf-galaxy paper, in collaboration with
Virginia Cuomo (Universidad de La Serena). Goal: enlarge the sample of dwarf
barred galaxies with a Tremaine-Weinberg (TW) bar pattern speed from three
(all Virgo, Cuomo+2022, 2024) to tens, across environments.

Plan agreed in the kick-off meeting (Oct 2026):

1. DESI DR1 star-forming galaxies -> dwarfs; cross-check bars with the DESI
   Legacy Imaging bar catalogue (and Galaxy Zoo DESI).
2. Cross-match with the ESO archive for MUSE data to refine the sample.
3. Apply the TW method to the final sample (+ one galaxy with new ROSIE-IFU data).

Documents: [project plan](docs/01_project_plan.md) ·
[sample selection](docs/02_sample_selection.md) ·
[TW method and validation](docs/03_tw_method.md) ·
[literature](docs/04_literature.md) · [data access](docs/05_data_access.md) ·
[kick-off notes](docs/meetings/2026-10-23_kickoff.md) ·
[bar catalogue inspection](docs/notes/bar_catalog_inspection.md)

## Pipeline

```
scripts/download_data.sh                 # FastSpecFit Iron (BGS), GZ DESI, SGA-2020 -> data/raw/
scripts/01_select_desi_sf_dwarfs.py      # paper sample (or FastSpecFit cuts) + resolution tier -> data/interim/01_sf_dwarfs.parquet
scripts/02_crossmatch_bars.py            # SGA bar catalogue + GZ DESI bars, priorities -> products/barred_dwarf_candidates.csv
scripts/03_query_muse.py                 # ESO TAP: MUSE cubes / raw frames   -> data/products/sample_with_muse.csv
scripts/02c_any_sample_bars_muse.py TBL  # route 2 with any DESI table (e.g. 200<cz<10300 km/s, no cuts): bars + MUSE, dwarfs by mass -> products/<name>_barred_muse_candidates.csv
scripts/02b_bar_catalog_muse.py          # route 2 without DESI: whole bar catalogue x MUSE, dwarfs by M_r, no SF requirement -> products/sga_barred_muse_candidates.csv
scripts/04_measure_geometry.py           # LS cutouts: disc PA, i, bar PA, a_bar, TW suitability -> sample_geometry.csv
scripts/download_muse_cubes.py --targetid ID   # ESO cubes -> data/raw/cubes/
scripts/inspect_cube.py CUBE --ra --dec  # field coverage of the bar, nucleus pixel position
scripts/05_stellar_kinematics.py CUBE    # Voronoi + pPXF stellar V maps      -> kin_<name>.fits
scripts/06_run_tw.py KIN --pa .. --incl ..  # TW: Omega_p, V_sys, R_cr, R + figures -> tw_results.csv
scripts/validate_tw_mock.py              # recover a known Omega_p from a synthetic barred disc
```

All thresholds live in `config/selection.yaml`. Each script prints its
selection funnel so the numbers can go straight into the paper/proposal.

## Notebook

`notebooks/01_vetting_muse_geometry.ipynb` walks through the next steps
interactively, in the order that costs the least effort: (1) load the
dwarfs and barred candidates; (2) download the ESO MUSE product lists once
and match all 1,782 resolvable dwarfs locally; (3) visually vet only the
MUSE-covered galaxies on Legacy Surveys cutouts (saved to
`products/vetting.csv`); (4) optional Galaxy Zoo DESI bars; (5) disc/bar
geometry and TW suitability for the vetted galaxies; (6) a TW demo on a
synthetic disc. Sections 2-5 need internet access. Works on Python >= 3.9.

`notebooks/02_route2_candidates.ipynb` works on the output of
`scripts/02c_any_sample_bars_muse.py`: cutouts, geometry and vetting of the
priority candidates, an isophote sweep of all MUSE-covered dwarfs for bars
the SGA catalogue cannot see, and the TW-ready list. Copy either notebook to
a working name before running so `git pull` never conflicts.

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt  (Python >= 3.9)
pytest                      # 16 tests: TW recovery on mocks, geometry, DESI selection logic
python scripts/validate_tw_mock.py      # -> figures/tw_mock_validation.png
cp DESI_dwarfs.csv data/raw/DESI_dwarfs_paper_sample.csv   # paper sample (not in git)
bash scripts/download_data.sh           # GZ DESI, SGA-2020 (FastSpecFit only for --from-fastspecfit)
python scripts/01_select_desi_sf_dwarfs.py && python scripts/02_crossmatch_bars.py
```

## Status (23 Oct 2026)

* Steps 1-2 run: from the paper's 14,786 star-forming dwarfs, 797 are SGA
  galaxies, 242 are in the bar catalogue, 131 have bar ellipticity >= 0.3, 66 also
  have a bar more elongated than the disc (and <= 0.7), 61 have 25-70 deg
  inclination; 40 are at z < 0.012 (priority 1-2). List: `products/barred_dwarf_candidates.csv`.
* Package `bartw/` written and unit-tested; TW code recovers the input
  pattern speed of synthetic barred discs to < 2% (noiseless) and shows the
  expected PA-dominated error budget.
* Bar catalogue (89,780 SGA galaxies) inspected: `bar_length` unit unknown,
  no bar PA; used as a bar flag only until clarified with the authors.
* Steps 2b-4 (Galaxy Zoo DESI bars, ESO archive query, Legacy Surveys
  geometry) need network access the setup session did not have: run
  `scripts/download_data.sh`, then scripts 02-04 locally.
* First archival result (26 Oct): 21 star-forming dwarfs at z < 0.012 have
  MUSE data; the imaging finds one bar, aligned with the major axis, so zero
  TW targets from the star-forming route. Next: z < 0.0333, all dwarfs
  regardless of star formation (`scripts/02b_bar_catalog_muse.py`), and a
  MUSE proposal list from the 40 priority candidates.
* Open items for Virginia: ROSIE galaxy details; MUSE programmes to prioritise;
  bar catalogue definitions.

## Layout

```
bartw/        python package (desi, morphology, bar_catalog, muse, photometry, geometry, kinematics, tw, mock, plotting)
scripts/      numbered pipeline steps
notebooks/    interactive companion (vetting, MUSE query, geometry)
config/       selection.yaml
docs/         plan, selection, method, literature, data access, meeting notes
tests/        pytest suite
data/         raw/ interim/ products/  (git-ignored)
figures/      diagnostic figures (only the mock validation is tracked)
```
