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
scripts/01_select_desi_sf_dwarfs.py      # quality + z + mass + BPT/WHAN      -> data/interim/01_sf_dwarfs.parquet
scripts/02_crossmatch_bars.py            # SGA bar catalogue + GZ DESI bars   -> data/interim/02_barred_candidates.parquet
scripts/03_query_muse.py                 # ESO TAP: MUSE cubes / raw frames   -> data/products/sample_with_muse.csv
scripts/04_measure_geometry.py           # LS cutouts: disc PA, i, bar PA, a_bar, TW suitability -> sample_geometry.csv
scripts/05_stellar_kinematics.py CUBE    # Voronoi + pPXF stellar V maps      -> kin_<name>.fits
scripts/06_run_tw.py KIN --pa .. --incl ..  # TW: Omega_p, V_sys, R_cr, R + figures -> tw_results.csv
scripts/validate_tw_mock.py              # recover a known Omega_p from a synthetic barred disc
```

All thresholds live in `config/selection.yaml`. Each script prints its
selection funnel so the numbers can go straight into the paper/proposal.

## Quick start

```bash
pip install -e .            # or: pip install -r requirements.txt
pytest                      # 16 tests: TW recovery on mocks, geometry, DESI selection logic
python scripts/validate_tw_mock.py      # -> figures/tw_mock_validation.png
bash scripts/download_data.sh           # ~6 GB; then run scripts 01-04 in order
```

## Status (23 Oct 2026)

* Package `bartw/` written and unit-tested; TW code recovers the input
  pattern speed of synthetic barred discs to < 2% (noiseless) and shows the
  expected PA-dominated error budget.
* Bar catalogue (89,780 SGA galaxies) inspected: `bar_length` unit unknown,
  no bar PA; used as a bar flag only until clarified with the authors.
* Steps 1-4 not yet run on real data: the setup session had no access to the
  DESI/Zenodo/ESO hosts. Run `scripts/download_data.sh` locally.
* Open items for Virginia: ROSIE galaxy details; MUSE programmes to prioritise;
  bar catalogue definitions.

## Layout

```
bartw/        python package (desi, morphology, bar_catalog, muse, photometry, geometry, kinematics, tw, mock, plotting)
scripts/      numbered pipeline steps
config/       selection.yaml
docs/         plan, selection, method, literature, data access, meeting notes
tests/        pytest suite
data/         raw/ interim/ products/  (git-ignored)
figures/      diagnostic figures (only the mock validation is tracked)
```
