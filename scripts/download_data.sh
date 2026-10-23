#!/usr/bin/env bash
# Fetch the public catalogues the pipeline needs into data/raw/.
# Run from the repo root. Sizes are approximate. Re-running skips existing files.
set -euo pipefail
RAW=data/raw
mkdir -p "$RAW"

fetch () {  # fetch URL DEST
  if [ -s "$2" ]; then echo "have $2"; else echo "-> $2"; curl -L --fail --retry 4 -o "$2" "$1"; fi
}

# 1. DESI DR1 FastSpecFit "Iron" VAC, BGS (main/bright) -- ~3 GB.
#    Index: https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/
#    Check the version directory (v3.0 expected for DR1) and adjust below.
FSF_VER=${FSF_VER:-v3.0}
fetch "https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/${FSF_VER}/catalogs/fastspec-iron-main-bright.fits" \
      "$RAW/fastspec-iron-main-bright.fits"

# 2. Galaxy Zoo DESI (Walmsley+2023), Zenodo 8360385 -- the "friendly" catalogue, ~2 GB parquet.
#    Browse https://zenodo.org/records/8360385 for the exact filename if this 404s.
fetch "https://zenodo.org/records/8360385/files/gz_desi_deep_learning_catalog_friendly.parquet?download=1" \
      "$RAW/gz_desi_deep_learning_catalog_friendly.parquet"

# 3. Siena Galaxy Atlas 2020 (disc PA, b/a, D26 for the bar-catalogue galaxies) -- ~1 GB.
fetch "https://portal.nersc.gov/project/cosmo/data/sga/2020/SGA-2020.fits" "$RAW/SGA-2020.fits"

# 4. Bar catalogue from V. Cuomo (not public): copy it by hand to
#    $RAW/desi_legacy_bars_sga.fits  (89,780 rows; SGA_ID, bar_length, bar_ellipticity)

echo "done. Now: python scripts/01_select_desi_sf_dwarfs.py"
