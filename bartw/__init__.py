"""bartw: bar pattern speeds of DESI star-forming dwarf galaxies.

Pipeline (see README.md):

1. :mod:`bartw.desi`        -- star-forming dwarfs from the DESI DR1 FastSpecFit (Iron) VAC
2. :mod:`bartw.morphology`  -- barred subsample from Galaxy Zoo DESI
3. :mod:`bartw.muse`        -- cross-match with the ESO archive for MUSE cubes
4. :mod:`bartw.photometry`  -- disc/bar geometry from Legacy Survey imaging
5. :mod:`bartw.geometry`    -- Tremaine-Weinberg applicability
6. :mod:`bartw.tw`          -- Tremaine-Weinberg pattern speed, corotation, rotation rate
7. :mod:`bartw.mock`        -- synthetic barred discs used to validate :mod:`bartw.tw`
"""

__version__ = "0.1.0"
