"""Class for reading and writing beamfits files."""
import numpy as np
from astropy.io import fits
from uvbeam import UVBeam


class UVFITS(UVBeam):
    """
    Defines a beamfits-specific subclass of UVBeam for reading and writing beamfits files.
    This class should not be interacted with directly, instead use the read_beamfits
    and write_beamfits methods on the UVBeam class.

    """
