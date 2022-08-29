# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Definition for BeamInterface object."""
import warnings

from .analytic_beam import AnalyticBeam
from .uvbeam import UVBeam


class BeamInterface:
    """
    Definition for a unified beam interface.

    This object provides a unified interface for UVBeam and AnalyticBeam objects
    to compute beam response values in any direction.

    Attributes
    ----------
    beam : pyuvdata.UVBeam or pyuvdata.AnalyticBeam
        Beam object to use for computations
    beam_type : str
        The beam type, either "efield" or "power".

    """

    def __init__(self, beam, beam_type=None):
        if not isinstance(beam, UVBeam) or isinstance(beam, AnalyticBeam):
            raise ValueError("beam must be a UVBeam or an AnalyticBeam instance.")
        self.beam = beam
        if isinstance(beam, UVBeam):
            if beam_type is None or beam_type == beam.beam_type:
                self.beam_type = beam.beam_type
            elif beam_type == "power":
                warnings.Warn(
                    "`beam` is an efield UVBeam but `beam_type` is specified as "
                    "'power'. Converting efield beam to power."
                )
                self.beam.efield_to_power()
            else:
                raise ValueError(
                    "`beam` is a power UVBeam but `beam_type` is specified as 'efield'."
                    "It's not possible to convert a power beam to an efield beam, "
                    "either provide an efield UVBeam or do not specify `beam_type`."
                )
        else:
            self.beam_type = beam_type
