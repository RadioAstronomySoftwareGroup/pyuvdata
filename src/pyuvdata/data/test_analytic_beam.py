# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Define an AnalyticBeam subclass for testing the AnalyticBeam plugin code."""

from dataclasses import InitVar, dataclass, field
from typing import Literal

import numpy.typing as npt

from ..uvbeam.analytic_beam import AnalyticBeam


@dataclass(kw_only=True)
class AnalyticTest(AnalyticBeam):
    """A test class to support testing the AnalyticBeam plugin code."""

    radius: float
    feed_array: npt.NDArray[str] | None = field(default=None, repr=False, compare=False)
    x_orientation: Literal["east", "north"] = field(
        default="east", repr=False, compare=False
    )

    include_cross_pols: InitVar[bool] = True

    basis_vector_type = "az_za"

    def _efield_eval(
        self,
        *,
        az_array: npt.NDArray[float],
        za_array: npt.NDArray[float],
        freq_array: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_array.size, freq_array.size)

        return data_array

    def _power_eval(
        self,
        *,
        az_array: npt.NDArray[float],
        za_array: npt.NDArray[float],
        freq_array: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(
            az_array.size, freq_array.size, beam_type="power"
        )

        return data_array
