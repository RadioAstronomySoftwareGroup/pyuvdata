# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Define an AnalyticBeam subclass for testing the AnalyticBeam plugin code."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as nptype

from ..analytic_beam import UnpolarizedAnalyticBeam


@dataclass(kw_only=True)
class CosPowerTest(UnpolarizedAnalyticBeam):
    """A test class to support testing the AnalyticBeam plugin code."""

    width: float

    def _power_eval(
        self,
        *,
        az_grid: nptype.NDArray[np.floating],
        za_grid: nptype.NDArray[np.floating],
        f_grid: nptype.NDArray[np.floating],
    ) -> nptype.NDArray[np.floating]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")

        for pol_i in np.arange(self.Npols):
            # For power beams the first axis is shallow because we don't have to worry
            # about polarization.
            data_array[0, pol_i, :, :] = np.cos(self.width * za_grid) ** 2

        return data_array


@dataclass(kw_only=True)
class CosEfieldTest(UnpolarizedAnalyticBeam):
    """A test class to support testing the AnalyticBeam plugin code."""

    width: float

    def _efield_eval(
        self,
        *,
        az_grid: nptype.NDArray[np.floating],
        za_grid: nptype.NDArray[np.floating],
        f_grid: nptype.NDArray[np.floating],
    ) -> nptype.NDArray[np.floating]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_grid.shape)

        for feed_i in np.arange(self.Nfeeds):
            # For power beams the first axis is shallow because we don't have to worry
            # about polarization.
            data_array[0, feed_i, :, :] = np.cos(self.width * za_grid) / np.sqrt(2)
            data_array[1, feed_i, :, :] = np.cos(self.width * za_grid) / np.sqrt(2)

        return data_array
