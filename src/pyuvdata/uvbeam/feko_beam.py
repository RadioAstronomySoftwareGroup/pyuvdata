# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading beam FEKO files."""

import os

import numpy as np
import numpy.typing as npt
from docstring_parser import DocstringStyle

from .. import utils as uvutils
from ..docstrings import copy_replace_short_description
from .uvbeam import UVBeam

__all__ = ["FEKOBeam"]


class FEKOBeam(UVBeam):
    """
    Defines a FEKO-specific subclass of UVBeam for reading FEKO ffe files.

    This class should not be interacted with directly, instead use the
    read_feko_beam method on the UVBeam class.

    """

    @copy_replace_short_description(
        UVBeam.read_feko_beam, style=DocstringStyle.NUMPYDOC
    )
    def read_feko_beam(
        self,
        filename,
        *,
        telescope_name: str,
        feed_name: str,
        feed_version: str,
        model_name: str,
        model_version: str,
        beam_type: str = "power",
        feed_pol: str | None = "x",
        feed_angle: npt.NDArray[float] | None = None,
        mount_type: str | None = "fixed",
        history: str | None,
        reference_impedance: npt.NDArray[float] | None = None,
        extra_keywords: dict | None,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
        check_auto_power: bool = True,
        fix_auto_power: bool = True,
    ):
        """Read in a FEKO ffe file."""
        # handle defaults from generic read/from_file
        if feed_pol is None:
            feed_pol = "x"
        if history is None:
            history = ""

        if isinstance(feed_pol, np.ndarray):
            if len(feed_pol.shape) > 1:
                raise ValueError("feed_pol can not be a multi-dimensional array")
            feed_pol = feed_pol.tolist()
        if isinstance(feed_pol, list | tuple):
            if len(feed_pol) != 1:
                raise ValueError("feed_pol must have exactly one element")
            feed_pol = feed_pol[0]

        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

        self.telescope_name = telescope_name
        self.feed_name = feed_name
        self.feed_version = feed_version
        self.model_name = model_name
        self.model_version = model_version
        self.history = history
        if not uvutils.history._check_history_version(
            self.history, self.pyuvdata_version_str
        ):
            self.history += self.pyuvdata_version_str

        if reference_impedance is not None:
            self.reference_impedance = float(reference_impedance)
        if extra_keywords is not None:
            self.extra_keywords = extra_keywords

        if mount_type is not None:
            self.mount_type = mount_type

        self.feed_array = np.asarray(feed_pol).reshape(-1)
        self.Nfeeds = 1

        if feed_angle is not None:
            self.feed_angle = np.asarray(feed_angle).reshape(-1)

        if beam_type == "power":
            self.Naxes_vec = 1
            self.Npols = 1

            self.polarization_array = np.array(
                [uvutils.polstr2num(feed_pol + feed_pol)]
            )
            self._set_power()
        else:
            self.Naxes_vec = 2
            self.Ncomponents_vec = 2

            self._set_efield()

        self.data_normalization = "physical"
        self.antenna_type = "simple"

        self.pixel_coordinate_system = "az_za"
        self._set_cs_params()

        with open(filename) as fh:
            # there is a blank row at the start of every frequency
            data_chunks = fh.read()[1:].split("\n\n")

        # get the first header line to get the column names
        header_line = (data_chunks[0].split("\n"))[8]
        column_names = header_line.split('"')[1::2]
        theta_col = np.where(np.array(column_names) == "Theta")[0][0]
        phi_col = np.where(np.array(column_names) == "Phi")[0][0]

        # skips the 9 lines of header text for each chunk
        data_all = [i.splitlines()[9:] for i in data_chunks]

        frequencies = [
            float(i.split("Frequency")[1].split()[1]) for i in data_chunks[:-1]
        ]
        self.Nfreqs = len(frequencies)
        self.freq_array = np.zeros(self.Nfreqs)
        self.freq_array = np.array(frequencies)
        self.bandpass_array = np.ones(self.Nfreqs)

        theta_mag_col = np.where(np.array(column_names) == "Gain(Theta)")[0][0]
        theta_real_col = np.where(np.array(column_names) == "Re(Etheta)")[0][0]
        theta_imag_col = np.where(np.array(column_names) == "Im(Etheta)")[0][0]
        phi_mag_col = np.where(np.array(column_names) == "Gain(Phi)")[0][0]
        phi_real_col = np.where(np.array(column_names) == "Re(Ephi)")[0][0]
        phi_imag_col = np.where(np.array(column_names) == "Im(Ephi)")[0][0]

        for i in range(len(self.freq_array)):
            thisdata = np.array(
                [list(map(float, data.split())) for data in data_all[i]]
            )
            if i == 0:
                # theta is always exported in degs
                theta_data = np.radians(thisdata[:, theta_col])
                # phi is always exported in degs
                phi_data = np.radians(thisdata[:, phi_col])

                theta_axis = np.sort(np.unique(theta_data))
                phi_axis = np.sort(np.unique(phi_data))

                if theta_axis.size * phi_axis.size != theta_data.size:
                    raise ValueError("Data does not appear to be on a grid")

                theta_data = theta_data.reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                phi_data = phi_data.reshape((theta_axis.size, phi_axis.size), order="F")

                if not uvutils.tools._test_array_constant_spacing(
                    theta_axis, tols=self._axis2_array.tols
                ):
                    raise ValueError(
                        "Data does not appear to be regularly gridded in zenith angle"
                    )

                if not uvutils.tools._test_array_constant_spacing(
                    phi_axis, tols=self._axis1_array.tols
                ):
                    raise ValueError(
                        "Data does not appear to be regularly gridded in azimuth angle"
                    )
                self.axis1_array = phi_axis
                self.Naxes1 = self.axis1_array.size
                self.axis2_array = theta_axis
                self.Naxes2 = self.axis2_array.size

                if self.beam_type == "power":
                    self.data_array = np.zeros(
                        self._data_array.expected_shape(self), dtype=np.float64
                    )
                else:
                    self.data_array = np.zeros(
                        self._data_array.expected_shape(self), dtype=np.complex128
                    )

            # get beam
            if self.beam_type == "power":
                name = "Gain(Total)"
                this_col = np.where(np.array(column_names) == name)[0]
                data_col = this_col.tolist()
                power_beam1 = 10 ** (thisdata[:, data_col] / 10).reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                self.data_array[0, 0, i, :, :] = power_beam1

            else:
                self.basis_vector_array = np.zeros(
                    (self.Naxes_vec, self.Ncomponents_vec, self.Naxes2, self.Naxes1)
                )
                self.basis_vector_array[0, 0, :, :] = 1.0
                self.basis_vector_array[1, 1, :, :] = 1.0

                # the E field magnitudes are taken as the sqrt of the power gain
                # so that the input power and Zo are taken into account. Taking
                # the magnitude of the E-fields from the real and imag parts
                # doesn't account for the input power & will need that information
                # from different output file of FEKO
                theta_mag = np.sqrt(10 ** (thisdata[:, theta_mag_col] / 10)).reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                phi_mag = np.sqrt(10 ** (thisdata[:, phi_mag_col] / 10)).reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                theta_phase = np.angle(
                    thisdata[:, theta_real_col] + 1j * thisdata[:, theta_imag_col]
                )
                phi_phase = np.angle(
                    thisdata[:, phi_real_col] + 1j * thisdata[:, phi_imag_col]
                )

                theta_phase = theta_phase.reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                phi_phase = phi_phase.reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )

                theta_beam = theta_mag * np.exp(1j * theta_phase)
                phi_beam = phi_mag * np.exp(1j * phi_phase)

                self.data_array[0, 0, i, :, :] = phi_beam
                self.data_array[1, 0, i, :, :] = theta_beam

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
