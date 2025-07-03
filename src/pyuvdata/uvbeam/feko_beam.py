# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading beam FEKO files."""

import os
import warnings

import numpy as np

from .. import utils as uvutils
from .uvbeam import UVBeam

__all__ = ["FEKOBeam"]


class FEKOBeam(UVBeam):
    """
    Defines a FEKO-specific subclass of UVBeam for reading FEKO ffe files.

    This class should not be interacted with directly, instead use the
    read_feko_beam method on the UVBeam class.

    """

    def nametopol(self, fname):
        """
        Get name of the y file  from the main filename.

        Parameters
        ----------
        fname : str
            Filename to parse.

        Returns
        -------
        str
            New file name.
        """
        fnew = fname.replace("x", "y")

        return fnew

    def read_feko_beam(
        self,
        filename,
        *,
        beam_type="power",
        feed_pol="x",
        feed_angle=None,
        mount_type=None,
        frequency=None,
        telescope_name=None,
        feed_name=None,
        feed_version=None,
        model_name=None,
        model_version=None,
        history="",
        reference_impedance=None,
        extra_keywords=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        check_auto_power=True,
        fix_auto_power=True,
    ):
        """
        Read in data from a FEKO ffe file.

        Parameters
        ----------
        filename : str
            The FEKO file to read from.
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        feed_pol : str
            The feed polarization that the files corresponds to, e.g. x, y, r or l.
            Defaults to 'x'.
        feed_angle : float
            Position angle of the feed, units of radians. A feed angle of 0 is
            typically oriented toward zenith for steerable antennas, otherwise toward
            north for fixed antennas (e.g., HERA, LWA). More details on this can be
            found on the "Conventions" page of the docs.
        mount_type : str
            Antenna mount type, which describes the optics of the antenna in question.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination)
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.
        frequency : float or list of float
            The frequency or list of frequencies corresponding to the filename(s).
            This is assumed to be in the same order as the files.
            If not passed, the code attempts to parse it from the filenames.
        telescope_name : str
            The name of the telescope corresponding to the filename(s).
        feed_name : str
            The name of the feed corresponding to the filename(s).
        feed_version : str
            The version of the feed corresponding to the filename(s).
        model_name : str
            The name of the model corresponding to the filename(s).
        model_version : str
            The version of the model corresponding to the filename(s).
        history : str
            A string detailing the history of the filename(s).
        reference_impedance : float, optional
            The reference impedance of the model(s).
        extra_keywords : dict, optional
            A dictionary containing any extra_keywords.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as
            required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after reading in the file.
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.
        """
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

        with open(filename) as out_file:
            line = out_file.readlines()[9].strip()  # Get the line with column names
            column_names = line.split('"')[1::2]

        theta_col = np.where(np.array(column_names) == "Theta")[0][0]
        phi_col = np.where(np.array(column_names) == "Phi")[0][0]

        with open(filename) as fh:
            data_chunks = fh.read()[1:].split(
                "\n\n"
            )  ## avoiding the row=1; there is a blank row at the start of every file
        data_all = [
            i.splitlines()[9:] for i in data_chunks
        ]  ## skips the 9 lines of text in each chunk

        filename2 = self.nametopol(filename)
        with open(filename2) as fh:
            data_chunks = fh.read()[1:].split(
                "\n\n"
            )  ## avoiding the row=1;there is a blank row at the start of every file
        data_all2 = [
            i.splitlines()[9:] for i in data_chunks
        ]  ## skips the 9 lines of text in each chunk

        frequency = [
            float(i.split("Frequency")[1].split()[1]) for i in data_chunks[:-1]
        ]
        self.Nfreqs = len(frequency)
        self.freq_array = np.zeros(self.Nfreqs)
        self.freq_array = np.array(frequency)
        self.bandpass_array = np.zeros(self.Nfreqs)

        data_each = np.zeros((len(self.freq_array), np.shape(data_all[0])[0], 9))
        data_each2 = np.zeros((len(self.freq_array), np.shape(data_all2[0])[0], 9))

        for i in range(len(self.freq_array)):
            data_each[i, :, :] = np.array(
                [list(map(float, data.split())) for data in data_all[i]]
            )
            data_each2[i, :, :] = np.array(
                [list(map(float, data.split())) for data in data_all2[i]]
            )
            if i == 0:
                theta_data = np.radians(
                    data_each[i, :, theta_col]
                )  ## theta is always exported in degs
                phi_data = np.radians(
                    data_each[i, :, phi_col]
                )  ## phi is always exported in degs

                theta_axis = np.sort(np.unique(theta_data))
                phi_axis = np.sort(np.unique(phi_data))

                if not theta_axis.size * phi_axis.size == theta_data.size:
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
                    # type depends on whether cross pols are present
                    # (if so, complex, else float)
                    if complex in self._data_array.expected_type:
                        dtype_use = np.complex128
                    else:
                        dtype_use = np.float64
                    self.data_array = np.zeros(
                        self._data_array.expected_shape(self), dtype=dtype_use
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
                power_beam1 = 10 ** (data_each[i, :, data_col] / 10).reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                self.data_array[0, 0, i, :, :] = power_beam1

            else:
                self.basis_vector_array = np.zeros(
                    (self.Naxes_vec, self.Ncomponents_vec, self.Naxes2, self.Naxes1)
                )
                self.basis_vector_array[0, 0, :, :] = 1.0
                self.basis_vector_array[1, 1, :, :] = 1.0

                theta_mag_col = np.where(np.array(column_names) == "Gain(Theta)")[0][0]
                theta_real_col = np.where(np.array(column_names) == "Re(Etheta)")[0][0]
                theta_imag_col = np.where(np.array(column_names) == "Im(Etheta)")[0][0]
                phi_mag_col = np.where(np.array(column_names) == "Gain(Phi)")[0][0]
                phi_real_col = np.where(np.array(column_names) == "Re(Ephi)")[0][0]
                phi_imag_col = np.where(np.array(column_names) == "Im(Ephi)")[0][0]

                theta_mag = np.sqrt(
                    10 ** (data_each[i, :, theta_mag_col] / 10)
                ).reshape((theta_axis.size, phi_axis.size), order="F")
                phi_mag = np.sqrt(10 ** (data_each[i, :, phi_mag_col] / 10)).reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                # theta_phase = np.angle(
                #    data_each[i, :, theta_real_col] + 1j * data_c1[:, theta_imag_col]
                # )
                # phi_phase = np.angle(
                #    data_each[i, :, phi_real_col] + 1j * data_c1[:, phi_imag_col]
                # )
                theta_phase = np.angle(
                    data_each[i, :, theta_real_col]
                    + 1j * data_each[i, :, theta_imag_col]
                )
                phi_phase = np.angle(
                    data_each[i, :, phi_real_col] + 1j * data_each[i, :, phi_imag_col]
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

                theta_mag2 = np.sqrt(
                    10 ** (data_each2[i, :, theta_mag_col] / 10)
                ).reshape((theta_axis.size, phi_axis.size), order="F")
                phi_mag2 = np.sqrt(10 ** (data_each2[i, :, phi_mag_col] / 10)).reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                theta_phase2 = np.angle(
                    data_each2[i, :, theta_real_col]
                    + 1j * data_each2[i, :, theta_imag_col]
                )
                phi_phase2 = np.angle(
                    data_each2[i, :, phi_real_col] + 1j * data_each2[i, :, phi_imag_col]
                )

                theta_phase2 = theta_phase2.reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )
                phi_phase2 = phi_phase2.reshape(
                    (theta_axis.size, phi_axis.size), order="F"
                )

                theta_beam2 = theta_mag2 * np.exp(1j * theta_phase2)
                phi_beam2 = phi_mag2 * np.exp(1j * phi_phase2)

                self.data_array[0, 1, i, :, :] = phi_beam2
                self.data_array[1, 1, i, :, :] = theta_beam2

        self.bandpass_array[0] = 1

        if frequency is None:
            warnings.warn(
                f"No frequency provided. Detected frequency is: {self.freq_array} Hz"
            )

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
