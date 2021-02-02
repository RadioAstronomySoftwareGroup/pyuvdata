# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading beam CST files."""
import re
import warnings

import numpy as np

from .uvbeam import UVBeam
from .. import utils as uvutils

__all__ = ["CSTBeam"]


class CSTBeam(UVBeam):
    """
    Defines a CST-specific subclass of UVBeam for reading CST text files.

    This class should not be interacted with directly, instead use the
    read_cst_beam method on the UVBeam class.

    """

    def name2freq(self, fname):
        """
        Extract frequency from the filename.

        Assumes the file name contains a substring with the frequency channel
        in MHz that the data represents.
        e.g. "HERA_Sim_120.87MHz.txt" should yield 120.87e6

        Parameters
        ----------
        fname : str
            Filename to parse.

        Returns
        -------
        float
            Frequency extracted from filename in Hz.
        """
        fi = fname.rfind("Hz")
        frequency = float(re.findall(r"\d*\.\d+|\d+", fname[:fi])[-1])

        si_prefix = fname[fi - 1]
        si_dict = {"k": 1e3, "M": 1e6, "G": 1e9}
        if si_prefix in si_dict.keys():
            frequency = frequency * si_dict[si_prefix]

        return frequency

    def read_cst_beam(
        self,
        filename,
        beam_type="power",
        feed_pol="x",
        rotate_pol=True,
        frequency=None,
        telescope_name=None,
        feed_name=None,
        feed_version=None,
        model_name=None,
        model_version=None,
        history="",
        x_orientation=None,
        reference_impedance=None,
        extra_keywords=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in data from a cst file.

        Parameters
        ----------
        filename : str
            The cst file to read from.
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        feed_pol : str
            The feed or polarization or list of feeds or polarizations the
            files correspond to.
            Defaults to 'x' (meaning x for efield or xx for power beams).
        rotate_pol : bool
            If True, assume the structure in the simulation is symmetric under
            90 degree rotations about the z-axis (so that the y polarization can be
            constructed by rotating the x polarization or vice versa).
            Default: True if feed_pol is a single value or a list with all
            the same values in it, False if it is a list with varying values.
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
        x_orientation : str, optional
            Orientation of the physical dipole corresponding to what is
            labelled as the x polarization. Options are "east" (indicating
            east/west orientation) and "north" (indicating north/south orientation)
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

        """
        self.telescope_name = telescope_name
        self.feed_name = feed_name
        self.feed_version = feed_version
        self.model_name = model_name
        self.model_version = model_version
        self.history = history
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if x_orientation is not None:
            self.x_orientation = x_orientation
        if reference_impedance is not None:
            self.reference_impedance = float(reference_impedance)
        if extra_keywords is not None:
            self.extra_keywords = extra_keywords

        if beam_type == "power":
            self.Naxes_vec = 1

            if feed_pol == "x":
                feed_pol = "xx"
            elif feed_pol == "y":
                feed_pol = "yy"

            if rotate_pol:
                rot_pol_dict = {"xx": "yy", "yy": "xx", "xy": "yx", "yx": "xy"}
                pol2 = rot_pol_dict[feed_pol]
                self.polarization_array = np.array(
                    [uvutils.polstr2num(feed_pol), uvutils.polstr2num(pol2)]
                )
            else:
                self.polarization_array = np.array([uvutils.polstr2num(feed_pol)])

            self.Npols = len(self.polarization_array)
            self._set_power()
        else:
            self.Naxes_vec = 2
            self.Ncomponents_vec = 2
            if rotate_pol:
                if feed_pol == "x":
                    self.feed_array = np.array(["x", "y"])
                else:
                    self.feed_array = np.array(["y", "x"])
            else:
                if feed_pol == "x":
                    self.feed_array = np.array(["x"])
                else:
                    self.feed_array = np.array(["y"])
            self.Nfeeds = self.feed_array.size
            self._set_efield()

        self.data_normalization = "physical"
        self.antenna_type = "simple"

        self.Nfreqs = 1
        self.Nspws = 1
        self.freq_array = np.zeros((self.Nspws, self.Nfreqs))
        self.bandpass_array = np.zeros((self.Nspws, self.Nfreqs))

        self.spw_array = np.array([0])
        self.pixel_coordinate_system = "az_za"
        self._set_cs_params()

        out_file = open(filename, "r")
        line = out_file.readline().strip()  # Get the first line
        out_file.close()
        raw_names = line.split("]")
        raw_names = [raw_name for raw_name in raw_names if not raw_name == ""]
        column_names = []
        units = []
        for raw_name in raw_names:
            column_name, unit = tuple(raw_name.split("["))
            column_names.append("".join(column_name.lower().split(" ")))
            units.append(unit.lower().strip())

        data = np.loadtxt(filename, skiprows=2)

        theta_col = np.where(np.array(column_names) == "theta")[0][0]
        phi_col = np.where(np.array(column_names) == "phi")[0][0]

        if "deg" in units[theta_col]:
            theta_data = np.radians(data[:, theta_col])
        else:
            theta_data = data[:, theta_col]
        if "deg" in units[phi_col]:
            phi_data = np.radians(data[:, phi_col])
        else:
            phi_data = data[:, phi_col]

        theta_axis = np.sort(np.unique(theta_data))
        phi_axis = np.sort(np.unique(phi_data))
        if not theta_axis.size * phi_axis.size == theta_data.size:
            raise ValueError("Data does not appear to be on a grid")

        theta_data = theta_data.reshape((theta_axis.size, phi_axis.size), order="F")
        phi_data = phi_data.reshape((theta_axis.size, phi_axis.size), order="F")

        delta_theta = np.diff(theta_axis)
        if not np.isclose(np.max(delta_theta), np.min(delta_theta)):
            raise ValueError(
                "Data does not appear to be regularly gridded in zenith angle"
            )
        delta_theta = delta_theta[0]

        delta_phi = np.diff(phi_axis)
        if not np.isclose(np.max(delta_phi), np.min(delta_phi)):
            raise ValueError(
                "Data does not appear to be regularly gridded in azimuth angle"
            )
        delta_phi = delta_phi[0]

        self.axis1_array = phi_axis
        self.Naxes1 = self.axis1_array.size
        self.axis2_array = theta_axis
        self.Naxes2 = self.axis2_array.size

        if self.beam_type == "power":
            # type depends on whether cross pols are present
            # (if so, complex, else float)
            self.data_array = np.zeros(
                self._data_array.expected_shape(self),
                dtype=self._data_array.expected_type,
            )
        else:
            self.data_array = np.zeros(
                self._data_array.expected_shape(self), dtype=np.complex128
            )

        if frequency is not None:
            self.freq_array[0] = frequency
        else:
            self.freq_array[0] = self.name2freq(filename)

        if rotate_pol:
            # for second polarization, rotate by pi/2
            rot_phi = phi_data + np.pi / 2
            rot_phi[np.where(rot_phi >= 2 * np.pi)] -= 2 * np.pi
            roll_rot_phi = np.roll(rot_phi, int((np.pi / 2) / delta_phi), axis=1)
            if not np.allclose(roll_rot_phi, phi_data):
                raise ValueError("Rotating by pi/2 failed")

            # theta is not affected by the rotation

        # get beam
        if self.beam_type == "power":

            data_col_enum = ["abs(e)", "abs(v)"]
            data_col = []
            for name in data_col_enum:
                this_col = np.where(np.array(column_names) == name)[0]
                if this_col.size > 0:
                    data_col = data_col + this_col.tolist()
            if len(data_col) == 0:
                raise ValueError("No power column found in file: {}".format(filename))
            elif len(data_col) > 1:
                raise ValueError(
                    "Multiple possible power columns found in file: {}".format(filename)
                )
            data_col = data_col[0]
            power_beam1 = (
                data[:, data_col].reshape((theta_axis.size, phi_axis.size), order="F")
                ** 2.0
            )

            self.data_array[0, 0, 0, 0, :, :] = power_beam1

            if rotate_pol:
                # rotate by pi/2 for second polarization
                power_beam2 = np.roll(power_beam1, int((np.pi / 2) / delta_phi), axis=1)
                self.data_array[0, 0, 1, 0, :, :] = power_beam2
        else:
            self.basis_vector_array = np.zeros(
                (self.Naxes_vec, self.Ncomponents_vec, self.Naxes2, self.Naxes1)
            )
            self.basis_vector_array[0, 0, :, :] = 1.0
            self.basis_vector_array[1, 1, :, :] = 1.0

            theta_mag_col = np.where(np.array(column_names) == "abs(theta)")[0][0]
            theta_phase_col = np.where(np.array(column_names) == "phase(theta)")[0][0]
            phi_mag_col = np.where(np.array(column_names) == "abs(phi)")[0][0]
            phi_phase_col = np.where(np.array(column_names) == "phase(phi)")[0][0]

            theta_mag = data[:, theta_mag_col].reshape(
                (theta_axis.size, phi_axis.size), order="F"
            )
            phi_mag = data[:, phi_mag_col].reshape(
                (theta_axis.size, phi_axis.size), order="F"
            )
            if "deg" in units[theta_phase_col]:
                theta_phase = np.radians(data[:, theta_phase_col])
            else:
                theta_phase = data[:, theta_phase_col]
            if "deg" in units[phi_phase_col]:
                phi_phase = np.radians(data[:, phi_phase_col])
            else:
                phi_phase = data[:, phi_phase_col]
            theta_phase = theta_phase.reshape(
                (theta_axis.size, phi_axis.size), order="F"
            )
            phi_phase = phi_phase.reshape((theta_axis.size, phi_axis.size), order="F")

            theta_beam = theta_mag * np.exp(1j * theta_phase)
            phi_beam = phi_mag * np.exp(1j * phi_phase)

            self.data_array[0, 0, 0, 0, :, :] = phi_beam
            self.data_array[1, 0, 0, 0, :, :] = theta_beam

            if rotate_pol:
                # rotate by pi/2 for second polarization
                theta_beam2 = np.roll(theta_beam, int((np.pi / 2) / delta_phi), axis=1)
                phi_beam2 = np.roll(phi_beam, int((np.pi / 2) / delta_phi), axis=1)
                self.data_array[0, 0, 1, 0, :, :] = phi_beam2
                self.data_array[1, 0, 1, 0, :, :] = theta_beam2

        self.bandpass_array[0] = 1

        if frequency is None:
            warnings.warn(
                "No frequency provided. Detected frequency is: "
                "{freqs} Hz".format(freqs=self.freq_array)
            )

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
