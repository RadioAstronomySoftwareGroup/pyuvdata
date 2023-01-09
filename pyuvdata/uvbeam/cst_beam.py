# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading beam CST files."""
import os
import re
import warnings
from typing import Callable

import numpy as np

from .. import utils as uvutils
from .uvbeam import UVBeam

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
        use_future_array_shapes=False,
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
        check_auto_power=True,
        fix_auto_power=True,
        thetaphi_to_azza_fnc: Callable | None = None,
    ):
        """
        Read in data from a cst file.

        Parameters
        ----------
        filename : str
            The cst file to read from.
        beam_type : str
            What beam_type to read in ('power' or 'efield').
        use_future_array_shapes : bool
            Option to convert to the future planned array shapes before the changes go
            into effect by removing the spectral window axis.
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
        check_auto_power : bool
            For power beams, check whether the auto polarization beams have non-zero
            imaginary values in the data_array (which should not mathematically exist).
        fix_auto_power : bool
            For power beams, if auto polarization beams with imaginary values are found,
            fix those values so that they are real-only in data_array.
        thetaphi_to_azza_fnc : callable
            Since CST can output theta/phi coordinates in a number of conventions, you
            can provide a function to convert from the input theta/phi to azimuth/
            zenith-angle, as required by UVBeam. The signature of the function should
            be ``az, za = fnc(theta, phi)``. Theta and Phi will be in radians.

        """
        # update filename attribute
        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

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
        self.freq_array = np.zeros((1, self.Nfreqs))
        self.bandpass_array = np.zeros((1, self.Nfreqs))

        if not use_future_array_shapes:
            self.Nspws = 1
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

        theta_col = column_names.index("theta")
        phi_col = column_names.index("phi")

        for icol, un in enumerate(units):
            if "deg" in un:
                data[:, icol] = np.radians(data[:, icol])

        theta_data = data[:, theta_col]
        phi_data = data[:, phi_col]

        if thetaphi_to_azza_fnc is not None:
            phi_data, theta_data = thetaphi_to_azza_fnc(theta_data, phi_data)

        if theta_data.max() > np.pi * 1.00001 or theta_data.min() < 0:
            raise ValueError(
                "Some zenith-angles are outside (0, pi). Range = "
                f"{theta_data.min(), theta_data.max()}. "
                "Check your thetaphi_to_azza_fnc."
            )

        if phi_data.max() > phi_data.min() + 2 * np.pi:
            raise ValueError(
                "The azimuthal angle has a range greater than 2pi: "
                f"{phi_data.min(), phi_data.max()}. "
                "Check your thetaphi_to_azza_fnc."
            )

        phi_data %= 2 * np.pi

        theta_axis = np.sort(np.unique(theta_data))
        phi_axis = np.sort(np.unique(phi_data))

        # Get the order of the data in which phi increases on the inner loop and theta
        # on the outer.
        sort_idx = np.lexsort((phi_data, theta_data))
        theta_data = theta_data[sort_idx]
        phi_data = phi_data[sort_idx]
        data = data[sort_idx]
        data[:, theta_col] = theta_data
        data[:, phi_col] = phi_data

        if self.beam_type == "power":
            data_col_enum = ["abs(e)", "abs(v)"]
            data_col = []
            for name in data_col_enum:
                this_col = np.where(np.array(column_names) == name)[0]
                if this_col.size > 0:
                    data_col = data_col + this_col.tolist()
            if len(data_col) == 0:
                raise ValueError(f"No power column found in file: {filename}")
            elif len(data_col) > 1:
                raise ValueError(
                    f"Multiple possible power columns found in file: {filename}"
                )
            data_col = data_col[0]

        tupled = list(zip(theta_data, phi_data))
        n = len(tupled)
        new_data = np.zeros((len(theta_axis) * len(phi_axis), data.shape[1]))
        i = 0
        idx = -1
        for za in theta_axis:
            for az in phi_axis:
                if (idx + 1) < n and (za, az) == tupled[idx + 1]:
                    # best case scenario is everything is in grid already.
                    # For most beams, the large majority of iterations will hit this.
                    idx = idx + 1
                    new_data[i] = data[idx]
                else:
                    try:
                        # Find the index of these coordinates
                        # There may be multiple indices with the same az/za coords,
                        # because of poles -- here we just take the first one.
                        # NOTE: this is specifically *az/za* coordinates -- so even
                        # though (az, 0) is the same *location* for all values of az,
                        # it is represented by different coordinates, and we therefore
                        # will take each of them, not just the first (this is important
                        # because the phase
                        # differs for different azimuth values at zenith). Conversely,
                        # for an input coordinate system that represents some location
                        # more than once (which is bound to happen for any particular
                        # choice of regular grid in spherical coordinates), we need only
                        # use one of the coordinates (the others will by necessity
                        # contain the same information).
                        idx = tupled.index((za, az))
                        new_data[i] = data[idx]
                    except ValueError as err:
                        # These co-ordinates do not exist.

                        # However, if the az/za coordinates don't exist and we're at a
                        # pole, we can potentially deduce the data from other
                        # coordinates at the pole. In general, the beam at (az, 0) is
                        # the conjugate of the beam at (az + pi, 0). That is, when
                        # tracing a meridian through the pole, the amplitude is
                        # perfectly smooth (so all az at the zenith have the same
                        # amplitude), but the phase "flips" right at the pole as you
                        # pass through.
                        if za == 0 or za == np.pi:
                            if beam_type == "power":
                                # We only care about abs() columns
                                zero_idx = np.where(theta_data == 0)[0][0]
                                new_data[i, data_col] = data[zero_idx, data_col]
                            else:
                                # We care about all the columns.
                                idx = np.where(
                                    np.isclose(theta_data, za)
                                    & np.isclose(phi_data, (az + np.pi) % (2 * np.pi))
                                )[0]
                                if len(idx) == 0:
                                    raise ValueError(
                                        f"No coordinate data for za={za} with az={az}"
                                    ) from err
                                idx = idx[0]

                                for icol, name in enumerate(column_names):
                                    if "phase" in name:  # flip the phase.
                                        new_data[i, icol] = (
                                            data[idx, icol] + np.pi
                                        ) % (2 * np.pi)
                                    elif name == "theta":
                                        new_data[i, icol] = za
                                    elif name == "phi":
                                        new_data[i, icol] = az
                                    else:
                                        new_data[i, icol] = data[idx, icol]
                        else:
                            raise err

                i += 1

        delta_phi = phi_axis[1] - phi_axis[0]
        data = new_data.reshape((theta_axis.size, phi_axis.size, -1)).transpose(
            (2, 0, 1)
        )

        theta_data = data[theta_col]
        phi_data = data[phi_col]

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
            power_beam1 = data[data_col] ** 2.0

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

            theta_mag = data[theta_mag_col]
            phi_mag = data[phi_mag_col]
            theta_phase = data[theta_phase_col]
            phi_phase = data[phi_phase_col]

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

        if use_future_array_shapes:
            self.use_future_array_shapes()

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
