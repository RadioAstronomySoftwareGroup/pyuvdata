# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading beam CST files."""
import os
import re
import warnings

import numpy as np

from .. import utils as uvutils
from .uvbeam import UVBeam, _future_array_shapes_warning

__all__ = ["FEKOBeam"]

class FEKOBeam(UVBeam):
    """
    Defines a FEKO-specific subclass of UVBeam for reading FEKO ffe files.

    This class should not be interacted with directly, instead use the
    read_feko_beam method on the UVBeam class.

    """
 def read_feko_beam(
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
    ):
        """
        Read in data from a FEKO ffe file.

        Parameters
        ----------
        filename : str
            The FEKO file to read from.
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


    out_file = open(feko_file,"r")
    line = out_file.readlines()[9].strip() # Get the line with column names
    out_file.close()
    column_names = line.split("\"")[1::2]
    
    with open(feko_file, 'r') as fh:
        data_chunks = fh.read().split('\n\n')


    data = data_chunks[1].splitlines()[9:]
    data_c1 = np.array([list(map(float,data.split())) for data in data])

    theta_col = np.where(np.array(column_names) == "Theta")[0][0]
    phi_col = np.where(np.array(column_names) == "Phi")[0][0]

    theta_data = np.radians(data_c1[:, theta_col])  ## theta is always exported in degs
    phi_data = np.radians(data_c1[:, phi_col])      ## phi is always exported in degs
    

    theta_axis = np.sort(np.unique(theta_data))
    phi_axis = np.sort(np.unique(phi_data))


    if not theta_axis.size * phi_axis.size == theta_data.size:
        raise ValueError("Data does not appear to be on a grid")

    theta_data = theta_data.reshape((theta_axis.size, phi_axis.size), order="F")
    phi_data = phi_data.reshape((theta_axis.size, phi_axis.size), order="F")

    if not uvutils._test_array_constant_spacing(theta_axis, self._axis2_array.tols):
        raise ValueError(
            "Data does not appear to be regularly gridded in zenith angle"
        )

    if not uvutils._test_array_constant_spacing(phi_axis, self._axis1_array.tols):
        raise ValueError(
            "Data does not appear to be regularly gridded in azimuth angle"
        )
    delta_phi = phi_axis[1] - phi_axis[0]

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
        self.freq_array[0] = [float(i.split('Frequency')[1].split()[1]) for i in data_chunks[:-1]]

    if rotate_pol:
        # for second polarization, rotate by pi/2
        rot_phi = phi_data + np.pi / 2
        rot_phi[np.where(rot_phi >= 2 * np.pi)] -= 2 * np.pi
        roll_rot_phi = np.roll(rot_phi, int((np.pi / 2) / delta_phi), axis=1)
        if not np.allclose(roll_rot_phi, phi_data):
            raise ValueError("Rotating by pi/2 failed")

    # get beam
    if self.beam_type == "power":
        name = "Gain(Total)"
        this_col = np.where(np.array(column_names) == name)[0]
        data_col = this_col.tolist()
        power_beam1 =  10**(data_c1[:,data_col]/10).reshape((theta_axis.size, phi_axis.size), order="F")

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

        theta_mag_col = np.where(np.array(column_names) == "Gain(Theta)")[0][0]
        theta_real_col = np.where(np.array(column_names) == "Re(Etheta)")[0][0]
        theta_imag_col = np.where(np.array(column_names) == "Im(Etheta)")[0][0]
        phi_mag_col = np.where(np.array(column_names) == "Gain(Phi)")[0][0]
        phi_real_col = np.where(np.array(column_names) == "Re(Ephi)")[0][0]
        phi_imag_col = np.where(np.array(column_names) == "Im(Ephi)")[0][0]

        theta_mag = np.sqrt(10**(data_c1[:, theta_mag_col]/10)).reshape(
            (theta_axis.size, phi_axis.size), order="F"
        )
        phi_mag = np.sqrt(10**(data_c1[:, phi_mag_col]/10)).reshape(
            (theta_axis.size, phi_axis.size), order="F"
        )
        theta_phase = np.angle(data_c1[:, theta_real_col] + 1j * data_c1[:, theta_imag_col])
        phi_phase = np.angle(data_c1[:, phi_real_col] +1j *data_c1[:, phi_imag_col])
        
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

    if use_future_array_shapes:
        self.use_future_array_shapes()
    else:
        warnings.warn(_future_array_shapes_warning, DeprecationWarning)

    if run_check:
        self.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            check_auto_power=check_auto_power,
            fix_auto_power=fix_auto_power,
        )
