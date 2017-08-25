import os
import sys
import re
import numpy as np
from uvbeam import UVBeam
import utils as uvutils


class CSTBeam(UVBeam):
    """
    Defines a CST-specific subclass of UVBeam for reading CST text files.
    This class should not be interacted with directly, instead use the
    read_cst_power method on the UVBeam class.

    Assumes the structure in the simulation was symmetric under
    45 degree rotations about the z-axis.
    """

    def read_cst_power(self, filelist, frequencies=None, telescope_name=None,
                       feed_name=None, feed_version=None, model_name=None, model_version=None,
                       history='', run_check=True, run_check_acceptability=True):

        self.telescope_name = telescope_name
        self.feed_name = feed_name
        self.feed_version = feed_version
        self.model_name = model_name
        self.model_version = model_version
        self.history = history
        if not uvutils.check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        self.set_power()
        self.data_normalization = 'physical'
        self.Naxes_vec = 1
        self.antenna_type = 'simple'

        self.Nfreqs = len(filelist)
        self.Nspws = 1

        self.freq_array = []
        self.bandpass_array = []

        self.spw_array = np.array([0])
        self.polarization_array = np.array([-5, -6])
        self.Npols = len(self.polarization_array)
        self.pixel_coordinate_system = 'az_za'
        self.set_cs_params()

        for freq_i, fname in enumerate(filelist):
            data = np.loadtxt(fname, skiprows=2)

            theta_data = np.radians(data[:, 0])
            phi_data = np.radians(data[:, 1])

            theta_axis = np.sort(np.unique(theta_data))
            phi_axis = np.sort(np.unique(phi_data))
            if not theta_axis.size * phi_axis.size == theta_data.size:
                raise ValueError('Data does not appear to be on a grid')

            delta_theta = np.diff(theta_axis)
            if not np.isclose(np.max(delta_theta), np.min(delta_theta)):
                raise ValueError('Data does not appear to be regularly gridded in zenith angle')
            delta_theta = delta_theta[0]

            delta_phi = np.diff(phi_axis)
            if not np.isclose(np.max(delta_phi), np.min(delta_phi)):
                raise ValueError('Data does not appear to be regularly gridded in azimuth angle')
            delta_phi = delta_phi[0]

            if freq_i == 0:
                self.axis1_array = phi_axis
                self.Naxes1 = self.axis1_array.size
                self.axis2_array = theta_axis
                self.Naxes2 = self.axis2_array.size

                self.data_array = np.zeros(self._data_array.expected_shape(self), dtype=np.float)
            else:
                if not np.allclose(self.axis1_array, phi_axis):
                    raise ValueError('Phi values for file {f} are different '
                                     'than for previous files'.format(f=fname))
                if not np.allclose(self.axis2_array, theta_axis):
                    raise ValueError('Theta values for file {f} are different '
                                     'than for previous files'.format(f=fname))

            # reshape beam
            power_beam1 = (data[:, 2].reshape((theta_axis.size, phi_axis.size), order='F')) ** 2.

            # for second polarization, rotate by pi/2
            rot_phi = phi_axis + np.pi / 2
            rot_phi[np.where(rot_phi >= 2 * np.pi)] -= 2 * np.pi
            roll_rot_phi = np.roll(rot_phi, int((np.pi / 2) / delta_phi))
            if not np.allclose(roll_rot_phi, phi_axis):
                raise ValueError('Rotating by pi/2 failed')
            power_beam2 = np.roll(power_beam1, int((np.pi / 2) / delta_phi), axis=0)

            if frequencies is not None:
                self.freq_array.append(frequencies[freq_i])
            else:
                self.freq_array.append(self.name2freq(fname))

            self.bandpass_array.append(1.)

            self.data_array[0, 0, 0, freq_i, :, :] = power_beam1
            self.data_array[0, 0, 1, freq_i, :, :] = power_beam2

        self.freq_array = np.array(self.freq_array)
        self.bandpass_array = np.array(self.bandpass_array)
        sort_inds = np.argsort(self.freq_array)

        self.data_array[0, 0, 0, :, :] = self.data_array[0, 0, 0, sort_inds, :]
        self.data_array[0, 0, 1, :, :] = self.data_array[0, 0, 1, sort_inds, :]
        self.bandpass_array = self.bandpass_array[sort_inds]

        self.freq_array.sort()
        self.freq_array = np.broadcast_to(self.freq_array, (self.Nspws, self.Nfreqs))
        self.bandpass_array = np.broadcast_to(self.bandpass_array, (self.Nspws, self.Nfreqs))

        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

    def name2freq(self, fname):
        """
        Method to extract the frequency from the file name, assuming the file name
        contains a substring with the frequency channel in MHz that the data represents.
        e.g. "HERA_Sim_120.87MHz.txt" should yield 120.87e6

        Args:
            fname: filename (string)

        Returns:
            extracted frequency
        """
        fi = fname.find('MHz')

        return float(re.findall('\d+?\.?\d*', fname[:fi])[0]) * 1e6
