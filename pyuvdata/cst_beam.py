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
    read_cst_beam method on the UVBeam class.

    Assumes the structure in the simulation was symmetric under
    45 degree rotations about the z-axis.
    """

    def read_cst_beam(self, filelist, beam_type='power', frequencies=None, telescope_name=None,
                      feed_name=None, feed_version=None, model_name=None, model_version=None,
                      history='', run_check=True, run_check_acceptability=True):

        """
        Read in data from a cst file.

        Args:
            filename: The cst file or list of files to read from.
            beam_type: what beam_type to read in ('power' or 'efield'). Defaults to 'power'.
            frequencies: the frequency or list of frequencies corresponding to the filename(s).
                If not passed, the code attempts to parse it from the filenames.
            telescope_name: the name of the telescope corresponding to the filename(s).
            feed_name: the name of the feed corresponding to the filename(s).
            feed_version: the version of the feed corresponding to the filename(s).
            model_name: the name of the model corresponding to the filename(s).
            model_version: the version of the model corresponding to the filename(s).
            history: A string detailing the history of the filename(s).
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        self.telescope_name = telescope_name
        self.feed_name = feed_name
        self.feed_version = feed_version
        self.model_name = model_name
        self.model_version = model_version
        self.history = history
        if not uvutils.check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        if beam_type is 'power':
            self.set_power()
            self.Naxes_vec = 1
            self.polarization_array = np.array([-5, -6])
            self.Npols = len(self.polarization_array)
        else:
            self.set_efield()
            self.Naxes_vec = 2
            self.feed_array = np.array(['x', 'y'])
            self.Nfeeds = len(self.feed_array)

        self.data_normalization = 'physical'
        self.antenna_type = 'simple'

        self.Nfreqs = len(filelist)
        self.Nspws = 1

        self.freq_array = []
        self.bandpass_array = []

        self.spw_array = np.array([0])
        self.pixel_coordinate_system = 'az_za'
        self.set_cs_params()

        for freq_i, fname in enumerate(filelist):
            out_file = open(fname, 'r')
            line = out_file.readline().strip()  # Get the first line
            out_file.close()
            raw_names = line.split(']')
            raw_names = [raw_name for raw_name in raw_names if not raw_name == '']
            column_names = []
            units = []
            for raw_name in raw_names:
                column_name, unit = tuple(raw_name.split('['))
                column_names.append(''.join(column_name.lower().split(' ')))
                units.append(unit.lower().strip())

            data = np.loadtxt(fname, skiprows=2)

            theta_col = np.where(np.array(column_names) == 'theta')[0][0]
            phi_col = np.where(np.array(column_names) == 'phi')[0][0]

            if 'deg' in units[theta_col]:
                theta_data = np.radians(data[:, theta_col])
            else:
                theta_data = data[:, theta_col]
            if 'deg' in units[phi_col]:
                phi_data = np.radians(data[:, phi_col])
            else:
                phi_data = data[:, phi_col]

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

                if self.beam_type == 'power':
                    self.data_array = np.zeros(self._data_array.expected_shape(self), dtype=np.float)
                else:
                    self.data_array = np.zeros(self._data_array.expected_shape(self), dtype=np.complex)
            else:
                if not np.allclose(self.axis1_array, phi_axis):
                    raise ValueError('Phi values for file {f} are different '
                                     'than for previous files'.format(f=fname))
                if not np.allclose(self.axis2_array, theta_axis):
                    raise ValueError('Theta values for file {f} are different '
                                     'than for previous files'.format(f=fname))

            if frequencies is not None:
                self.freq_array.append(frequencies[freq_i])
            else:
                self.freq_array.append(self.name2freq(fname))

            # for second polarization, rotate by pi/2
            rot_phi = phi_axis + np.pi / 2
            rot_phi[np.where(rot_phi >= 2 * np.pi)] -= 2 * np.pi
            roll_rot_phi = np.roll(rot_phi, int((np.pi / 2) / delta_phi))
            if not np.allclose(roll_rot_phi, phi_axis):
                raise ValueError('Rotating by pi/2 failed')

            # get beam
            if beam_type is 'power':
                data_col = np.where(np.array(column_names) == 'abs(v)')[0][0]
                power_beam1 = data[:, data_col].reshape((theta_axis.size, phi_axis.size), order='F') ** 2.

                self.data_array[0, 0, 0, freq_i, :, :] = power_beam1

                # rotate by pi/2 for second polarization
                power_beam2 = np.roll(power_beam1, int((np.pi / 2) / delta_phi), axis=0)
                self.data_array[0, 0, 1, freq_i, :, :] = power_beam2
            else:
                self.basis_vector_array = np.zeros((self.Naxes_vec, 2, self.Naxes2, self.Naxes1))
                self.basis_vector_array[0, 0, :, :] = 1.0
                self.basis_vector_array[1, 1, :, :] = 1.0

                theta_mag_col = np.where(np.array(column_names) == 'abs(theta)')[0][0]
                theta_phase_col = np.where(np.array(column_names) == 'phase(theta)')[0][0]
                phi_mag_col = np.where(np.array(column_names) == 'abs(phi)')[0][0]
                phi_phase_col = np.where(np.array(column_names) == 'phase(phi)')[0][0]

                theta_mag = data[:, theta_mag_col].reshape((theta_axis.size, phi_axis.size), order='F')
                phi_mag = data[:, phi_mag_col].reshape((theta_axis.size, phi_axis.size), order='F')
                if 'deg' in units[theta_phase_col]:
                    theta_phase = np.radians(data[:, theta_phase_col])
                else:
                    theta_phase = data[:, theta_phase_col]
                if 'deg' in units[phi_phase_col]:
                    phi_phase = np.radians(data[:, phi_phase_col])
                else:
                    phi_phase = data[:, phi_phase_col]
                theta_phase = theta_phase.reshape((theta_axis.size, phi_axis.size), order='F')
                phi_phase = theta_phase.reshape((theta_axis.size, phi_axis.size), order='F')

                theta_beam = theta_mag * np.exp(1j * theta_phase)
                phi_beam = theta_mag * np.exp(1j * theta_phase)

                self.data_array[0, 0, 0, freq_i, :, :] = phi_beam
                self.data_array[1, 0, 0, freq_i, :, :] = theta_beam

                # rotate by pi/2 for second polarization
                theta_beam2 = np.roll(theta_beam, int((np.pi / 2) / delta_phi), axis=0)
                phi_beam2 = np.roll(phi_beam, int((np.pi / 2) / delta_phi), axis=0)
                self.data_array[0, 0, 1, freq_i, :, :] = theta_beam2
                self.data_array[1, 0, 1, freq_i, :, :] = phi_beam2

            self.bandpass_array.append(1.)

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
