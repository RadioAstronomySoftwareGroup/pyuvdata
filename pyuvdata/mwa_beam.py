# -- mode: python; coding: utf-8 --
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
from __future__ import absolute_import, division, print_function
import numpy as np
import warnings
from . import UVBeam


class MWABeam(UVBeam):
    """
    Defines an MWA-specific subclass of UVBeam for representing MWA beams.

    This class should not be interacted with directly, instead use the
    read_mwa_beam method on the UVBeam class.

    This is based on https://github.com/MWATelescope/mwa_pb/ but we don’t import
    that module because it’s not python 3 compatible
    """

    def _read_metadata(h5filepath):
        """
        Get metadata (frequencies, polarizations, dipole numbers) from input file

        Parameters
        ----------
        h5filepath : str
            path to input h5 file containing the MWA full embedded element spherical
            harmonic modes.

        Returns
        -------
        freqs : array of int
            Frequencies in Hz present in the file.
        pol_names : list of str
            Polarizations present in the file.
        dipoles :
            Dipoles present in the file.
        """
        pol_names = set()
        dipole_names = set()
        freqs_hz = set()
        other_names = []
        with h5py.File(h5filepath, 'r') as file:
            for name in file.keys():
                if name.startswith('X') or name.startswith('Y'):
                    pol = name[0]
                    dipole, freq = name[1:].split('_')
                    pol_names.add(pol)
                    dipole_names.add(dipole)
                    freqs_hz.add(np.int(freq))
                else:
                    other_names.append(name)

        pol_names = sorted(list(pol_names))

        dipole_names = np.array(sorted(list(dipole_names)))

        freqs_hz = np.array(sorted(list(freqs)))

        return freqs_hz, pol_names, dipole_names

    def _get_beam_modes(h5filepath, freqs_hz, pol_names, dipole_names, delays):
        """
        Get beam modes from input file

        Parameters
        ----------
        h5filepath : str
            path to input h5 file containing the MWA full embedded element spherical
            harmonic modes.

        Returns
        -------
        freqs : array of int
            Frequencies in Hz present in the file.
        pols : list of str
            Polarizations present in the file.
        dipoles :
            Dipoles present in the file.
        """
        beam_modes = {}
        for pol_i, pol in enumerate(pol_names):
            beam_modes[pol] = {}
            for freq in freqs_hz:
                # Calculate complex excitation voltages
                # convert delay to phase (What is 435e-12?)
                phases = 2 * np.pi * freq * (-delays[pol_i, :]) * 435e-12
                # complex excitation col voltage
                Vcplx = amps[pol_i, :] * np.exp(1.0j * phases)

                Q1_accum = np.zeros(max_length[freq], dtype=np.complex128)
                Q2_accum = np.zeros(max_length[freq], dtype=np.complex128)

                # Read in modes
                Q_modes_all = h5file['modes'][()].T
                Nmax = 0
                M_accum = None
                N_accum = None
                for dp_i, dp in enumerate(dipole_names):
                    # re-initialise Q1 and Q2 for every dipole
                    Q1 = np.zeros(max_length[freq], dtype=np.complex128)
                    Q2 = np.zeros(max_length[freq], dtype=np.complex128)

                    # select spherical wave table
                    name = pol + dp + '_' + str(freq)
                    Q_all = h5file[name][()].T

                    # current length
                    my_len = np.max(Q_all.shape)
                    my_len_half = my_len // 2

                    # Get modes for this dipole
                    Q_modes = Q_modes_all[0:my_len, :]

                    # convert Qall to M, N, Q1, Q2 vectors for processing

                    # find s=1 and s=2 indices
                    # only find s1 and s2 for this dipole
                    s1 = Q_modes[0:my_len, 0] <= 1
                    s2 = Q_modes[0:my_len, 0] > 1

                    # grab m,n vectors
                    M = Q_modes[s1, 1]
                    N = Q_modes[s1, 2]

                    # update to the larger M and N
                    if np.max(N) > Nmax:
                        M_accum = M
                        N_accum = N
                        Nmax = np.max(N_accum)

                    # grab Q1mn and Q2mn and make them complex
                    Q1[0:my_len_half] = Q_all[s1, 0] * np.exp(1.0j * np.deg2rad(Q_all[s1, 1]))
                    Q2[0:my_len_half] = Q_all[s2, 0] * np.exp(1.0j * np.deg2rad(Q_all[s2, 1]))

                    # accumulate Q1 and Q2, scaled by excitation voltage
                    Q1_accum = Q1_accum + Q1 * Vcplx[dp_i]
                    Q2_accum = Q2_accum + Q2 * Vcplx[dp_i]

                beam_modes[pol][freq] = {'Q1': Q1_accum, 'Q2': Q2_accum,
                                         'M': M_accum, 'N': N_accum}
        self.beam_modes = beam_modes

    def get_response(self, pol_names, phi_arr, theta_arr):
        """
        Calculate full Jones matrix response (E-field) of beam for
        one or more spherical coordinates

        Parameters
        ----------
        phi_arr : float or array of float
            azimuth angles (radians), north through east.
        theta_arr : float or array of float
            zenith angles (radian)

        Returns
        -------
        Jones : array of float
            Jones vectors, shape (2, Npol, Npts), e.g.
                [J_11=Xtheta J_12=Xphi]
                [J_21=Ytheta J_21=Yphi]
        """
        Jones = np.zeros((2, 2, phi_arr.size, theta_arr.size), dtype=np.complex128)

        phi_arr = math.pi / 2 - phi_arr  # Convert to East through North (FEKO coords)
        phi_arr[phi_arr < 0] += 2 * math.pi  # 360 wrap

        for pol_i, pol in enumerate(pol_names):
            M = self.beam_modes[pol]['M']
            N = self.beam_modes[pol]['N']
            Q1 = self.beam_modes[pol]['Q1']
            Q2 = self.beam_modes[pol]['Q2']

    def read_mwa_beam(h5filepath, delays=None, amplitudes=None):

        freqs_hz, pol_names, dipole_names = self._read_metadata(h5filepath)
        dipole_nums = np.array([np.int(name) - 1 for name in dipole_names])

        n_dp = dipole_names.size
        n_pol = len(pols)

        if delays is None:
            delays = np.zeros([n_pol, n_dp])

        if amplitudes is None:
            amplitudes = np.ones([n_pol, n_dp])

        if amplitudes.shape != (n_pol, n_dp):
            raise ValueError('amplitudes must be shape ({np, nd})'.format(np=n_pol, nd=n_dp))

        if delays.shape != (n_pol, n_dp):
            raise ValueError('delays must be shape ({np, nd})'.format(np=n_pol, nd=n_dp))

        beam_modes = self._get_beam_modes(h5filepath, freqs_hz, pol_names,
                                          dipole_names, delays)
