# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Read in the Sujinto et al. full embedded element MWA Beam."""
import warnings

import numpy as np
import h5py
from scipy.special import factorial, lpmv  # associated Legendre function

from . import UVBeam
from .. import utils as uvutils

__all__ = ["P1sin", "P1sin_array", "MWABeam"]


def P1sin(nmax, theta):
    """
    Create the Legendre function flavors for FF expansion using spherical waves.

    Note this is not vectorized so is a bit slow, but it handles the special
    case of theta = 0 and pi. We primarily use the vectorized version
    (`P1sin_array`), but call this to handle the special cases.

    See:
    Calculating Far-Field Radiation Based on FEKO Spherical Wave Coefficients,
    draft 10 June 2015.
    Available at pyuvdata/docs/references/Far_field_spherical_FEKO_draft2.pdf
    This memo gives a full description of the equations implemented here,
    including descriptions of the approximations and numerical approaches used.
    In line comments below are helpful reminders, but see the memo for the full
    detail.
    Also see Sokolowski, M. et al, "Calibration and Stokes Imaging with Full
    Embedded Element Primary Beam Model for the Murchison Widefield Array",
    PASA, 2017 (10.1017/pasa.2017.54) for details specific to the MWA.

    Parameters
    ----------
    nmax : int
        Maximum n from FEKO Q1mn and Q2mn, n must be >=1
    theta : float
        The argument of the cosine or sine function used in the associated
        Legendre functions, in radians.

    Returns
    -------
    P_sin : array of float
        P_{n}^{abs(m)}(cos(theta))/sin(theta) with FEKO order M,N.
        Shape (nmax ** 2 + 2 * nmax).
    P1 : array of float
        P_{n}^{abs(m)+1}(cos(theta)) with FEKO order M,N.
        Shape (nmax ** 2 + 2 * nmax).

    """
    # initialize for nmax, we have 2(1+...+nmax)+nmax=nmax^2+2*nmax long array
    P_sin = np.zeros((nmax ** 2 + 2 * nmax))
    P1 = np.zeros((nmax ** 2 + 2 * nmax))

    # theta arguments
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    delta_cos = 1e-6  # for slope estimation

    # step from 1 to nmax
    for n in range(1, nmax + 1):
        # legendre P_{n}^{abs(m)=0...n} (cos_th)
        orders = np.arange(0, n + 1)
        orders = orders.reshape(n + 1, 1)
        P = lpmv(orders, n, cos_th)

        # THESE ARE THE SAME:
        # legendre(2,0:0.1:0.2) (matlab)
        # scipy:
        # a=np.arange(0,3)
        # a=a.reshape(3,1)
        # lpmv(b,2,np.arange(0,0.3,0.1))

        # P_{n}^{abs(m)+1} (cos_th)
        Pm1 = np.append(P[1::], 0)
        Pm1 = Pm1.reshape(len(Pm1), 1)

        # P_{n}^{abs(m)}(cos_th)/sin_th
        Pm_sin = np.zeros((n + 1, 1))  # initialize

        if cos_th == 1:
            # special treatment depending on m;
            # for m=0, Pm_sin=inf so, the product m*Pm_sin is zero;
            # for m=1, we need a substitution
            # m>=2, value is 0, so initial values are OK
            # The first approach, to just use the analytical derivative
            # is not stable for n>~45
            # Instead use slope estimate with a small delta_cos
            # Pn(cos x)/sin x = -dPn(cos_th)/dcos_th
            Pm_cos_delta_cos = lpmv(orders, n, cos_th - delta_cos)
            # backward difference
            Pm_sin[1, 0] = -(P[0] - Pm_cos_delta_cos[0]) / delta_cos

        elif cos_th == -1:
            # The first approach, to just use the analytical derivative
            # is not stable for n>~45
            # Instead use slope estimate with a small delta_cos
            # Pn(cos x)/sin x = -dPn(cos_th)/dcos_th
            Pm_cos_delta_cos = lpmv(orders, n, cos_th - delta_cos)
            # forward difference
            Pm_sin[1, 0] = -(Pm_cos_delta_cos[0] - P[0]) / delta_cos
        else:
            Pm_sin = P / sin_th

        # accumulate Psin and P1 for the m values
        ind_start = (n - 1) ** 2 + 2 * (n - 1)  # start index to populate
        ind_stop = n ** 2 + 2 * n  # stop index to populate
        # assign
        P_sin[np.arange(ind_start, ind_stop)] = np.append(
            np.flipud(Pm_sin[1::, 0]), Pm_sin
        )
        P1[np.arange(ind_start, ind_stop)] = np.append(np.flipud(Pm1[1::, 0]), Pm1)

    return P_sin, P1


def P1sin_array(nmax, theta):
    """
    Calculate P^abs(m)_n(cos(theta))/sin(theta) and P^(abs(m)+1)_n(cos(theta)).

    Similar to the "P1sin" function, but calculates for all theta in one go.
    At the end of the function, patches are made using the original P1sin function
    to solve the 0/0 issue.


    Parameters
    ----------
    nmax : int
        Maximum n from FEKO Q1mn and Q2mn, n must be >=1
    theta : array of float
        The argument of the cosine or sine functions used in the associated
        Legendre functions, in radians.

    Returns
    -------
    P_sin : array of float
        P_{n}^{abs(m)}(cos(theta))/sin(theta) with FEKO order M,N.
        Shape (nmax ** 2 + 2 * nmax, theta.size).
    P1 : array of float
        P_{n}^{abs(m)+1}(cos(theta)) with FEKO order M,N.
        Shape (nmax ** 2 + 2 * nmax, theta.size).

    """
    cos_th = np.cos(theta)
    sin_theta = np.sin(theta)

    # Make sure that we don't divide by 0 (sin(0) = sin(pi) = 0 ) proper results
    # are inserted at the end of this function. Set to NaN for now
    sin_theta[(theta == 0) | (theta == np.pi)] = np.NaN

    # create at forehand
    P_sin = np.zeros((nmax ** 2 + 2 * nmax, np.size(theta)))
    P1 = np.zeros((nmax ** 2 + 2 * nmax, np.size(theta)))
    for n in range(1, nmax + 1):
        # legendre P_{n}^{abs(m)=0...n} (cos_th)
        orders = np.arange(0, n + 1)
        orders = orders.reshape(n + 1, 1)

        # fetch entire matrix in one go (for a particular n)
        # in theory, fetching for all n in one go should also be possible
        P = lpmv(orders, n, cos_th)

        # P_{n}^{abs(m)+1} (cos_th)
        Pm1 = np.vstack([P[1::, :], np.zeros((1, np.size(theta)))])

        # P_{n}^{abs(m)}(u)/sin_th
        Pm_sin = P / sin_theta

        # accumulate Psin and P1 for the m values
        # start index to populate
        ind_start = (n - 1) ** 2 + 2 * (n - 1)
        # stop index to populate
        ind_stop = n ** 2 + 2 * n
        # assign
        P_sin[np.arange(ind_start, ind_stop), :] = np.vstack(
            [np.flipud(Pm_sin[1::, :]), Pm_sin]
        )
        P1[np.arange(ind_start, ind_stop), :] = np.vstack([np.flipud(Pm1[1::, :]), Pm1])

    # fix for theta = 0 and theta = pi
    # (properly handled in P1sin, so use that function)
    P_sin[:, theta == 0] = np.array([P1sin(nmax, 0)[0]]).transpose()
    P_sin[:, theta == np.pi] = np.array([P1sin(nmax, np.pi)[0]]).transpose()

    return P_sin.transpose(), P1.transpose()


class MWABeam(UVBeam):
    """
    Defines an MWA-specific subclass of UVBeam for representing MWA beams.

    This class should not be interacted with directly, instead use the
    read_mwa_beam method on the UVBeam class.

    This is based on https://github.com/MWATelescope/mwa_pb/ but we don’t import
    that module because it’s not python 3 compatible.

    Note that the azimuth convention in for the UVBeam object is different than the
    azimuth convention in the mwa_pb repo. In that repo, the azimuth convention is
    changed from the native FEKO convention (the FEKO convention is the same as the
    UVBeam convention). The convention in the mwa_pb repo has a different zero point
    and a different direction (so it is in a left handed coordinate system).

    """

    def _read_metadata(self, h5filepath):
        """
        Get metadata (frequencies, polarizations, dipole numbers) from input file.

        Parameters
        ----------
        h5filepath : str
            path to input h5 file containing the MWA full embedded element spherical
            harmonic modes.

        Returns
        -------
        freqs_hz : array of int
            Frequencies in Hz present in the file.
        pol_names : list of str
            Polarizations present in the file.
        dipole_names :
            Dipoles names present in the file.
        max_length : dict
            Dictionary keyed on pol and freq, giving max number of modes in the
            file for each pol and freq.
        """
        pol_names = set()
        dipole_names = set()
        freqs_hz = set()
        other_names = []
        max_length = {}
        with h5py.File(h5filepath, "r") as h5f:
            for name in h5f.keys():
                if name.startswith("X") or name.startswith("Y"):
                    pol = name[0]
                    dipole, freq = name[1:].split("_")
                    pol_names.add(pol)
                    dipole_names.add(dipole)
                    freq = np.int64(freq)
                    freqs_hz.add(freq)

                    if pol not in max_length:
                        max_length[pol] = {}

                    this_length = h5f[name].shape[1] // 2
                    if freq not in max_length[pol]:
                        max_length[pol][freq] = this_length
                    elif this_length > max_length[pol][freq]:
                        max_length[pol][freq] = this_length

                else:
                    other_names.append(name)

        pol_names = sorted(pol_names)

        dipole_names = np.array(sorted(dipole_names))

        freqs_hz = np.array(sorted(freqs_hz))

        return freqs_hz, pol_names, dipole_names, max_length

    def _get_beam_modes(
        self,
        h5filepath,
        freqs_hz,
        pol_names,
        dipole_names,
        max_length,
        delays,
        amplitudes,
    ):
        """
        Get beam modes from input file and save as a dict to the object.

        Parameters
        ----------
        h5filepath : str
            path to input h5 file containing the MWA full embedded element spherical
            harmonic modes.
        freqs_hz : array of int
            Frequencies in Hz to get modes for. Must be present in the file.
        pol_names : list of str
            Polarizations  to get modes for. Must be present in the file.
        dipole_names : array of str
            Dipoles names present in the file.
        max_length : dict
            Dictionary keyed on pol and freq, giving max number of modes in the
            file for each pol and freq.
        delays : array of ints
            Array of MWA beamformer delay steps. Should be shape (n_pols, n_dipoles).
        amplitudes : array of floats
            Array of dipole amplitudes, these are absolute values
            (i.e. relatable to physical units). Should be shape (n_pols, n_dipoles).

        Returns
        -------
        beam_modes : dict
            A multi-level dict keyed on (in order) pol, freq, mode name (Q1, Q2, M, N).
        """
        beam_modes = {}
        for pol_i, pol in enumerate(pol_names):
            beam_modes[pol] = {}
            for freq in freqs_hz:
                # Calculate complex excitation voltages
                # convert delay to phase
                # 435e-12 is the delay step size in seconds (435 picosec)
                phases = 2 * np.pi * freq * (-delays[pol_i, :]) * 435e-12
                # complex excitation col voltage
                Vcplx = amplitudes[pol_i, :] * np.exp(1.0j * phases)

                Q1_accum = np.zeros(max_length[pol][freq], dtype=np.complex128)
                Q2_accum = np.zeros(max_length[pol][freq], dtype=np.complex128)

                # Read in modes
                with h5py.File(h5filepath, "r") as h5f:
                    Q_modes_all = h5f["modes"][()].T
                    Nmax = 0
                    M_accum = None
                    N_accum = None
                    for dp_i, dp in enumerate(dipole_names):
                        # re-initialise Q1 and Q2 for every dipole
                        Q1 = np.zeros(max_length[pol][freq], dtype=np.complex128)
                        Q2 = np.zeros(max_length[pol][freq], dtype=np.complex128)

                        # select spherical wave table
                        name = pol + dp + "_" + str(freq)
                        Q_all = h5f[name][()].T

                        # current length
                        my_len = np.max(Q_all.shape)
                        my_len_half = my_len // 2

                        # Get modes for this dipole
                        Q_modes = Q_modes_all[0:my_len, :]

                        # convert Qall to M, N, Q1, Q2 vectors for processing

                        # find s=1 and s=2 indices
                        # only find s1 and s2 for this dipole
                        # s = 1 and s = 2 refer to TE and TM modes, respectively
                        # see the Far_field_spherical_FEKO_draft2 memo under
                        # pyuvdata/docs/references/
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
                        Q1[0:my_len_half] = Q_all[s1, 0] * np.exp(
                            1.0j * np.deg2rad(Q_all[s1, 1])
                        )
                        Q2[0:my_len_half] = Q_all[s2, 0] * np.exp(
                            1.0j * np.deg2rad(Q_all[s2, 1])
                        )

                        # accumulate Q1 and Q2, scaled by excitation voltage
                        Q1_accum = Q1_accum + Q1 * Vcplx[dp_i]
                        Q2_accum = Q2_accum + Q2 * Vcplx[dp_i]

                    beam_modes[pol][freq] = {
                        "Q1": Q1_accum,
                        "Q2": Q2_accum,
                        "M": M_accum,
                        "N": N_accum,
                    }
        return beam_modes

    def _get_response(self, freqs_hz, pol_names, beam_modes, phi_arr, theta_arr):
        """
        Calculate full Jones matrix response (E-field) of beam on a regular az/za grid.

        Parameters
        ----------
        freqs_hz : array of int
            Frequencies in Hz to get modes for. Must be present in the file.
        pol_names : list of str
            Polarizations  to get modes for. Must be present in the file.
        beam_modes : dict
            A multi-level dict keyed on (in order) pol, freq, mode name (Q1, Q2, M, N).
        phi_arr : float or array of float
            azimuth angles (radians), east through north.
        theta_arr : float or array of float
            zenith angles (radian)

        Returns
        -------
        jones : array of float
            jones vectors, shape (Npol, 2, Nfreq, Nphi, Ntheta), e.g.
                [J_11=Xtheta J_12=Xphi]
                [J_21=Ytheta J_21=Yphi]

        """
        jones = np.zeros(
            (len(pol_names), 2, freqs_hz.size, phi_arr.size, theta_arr.size),
            dtype=np.complex128,
        )

        for pol_i, pol in enumerate(pol_names):
            for freq_i, freq in enumerate(freqs_hz):
                M = beam_modes[pol][freq]["M"]
                N = beam_modes[pol][freq]["N"]
                Q1 = beam_modes[pol][freq]["Q1"]
                Q2 = beam_modes[pol][freq]["Q2"]

                # form P(cos(theta))/(sin\theta) and P^{m+1}(cos(theta))with
                # FEKO M,N order
                nmax = int(np.max(N))
                assert (
                    np.max(N) - nmax == 0
                ), "The maximum of N should be an integer value!"

                # calculate equation C_mn from equation 4 of
                # pyuvdata/docs/references/Far_field_spherical_FEKO_draft2.pdf
                # These are the normalization factors for the associated
                # Legendre function of order n and rank abs(m)
                C_MN = (
                    0.5 * (2 * N + 1) * factorial(N - abs(M)) / factorial(N + abs(M))
                ) ** 0.5

                # 1 for M<=0, -1 for odd M>0
                MabsM = np.ones(M.shape)
                MabsM[(M > 0) & (M % 2 != 0)] = -1

                # nomenclature:
                # T and P are the sky polarisations theta and phi
                # theta and phi are direction coordinates

                phi_comp = np.ascontiguousarray(
                    np.exp(1.0j * np.outer(phi_arr, range(-nmax, nmax + 1)))
                )

                (P_sin, P1) = P1sin_array(nmax, theta_arr)
                M_u = np.outer(np.cos(theta_arr), np.abs(M))
                phi_const = C_MN * MabsM / (N * (N + 1)) ** 0.5

                emn_T = (
                    (1.0j) ** N * (P_sin * (M_u * Q2 - M * Q1) + Q2 * P1) * phi_const
                )
                emn_P = (
                    (1.0j) ** (N + 1)
                    * (P_sin * (M * Q2 - Q1 * M_u) - Q1 * P1)
                    * phi_const
                )

                # Use a matrix multiplication to calculate Emn_P and Emn_T.
                # Sum results of Emn_P and emn_T for each unique M
                emn_P_sum = np.zeros(
                    (len(theta_arr), 2 * nmax + 1), dtype=np.complex128
                )
                emn_T_sum = np.zeros(
                    (len(theta_arr), 2 * nmax + 1), dtype=np.complex128
                )
                for m in range(-nmax, nmax + 1):
                    emn_P_sum[:, m + nmax] = np.sum(emn_P[:, M == m], axis=1)
                    emn_T_sum[:, m + nmax] = np.sum(emn_T[:, M == m], axis=1)

                Sigma_P = np.inner(phi_comp, emn_P_sum)
                Sigma_T = np.inner(phi_comp, emn_T_sum)

                jones[pol_i, 0, freq_i] = Sigma_T
                jones[pol_i, 1, freq_i] = -Sigma_P

        return jones

    def read_mwa_beam(
        self,
        h5filepath,
        delays=None,
        amplitudes=None,
        pixels_per_deg=5,
        freq_range=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read in the full embedded element MWA beam.

        Parameters
        ----------
        h5filepath : str
            path to input h5 file containing the MWA full embedded element spherical
            harmonic modes. Download via
            `wget http://cerberus.mwa128t.org/mwa_full_embedded_element_pattern.h5`
        delays : array of ints
            Array of MWA beamformer delay steps. Should be shape (n_pols, n_dipoles).
        amplitudes : array of floats
            Array of dipole amplitudes, these are absolute values
            (i.e. relatable to physical units).
            Should be shape (n_pols, n_dipoles).
        pixels_per_deg : float
            Number of theta/phi pixels per degree. Sets the resolution of the beam.
        freq_range : array_like of float
            Range of frequencies to include in Hz, defaults to all available
            frequencies. Must be length 2.
        run_check : bool
            Option to check for the existence and proper shapes of
            required parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            required parameters after reading in the file.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the amplitudes or delays are the wrong shape or there are delays
            greater than 32 or delays are not integer types.
            If the frequency range doesn't include any
            available frequencies.

        """
        freqs_hz, pol_names, dipole_names, max_length = self._read_metadata(h5filepath)

        n_dp = dipole_names.size
        n_pol = len(pol_names)

        if delays is None:
            delays = np.zeros([n_pol, n_dp], dtype="int")
        else:
            if not np.issubdtype(delays.dtype, np.integer):
                raise ValueError("Delays must be integers.")

        if amplitudes is None:
            amplitudes = np.ones([n_pol, n_dp])

        if amplitudes.shape != (n_pol, n_dp):
            raise ValueError(
                "amplitudes must be shape ({npol}, {nd})".format(npol=n_pol, nd=n_dp)
            )

        if delays.shape != (n_pol, n_dp):
            raise ValueError(
                "delays must be shape ({npol}, {nd})".format(npol=n_pol, nd=n_dp)
            )

        if (delays > 32).any():
            raise ValueError(
                "There are delays greater than 32: {delays}".format(delays=delays)
            )

        # check for terminated dipoles and reset delays and amplitudes
        terminated = delays == 32
        if (terminated).any():
            warnings.warn(
                "There are some terminated dipoles "
                "(delay setting 32). Setting the amplitudes and "
                "delays of terminated dipoles to zero."
            )
            delays[terminated] = 0
            amplitudes[terminated] = 0

        if freq_range is not None:
            if np.array(freq_range).size != 2:
                raise ValueError("freq_range must have 2 elements.")
            freqs_use = freqs_hz[
                np.nonzero((freqs_hz >= freq_range[0]) & (freqs_hz <= freq_range[1]))
            ]
            if freqs_use.size < 1:
                raise ValueError(
                    "No frequencies available in freq_range. "
                    "Available frequencies are between {fmin} Hz "
                    "and {fmax} Hz".format(fmin=np.min(freqs_hz), fmax=np.max(freqs_hz))
                )
            if freqs_use.size < 2:
                warnings.warn("Only one available frequency in freq_range.")
        else:
            freqs_use = freqs_hz

        beam_modes = self._get_beam_modes(
            h5filepath,
            freqs_hz,
            pol_names,
            dipole_names,
            max_length,
            delays,
            amplitudes,
        )

        n_phi = np.floor(360 * pixels_per_deg)
        n_theta = np.floor(90 * pixels_per_deg) + 1
        theta_arr = np.deg2rad(np.arange(0, n_theta) / pixels_per_deg)
        phi_arr = np.deg2rad(np.arange(0, n_phi) / pixels_per_deg)

        jones = self._get_response(freqs_use, pol_names, beam_modes, phi_arr, theta_arr)

        # work out zenith normalization
        # (MWA beams are peak normalized to 1 when pointed at zenith)

        # start filling in the object
        self.telescope_name = "MWA"
        self.feed_name = "MWA"
        self.feed_version = "1.0"
        self.model_name = "full embedded element"
        self.model_version = "1.0"
        self.history = (
            "Sujito et al. full embedded element beam, derived from "
            "https://github.com/MWATelescope/mwa_pb/"
        )

        delay_str_list = []
        gain_str_list = []
        for pol in range(n_pol):
            delay_str_list.append(
                "[" + ", ".join([str(x) for x in delays[pol, :]]) + "]"
            )
            gain_str_list.append(
                "[" + ", ".join([str(x) for x in amplitudes[pol, :]]) + "]"
            )
        delay_str = "[" + ", ".join(delay_str_list) + "]"
        gain_str = "[" + ", ".join(gain_str_list) + "]"

        self.history += "  delays set to " + delay_str + "  gains set to " + gain_str
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        self.x_orientation = "east"

        self._set_efield()
        self.Naxes_vec = 2
        self.Ncomponents_vec = 2
        self.feed_array = np.array([str(pol.lower()) for pol in pol_names])
        self.Nfeeds = self.feed_array.size

        self.data_normalization = "physical"

        # for now this returns a simple beam because it requires amps & delays
        # to make the beam
        self.antenna_type = "simple"

        self.Nspws = 1
        self.spw_array = np.array([0])
        self.Nfreqs = freqs_use.size
        self.freq_array = freqs_use.astype(np.float64)
        self.freq_array = self.freq_array[np.newaxis, :]
        self.bandpass_array = np.ones((self.Nspws, self.Nfreqs))

        self.pixel_coordinate_system = "az_za"
        self._set_cs_params()

        self.axis1_array = phi_arr
        self.Naxes1 = self.axis1_array.size
        self.axis2_array = theta_arr
        self.Naxes2 = self.axis2_array.size

        # The array that come from `_get_response` has shape shape
        # (Npol, 2, Nfreq, Nphi, Ntheta)
        # UVBeam wants shape
        # ('Naxes_vec', 'Nspws', 'Nfeeds', 'Nfreqs', 'Naxes2', 'Naxes1')
        # where the Naxes_vec dimension lines up with the 2 from `_get_response`,
        # Nfeeds is UVBeam's Npol for E-field beams,
        # and axes (2, 1) correspond to (theta, phi)
        # Then add an empty dimension for Nspws.
        self.data_array = np.transpose(jones, axes=[1, 0, 2, 4, 3])
        self.data_array = self.data_array[:, np.newaxis, :, :, :, :]

        self.basis_vector_array = np.zeros(
            (self.Naxes_vec, self.Ncomponents_vec, self.Naxes2, self.Naxes1)
        )
        self.basis_vector_array[0, 0, :, :] = 1.0
        self.basis_vector_array[1, 1, :, :] = 1.0

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
