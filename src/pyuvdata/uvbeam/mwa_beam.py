# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Read in the Sujinto et al. full embedded element MWA Beam."""

import os
import warnings

import h5py
import numpy as np
from astropy import constants
from astropy.io import fits
from docstring_parser import DocstringStyle
from scipy.special import factorial, lpmv  # associated Legendre function

from .. import utils
from ..data import DATA_PATH
from ..docstrings import copy_replace_short_description
from . import UVBeam

__all__ = ["P1sin", "P1sin_array", "MWABeam"]

# dipole spacing in meters
MWA_DIPOLE_SPACING_M = 1.1

# 435 picoseconds is base delay length unit
MWA_BASE_DELAY_S = 4.35e-10

MWA_NFEED = 2
MWA_NDIPOLE = 16


def P1sin(nmax, theta):  # noqa N802
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
    P_sin = np.zeros(nmax**2 + 2 * nmax)
    P1 = np.zeros(nmax**2 + 2 * nmax)

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
            Pm_sin[1, 0] = -(P[0, 0] - Pm_cos_delta_cos[0, 0]) / delta_cos

        elif cos_th == -1:
            # The first approach, to just use the analytical derivative
            # is not stable for n>~45
            # Instead use slope estimate with a small delta_cos
            # Pn(cos x)/sin x = -dPn(cos_th)/dcos_th
            Pm_cos_delta_cos = lpmv(orders, n, cos_th - delta_cos)
            # forward difference
            Pm_sin[1, 0] = -(Pm_cos_delta_cos[0, 0] - P[0, 0]) / delta_cos
        else:
            Pm_sin = P / sin_th

        # accumulate Psin and P1 for the m values
        ind_start = (n - 1) ** 2 + 2 * (n - 1)  # start index to populate
        ind_stop = n**2 + 2 * n  # stop index to populate
        # assign
        P_sin[np.arange(ind_start, ind_stop)] = np.append(
            np.flipud(Pm_sin[1::, 0]), Pm_sin
        )
        P1[np.arange(ind_start, ind_stop)] = np.append(np.flipud(Pm1[1::, 0]), Pm1)

    return P_sin, P1


def P1sin_array(nmax, theta):  # noqa N802
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
    sin_theta[(theta == 0) | (theta == np.pi)] = np.nan

    # create at forehand
    P_sin = np.zeros((nmax**2 + 2 * nmax, np.size(theta)))
    P1 = np.zeros((nmax**2 + 2 * nmax, np.size(theta)))
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
        ind_stop = n**2 + 2 * n
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


def _get_freq_inds_use(freqs_hz, freq_range=None):
    """
    Get the frequencies to use.

    Parameters
    ----------
    freqs_hz : array of float
        Frequencies in file.
    freq_range : tuple of float in Hz
        If given, the lower and upper limit of the frequencies to read in. Default
        is to use all frequencies.

    Returns
    -------
    freq_inds_use : array of int
        Indices into the freqs_hz to use.
    """
    if freq_range is not None:
        if np.array(freq_range).size != 2:
            raise ValueError("freq_range must have 2 elements.")
        freqs_mask = utils.tools._is_between(
            freqs_hz, np.asarray(freq_range)[np.newaxis]
        )
        freq_inds_use = np.nonzero(freqs_mask)[0]
        if freq_inds_use.size < 1:
            raise ValueError(
                "No frequencies available in freq_range. "
                f"Available frequencies are between {np.min(freqs_hz)} Hz "
                f"and {np.max(freqs_hz)} Hz"
            )
        if freq_inds_use.size < 2:
            warnings.warn("Only one available frequency in freq_range.")
    else:
        freq_inds_use = np.arange(freqs_hz.size)

    return freq_inds_use


class MWABeam(UVBeam):
    """
    Defines an MWA-specific subclass of UVBeam for representing MWA beams.

    This class should not be interacted with directly, instead use the
    read_mwa_beam method on the UVBeam class.

    This is based on https://github.com/MWATelescope/mwa_pb/ and offers support
    for either the Fully Embedded Element (FEE) model or the Average Embedded
    Element (AEE) model. The FEE model is the most current model and is generally
    considered to be the best match to the instrument beam, but we provide the
    older AEE model for comparison as well.

    Note that the azimuth convention for the UVBeam object is different than the
    azimuth convention in the mwa_pb repo. In that repo, the azimuth convention
    is changed from the native FEKO convention that the underlying data file is
    in. The FEKO convention that the data file is in is the same as the UVBeam
    convention, so we do not need to do a conversion here. The convention in the
    mwa_pb repo is North through East, so it has a different zero point and a
    different direction (so it is in a left handed coordinate system looking
    down at the beam, a right handed coordinate system looking up at the sky).

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
        feed_names : list of str
            Polarizations present in the file.
        dipole_names :
            Dipoles names present in the file.
        max_length : dict
            Dictionary keyed on pol and freq, giving max number of modes in the
            file for each pol and freq.
        """
        feed_names = set()
        dipole_names = set()
        freqs_hz = set()
        other_names = []
        max_length = {}
        with h5py.File(h5filepath, "r") as h5f:
            for name in h5f:
                if name.startswith("X") or name.startswith("Y"):
                    pol = name[0]
                    dipole, freq = name[1:].split("_")
                    feed_names.add(pol)
                    dipole_names.add(dipole)
                    freq = np.int64(freq)
                    freqs_hz.add(freq)

                    if pol not in max_length:
                        max_length[pol] = {}

                    this_length = h5f[name].shape[1] // 2
                    if (
                        freq not in max_length[pol]
                        or this_length > max_length[pol][freq]
                    ):
                        max_length[pol][freq] = this_length

                else:
                    other_names.append(name)

        feed_names = sorted(feed_names)

        dipole_names = np.asarray(sorted(dipole_names, key=int))

        freqs_hz = np.array(sorted(freqs_hz))

        return freqs_hz, feed_names, dipole_names, max_length

    def _get_beam_modes(
        self,
        *,
        h5filepath,
        freqs_hz,
        feed_names,
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
        feed_names : list of str
            Polarizations  to get modes for. Must be present in the file.
        dipole_names : array of str
            Dipoles names present in the file.
        max_length : dict
            Dictionary keyed on pol and freq, giving max number of modes in the
            file for each pol and freq.
        delays : array of ints
            Array of MWA beamformer delay steps. Should be shape (n_feeds, n_dipoles).
        amplitudes : array of floats
            Array of dipole amplitudes, these are absolute values
            (i.e. relatable to physical units). Should be shape (n_feeds, n_dipoles).

        Returns
        -------
        beam_modes : dict
            A multi-level dict keyed on (in order) pol, freq, mode name (Q1, Q2, M, N).
        """
        beam_modes = {}
        for pol_i, pol in enumerate(feed_names):
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

    def _get_response(self, *, freqs_hz, feed_names, beam_modes, az_array, za_array):
        """
        Calculate full Jones matrix response (E-field) of beam on a regular az/za grid.

        Parameters
        ----------
        freqs_hz : array of int
            Frequencies in Hz to get modes for. Must be present in the file.
        feed_names : list of str
            Polarizations  to get modes for. Must be present in the file.
        beam_modes : dict
            A multi-level dict keyed on (in order) pol, freq, mode name (Q1, Q2, M, N).
        az_array : float or array of float
            azimuth angles (radians), east through north.
        za_array : float or array of float
            zenith angles (radian)

        Returns
        -------
        jones : array of float
            jones vectors, shape (Npol, 2, Nfreq, Nphi, Ntheta), e.g.
                [J_11=Xtheta J_12=Xphi]
                [J_21=Ytheta J_21=Yphi]

        """
        jones = np.zeros(
            (len(feed_names), 2, freqs_hz.size, az_array.size, za_array.size),
            dtype=np.complex128,
        )

        for pol_i, pol in enumerate(feed_names):
            for freq_i, freq in enumerate(freqs_hz):
                M = beam_modes[pol][freq]["M"]
                N = beam_modes[pol][freq]["N"]
                Q1 = beam_modes[pol][freq]["Q1"]
                Q2 = beam_modes[pol][freq]["Q2"]

                # form P(cos(theta))/(sin\theta) and P^{m+1}(cos(theta))with
                # FEKO M,N order
                nmax = int(np.max(N))
                if np.max(N) - nmax != 0:  # pragma: no cover
                    raise RuntimeError(
                        "Something went wrong in mwa_beam._get_response. Please "
                        "file an issue in our GitHub issue log so that we can help: "
                        "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues."
                        " Developer info: The maximum of N is not an integer value"
                    )

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
                    np.exp(1.0j * np.outer(az_array, range(-nmax, nmax + 1)))
                )

                (P_sin, P1) = P1sin_array(nmax, za_array)
                M_u = np.outer(np.cos(za_array), np.abs(M))
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
                emn_P_sum = np.zeros((len(za_array), 2 * nmax + 1), dtype=np.complex128)
                emn_T_sum = np.zeros((len(za_array), 2 * nmax + 1), dtype=np.complex128)
                for m in range(-nmax, nmax + 1):
                    emn_P_sum[:, m + nmax] = np.sum(emn_P[:, m == M], axis=1)
                    emn_T_sum[:, m + nmax] = np.sum(emn_T[:, m == M], axis=1)

                Sigma_P = np.inner(phi_comp, emn_P_sum)
                Sigma_T = np.inner(phi_comp, emn_T_sum)

                # we do not want a minus sign on Sigma_P unlike in mwa_pb because
                # that minus sign is associated with the coordinate conversion
                # they do that we do not want.
                jones[pol_i, 0, freq_i] = Sigma_P
                jones[pol_i, 1, freq_i] = Sigma_T

        return jones

    def _read_fee_jones(
        self,
        h5filepath,
        *,
        delays=None,
        amplitudes=None,
        pixels_per_deg=5,
        freq_range=None,
    ):
        """Read in the full embedded element MWA beam."""
        # update filename attribute
        basename = os.path.basename(h5filepath)
        self.filename = [basename]
        self._filename.form = (1,)

        freqs_hz, feed_names, dipole_names, max_length = self._read_metadata(h5filepath)

        if [str(feed.lower()) for feed in feed_names] != ["x", "y"]:
            raise ValueError(
                f"Did not find expected feed names in FEE file {h5filepath}"
            )

        freqs_inds_use = _get_freq_inds_use(freqs_hz, freq_range=freq_range)
        freqs_use = freqs_hz[freqs_inds_use]

        beam_modes = self._get_beam_modes(
            h5filepath=h5filepath,
            freqs_hz=freqs_hz,
            feed_names=feed_names,
            dipole_names=dipole_names,
            max_length=max_length,
            delays=delays,
            amplitudes=amplitudes,
        )

        n_phi = np.floor(360 * pixels_per_deg)
        n_theta = np.floor(90 * pixels_per_deg) + 1
        za_array = np.deg2rad(np.arange(0, n_theta) / pixels_per_deg)
        az_array = np.deg2rad(np.arange(0, n_phi) / pixels_per_deg)

        jones = self._get_response(
            freqs_hz=freqs_use,
            feed_names=feed_names,
            beam_modes=beam_modes,
            az_array=az_array,
            za_array=za_array,
        )

        # The array that come from `_get_response` has shape shape
        # (Npol, 2, Nfreq, Nphi, Ntheta)
        # UVBeam wants shape
        # ('Naxes_vec', 1, 'Nfeeds', 'Nfreqs', 'Naxes2', 'Naxes1')
        # where the Naxes_vec dimension lines up with the 2 from `_get_response`,
        # Nfeeds is UVBeam's Npol for E-field beams,
        # and axes (2, 1) correspond to (theta, phi)
        # Then add an empty dimension for Nspws.
        jones = np.transpose(jones, axes=[1, 0, 2, 4, 3])

        return jones, freqs_use, za_array, az_array

    def _read_aee_jones(
        self,
        jonesfile,
        *,
        zfile,
        delays=None,
        amplitudes=None,
        freq_range=None,
        include_cross_feed_coupling=True,
    ):
        """Read in the average embedded element MWA beam."""
        # update filename attribute
        jones_basename = os.path.basename(jonesfile)
        z_basename = os.path.basename(zfile)
        self.filename = [jones_basename, z_basename]
        self._filename.form = (2,)

        n_feed = MWA_NFEED
        n_dipole = MWA_NDIPOLE

        with fits.open(jonesfile) as jfile:
            n_freqs = len(jfile)
            freqs_hz = np.ndarray(n_freqs, dtype=float)
            for f_ind in range(n_freqs):
                freqs_hz[f_ind] = jfile[f_ind].header["freq"]
            raw_theta = jfile[0].data[:, 0]
            raw_phi = jfile[0].data[:, 1]

        freqs_inds_use = _get_freq_inds_use(freqs_hz, freq_range=freq_range)
        freqs_use = freqs_hz[freqs_inds_use]
        n_freqs = freqs_inds_use.size

        theta = np.unique(raw_theta)
        phi = np.unique(raw_phi)
        n_theta = theta.size
        n_phi = phi.size
        theta_grid = raw_theta.reshape(n_phi, n_theta).T
        phi_grid = raw_phi.reshape(n_phi, n_theta).T

        if not np.allclose(theta, theta_grid[:, 0]):
            raise ValueError("reshaping theta did not work as expected")
        if not np.allclose(phi, phi_grid[0, :]):
            raise ValueError("reshaping phi did not work as expected")

        # convert theta, phi to radians, rename
        az_grid = np.deg2rad(phi_grid)
        za_grid = np.deg2rad(theta_grid)
        az_array = np.deg2rad(phi)
        za_array = np.deg2rad(theta)

        # this is just the dipole jones for now, but will be updated later
        aee_jones = np.ndarray((2, 2, n_freqs, n_theta, n_phi), dtype=complex)

        with fits.open(jonesfile) as jfile:
            # This is the comment from the JMatrix FITS file:
            # Cols: theta phi  real(Jxt(t,p)) imag(Jxt(t,p)) real(Jxp(t,p))
            # imag(Jxp(t,p)) real(Jyt(t,p)) imag(Jyt(t,p)) real(Jyp(t,p))
            # imag(Jyp(t,p)))
            # Where theta is the zenith angle, phi is angle measured clockwise
            # from +east direction looking down
            # Jxt is the Jones mapping unit vec in theta (t) direction to the x
            # (east-west) dipole etc

            for fi_arr, f_ind in enumerate(freqs_inds_use):
                data = jfile[f_ind].data

                if not np.allclose(raw_theta, data[:, 0]):
                    raise ValueError("Inconsistent theta values across frequecies")
                if not np.allclose(raw_phi, data[:, 1]):
                    raise ValueError("Inconsistent theta values across frequecies")

                aee_jones[1, 0, fi_arr] = (
                    (data[:, 2] + 1j * data[:, 3]).reshape(n_phi, n_theta).T
                )
                aee_jones[0, 0, fi_arr] = (
                    (data[:, 4] + 1j * data[:, 5]).reshape(n_phi, n_theta).T
                )
                aee_jones[1, 1, fi_arr] = (
                    (data[:, 6] + 1j * data[:, 7]).reshape(n_phi, n_theta).T
                )
                aee_jones[0, 1, fi_arr] = (
                    (data[:, 8] + 1j * data[:, 9]).reshape(n_phi, n_theta).T
                )

        # Now we have to work out the apperature array factor -- the factor
        # to multiply the average dipole jones by to account for all the
        # apperature array stuff: geometric dipole delays, delay line settings,
        # dipole impedances, dipole couplings

        # set up the dipole centers on a 4x4 grid with the dipole spacing for
        # geometric delay calculation
        x_centers, y_centers = np.meshgrid(
            np.arange(4) * MWA_DIPOLE_SPACING_M,
            np.flip(np.arange(4)) * MWA_DIPOLE_SPACING_M,
        )
        x_centers = (x_centers - x_centers.mean()).flatten()
        y_centers = (y_centers - y_centers.mean()).flatten()
        z_centers = np.zeros(16, dtype=float)

        # calculate delays for dipoles relative to the center of the tile
        # az runs from East to North
        east_direction_cos = np.sin(za_grid) * np.cos(az_grid)
        north_direction_cos = np.sin(za_grid) * np.sin(az_grid)
        radial_direction_cos = np.cos(za_grid)

        geometric_dipole_delay = (
            np.outer(east_direction_cos.flatten(), x_centers)
            + np.outer(north_direction_cos.flatten(), y_centers)
            + np.outer(radial_direction_cos.flatten(), z_centers)
        )

        delays_sec = delays * MWA_BASE_DELAY_S

        lna_impedance_file = os.path.join(
            DATA_PATH, "mwa_config_data", "mwa_lna_impedance.txt"
        )

        # the LNA impedances were measured on a different grid than the beam file
        # interpolate the impedances to the frequencies we care about
        data = np.genfromtxt(lna_impedance_file, skip_header=1)
        impedance_freq = data[:, 0]
        lna_impedance = data[:, 1] + 1j * data[:, 2]
        lna_impedance_use = np.interp(
            freqs_use, impedance_freq, np.real(lna_impedance)
        ) + 1j * np.interp(freqs_use, impedance_freq, np.imag(lna_impedance))

        # read the dipole coupling out of the Zmatrix file. these are 32 x 32
        # as all dipole pols can couple to all other dipole pols
        dipole_coupling = np.zeros((n_freqs, n_dipole * 2, n_dipole * 2), dtype=complex)
        with fits.open(zfile) as zf:
            # Comment from ZMatrix FITS file and related code:
            # Data are 32x32x2 cubes per ext. 1st plane Mag, 2nd plane phase
            # ordering in Z matrix is 0-15:Y, 16-31:X
            if not len(zf) == freqs_hz.size:
                raise ValueError(
                    "Zmatrix file does not have as the same number of frequencies"
                    "as Jmatrix file."
                )
            for fi_arr, f_ind in enumerate(freqs_inds_use):
                if not np.isclose(zf[f_ind].header["freq"], freqs_hz[f_ind]):
                    raise ValueError(
                        f"Zmatrix {f_ind}th freq does not match Jmatrix file."
                    )
                data = zf[f_ind].data
                # convert to a proper complex number
                data = data[0, :, :] * (
                    np.cos(data[1, :, :]) + 1j * np.sin(data[1, :, :])
                )
                # move blocks around so we have X dipoles then Y dipoles
                dipole_coupling[fi_arr, :n_dipole, :n_dipole] = data[
                    n_dipole:, n_dipole:
                ]
                dipole_coupling[fi_arr, :n_dipole, n_dipole:] = data[
                    n_dipole:, :n_dipole
                ]
                dipole_coupling[fi_arr, n_dipole:, :n_dipole] = data[
                    :n_dipole, n_dipole:
                ]
                dipole_coupling[fi_arr, n_dipole:, n_dipole:] = data[
                    :n_dipole, :n_dipole
                ]

        n_couple = n_dipole * 2
        if not include_cross_feed_coupling:
            # this is what FHD does. But by construction it excludes x <-> y coupling
            # because rather than a 32 x 32 matrix we have (2 x 16 x 16)
            n_couple = n_dipole
            same_feed_coupling = np.zeros(
                (2, n_freqs, n_dipole, n_dipole), dtype=complex
            )
            same_feed_coupling[0] = dipole_coupling[:, :n_dipole, :n_dipole]
            same_feed_coupling[1] = dipole_coupling[:, n_dipole:, n_dipole:]
            dipole_coupling = same_feed_coupling

        z_total = dipole_coupling.copy()
        for fi in range(n_freqs):
            if include_cross_feed_coupling:
                z_total[fi] += np.eye(n_couple) * lna_impedance_use[fi]
            else:
                for feed_i in range(n_feed):
                    z_total[feed_i, [fi]] += np.eye(n_couple) * lna_impedance_use[fi]

        # invert z to calculate currents (as voltage/impedance)
        inv_z = np.linalg.inv(z_total)

        # signs in delayline term were adjusted to make the pointings go the right way
        apparray_factor = np.zeros((2, n_freqs, n_theta, n_phi), dtype=complex)
        for fi, this_f in enumerate(freqs_use):
            k_conv = (2.0 * np.pi) * (this_f / constants.c.to("m/s").value)
            delayline_term = np.exp(
                -1j * 2.0 * np.pi * delays_sec * this_f * amplitudes
            )
            if include_cross_feed_coupling:
                port_current = np.dot(inv_z[fi], delayline_term.reshape(32)).reshape(
                    2, 16
                )

            for feed_i in range(n_feed):
                if include_cross_feed_coupling:
                    port_current_use = port_current[feed_i]
                else:
                    port_current_use = np.dot(inv_z[feed_i, fi], delayline_term[feed_i])

                for dip_i in range(n_dipole):
                    geometric_dipole_factor = np.exp(
                        1j * k_conv * geometric_dipole_delay[:, dip_i]
                    )
                    # sum product of geometric factor & port current across dipoles
                    apparray_factor[feed_i, fi] += (
                        geometric_dipole_factor * port_current_use[dip_i]
                    ).reshape((n_theta, n_phi))

        for vec_i in range(2):
            aee_jones[vec_i] *= apparray_factor

        return aee_jones, freqs_use, za_array, az_array

    @copy_replace_short_description(UVBeam.read_mwa_beam, style=DocstringStyle.NUMPYDOC)
    def read_mwa_beam(
        self,
        filename,
        *,
        model_type=None,
        zfile=None,
        delays=None,
        amplitudes=None,
        pixels_per_deg=5,
        freq_range=None,
        include_cross_feed_coupling=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        check_auto_power=True,
        fix_auto_power=True,
    ):
        """Read in MWA beam."""
        n_feed = MWA_NFEED
        n_dipole = MWA_NDIPOLE

        if model_type is None:
            _, extension = os.path.splitext(filename)
            if extension == ".fits":
                model_type = "aee"
            elif extension == ".hdf5" or extension == ".h5":
                model_type = "fee"
            else:
                raise ValueError(
                    "model_type could not be determined for MWA beam file, use "
                    "the model_type keyword to specify the type (one of 'fee' "
                    f"or 'aee'). Filename is: {filename}"
                )

        if model_type == "aee" and zfile is None:
            raise ValueError(
                "zfile must be supplied for average embedded element MWA beam."
            )

        if delays is None:
            delays = np.zeros([n_feed, n_dipole], dtype="int")
        else:
            if not np.issubdtype(delays.dtype, np.integer):
                raise ValueError("Delays must be integers.")

        if amplitudes is None:
            amplitudes = np.ones([n_feed, n_dipole])

        if amplitudes.shape != (n_feed, n_dipole):
            raise ValueError(f"amplitudes must be shape ({n_feed}, {n_dipole})")

        if delays.shape != (n_feed, n_dipole):
            raise ValueError(f"delays must be shape ({n_feed}, {n_dipole})")

        if (delays > 32).any():
            raise ValueError(f"There are delays greater than 32: {delays}")

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

        if model_type == "fee":
            jones, freq_array, za_array, az_array = self._read_fee_jones(
                filename,
                delays=delays,
                amplitudes=amplitudes,
                pixels_per_deg=pixels_per_deg,
                freq_range=freq_range,
            )
        else:
            jones, freq_array, za_array, az_array = self._read_aee_jones(
                filename,
                zfile=zfile,
                delays=delays,
                amplitudes=amplitudes,
                freq_range=freq_range,
                include_cross_feed_coupling=include_cross_feed_coupling,
            )
        # start filling in the object
        self.telescope_name = "MWA"
        self.feed_name = "MWA"
        self.feed_version = "1.0"
        if model_type == "fee":
            self.model_name = "full embedded element"
        else:
            self.model_name = "average embedded element"
        self.model_version = "1.0"
        self.history = (
            "Sujito et al. full embedded element beam, derived from "
            "https://github.com/MWATelescope/mwa_pb/"
        )

        delay_str_list = []
        gain_str_list = []
        for pol in range(n_feed):
            delay_str_list.append(
                "[" + ", ".join([str(x) for x in delays[pol, :]]) + "]"
            )
            gain_str_list.append(
                "[" + ", ".join([str(x) for x in amplitudes[pol, :]]) + "]"
            )
        delay_str = "[" + ", ".join(delay_str_list) + "]"
        gain_str = "[" + ", ".join(gain_str_list) + "]"

        self.history += "  delays set to " + delay_str + "  gains set to " + gain_str
        if not utils.history._check_history_version(
            self.history, self.pyuvdata_version_str
        ):
            self.history += self.pyuvdata_version_str

        self._set_efield()
        self.Naxes_vec = 2
        self.Ncomponents_vec = 2
        self.feed_array = np.array(["x", "y"])
        self.feed_angle = np.where(self.feed_array == "x", np.pi / 2, 0.0)
        self.Nfeeds = self.feed_array.size
        self.mount_type = "phased"

        self.data_normalization = "physical"

        # for now this returns a simple beam because it requires amps & delays
        # to make the beam
        self.antenna_type = "simple"

        self.Nfreqs = freq_array.size
        self.freq_array = freq_array.astype(np.float64)
        self.bandpass_array = np.ones(self.Nfreqs)

        self.pixel_coordinate_system = "az_za"
        self._set_cs_params()

        self.axis1_array = az_array
        self.Naxes1 = self.axis1_array.size
        self.axis2_array = za_array
        self.Naxes2 = self.axis2_array.size

        self.data_array = jones

        self.basis_vector_array = np.zeros(
            (self.Naxes_vec, self.Ncomponents_vec, self.Naxes2, self.Naxes1)
        )
        self.basis_vector_array[0, 0, :, :] = 1.0
        self.basis_vector_array[1, 1, :, :] = 1.0

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )
