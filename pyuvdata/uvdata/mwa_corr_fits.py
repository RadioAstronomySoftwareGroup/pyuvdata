# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading MWA correlator FITS files."""

import itertools
import os
import warnings

import h5py
import numpy as np
from astropy import constants as const
from astropy.io import fits
from astropy.time import Time
from docstring_parser import DocstringStyle
from scipy.integrate import simpson
from scipy.special import erf

from pyuvdata.data import DATA_PATH

from .. import _corr_fits
from .. import telescopes as uvtel
from .. import utils as uvutils
from ..docstrings import copy_replace_short_description
from .uvdata import UVData, _future_array_shapes_warning

__all__ = ["input_output_mapping", "MWACorrFITS"]


def input_output_mapping():
    """Build a mapping dictionary from pfb input to output numbers."""
    # the polyphase filter bank maps inputs to outputs, which the MWA
    # correlator then records as the antenna indices.
    # the following is taken from mwa_build_lfiles/mwac_utils.c
    # inputs are mapped to outputs via pfb_mapper as follows
    # (from mwa_build_lfiles/antenna_mapping.h):
    # floor(index/4) + index%4 * 16 = input
    # for the first 64 outputs, pfb_mapper[output] = input
    return _corr_fits.input_output_mapping()


def read_metafits(
    file,
    *,
    mwax=None,
    flag_init=None,
    start_flag=None,
    start_time=None,
    telescope_info_only=False,
):
    # get information from metafits file
    with fits.open(file, memmap=True) as meta:
        meta_hdr = meta[0].header

        telescope_name = meta_hdr.pop("TELESCOP")
        instrument = meta_hdr.pop("INSTRUME")

        # get antenna data from metafits file table
        meta_tbl = meta[1].data

        # because of polarization, each antenna # is listed twice
        antenna_inds = meta_tbl["Antenna"][1::2]
        antenna_numbers = meta_tbl["Tile"][1::2]
        antenna_names = meta_tbl["TileName"][1::2]
        flagged_ant_inds = antenna_inds[meta_tbl["Flag"][1::2] == 1]
        cable_lens = np.asarray(meta_tbl["Length"][1::2]).astype(np.str_)
        dig_gains = meta_tbl["Gains"][1::2, :].astype(np.float64)

        # get antenna postions in enu coordinates
        antenna_positions = np.zeros((len(antenna_numbers), 3))
        antenna_positions[:, 0] = meta_tbl["East"][1::2]
        antenna_positions[:, 1] = meta_tbl["North"][1::2]
        antenna_positions[:, 2] = meta_tbl["Height"][1::2]

        mwa_telescope_obj = uvtel.get_telescope("mwa")

        # convert antenna positions from enu to ecef
        # antenna positions are "relative to
        # the centre of the array in local topocentric \"east\", \"north\",
        # \"height\". Units are meters."
        antenna_positions_ecef = uvutils.ECEF_from_ENU(
            antenna_positions, *mwa_telescope_obj.telescope_location_lat_lon_alt
        )
        # make antenna positions relative to telescope location
        antenna_positions = (
            antenna_positions_ecef - mwa_telescope_obj.telescope_location
        )

        # reorder antenna parameters from metafits ordering
        reordered_inds = antenna_inds.argsort()
        antenna_numbers = antenna_numbers[reordered_inds]
        antenna_names = list(antenna_names[reordered_inds])
        antenna_positions = antenna_positions[reordered_inds, :]
        cable_lens = cable_lens[reordered_inds]
        dig_gains = dig_gains[reordered_inds, :]

        if telescope_info_only:
            return {
                "telescope_name": telescope_name,
                "telescope_location": mwa_telescope_obj.telescope_location,
                "instrument": instrument,
                "antenna_numbers": antenna_numbers,
                "antenna_names": antenna_names,
                "antenna_positions": antenna_positions,
            }

        if None in [mwax, flag_init, start_flag, start_time]:
            raise ValueError(
                "mwax, flag_init, start_flag and start_time must all be passed if the "
                "`telescope_info_only` parameter is False"
            )

        # get a list of coarse channels
        coarse_chans = meta_hdr["CHANNELS"].split(",")
        coarse_chans = np.array(sorted(int(i) for i in coarse_chans))
        # fine channel width
        channel_width = float(meta_hdr.pop("FINECHAN") * 1000)
        # number of fine channels in observation
        obs_num_fine_chans = meta_hdr["NCHANS"]
        # calculate number of fine channels per coarse channel
        coarse_num_fine_chans = obs_num_fine_chans / len(coarse_chans)

        # center frequency of first fine channel of center coarse channel in hertz
        # For the legacy correlator, the metafits file includes the observation
        # frequency center, which is the center frequency of the first fine
        # channel of the center coarse channel. (If there are an even number of
        # coarse channels, the center channel is to the right).
        # For mwax, the center frequency of the first fine channel of a coarse
        # channel is the leftmost edge of the coarse channel if the number of
        # fine channels per coarse channel is even. Otherwise it is offset by
        # half of the fine channel width.
        if mwax:
            # calculate coarse channel width in MHz
            coarse_chan_width = meta_hdr["BANDWDTH"] / len(coarse_chans)
            # coarse channel center freq is channel number * coarse channel width
            center_coarse_chan_center = meta_hdr["CENTCHAN"] * coarse_chan_width * 1e6
            # calculate center of first fine channel; this works if the number of
            # fine channels is even or odd
            obs_freq_center = (
                center_coarse_chan_center
                - int(coarse_num_fine_chans / 2) * channel_width
            )
        else:
            obs_freq_center = meta_hdr["FREQCENT"] * 1e6

        # frequency averaging factor
        avg_factor = meta_hdr["NAV_FREQ"]

        # integration time in seconds
        int_time = meta_hdr["INTTIME"]

        # pointing center in degrees
        ra_deg = meta_hdr["RA"]
        dec_deg = meta_hdr["DEC"]
        ra_rad = np.pi * ra_deg / 180
        dec_rad = np.pi * dec_deg / 180

        # set start_flag with goodtime
        if flag_init and start_flag == "goodtime":
            # ppds file does not contain this key
            if "GOODTIME" not in meta_hdr:
                raise ValueError(
                    "To use start_flag='goodtime', a .metafits file must be "
                    "submitted"
                )
            if meta_hdr["GOODTIME"] > start_time:
                start_flag = meta_hdr["GOODTIME"] - start_time
                # round start_flag up to nearest multiple of int_time
                if start_flag % int_time > 0:
                    start_flag = (1 + int(start_flag / int_time)) * int_time
            else:
                start_flag = 0.0

        if "HISTORY" in meta_hdr:
            history = str(meta_hdr["HISTORY"])
            meta_hdr.remove("HISTORY", remove_all=True)
        else:
            history = ""

        object_name = meta_hdr.pop("FILENAME")

        # if not mwax, remove mwax-specific keys
        mwax_keys_to_skip = []
        if not mwax:
            mwax_keys_to_skip = [
                "DELAYMOD",
                "DELDESC",
                "CABLEDEL",
                "GEODEL",
                "CALIBDEL",
            ]
        # store remaining keys in extra keywords
        meta_extra_keywords = uvutils._get_fits_extra_keywords(
            meta_hdr, keywords_to_skip=["DATE-OBS"] + mwax_keys_to_skip
        )

    meta_dict = {
        "telescope_name": telescope_name,
        "telescope_location": mwa_telescope_obj.telescope_location,
        "instrument": instrument,
        "antenna_inds": antenna_inds,
        "antenna_numbers": antenna_numbers,
        "antenna_names": antenna_names,
        "antenna_positions": antenna_positions,
        "flagged_ant_inds": flagged_ant_inds,
        "int_time": int_time,
        "start_flag": start_flag,
        "obs_freq_center": obs_freq_center,
        "avg_factor": avg_factor,
        "coarse_chans": coarse_chans,
        "coarse_num_fine_chans": coarse_num_fine_chans,
        "channel_width": channel_width,
        "dig_gains": dig_gains,
        "cable_lens": cable_lens,
        "ra_rad": ra_rad,
        "dec_rad": dec_rad,
        "history": history,
        "object_name": object_name,
        "extra_keywords": meta_extra_keywords,
    }

    return meta_dict


def sighat_vector(x):
    """
    Generate quantized sigma using Van Vleck relation.

    For an explanation of the Van Vleck relations used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    x : numpy array
        Array of sigma inputs.

    Returns
    -------
    sighat : numpy array
        Array of corresponding sigmas of quantized values.

    """
    yy = np.arange(7)[:, np.newaxis]
    z = (2 * yy + 1) * erf((yy + 0.5) / (x * np.sqrt(2)))
    z = z.sum(axis=0)
    sighat = np.sqrt(7**2 - z)
    return sighat


def sighat_vector_prime(x):
    """
    Calculate the derivative of sighat_vector.

    Parameters
    ----------
    x : numpy array
        Array of sigma inputs.

    Returns
    -------
    sighat : numpy array
        Array of corresponding derivatives with respect to sigma inputs.

    """
    yy = np.arange(7)[:, np.newaxis]
    z = (
        (2 * yy + 1)
        * (yy + 0.5)
        * np.exp(-((yy + 0.5) ** 2) / (2 * (x**2)))
        / (np.sqrt(2 * np.pi) * (x**2))
    )
    sighat_prime = z.sum(axis=0)
    sighat_prime /= sighat_vector(x)
    return sighat_prime


def corrcorrect_simps(rho, sig1, sig2):
    """
    Generate quantized kappa using the Van Vleck relation.

    For an explanation of the Van Vleck relations used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    rho : numpy array
        Array of rho inputs.
    sig1 : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2: numpy array
        Array of sigma inputs corresponding to antenna 2.

    Returns
    -------
    integrated_khat : numpy array
        Array of cross-correlations of quantized values.

    """
    x = np.linspace(0, rho, 11, dtype=np.float64)
    khat = np.zeros((11, rho.size), dtype=np.float64)
    khat = _corr_fits.get_khat(x, sig1, sig2)
    integrated_khat = simpson(khat, x=x, axis=0)
    return integrated_khat


def corrcorrect_vect_prime(rho, sig1, sig2):
    """
    Calculate the derivative of corrcorrect_simps.

    Parameters
    ----------
    rho : numpy array
        Array of rho inputs.
    sig1 : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2: numpy array
        Array of sigma inputs corresponding to antenna 2.

    """
    return _corr_fits.get_khat(rho, sig1, sig2)


def van_vleck_autos(sighat_arr):
    """
    Use Newton's method to solve the inverse of sighat_vector.

    For an explanation of the Van Vleck corrections used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    sighat_arr : numpy array
        Array of quantized sigma to be corrected.

    Returns
    -------
    sighat_arr : numpy array
        Array of Van Vleck corrected scaled auto-correlations.

    """
    # cut off small sigmas that will not converge
    cutoff_inds = np.where(sighat_arr > 0.5)[0]
    sighat = sighat_arr[cutoff_inds]
    if len(sighat) > 0:
        guess = np.copy(sighat)
        inds = np.where(np.abs(sighat_vector(guess) - sighat) > 1e-10)[0]
        while len(inds) != 0:
            guess[inds] -= (
                sighat_vector(guess[inds]) - sighat[inds]
            ) / sighat_vector_prime(guess[inds])
            inds = np.where(np.abs(sighat_vector(guess) - sighat) > 1e-10)[0]
        sighat_arr[cutoff_inds] = guess

    return sighat_arr


def van_vleck_crosses_int(k_arr, sig1_arr, sig2_arr, cheby_approx):
    """
    Use Newton's method to solve the inverse of corrcorrect_simps.

    For an explanation of the Van Vleck corrections used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    k_arr : numpy array
        Array of quantized kappa to be corrected.
    sig1_arr : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2_arr: numpy array
        Array of sigma inputs corresponding to antenna 2.
    cheby_approx : bool
        Flag to warn if chebyshev approximation is being used.

    Returns
    -------
    k_arr : numpy array
        Array of Van Vleck corrected scaled cross-correlations.

    """
    nonzero_inds = np.where((k_arr != 0) & (sig1_arr != 0) & (sig2_arr != 0))[0]
    if len(nonzero_inds) > 0.0:
        if cheby_approx:
            warnings.warn(
                str(len(nonzero_inds))
                + " values are being corrected with the van vleck integral"
            )
        neg_inds = np.where(k_arr < 0.0)[0]
        khat = np.abs(k_arr[nonzero_inds])
        sig1 = sig1_arr[nonzero_inds]
        sig2 = sig2_arr[nonzero_inds]
        x0 = khat / (sig1 * sig2)
        corr = corrcorrect_simps(x0, sig1, sig2) - khat
        x0 -= corr / corrcorrect_vect_prime(x0, sig1, sig2)
        inds = np.where(np.abs(corr) > 1e-8)[0]
        while len(inds) != 0:
            corr = corrcorrect_simps(x0[inds], sig1[inds], sig2[inds]) - khat[inds]
            x0[inds] -= corr / corrcorrect_vect_prime(x0[inds], sig1[inds], sig2[inds])
            inds2 = np.where(np.abs(corr) > 1e-8)[0]
            inds = inds[inds2]
        k_arr[nonzero_inds] = x0 * sig1 * sig2
        k_arr[neg_inds] = np.negative(k_arr[neg_inds])

    return k_arr


def van_vleck_crosses_cheby(
    khat,
    sig1,
    sig2,
    broad_inds,
    rho_coeff,
    sv_inds_right1,
    sv_inds_right2,
    ds1,
    ds2,
    cheby_approx,
):
    """
    Compute a chebyshev approximation of corrcorrect_simps.

    Uses a bilinear interpolation to find chebyshev coefficients. Assumes distance
    between points of interpolation grid is 0.01. If sig1 or sig2 falls outside
    the interpolation grid, the corresponding values are corrected using
    van_vleck_crosses_int.

    For an explanation of the Van Vleck corrections used and their implementation
    in this code, see the memos at
    https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
    https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

    Parameters
    ----------
    khat : numpy array
        Array of quantized kappa to be corrected.
    sig1 : numpy array
        Array of sigma inputs corresponding to antenna 1.
    sig2: numpy array
        Array of sigma inputs corresponding to antenna 2.
    broad_inds : numpy array
        Array indexing sigmas within the chebyshev approximation range.
    rho_coeff : numpy array
        Array of chebyshev polynomial coefficients.
    sv_inds_right1 : numpy array
        Array of right indices for sig1 for bilinear interpolation.
    sv_inds_right2 : numpy array
        Array of right indices for sig2 for bilinear interpolation.
    ds1 : numpy array
        Distance between sig1 and right-indexed value for bilinear interpolation.
    ds2 : numpy array
        Distance between sig2 and right-indexed value for bilinear interpolation.
    cheby_approx : bool
        Flag to warn if chebyshev approximation is being used.

    Returns
    -------
    khat : numpy array
        Array of Van Vleck corrected scaled cross-correlations.

    """
    kap = np.array([khat[broad_inds].real, khat[broad_inds].imag])
    _corr_fits.van_vleck_cheby(kap, rho_coeff, sv_inds_right1, sv_inds_right2, ds1, ds2)
    khat[broad_inds] = (kap[0, :] + 1j * kap[1, :]) * (
        sig1[broad_inds] * sig2[broad_inds]
    )
    khat[~broad_inds] = van_vleck_crosses_int(
        khat.real[~broad_inds], sig1[~broad_inds], sig2[~broad_inds], cheby_approx
    ) + 1j * van_vleck_crosses_int(
        khat.imag[~broad_inds], sig1[~broad_inds], sig2[~broad_inds], cheby_approx
    )

    return khat


class MWACorrFITS(UVData):
    """
    UVData subclass for reading MWA correlator fits files.

    This class should not be interacted with directly; instead use the
    read_mwa_corr_fits method on the UVData class.
    """

    def correct_cable_length(self, cable_lens, ant_1_inds, ant_2_inds):
        """
        Apply a cable length correction to the data array.

        Parameters
        ----------
        cable_lens : list of strings
            A list of strings containing the cable lengths for each antenna.
        ant_1_inds : array
            An array of indices for antenna 1
        ant_2_inds : array
            An array of indices for antenna 2

        """
        # as of version 0.29.X cython does not handle numpy arrays of strings
        # particularly efficiently. Casting to bytes, then into this demonic
        # form is a workaround found here: https://stackoverflow.com/a/28777163
        cable_lens = np.asarray(cable_lens).astype(np.string_)
        cable_lens = cable_lens.view("uint8").reshape(
            cable_lens.size, cable_lens.dtype.itemsize
        )
        # from MWA_Tools/CONV2UVFITS/convutils.h
        cable_len_diffs = _corr_fits.get_cable_len_diffs(
            ant_1_inds, ant_2_inds, cable_lens
        )
        self.data_array *= np.exp(
            -1j
            * 2
            * np.pi
            * cable_len_diffs.reshape(self.Nblts, 1)
            / const.c.to("m/s").value
            * self.freq_array.reshape(1, self.Nfreqs)
        )[:, :, None]
        history_add_string = " Applied cable length correction."
        self.history += history_add_string

    def flag_init(
        self,
        num_fine_chan,
        edge_width=80e3,
        start_flag=2.0,
        end_flag=0.0,
        flag_dc_offset=True,
    ):
        """
        Apply routine flagging to the MWA Correlator FITS file data.

        Includes options to flag the coarse channel edges, beginning and end
        of obs, as well as the center fine channel of each coarse channel.

        Parameters
        ----------
        edge_width: float
            The width to flag on the edge of each coarse channel, in hz. Set to
            0 for no edge flagging.
        start_flag: float
            The number of seconds to flag at the beginning of the observation.
            Set to 0 for no flagging.
        end_flag: floats
            The number of seconds to flag at the end of the observation. Set to
            0 for no flagging.
        flag_dc_offset: bool
            Set to True to flag the center fine channel of each coarse channel.

        Raises
        ------
        ValueError
            If edge_width is not an integer multiple of the channel_width of
            the data (0 also acceptable).
            If start_flag is not an integer multiple of the integration time
            (0 also acceptable).
            If end_flag is not an integer multiple of the integration time
            (0 also acceptable).

        """
        if (edge_width % self.channel_width[0]) > 0:
            raise ValueError(
                "The edge_width must be an integer multiple of the "
                "channel_width of the data or zero."
            )
        if (start_flag % self.integration_time[0]) > 0:
            raise ValueError(
                "The start_flag must be an integer multiple of the "
                "integration_time of the data or zero."
            )
        if (end_flag % self.integration_time[0]) > 0:
            raise ValueError(
                "The end_flag must be an integer multiple of the "
                "integration_time of the data or zero."
            )

        num_ch_flag = int(edge_width / self.channel_width[0])
        num_start_flag = int(start_flag / self.integration_time[0])
        num_end_flag = int(end_flag / self.integration_time[0])

        shape = self.flag_array.shape
        reshape = [self.Ntimes, self.Nbls, self.Nfreqs, self.Npols]

        self.flag_array = (
            self.flag_array
            if (shape == reshape)
            else np.reshape(self.flag_array, reshape)
        )

        bad_chan_inds = []
        if num_ch_flag > 0:
            for ch_count in range(num_ch_flag):
                # count up from the left
                left_chans = list(range(ch_count, self.Nfreqs, num_fine_chan))
                # count down from the right
                right_chans = list(range(self.Nfreqs - 1 - ch_count, 0, -num_fine_chan))
                bad_chan_inds += left_chans + right_chans

        if flag_dc_offset:
            bad_chan_inds += list(range(num_fine_chan // 2, self.Nfreqs, num_fine_chan))

        if len(bad_chan_inds) != 0:
            self.flag_array[:, :, bad_chan_inds, :] = True

        if (num_start_flag > 0) or (num_end_flag > 0):
            if num_start_flag > 0:
                self.flag_array[:num_start_flag] = True
            if num_end_flag > 0:
                self.flag_array[-num_end_flag:] = True
            self.flag_array = np.reshape(self.flag_array, shape)

        self.flag_array = (
            self.flag_array
            if (shape == reshape)
            else np.reshape(self.flag_array, shape)
        )

    def _read_fits_file(
        self,
        filename,
        time_array,
        file_nums,
        num_fine_chans,
        int_time,
        mwax,
        map_inds,
        conj,
        pol_index_array,
    ):
        """
        Read the fits file and populate into memory.

        This is an internal function and should not regularly be called except
        by read_mwa_corr_fits function.

        It is designed to close the fits files, headers, and all associated pointers.
        Without this read in a function, reading files has a large memory footprint.

        Parameters
        ----------
        filename : str
            The mwa gpubox fits file to read
        time_array : array of floats
            The time_array object constructed during read_mwa_corr_fits call
        file_nums : array
            List of included file numbers ordered by coarse channel
        num_fine_chans : int
            Number of fine channels in each data file
        int_time : float
            The integration time of each observation.
        map_inds : array
            Indices for reordering data_array from weird correlator packing.
        conj : array
            Indices for conjugating data_array from weird correlator packing.
        pol_index_array : array
            Indices for reordering polarizations to the 'AIPS' convention

        """
        # get the file number from the file name
        if mwax:
            file_num = int(filename.split("_")[-2][-3:])
        else:
            file_num = int(filename.split("_")[-2][-2:])
        # map file number to frequency index
        freq_ind = np.where(file_nums == file_num)[0][0] * num_fine_chans
        # get a coarse channel index for flag array
        coarse_ind = np.where(file_nums == file_num)[0][0]
        # create an intermediate array for data
        if mwax:
            coarse_chan_data = np.zeros(
                (self.Ntimes, self.Nbls, num_fine_chans * self.Npols),
                dtype=np.complex64,
            )
        else:
            coarse_chan_data = np.zeros(
                (self.Ntimes, num_fine_chans, self.Nbls * self.Npols),
                dtype=np.complex64,
            )
        with fits.open(filename, mode="denywrite") as hdu_list:
            # if mwax, data is in every other hdu
            if mwax:
                hdu_list = hdu_list[1::2]
            for hdu in hdu_list:
                # entry 0 is a header, so we skip it.
                if hdu.data is None:
                    continue
                time = (
                    hdu.header["TIME"]
                    + hdu.header["MILLITIM"] / 1000.0
                    + int_time / 2.0
                )
                time_ind = np.where(time_array == time)[0][0]
                # dump data into matrix
                # and take data from real to complex numbers
                coarse_chan_data.view(np.float32)[time_ind, :, :] = hdu.data
                # fill nsample and flag arrays
                # think about using the mwax weights array in the future
                self.nsample_array[
                    time_ind, :, freq_ind : freq_ind + num_fine_chans, :
                ] = 1.0
                self.flag_array[time_ind, :, coarse_ind, :] = False
        if not mwax:
            # do mapping and reshaping here to avoid copying whole data_array
            np.take(coarse_chan_data, map_inds, axis=2, out=coarse_chan_data)
            # conjugate data
            coarse_chan_data[:, :, conj] = np.conj(coarse_chan_data[:, :, conj])
        # reshape
        if mwax:
            coarse_chan_data = coarse_chan_data.reshape(
                (self.Ntimes, self.Nbls, num_fine_chans, self.Npols)
            )
        else:
            coarse_chan_data = coarse_chan_data.reshape(
                (self.Ntimes, num_fine_chans, self.Nbls, self.Npols)
            )
            coarse_chan_data = np.swapaxes(coarse_chan_data, 1, 2)
        coarse_chan_data = coarse_chan_data.reshape(
            self.Nblts, num_fine_chans, self.Npols
        )
        # reorder pols here to avoid memory spike from self.reorder_pols
        np.take(coarse_chan_data, pol_index_array, axis=-1, out=coarse_chan_data)
        # make a mask where data actually is so coarse channels that
        # are split into two files don't overwrite eachother
        data_mask = coarse_chan_data != 0
        self.data_array[:, freq_ind : freq_ind + num_fine_chans, :][data_mask] = (
            coarse_chan_data[data_mask]
        )

        return

    def _read_flag_file(self, filename, file_nums, num_fine_chans):
        """
        Read aoflagger flag file into flag_array.

        Parameters
        ----------
        filename : str
            The aoflagger fits file to read.
        file_nums : array
            List of included file numbers ordered by coarse channel.
        num_fine_chans : int
            Number of fine channels in each data file.

        """
        flag_num = int(filename.split("_")[-1][0:2])
        # map file number to frequency index
        freq_ind = np.where(file_nums == flag_num)[0][0] * num_fine_chans
        with fits.open(filename, mode="denywrite") as aoflags:
            flags = aoflags[1].data.field("FLAGS")
        # some flag files are longer than data; crop the ends
        flags = flags[: self.Nblts, :]
        # some flag files are shorter than data; assume same end time
        blt_ind = self.Nblts - len(flags)
        flags = flags[:, :, np.newaxis]
        self.flag_array[blt_ind:, freq_ind : freq_ind + num_fine_chans, :] = (
            np.logical_or(
                self.flag_array[blt_ind:, freq_ind : freq_ind + num_fine_chans, :],
                flags,
            )
        )

    def van_vleck_correction(
        self, ant_1_inds, ant_2_inds, flagged_ant_inds, cheby_approx, data_array_dtype
    ):
        """
        Apply a van vleck correction to the data array.

        For an explanation of the Van Vleck corrections used and their implementation
        in this code, see the memos at
        https://github.com/EoRImaging/Memos/blob/master/PDFs/007_Van_Vleck_A.pdf and
        https://github.com/EoRImaging/Memos/blob/master/PDFs/008_Van_Vleck_B.pdf

        Parameters
        ----------
        ant_1_inds : array
            An array of indices for antenna 1.
        ant_2_inds : array
            An array of indices for antenna 2.
        flagged_ant_inds : numpy array of type int
            List of indices of flagged antennas.
        cheby_approx : bool
            Option to implement the van vleck correction with a chebyshev polynomial.
            approximation.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as.

        """
        history_add_string = " Applied Van Vleck correction."
        # reshape to (nbls, ntimes, nfreqs, npols)
        self.data_array = self.data_array.reshape(
            self.Ntimes, self.Nbls, self.Nfreqs, self.Npols
        )
        self.data_array = np.swapaxes(self.data_array, 0, 1)
        # combine axes
        self.data_array = self.data_array.reshape(
            (self.Nbls, self.Nfreqs * self.Ntimes, self.Npols)
        )
        # need data array to have 64 bit precision
        # work on this in the future to only change precision where necessary
        if self.data_array.dtype != np.complex128:
            self.data_array = self.data_array.astype(np.complex128)

        # scale the data
        # number of samples per fine channel is equal to channel width (Hz)
        # multiplied be the integration time (s)
        # circular symmetry gives a factor of two
        nsamples = self.channel_width[0] * self.integration_time[0] * 2
        self.data_array /= nsamples
        # get indices for autos
        autos = np.where(
            self.ant_1_array[0 : self.Nbls] == self.ant_2_array[0 : self.Nbls]
        )[0]
        # get indices for crosses
        crosses = np.where(
            self.ant_1_array[0 : self.Nbls] != self.ant_2_array[0 : self.Nbls]
        )[0]
        # find polarizations
        xx = np.where(self.polarization_array == -5)[0][0]
        yy = np.where(self.polarization_array == -6)[0][0]
        xy = np.where(self.polarization_array == -7)[0][0]
        yx = np.where(self.polarization_array == -8)[0][0]
        pols = np.array([yy, xx])
        # square root autos
        auto_inds = autos[:, np.newaxis]
        self.data_array.real[auto_inds, :, pols] = np.sqrt(
            self.data_array.real[auto_inds, :, pols]
        )
        # get unflagged autos
        good_autos = np.delete(autos, flagged_ant_inds)
        sighat = self.data_array.real[good_autos[:, np.newaxis], :, pols].flatten()
        # correct autos
        sigma = van_vleck_autos(sighat)
        self.data_array.real[good_autos[:, np.newaxis], :, pols] = sigma.reshape(
            len(good_autos), len(pols), self.Ntimes * self.Nfreqs
        )
        # get good crosses
        bad_ant_inds = np.nonzero(
            np.logical_or(
                np.isin(ant_1_inds[0 : self.Nbls], flagged_ant_inds),
                np.isin(ant_2_inds[0 : self.Nbls], flagged_ant_inds),
            )
        )[0]
        crosses = np.delete(crosses, np.nonzero(np.isin(crosses, bad_ant_inds))[0])
        # correct crosses
        if cheby_approx:
            history_add_string += " Used Van Vleck Chebychev approximation."
            # load in interpolation files
            with h5py.File(DATA_PATH + "/mwa_config_data/Chebychev_coeff.h5", "r") as f:
                rho_coeff = f["rho_data"][:]
            with h5py.File(DATA_PATH + "/mwa_config_data/sigma1.h5", "r") as f:
                sig_vec = f["sig_data"][:]
            sigs = self.data_array.real[autos[:, np.newaxis], :, pols]
            # find sigmas within interpolation range
            in_inds = np.logical_and(sigs > 0.9, sigs <= 4.5)
            # get indices and distances for bilinear interpolation
            sv_inds_right = np.zeros(in_inds.shape, dtype=np.int64)
            ds = np.zeros(in_inds.shape)
            sv_inds_right[in_inds] = np.searchsorted(sig_vec, sigs[in_inds])
            ds[in_inds] = sig_vec[sv_inds_right[in_inds]] - sigs[in_inds]
            # get indices for sigmas corresponding to crosses
            sig1_inds = ant_1_inds[crosses]
            sig2_inds = ant_2_inds[crosses]
            # iterate over polarization
            pol_dict = {
                yy: [(yy, yy), (0, 0)],
                yx: [(yy, xx), (0, 1)],
                xy: [(xx, yy), (1, 0)],
                xx: [(xx, xx), (1, 1)],
            }
            for i in [xx, yy, xy, yx]:
                (pol1, pol2) = pol_dict[i][1]
                (sig1_pol, sig2_pol) = pol_dict[i][0]
                # broadcast in_inds
                broad_inds = np.logical_and(
                    in_inds[sig1_inds, pol1, :], in_inds[sig2_inds, pol2, :]
                )
                # broadcast indices and distances for bilinear interpolation
                sv_inds_right1 = sv_inds_right[sig1_inds, pol1, :][broad_inds]
                sv_inds_right2 = sv_inds_right[sig2_inds, pol2, :][broad_inds]
                ds1 = ds[sig1_inds, pol1, :][broad_inds]
                ds2 = ds[sig2_inds, pol2, :][broad_inds]
                self.data_array[crosses, :, i] = van_vleck_crosses_cheby(
                    self.data_array[crosses, :, i],
                    self.data_array.real[autos[sig1_inds], :, sig1_pol],
                    self.data_array.real[autos[sig2_inds], :, sig2_pol],
                    broad_inds,
                    rho_coeff,
                    sv_inds_right1,
                    sv_inds_right2,
                    ds1,
                    ds2,
                    cheby_approx,
                )
            # correct yx autos
            sig_inds = ant_1_inds[good_autos]
            broad_inds = np.logical_and(
                in_inds[sig_inds, 0, :], in_inds[sig_inds, 1, :]
            )
            sv_inds_right1 = sv_inds_right[sig_inds, 0, :][broad_inds]
            sv_inds_right2 = sv_inds_right[sig_inds, 1, :][broad_inds]
            ds1 = ds[sig_inds, 0, :][broad_inds]
            ds2 = ds[sig_inds, 1, :][broad_inds]
            self.data_array[good_autos, :, yx] = van_vleck_crosses_cheby(
                self.data_array[good_autos, :, yx],
                self.data_array.real[good_autos, :, yy],
                self.data_array.real[good_autos, :, xx],
                broad_inds,
                rho_coeff,
                sv_inds_right1,
                sv_inds_right2,
                ds1,
                ds2,
                cheby_approx,
            )
            # add back in frequency axis
            self.data_array = self.data_array.reshape(
                (self.Nbls, self.Ntimes, self.Nfreqs, self.Npols)
            )
        # solve integral directly
        else:
            # add back in frequency axis
            self.data_array = self.data_array.reshape(
                (self.Nbls, self.Ntimes, self.Nfreqs, self.Npols)
            )
            for k in crosses:
                auto1 = autos[ant_1_inds[k]]
                auto2 = autos[ant_2_inds[k]]
                for j in range(self.Nfreqs):
                    # get data
                    sig1 = self.data_array.real[
                        auto1, :, j, np.array([yy, yy, xx, xx])
                    ].flatten()
                    sig2 = self.data_array.real[
                        auto2, :, j, np.array([yy, xx, yy, xx])
                    ].flatten()
                    khat = self.data_array[
                        k, :, j, np.array([yy, yx, xy, xx])
                    ].flatten()
                    # correct real
                    kap = van_vleck_crosses_int(khat.real, sig1, sig2, cheby_approx)
                    self.data_array.real[k, :, j, np.array([yy, yx, xy, xx])] = (
                        kap.reshape(self.Npols, self.Ntimes)
                    )
                    # correct imaginary
                    kap = van_vleck_crosses_int(khat.imag, sig1, sig2, cheby_approx)
                    self.data_array.imag[k, :, j, np.array([yy, yx, xy, xx])] = (
                        kap.reshape(self.Npols, self.Ntimes)
                    )
            # correct yx autos
            for k in good_autos:
                for j in range(self.Nfreqs):
                    # get data
                    sig1 = self.data_array.real[k, :, j, yy]
                    sig2 = self.data_array.real[k, :, j, xx]
                    khat = self.data_array[k, :, j, yx]
                    # correct real
                    kap = van_vleck_crosses_int(khat.real, sig1, sig2, cheby_approx)
                    self.data_array.real[k, :, j, yx] = kap
                    # correct imaginary
                    kap = van_vleck_crosses_int(khat.imag, sig1, sig2, cheby_approx)
                    self.data_array.imag[k, :, j, yx] = kap
        # correct xy autos
        self.data_array[good_autos, :, :, xy] = np.conj(
            self.data_array[good_autos, :, :, yx]
        )
        # square autos
        self.data_array.real[auto_inds, :, :, pols] = (
            self.data_array.real[auto_inds, :, :, pols] ** 2
        )
        # reshape to (nblts, nfreqs, npols)
        self.data_array = np.swapaxes(self.data_array, 0, 1)
        self.data_array = self.data_array.reshape(self.Nblts, self.Nfreqs, self.Npols)
        # rescale the data
        self.data_array *= nsamples
        # return data array to desired precision
        if self.data_array.dtype != data_array_dtype:
            self.data_array = self.data_array.astype(data_array_dtype)
        self.history += history_add_string

    def _flag_small_auto_ants(
        self, nsamples, flag_small_auto_ants, ant_1_inds, ant_2_inds, flagged_ant_inds
    ):
        """
        Find and flag autocorrelations below a threshold.

        Specifically, look for autocorrelations < 0.5 * channel_width * int_time,
        as these have been found by the Van Vleck correction to indicate bad data.
        If flag_small_auto_ants is True, then antennas with autos below the
        threshold will be flagged completely. Otherwise, antennas will be flagged
        at only the times and frequencies at which their autos are below the threshold.

        Parameters
        ----------
        nsamples : int
            Twice the numkber of electric field samples in an autocorrelation; equal
            to 2 * channel_width * int_time. The auto divided by nsamples is equal to
            the expectation value of the electric field samples squared.
        flag_small_auto_ants : bool
            Keyword option to flag antenna entirely or only at specific times and
            frequencies.
        ant_1_inds : numpy array of type int
            Indices of antenna 1 corresponding the the baseline-time axis.
        ant_2_inds : numpy array of type int
            Indices of antenna 2 corresponding the the baseline-time axis.
        flagged_ant_inds : numpy array of type int
            List of indices of flagged antennas.

        Returns
        -------
        flagged_ant_inds : numpy array of type int
            Updated list of indices of flagged antennas.

        """
        # calculate threshold so that average cross multiply = 0.25
        threshold = 0.25 * nsamples
        # look for small autos and flag
        auto_inds = self.ant_1_array == self.ant_2_array
        autos = self.data_array.real[auto_inds, :, 0:2]
        autos = autos.reshape(self.Ntimes, self.Nants_data, self.Nfreqs, 2)
        # find autos below threshold
        small_auto_flags = np.logical_and(autos != 0, autos <= threshold)
        if flag_small_auto_ants:
            # find antenna indices for small sig ants and add to flagged_ant_inds
            ant_inds = np.unique(np.nonzero(small_auto_flags)[1])
            ant_inds = ant_inds[~np.in1d(ant_inds, flagged_ant_inds)]
            if len(ant_inds) != 0:
                self.history += (
                    " The following antennas were flagged by the Van Vleck correction: "
                    + str(ant_inds)
                    + "."
                )
                flagged_ant_inds = np.concatenate((flagged_ant_inds, ant_inds))
        else:
            # get flags for small auto ants and add to flag array
            small_auto_flags = np.logical_or(
                small_auto_flags[:, :, :, 0], small_auto_flags[:, :, :, 1]
            )
            # broadcast autos flags to corresponding crosses
            small_auto_flags = np.logical_or(
                small_auto_flags[:, ant_1_inds[: self.Nbls], :],
                small_auto_flags[:, ant_2_inds[: self.Nbls], :],
            )
            small_auto_flags = small_auto_flags.reshape(self.Nblts, self.Nfreqs)
            self.flag_array = np.logical_or(
                self.flag_array, small_auto_flags[:, :, np.newaxis]
            )
        return flagged_ant_inds

    def _get_pfb_shape(self, avg_factor, mwax):
        """
        Get pfb shape from file and apply appropriate averaging.

        Parameters
        ----------
        avg_factor : int
            Factor by which frequency channels have been averaged.

        Returns
        -------
        cb_array : numpy array of type float
            Array corresponding to pfb shape for a coarse band.
        """
        if mwax:
            if self.channel_width[0] == 40000:
                with h5py.File(
                    DATA_PATH + "/mwa_config_data/mwax_pfb_bandpass_40kHz.h5", "r"
                ) as f:
                    cb_array = f["coarse_band"][:]
            elif self.channel_width[0] == 80000:
                with h5py.File(
                    DATA_PATH + "/mwa_config_data/mwax_pfb_bandpass_80kHz.h5", "r"
                ) as f:
                    cb_array = f["coarse_band"][:]
            else:
                raise ValueError(
                    "mwax passband shapes are only available for 40 kHz and 80 kHz"
                    "fine channel widths. To request a passband for a different fine "
                    "channel width create an issue on the pyuvdata repository. To run "
                    "without a passband, resubmit with correct_coarse_band=False"
                )
        else:
            with h5py.File(
                DATA_PATH + "/mwa_config_data/MWA_rev_cb_10khz_doubles.h5", "r"
            ) as f:
                cb = f["coarse_band"][:]
            cb_array = cb.reshape(int(128 / avg_factor), int(avg_factor))
            cb_array = np.average(cb_array, axis=1)

        return cb_array

    def _correct_coarse_band(
        self,
        cb_num,
        ant_1_inds,
        ant_2_inds,
        cb_array,
        dig_gains,
        nsamples,
        num_fine_chans,
        correct_van_vleck,
        remove_coarse_band,
        remove_dig_gains,
    ):
        """
        Apply pfb, digital gain, and Van Vleck corrections to a coarse band.

        Parameters
        ----------
        cb_num : int
            Index of coarse band.
        ant_1_inds : numpy array of type int
            Indices of antenna 1 corresponding the the baseline-time axis.
        ant_2_inds : numpy array of type int
            Indices of antenna 2 corresponding the the baseline-time axis.
        cb_array : numpy array of type float
            Array corresponding to pfb shape for a coarse band.
        dig_gains : numpy array of type float
            Array corresponding to digital gains for each antenna and coarse band.
        nsamples : int
            Twice the numkber of electric field samples in an autocorrelation; equal
            to 2 * channel_width * int_time. The auto divided by nsamples is equal to
            the expectation value of the electric field sample squared.
        num_fine_chans : int
            Number of fine channels in each data file.
        correct_van_vleck : bool
            Option to apply Van Vleck correction to data.
        remove_coarse_band : bool
            Option to remove pfb coarse band shape from data.
        remove_dig_gains : bool
            Option to remove digital gains from data.

        """
        # get coarse band data as np.complex128
        cb_data = self.data_array[
            :, cb_num * num_fine_chans : (cb_num + 1) * num_fine_chans, :
        ].astype(np.complex128)
        # remove digital gains
        if remove_dig_gains:
            dig_gains1 = dig_gains[ant_1_inds, cb_num, np.newaxis, np.newaxis]
            dig_gains2 = dig_gains[ant_2_inds, cb_num, np.newaxis, np.newaxis]
            cb_data /= dig_gains1
            cb_data /= dig_gains2
        # remove coarse band
        if remove_coarse_band:
            cb_data /= cb_array[:num_fine_chans, np.newaxis]
        # put corrected data back into data array
        self.data_array[
            :, cb_num * num_fine_chans : (cb_num + 1) * num_fine_chans, :
        ] = cb_data

    def _apply_corrections(
        self,
        mwax,
        ant_1_inds,
        ant_2_inds,
        avg_factor,
        dig_gains,
        spw_inds,
        num_fine_chans,
        flagged_ant_inds,
        cheby_approx,
        data_array_dtype,
        flag_small_auto_ants,
        correct_van_vleck,
        remove_coarse_band,
        remove_dig_gains,
    ):
        """
        Prepare and apply pfb, digital gain, and Van Vleck corrections.

        Parameters
        ----------
        ant_1_inds : numpy array of type int
            Indices of antenna 1 corresponding the the baseline-time axis.
        ant_2_inds : numpy array of type int
            Indices of antenna 2 corresponding the the baseline-time axis.
        avg_factor : int
            Factor by which frequency channels have been averaged.
        dig_gains : array
            Array of digital gains with shape (Nants, Ncoarse_chans).
        spw_inds : array of type int
            Array of coarse band numbers.
        num_fine_chans : int
            Number of fine channels in each data file.
        flagged_ant_inds : numpy array of type int
            List of indices of flagged antennas.
        cheby_approx : bool
            Option to use chebyshev approximation for Van Vleck correction.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as.
        flag_small_auto_ants : bool
            Option to completely flag antennas found by _flag_small_auto_ants.
        correct_van_vleck : bool
            Option to apply Van Vleck correction to data.
        remove_coarse_band : bool
            Option to remove pfb coarse band shape from data.
        remove_dig_gains : bool
            Option to remove digital gains from data.

        Returns
        -------
        flagged_ant_inds : numpy array of type int
            Updated list of indices of flagged antennas

        """
        # get nsamples and check for small auto ants
        if correct_van_vleck:
            self.history += " Applied Van Vleck correction."
            # calculate number of samples going into real or imaginary part
            # factor of two comes from variables being circularly-symmetric
            nsamples = self.channel_width[0] * self.integration_time[0] * 2
            # look for small auto data and flag
            flagged_ant_inds = self._flag_small_auto_ants(
                nsamples, flag_small_auto_ants, ant_1_inds, ant_2_inds, flagged_ant_inds
            )
        else:
            nsamples = None
        # get digital gains
        if remove_dig_gains:
            self.history += " Divided out digital gains."
            # get gains for included coarse channels
            # During commissioning a shift in the bit selection in the digital
            # receiver was implemented which changed the data scaling by
            # a factor of 64. To be compatible with the earlier scaling scheme,
            # the digital gains are divided by a factor of 64 here.
            # For a more detailed explanation, see PR #908.
            dig_gains = dig_gains[:, spw_inds] / 64
        else:
            dig_gains = None
        # get pfb response shape
        if remove_coarse_band:
            self.history += " Divided out pfb coarse channel bandpass."
            cb_array = self._get_pfb_shape(avg_factor, mwax)
        else:
            cb_array = None

        # apply corrections to each coarse band
        for i in range(len(spw_inds)):
            self._correct_coarse_band(
                i,
                ant_1_inds,
                ant_2_inds,
                cb_array,
                dig_gains,
                nsamples,
                num_fine_chans,
                correct_van_vleck,
                remove_coarse_band,
                remove_dig_gains,
            )

        return flagged_ant_inds

    @copy_replace_short_description(
        UVData.read_mwa_corr_fits, style=DocstringStyle.NUMPYDOC
    )
    def read_mwa_corr_fits(
        self,
        filelist,
        use_aoflagger_flags=None,
        remove_dig_gains=True,
        remove_coarse_band=True,
        correct_cable_len=True,
        correct_van_vleck=False,
        cheby_approx=True,
        flag_small_auto_ants=True,
        phase_to_pointing_center=False,
        propagate_coarse_flags=True,
        flag_init=True,
        edge_width=80e3,
        start_flag="goodtime",
        end_flag=0.0,
        flag_dc_offset=True,
        remove_flagged_ants=True,
        background_lsts=True,
        read_data=True,
        data_array_dtype=np.complex64,
        nsample_array_dtype=np.float32,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=True,
        use_future_array_shapes=False,
        astrometry_library=None,
    ):
        """Read in MWA correlator gpu box files."""
        metafits_file = None
        ppds_file = None
        obs_id = None
        file_dict = {}
        start_time = 0.0
        end_time = 0.0
        included_file_nums = []
        included_flag_nums = []
        aoflagger_warning = False
        num_fine_chans = 0
        mwax = None

        # do datatype checks
        if data_array_dtype not in (np.complex64, np.complex128):
            raise ValueError("data_array_dtype must be np.complex64 or np.complex128")
        if nsample_array_dtype not in (np.float64, np.float32, np.float16):
            raise ValueError(
                "nsample_array_dtype must be one of: np.float64, np.float32, np.float16"
            )
        # do start_flag check
        if not isinstance(start_flag, (int, float)):
            if start_flag != "goodtime":
                raise ValueError("start_flag must be int or float or 'goodtime'")

        # set future array shapes
        self._set_future_array_shapes()

        # iterate through files and organize
        # create a list of included file numbers
        # find the first and last times that have data
        for filename in filelist:
            # update filename attribute
            basename = os.path.basename(filename)
            self.filename = uvutils._combine_filenames(self.filename, [basename])
            self._filename.form = (len(self.filename),)

            if filename.lower().endswith(".metafits"):
                # force only one metafits file
                if metafits_file is not None:
                    raise ValueError("multiple metafits files in filelist")
                metafits_file = filename
            elif filename.lower().endswith(".fits"):
                with fits.open(filename, memmap=True) as hdu_list:
                    hdunames = uvutils._fits_indexhdus(hdu_list)
                    if "PPDS" in hdunames.keys():
                        ppds_file = filename
                        ppd_meta_header = hdu_list[0].header
                        ppd_extra_keywords = uvutils._get_fits_extra_keywords(
                            ppd_meta_header,
                            keywords_to_skip=["DATE-OBS", "TELESCOP", "INSTRUME"],
                        )
                    else:
                        # check obsid
                        head0 = hdu_list[0].header
                        if obs_id is None:
                            obs_id = head0["OBSID"]
                        else:
                            if head0["OBSID"] != obs_id:
                                raise ValueError(
                                    "files from different observations submitted "
                                    "in same list"
                                )
                        # check if mwax
                        if mwax is None:
                            if "CORR_VER" in head0.keys():
                                mwax = True
                                # save mwax version #s into extra_keywords
                                self.extra_keywords["U2S_VER"] = head0["U2S_VER"]
                                self.extra_keywords["CBF_VER"] = head0["CBF_VER"]
                                self.extra_keywords["DB2F_VER"] = head0["DB2F_VER"]
                            else:
                                mwax = False
                        # check headers for first and last times containing data
                        headstart = hdu_list[1].header
                        headfin = hdu_list[-1].header
                        first_time = headstart["TIME"] + headstart["MILLITIM"] / 1000.0
                        last_time = headfin["TIME"] + headfin["MILLITIM"] / 1000.0
                        if start_time == 0.0:
                            start_time = first_time
                        # check that files with a timing offset can be aligned
                        elif np.abs(start_time - first_time) % head0["INTTIME"] != 0.0:
                            raise ValueError(
                                "coarse channel start times are misaligned by an "
                                "amount =that is not an integer multiple of the "
                                "integration time"
                            )
                        elif start_time > first_time:
                            start_time = first_time
                        if end_time < last_time:
                            end_time = last_time
                        # get number of fine channels in each coarse channel
                        if num_fine_chans == 0:
                            if mwax:
                                # number of fine channels is multiplied by 4 (pols)
                                # and by 2 (real and imaginary parts)
                                num_fine_chans = int(headstart["NAXIS1"] / 8)
                            else:
                                num_fine_chans = headstart["NAXIS2"]
                        else:
                            if mwax:
                                if num_fine_chans != int(headstart["NAXIS1"] / 8):
                                    raise ValueError(
                                        "files submitted have different numbers of "
                                        "fine channels"
                                    )
                            else:
                                if num_fine_chans != headstart["NAXIS2"]:
                                    raise ValueError(
                                        "files submitted have different numbers of "
                                        "fine channels"
                                    )

                        # get the file number from the file name;
                        # this will later be mapped to a coarse channel
                        if mwax:
                            file_num = int(filename.split("_")[-2][-3:])
                        else:
                            file_num = int(filename.split("_")[-2][-2:])
                        if file_num not in included_file_nums:
                            included_file_nums.append(file_num)
                        # organize files
                        if "data" not in file_dict.keys():
                            file_dict["data"] = [filename]
                        else:
                            file_dict["data"].append(filename)

                        # save bscale keyword
                        # look for bscale in the first hdu, as some data does not
                        # record it in the zeroth hdu
                        if not mwax:
                            if "SCALEFAC" not in self.extra_keywords.keys():
                                if "BSCALE" in headstart.keys():
                                    self.extra_keywords["SCALEFAC"] = headstart[
                                        "BSCALE"
                                    ]
                                else:
                                    # correlator did a divide by 4 before october 2014
                                    self.extra_keywords["SCALEFAC"] = 0.25

            # look for flag files
            elif filename.lower().endswith(".mwaf"):
                if use_aoflagger_flags is None:
                    use_aoflagger_flags = True
                flag_num = int(filename.split("_")[-1][0:2])
                included_flag_nums.append(flag_num)
                if use_aoflagger_flags is False and aoflagger_warning is False:
                    warnings.warn("mwaf files submitted with use_aoflagger_flags=False")
                    aoflagger_warning = True
                elif "flags" not in file_dict.keys():
                    file_dict["flags"] = [filename]
                else:
                    file_dict["flags"].append(filename)
            else:
                raise ValueError("only fits, metafits, and mwaf files supported")

        # checks:
        if metafits_file is None and ppds_file is None:
            raise ValueError("no metafits file submitted")
        elif metafits_file is None:
            metafits_file = ppds_file
        if "data" not in file_dict.keys():
            raise ValueError("no data files submitted")
        if "flags" not in file_dict.keys() and use_aoflagger_flags:
            raise ValueError(
                "no flag files submitted. Rerun with flag files or "
                "use_aoflagger_flags=False"
            )

        # reorder file numbers
        included_file_nums = sorted(included_file_nums)
        included_flag_nums = sorted(included_flag_nums)

        # first set parameters that are always true
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.vis_units = "uncalib"
        self.Npols = 4
        self.xorientation = "east"

        meta_dict = read_metafits(
            metafits_file,
            mwax=mwax,
            flag_init=flag_init,
            start_flag=start_flag,
            start_time=start_time,
            telescope_info_only=False,
        )

        self.telescope_name = meta_dict["telescope_name"]
        self.telescope_location = meta_dict["telescope_location"]
        self.instrument = meta_dict["instrument"]
        self.antenna_numbers = meta_dict["antenna_numbers"]
        self.antenna_names = meta_dict["antenna_names"]
        self.antenna_positions = meta_dict["antenna_positions"]
        self.history = meta_dict["history"]
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
        for key, value in meta_dict["extra_keywords"].items():
            self.extra_keywords[key] = value
        if ppds_file is not None:
            # get any unique ones from ppd file
            for key, value in ppd_extra_keywords.items():
                if key not in self.extra_keywords.keys():
                    self.extra_keywords[key] = value

        # set parameters from other parameters
        self.Nants_telescope = len(self.antenna_numbers)
        self.Nants_data = len(self.antenna_numbers)
        self.Nbls = int(
            len(self.antenna_numbers) * (len(self.antenna_numbers) + 1) / 2.0
        )
        if phase_to_pointing_center:
            # use another name to prevent name collision in phase call below
            cat_name = "unprojected"
        else:
            cat_name = meta_dict["object_name"]
        cat_id = self._add_phase_center(cat_name=cat_name, cat_type="unprojected")

        # build time array of centers
        time_array = np.arange(
            start_time + meta_dict["int_time"] / 2.0,
            end_time + meta_dict["int_time"] / 2.0 + meta_dict["int_time"],
            meta_dict["int_time"],
        )

        # convert to time to jd floats
        float_time_array = Time(time_array, format="unix", scale="utc").jd.astype(float)
        # build into time array
        self.time_array = np.repeat(float_time_array, self.Nbls)

        self.Ntimes = len(time_array)

        self.Nblts = int(self.Nbls * self.Ntimes)
        self.phase_center_id_array = np.zeros(self.Nblts, dtype=int) + cat_id

        # convert times to lst
        proc = self.set_lsts_from_time_array(
            background=background_lsts, astrometry_library=astrometry_library
        )

        self.integration_time = np.full((self.Nblts), meta_dict["int_time"])

        # make initial antenna arrays, where ant_1 <= ant_2
        # itertools.combinations_with_replacement returns
        # all pairs in the range 0...Nants_telescope
        # including pairs with the same number (e.g. (0,0) auto-correlation).
        # this is a little faster than having nested for-loops moving over the
        # upper triangle of antenna-pair combinations matrix.
        ant_1_array, ant_2_array = np.transpose(
            list(itertools.combinations_with_replacement(self.antenna_numbers, 2))
        )

        self.ant_1_array = np.tile(np.array(ant_1_array), self.Ntimes)
        self.ant_2_array = np.tile(np.array(ant_2_array), self.Ntimes)

        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )

        # make antenna index arrays
        ant_1_inds, ant_2_inds = np.transpose(
            list(itertools.combinations_with_replacement(np.arange(self.Nants_data), 2))
        )
        ant_1_inds = np.tile(np.array(ant_1_inds), self.Ntimes).astype(np.int_)
        ant_2_inds = np.tile(np.array(ant_2_inds), self.Ntimes).astype(np.int_)

        if not mwax:
            # coarse channel mapping for the legacy correlator:
            # channels in group 0-128 are assigned to files in order;
            # channels in group 129-155 are assigned in reverse order
            # that is, if the lowest channel is 127, it will be assigned to the
            # first file
            # channel 128 will be assigned to the second file
            # then the highest channel will be assigned to the third file
            # and the next hightest channel assigned to the fourth file, and so on
            mapped_coarse_chans = np.concatenate(
                (
                    meta_dict["coarse_chans"][meta_dict["coarse_chans"] <= 128],
                    np.flip(meta_dict["coarse_chans"][meta_dict["coarse_chans"] > 128]),
                )
            )
            ordered_file_nums = np.arange(len(meta_dict["coarse_chans"]))[
                np.argsort(mapped_coarse_chans)
            ]
            ordered_file_nums += 1
        else:
            # for mwax, the file numbers are the coarse channel numbers
            ordered_file_nums = meta_dict["coarse_chans"]
        file_mask = np.isin(ordered_file_nums, included_file_nums)
        # get included file numbers in coarse band order
        file_nums = ordered_file_nums[file_mask]
        self.Nfreqs = len(included_file_nums) * num_fine_chans

        # check that coarse channels are contiguous.
        spw_inds = np.nonzero(file_mask)[0]
        if np.any(np.diff(spw_inds) > 1):
            warnings.warn("coarse channels are not contiguous for this observation")
            # add spectral windows
            self._set_flex_spw()
            self.Nspws = len(spw_inds)
            self.spw_array = meta_dict["coarse_chans"][spw_inds]
            self.flex_spw_id_array = np.repeat(self.spw_array, num_fine_chans)
        else:
            # future proof: always set the fles_spw_id_array
            self.flex_spw_id_array = np.full(self.Nfreqs, self.spw_array[0], dtype=int)

        # warn user if not all coarse channels are included
        if len(included_file_nums) != len(meta_dict["coarse_chans"]):
            warnings.warn("some coarse channel files were not submitted")

        # build frequency array
        self.freq_array = np.zeros(self.Nfreqs)
        self.channel_width = np.full(self.Nfreqs, meta_dict["channel_width"])
        # Use the center frequency of the first fine channel of the center coarse
        # channel to get the frequency range for each included coarse channel.
        center_coarse_chan = int(len(meta_dict["coarse_chans"]) / 2)
        for i in range(len(spw_inds)):
            first_coarse_freq = (
                meta_dict["obs_freq_center"]
                + (spw_inds[i] - center_coarse_chan)
                * meta_dict["coarse_num_fine_chans"]
                * meta_dict["channel_width"]
            )
            last_coarse_freq = (
                first_coarse_freq + num_fine_chans * meta_dict["channel_width"]
            )
            self.freq_array[i * num_fine_chans : (i + 1) * num_fine_chans] = np.arange(
                first_coarse_freq, last_coarse_freq, meta_dict["channel_width"]
            )
        # for mwax, polarizations are ordered xx, xy, yx, yy
        if mwax:
            self.polarization_array = np.array([-5, -7, -8, -6])
        # otherwise, polarizations are ordered yy, yx, xy, xx
        else:
            self.polarization_array = np.array([-6, -8, -7, -5])
        # get index array for AIPS reordering
        pol_index_array = np.argsort(np.abs(self.polarization_array))
        # reorder polarization_array here to avoid memory spike from self.reorder_pols
        self.polarization_array = self.polarization_array[pol_index_array]

        if read_data:
            if not mwax:
                # build mapper from antenna numbers and polarizations to pfb inputs
                corr_ants_to_pfb_inputs = {}
                for i in range(len(meta_dict["antenna_inds"])):
                    for p in range(2):
                        corr_ants_to_pfb_inputs[(meta_dict["antenna_inds"][i], p)] = (
                            2 * i + p
                        )

                # for mapping, start with a pair of antennas/polarizations
                # this is the pair we want to find the data for
                # map the pair to the corresponding coarse pfb input indices
                # map the coarse pfb input indices to the fine pfb output indices
                # these are the indices for the data corresponding to the initial
                # antenna/pol pair

                # These two 1D arrays will be both C and F contiguous
                # but we are explicitly declaring C to be consistent with the rest
                # of the python which interacts with the C/Cython code.
                # generate a mapping index array
                map_inds = np.zeros((self.Nbls * self.Npols), dtype=np.int32, order="C")
                # generate a conjugation array
                conj = np.full(
                    (self.Nbls * self.Npols), False, dtype=np.bool_, order="C"
                )

                _corr_fits.generate_map(corr_ants_to_pfb_inputs, map_inds, conj)
            else:
                map_inds = None
                conj = None
            # create arrays for data, nsamples, and flags
            self.data_array = np.zeros(
                (self.Nblts, self.Nfreqs, self.Npols), dtype=data_array_dtype
            )
            self.nsample_array = np.zeros(
                (self.Ntimes, self.Nbls, self.Nfreqs, self.Npols),
                dtype=nsample_array_dtype,
            )
            self.flag_array = np.full(
                (self.Ntimes, self.Nbls, len(spw_inds), self.Npols), True
            )

            # read data files
            for filename in file_dict["data"]:
                self._read_fits_file(
                    filename,
                    time_array,
                    file_nums,
                    num_fine_chans,
                    meta_dict["int_time"],
                    mwax,
                    map_inds,
                    conj,
                    pol_index_array,
                )

            # propagate coarse flags
            if propagate_coarse_flags:
                self.flag_array = np.any(self.flag_array, axis=2)
                self.flag_array = np.repeat(
                    self.flag_array[:, :, np.newaxis, :], self.Nfreqs, axis=2
                )
            else:
                self.flag_array = np.repeat(self.flag_array, num_fine_chans, axis=2)

            if flag_init:
                self.flag_init(
                    num_fine_chans,
                    edge_width=edge_width,
                    start_flag=meta_dict["start_flag"],
                    end_flag=end_flag,
                    flag_dc_offset=flag_dc_offset,
                )

            # flag bad ants
            bad_ant_inds = np.logical_or(
                np.isin(ant_1_inds[: self.Nbls], meta_dict["flagged_ant_inds"]),
                np.isin(ant_2_inds[: self.Nbls], meta_dict["flagged_ant_inds"]),
            )
            self.flag_array[:, bad_ant_inds, :, :] = True
            # reshape arrays
            self.flag_array = self.flag_array.reshape(
                (self.Nblts, self.Nfreqs, self.Npols)
            )
            self.nsample_array = self.nsample_array.reshape(
                (self.Nblts, self.Nfreqs, self.Npols)
            )

            # When MWA data is cast to float for the correlator, the division
            # by 127 introduces small errors that are mitigated when the data
            # is cast back into integer.
            # this needs to happen before the van vleck correction
            if not mwax:
                self.data_array /= self.extra_keywords["SCALEFAC"]
                np.rint(self.data_array, out=self.data_array)

            # van vleck correction
            if correct_van_vleck:
                self.van_vleck_correction(
                    ant_1_inds,
                    ant_2_inds,
                    meta_dict["flagged_ant_inds"],
                    cheby_approx=cheby_approx,
                    data_array_dtype=data_array_dtype,
                )

            # apply corrections
            if np.any([correct_van_vleck, remove_coarse_band, remove_dig_gains]):
                meta_dict["flagged_ant_inds"] = self._apply_corrections(
                    mwax,
                    ant_1_inds,
                    ant_2_inds,
                    meta_dict["avg_factor"],
                    meta_dict["dig_gains"],
                    spw_inds,
                    num_fine_chans,
                    meta_dict["flagged_ant_inds"],
                    cheby_approx=cheby_approx,
                    data_array_dtype=data_array_dtype,
                    flag_small_auto_ants=flag_small_auto_ants,
                    correct_van_vleck=correct_van_vleck,
                    remove_coarse_band=remove_coarse_band,
                    remove_dig_gains=remove_dig_gains,
                )

            # rescale data
            # this needs to happen after the van vleck correction
            if not mwax:
                self.data_array *= self.extra_keywords["SCALEFAC"]

            # cable delay corrections
            if correct_cable_len:
                self.correct_cable_length(
                    meta_dict["cable_lens"], ant_1_inds, ant_2_inds
                )
            # add aoflagger flags to flag_array
            if use_aoflagger_flags:
                # throw an error if matching files not submitted
                if included_file_nums != included_flag_nums:
                    raise ValueError(
                        "flag file coarse bands do not match data file coarse bands"
                    )
                warnings.warn(
                    "coarse channel, start time, and end time flagging will default "
                    "to the more aggressive of flag_init and AOFlagger"
                )
                for filename in file_dict["flags"]:
                    self._read_flag_file(filename, file_nums, num_fine_chans)

            # to account for discrepancies between file conventions, in order
            # to be consistent with the uvw vector direction, all the data must
            # be conjugated
            np.conj(self.data_array, out=self.data_array)

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        self._set_app_coords_helper()

        # create self.uvw_array
        self.set_uvws_from_antenna_positions()

        # remove bad antennas
        # select must be called after lst thread is re-joined
        if remove_flagged_ants:
            good_ants = np.delete(
                np.array(self.antenna_numbers), meta_dict["flagged_ant_inds"]
            )
            self.select(antenna_nums=good_ants, run_check=False)

        # phasing
        if phase_to_pointing_center:
            self.phase(
                lon=meta_dict["ra_rad"],
                lat=meta_dict["dec_rad"],
                epoch="J2000",
                phase_frame="fk5",
                cat_name=meta_dict["object_name"],
            )

        # switch to current_array_shape
        if not use_future_array_shapes:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This method will be removed in version 3.0 when the "
                    "current array shapes are no longer supported.",
                )
                self.use_current_array_shapes()

        # check if object is self-consistent
        # uvws are calcuated using pyuvdata, so turn off the check for speed.
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )
