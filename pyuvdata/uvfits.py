# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing uvfits files.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import warnings
from astropy import constants as const
from astropy.time import Time
from astropy.io import fits

from . import UVData
from . import parameter as uvp
from . import utils as uvutils


class UVFITS(UVData):
    """
    Defines a uvfits-specific subclass of UVData for reading and writing uvfits files.
    This class should not be interacted with directly, instead use the read_uvfits
    and write_uvfits methods on the UVData class.

    Attributes:
        uvfits_required_extra: Names of optional UVParameters that are required
            for uvfits.
    """

    uvfits_required_extra = ['antenna_positions', 'gst0', 'rdate',
                             'earth_omega', 'dut1', 'timesys']

    def _get_parameter_data(self, vis_hdu, run_check_acceptability):
        """
        Internal function to read just the random parameters portion of the
        uvfits file (referred to as metadata).
        Separated from full read so that header, metadata and data can be read independently.
        """
        # astropy.io fits reader scales date according to relevant PZER0 (?)
        # uvfits standard is to have 2 DATE parameters, both floats:
        # DATE (full day) and _DATE (fractional day)
        # cotter uvfits files have one DATE that is a double
        # using data.par('date') is general -- it will add them together if there are 2
        self.time_array = vis_hdu.data.par('date')

        self.Ntimes = len(np.unique(self.time_array))

        # check if lst array is saved. It's not a standard metadata item in uvfits,
        # but if the file was written with pyuvdata it may be present (depending on pyuvdata version)
        if 'LST' in vis_hdu.data.parnames:
            # angles in uvfits files are stored in degrees, so convert to radians
            self.lst_array = np.deg2rad(vis_hdu.data.par('lst'))
            if run_check_acceptability:
                latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
                lst_array = uvutils.get_lst_for_time(self.time_array, latitude, longitude,
                                                     altitude)
                if not np.all(np.isclose(self.lst_array, lst_array, rtol=self._lst_array.tols[0],
                                         atol=self._lst_array.tols[1])):
                    warnings.warn("LST values stored in this file are not "
                                  "self-consistent with time_array and telescope "
                                  "location. Consider recomputing with "
                                  "utils.get_lst_for_time.")

        else:
            self.set_lsts_from_time_array()

        # if antenna arrays are present, use them. otherwise use baseline array
        if 'ANTENNA1' in vis_hdu.data.parnames and 'ANTENNA2' in vis_hdu.data.parnames:
            # Note: uvfits antennas are 1 indexed,
            # need to subtract one to get to 0-indexed
            self.ant_1_array = np.int32(vis_hdu.data.par('ANTENNA1')) - 1
            self.ant_2_array = np.int32(vis_hdu.data.par('ANTENNA2')) - 1
            subarray = np.int32(vis_hdu.data.par('SUBARRAY')) - 1
            # error on files with multiple subarrays
            if len(set(subarray)) > 1:
                raise ValueError('This file appears to have multiple subarray '
                                 'values; only files with one subarray are '
                                 'supported.')
        else:
            # cannot set this to be the baseline array because it uses the
            # 256 convention, not our 2048 convention
            bl_input_array = np.int64(vis_hdu.data.par('BASELINE'))

            # get antenna arrays based on uvfits baseline array
            self.ant_1_array, self.ant_2_array = \
                self.baseline_to_antnums(bl_input_array)

        # check for multi source files
        if 'SOURCE' in vis_hdu.data.parnames:
            source = vis_hdu.data.par('SOURCE')
            if len(set(source)) > 1:
                raise ValueError('This file has multiple sources. Only single '
                                 'source observations are supported.')

        # get self.baseline_array using our convention
        self.baseline_array = \
            self.antnums_to_baseline(self.ant_1_array,
                                     self.ant_2_array)
        self.Nbls = len(np.unique(self.baseline_array))

        # initialize internal variables based on the antenna lists
        self.Nants_data = int(
            len(np.unique(self.ant_1_array.tolist() + self.ant_2_array.tolist())))

        # read baseline vectors in units of seconds, return in meters
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.uvw_array = (-1) * (np.array(np.stack((vis_hdu.data.par('UU'),
                                                    vis_hdu.data.par('VV'),
                                                    vis_hdu.data.par('WW'))))
                                 * const.c.to('m/s').value).T

        if 'INTTIM' in vis_hdu.data.parnames:
            self.integration_time = np.asarray(vis_hdu.data.par('INTTIM'), dtype=np.float64)
        else:
            if self.Ntimes > 1:
                # assume that all integration times in the file are the same
                int_time = self._calc_single_integration_time()
                self.integration_time = (np.ones_like(self.time_array, dtype=np.float64)
                                         * int_time)
            else:
                raise ValueError('integration time not specified and only '
                                 'one time present')

    def _get_data(self, vis_hdu, antenna_nums, antenna_names, ant_str,
                  bls, frequencies, freq_chans, times, polarizations,
                  blt_inds, read_metadata, run_check, check_extra,
                  run_check_acceptability, keep_all_metadata):
        """
        Internal function to read just the visibility and flag data of the uvfits file.
        Separated from full read so that header, metadata and data can be read independently.
        """

        if self.time_array is None or read_metadata:
            # first read in random group parameters
            self._get_parameter_data(vis_hdu, run_check_acceptability)

        # figure out what data to read in
        blt_inds, freq_inds, pol_inds, history_update_string = \
            self._select_preprocess(antenna_nums, antenna_names, ant_str, bls,
                                    frequencies, freq_chans, times, polarizations, blt_inds)

        if blt_inds is not None:
            blt_frac = len(blt_inds) / float(self.Nblts)
        else:
            blt_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if pol_inds is not None:
            pol_frac = len(pol_inds) / float(self.Npols)
        else:
            pol_frac = 1

        min_frac = np.min([blt_frac, freq_frac, pol_frac])

        if min_frac == 1:
            # no select, read in all the data
            if vis_hdu.header['NAXIS'] == 7:
                raw_data_array = vis_hdu.data.data[:, 0, 0, :, :, :, :]
                assert(self.Nspws == raw_data_array.shape[1])

            else:
                # in many uvfits files the spw axis is left out,
                # here we put it back in so the dimensionality stays the same
                raw_data_array = vis_hdu.data.data[:, 0, 0, :, :, :]
                raw_data_array = raw_data_array[:, np.newaxis, :, :]
        else:
            # do select operations on everything except data_array, flag_array and nsample_array
            self._select_metadata(blt_inds, freq_inds, pol_inds, history_update_string,
                                  keep_all_metadata)

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                if vis_hdu.header['NAXIS'] == 7:
                    raw_data_array = vis_hdu.data.data[blt_inds, :, :, :, :, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert(self.Nspws == raw_data_array.shape[1])
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[blt_inds, :, :, :, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]
                if freq_frac < 1:
                    raw_data_array = raw_data_array[:, :, freq_inds, :, :]
                if pol_frac < 1:
                    raw_data_array = raw_data_array[:, :, :, pol_inds, :]
            elif freq_frac == min_frac:
                if vis_hdu.header['NAXIS'] == 7:
                    raw_data_array = vis_hdu.data.data[:, :, :, :, freq_inds, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert(self.Nspws == raw_data_array.shape[1])
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[:, :, :, freq_inds, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]

                if blt_frac < 1:
                    raw_data_array = raw_data_array[blt_inds, :, :, :, :]
                if pol_frac < 1:
                    raw_data_array = raw_data_array[:, :, :, pol_inds, :]
            else:
                if vis_hdu.header['NAXIS'] == 7:
                    raw_data_array = vis_hdu.data.data[:, :, :, :, :, pol_inds, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert(self.Nspws == raw_data_array.shape[1])
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[:, :, :, :, pol_inds, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]

                if blt_frac < 1:
                    raw_data_array = raw_data_array[blt_inds, :, :, :, :]
                if freq_frac < 1:
                    raw_data_array = raw_data_array[:, :, freq_inds, :, :]

        assert(len(raw_data_array.shape) == 5)
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.data_array = (raw_data_array[:, :, :, :, 0] - 1j * raw_data_array[:, :, :, :, 1])
        self.flag_array = (raw_data_array[:, :, :, :, 2] <= 0)
        self.nsample_array = np.abs(raw_data_array[:, :, :, :, 2])

        # check if object has all required UVParameters set
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

    def read_uvfits(self, filename, antenna_nums=None, antenna_names=None,
                    ant_str=None, bls=None, frequencies=None,
                    freq_chans=None, times=None, polarizations=None, blt_inds=None,
                    read_data=True, read_metadata=True,
                    run_check=True, check_extra=True, run_check_acceptability=True,
                    keep_all_metadata=True):
        """
        Read in header, metadata and data from a uvfits file. Supports reading
        only selected portions of the data.

        Args:
            filename: The uvfits file to read from.
            antenna_nums: The antennas numbers to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided. Ignored if read_data is False.
            antenna_names: The antennas names to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided. Ignored if read_data is False.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None. Ignored if read_data is False.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when reading data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y').
                See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be kept for both baselines (1,2) and (2,3) to return a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.
                Ignored if read_data is False.
            frequencies: The frequencies to include when reading data into the
                object. Ignored if read_data is False.
            freq_chans: The frequency channel numbers to include when reading
                data into the object. Ignored if read_data is False.
            times: The times to include when reading data into the object.
                Ignored if read_data is False.
            polarizations: The polarizations to include when reading data into
                the object. Ignored if read_data is False.
            blt_inds: The baseline-time indices to include when reading data into
                the object. This is not commonly used. Ignored if read_data is False.
            read_data: Read in the visibility and flag data. If set to false,
                only the basic header info and metadata (if read_metadata is True)
                will be read in. Results in an incompletely defined object
                (check will not pass). Default True.
            read_metadata: Read in metadata (times, baselines, uvws) as well as
                basic header info. Only used if read_data is False
                (metadata will be read if data is read). If both read_data and
                read_metadata are false, only basic header info is read in. Default True.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
                Ignored if read_data is False.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True. Ignored if read_data is False.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
                Ignored if read_data is False.
            keep_all_metadata: Option to keep all the metadata associated with antennas,
                even those that do not remain after the select option. Default is True.
        """
        if not read_data:
            run_check = False

        with fits.open(filename, memmap=True) as hdu_list:
            vis_hdu = hdu_list[0]  # assumes the visibilities are in the primary hdu
            vis_hdr = vis_hdu.header.copy()
            hdunames = uvutils._fits_indexhdus(hdu_list)  # find the rest of the tables

            # First get everything we can out of the header.
            self.set_phased()
            # check if we have an spw dimension
            if vis_hdr['NAXIS'] == 7:
                if vis_hdr['NAXIS5'] > 1:
                    raise ValueError('Sorry.  Files with more than one spectral'
                                     'window (spw) are not yet supported. A '
                                     'great project for the interested student!')

                self.Nspws = vis_hdr.pop('NAXIS5')

                self.spw_array = np.int32(uvutils._fits_gethduaxis(vis_hdu, 5)) - 1

                # the axis number for phase center depends on if the spw exists
                self.phase_center_ra_degrees = np.float(vis_hdr.pop('CRVAL6'))
                self.phase_center_dec_degrees = np.float(vis_hdr.pop('CRVAL7'))
            else:
                self.Nspws = 1
                self.spw_array = np.array([0])

                # the axis number for phase center depends on if the spw exists
                self.phase_center_ra_degrees = np.float(vis_hdr.pop('CRVAL5'))
                self.phase_center_dec_degrees = np.float(vis_hdr.pop('CRVAL6'))

            # get shapes
            self.Nfreqs = vis_hdr.pop('NAXIS4')
            self.Npols = vis_hdr.pop('NAXIS3')
            self.Nblts = vis_hdr.pop('GCOUNT')

            self.freq_array = uvutils._fits_gethduaxis(vis_hdu, 4)
            self.freq_array.shape = (self.Nspws,) + self.freq_array.shape
            self.channel_width = vis_hdr.pop('CDELT4')
            self.polarization_array = np.int32(uvutils._fits_gethduaxis(vis_hdu, 3))

            # other info -- not required but frequently used
            self.object_name = vis_hdr.pop('OBJECT', None)
            self.telescope_name = vis_hdr.pop('TELESCOP', None)
            self.instrument = vis_hdr.pop('INSTRUME', None)
            latitude_degrees = vis_hdr.pop('LAT', None)
            longitude_degrees = vis_hdr.pop('LON', None)
            altitude = vis_hdr.pop('ALT', None)
            self.x_orientation = vis_hdr.pop('XORIENT', None)
            blt_order_str = vis_hdr.pop('BLTORDER', None)
            if blt_order_str is not None:
                self.blt_order = tuple(blt_order_str.split(', '))
                if self.blt_order == ('bda',):
                    self._blt_order.form = (1,)
            self.history = str(vis_hdr.get('HISTORY', ''))
            if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
                self.history += self.pyuvdata_version_str

            while 'HISTORY' in vis_hdr.keys():
                vis_hdr.remove('HISTORY')

            self.vis_units = vis_hdr.pop('BUNIT', 'UNCALIB')
            self.phase_center_epoch = vis_hdr.pop('EPOCH', None)
            self.phase_center_frame = vis_hdr.pop('PHSFRAME', None)

            # remove standard FITS header items that are still around
            std_fits_substrings = ['SIMPLE', 'BITPIX', 'EXTEND', 'BLOCKED',
                                   'GROUPS', 'PCOUNT', 'BSCALE', 'BZERO', 'NAXIS',
                                   'PTYPE', 'PSCAL', 'PZERO', 'CTYPE', 'CRVAL',
                                   'CRPIX', 'CDELT', 'CROTA', 'CUNIT', 'DATE-OBS']
            for key in list(vis_hdr.keys()):
                for sub in std_fits_substrings:
                    if key.find(sub) > -1:
                        vis_hdr.remove(key)

            # find all the remaining header items and keep them as extra_keywords
            for key in vis_hdr:
                if key == 'COMMENT':
                    self.extra_keywords[key] = str(vis_hdr.get(key))
                elif key != '':
                    self.extra_keywords[key] = vis_hdr.get(key)

            # Next read the antenna table
            ant_hdu = hdu_list[hdunames['AIPS AN']]

            # stuff in the header
            if self.telescope_name is None:
                self.telescope_name = ant_hdu.header['ARRNAM']

            self.gst0 = ant_hdu.header['GSTIA0']
            self.rdate = ant_hdu.header['RDATE']
            self.earth_omega = ant_hdu.header['DEGPDY']
            self.dut1 = ant_hdu.header['UT1UTC']
            if 'TIMESYS' in ant_hdu.header.keys():
                self.timesys = ant_hdu.header['TIMESYS']
            else:
                # CASA misspells this one
                self.timesys = ant_hdu.header['TIMSYS']

            if 'FRAME' in ant_hdu.header.keys():
                xyz_telescope_frame = ant_hdu.header['FRAME']
            else:
                warnings.warn('Required Antenna frame keyword not set, '
                              'setting to ????')
                xyz_telescope_frame = '????'

            # get telescope location and antenna positions.
            # VLA incorrectly sets ARRAYX/ARRAYY/ARRAYZ to 0, and puts array center
            # in the antenna positions themselves
            if (np.isclose(ant_hdu.header['ARRAYX'], 0)
                    and np.isclose(ant_hdu.header['ARRAYY'], 0)
                    and np.isclose(ant_hdu.header['ARRAYZ'], 0)):
                x_telescope = np.mean(ant_hdu.data['STABXYZ'][:, 0])
                y_telescope = np.mean(ant_hdu.data['STABXYZ'][:, 1])
                z_telescope = np.mean(ant_hdu.data['STABXYZ'][:, 2])
                self.antenna_positions = (ant_hdu.data.field('STABXYZ')
                                          - np.array([x_telescope,
                                                      y_telescope,
                                                      z_telescope]))

            else:
                x_telescope = ant_hdu.header['ARRAYX']
                y_telescope = ant_hdu.header['ARRAYY']
                z_telescope = ant_hdu.header['ARRAYZ']
                # AIPS memo #117 says that antenna_positions should be relative to
                # the array center, but in a rotated ECEF frame so that the x-axis
                # goes through the local meridian.
                rot_ecef_positions = ant_hdu.data.field('STABXYZ')
                latitude, longitude, altitude = \
                    uvutils.LatLonAlt_from_XYZ(np.array([x_telescope, y_telescope, z_telescope]))
                self.antenna_positions = uvutils.ECEF_from_rotECEF(rot_ecef_positions,
                                                                   longitude)

            if xyz_telescope_frame == 'ITRF':
                self.telescope_location = np.array(
                    [x_telescope, y_telescope, z_telescope])
            else:
                if latitude_degrees is not None and longitude_degrees is not None and altitude is not None:
                    self.telescope_location_lat_lon_alt_degrees = (
                        latitude_degrees, longitude_degrees, altitude)

            # stuff in columns
            ant_names = ant_hdu.data.field('ANNAME').tolist()
            self.antenna_names = []
            for name in ant_names:
                self.antenna_names.append(name.replace('\x00!', ''))

            # subtract one to get to 0-indexed values rather than 1-indexed values
            self.antenna_numbers = ant_hdu.data.field('NOSTA') - 1

            self.Nants_telescope = len(self.antenna_numbers)

            if 'DIAMETER' in ant_hdu.columns.names:
                self.antenna_diameters = ant_hdu.data.field('DIAMETER')

            try:
                self.set_telescope_params()
            except ValueError as ve:
                warnings.warn(str(ve))

            if not read_data and not read_metadata:
                # don't read in the data or metadata. This means the object is incomplete,
                # but that may not matter for many purposes.
                return

            # Now read in the random parameter info
            self._get_parameter_data(vis_hdu, run_check_acceptability)

            if not read_data:
                # don't read in the data. This means the object is incomplete,
                # but that may not matter for many purposes.
                return

            # Now read in the data
            self._get_data(vis_hdu, antenna_nums, antenna_names, ant_str,
                           bls, frequencies, freq_chans, times, polarizations,
                           blt_inds, False, run_check, check_extra, run_check_acceptability,
                           keep_all_metadata)

    def read_uvfits_metadata(self, filename, run_check_acceptability=True):
        """
        Read in metadata (random parameter info) but not data from a uvfits file.

        This is useful when an object already has the associated header info and
        full visibility data isn't needed.

        Parameters
        ----------
        filename : str
            The uvfits file to read from.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file. Default is True.
        """

        if self.data_array is not None:
            raise ValueError('data_array is already defined, cannot read metadata')

        with fits.open(filename, memmap=True) as hdu_list:
            vis_hdu = hdu_list[0]  # assumes the visibilities are in the primary hdu

            self._get_parameter_data(vis_hdu, run_check_acceptability)

        del(vis_hdu)

    def read_uvfits_data(self, filename, antenna_nums=None, antenna_names=None,
                         ant_str=None, bls=None, frequencies=None,
                         freq_chans=None, times=None, polarizations=None,
                         blt_inds=None, read_metadata=True, run_check=True,
                         check_extra=True, run_check_acceptability=True,
                         keep_all_metadata=True):
        """
        Read in data but not header info from a uvfits file
        (useful for an object that already has the associated header info).

        Args:
            filename: The uvfits file to read from.
            antenna_nums: The antennas numbers to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided.
            antenna_names: The antennas names to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when reading data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y').
                See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be kept for both baselines (1,2) and (2,3) to return a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.
            frequencies: The frequencies to include when reading data into the
                object.
            freq_chans: The frequency channel numbers to include when reading
                data into the object.
            times: The times to include when reading data into the object.
            polarizations: The polarizations to include when reading data into
                the object.
            blt_inds: The baseline-time indices to include when reading data into
                the object. This is not commonly used.
            read_metadata: Option to read metadata even if it already exists
                (to ensure data and metadata match). Default is True.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
            keep_all_metadata: Option to keep all the metadata associated with antennas,
                even those that do not remain after the select option. Default is True.
        """

        with fits.open(filename, memmap=True) as hdu_list:
            vis_hdu = hdu_list[0]  # assumes the visibilities are in the primary hdu

            self._get_data(vis_hdu, antenna_nums, antenna_names, ant_str,
                           bls, frequencies, freq_chans, times, polarizations,
                           blt_inds, read_metadata, run_check, check_extra,
                           run_check_acceptability, keep_all_metadata)

        del(vis_hdu)

    def write_uvfits(self, filename, spoof_nonessential=False, write_lst=True,
                     force_phase=False, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Write the data to a uvfits file.

        Parameters
        ----------
        filename : str
            The uvfits file to write to.
        spoof_nonessential : bool
            Option to spoof the values of optional UVParameters that are not set
            but are required for uvfits files.
        write_lst : bool
            Option to write the LSTs to the metadata (random group parameters).
        force_phase : bool
            Option to automatically phase drift scan data to zenith of the first
            timestamp.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        """
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        if self.phase_type == 'phased':
            pass
        elif self.phase_type == 'drift':
            if force_phase:
                print('The data are in drift mode and do not have a '
                      'defined phase center. Phasing to zenith of the first '
                      'timestamp.')
                phase_time = Time(self.time_array[0], format='jd')
                self.phase_to_time(phase_time)
            else:
                raise ValueError('The data are in drift mode. '
                                 'Set force_phase to true to phase the data '
                                 'to zenith of the first timestamp before '
                                 'writing a uvfits file.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to drift or phased to '
                             'reflect the phasing status of the data')

        if self.Nfreqs > 1:
            freq_spacing = self.freq_array[0, 1:] - self.freq_array[0, :-1]
            if not np.isclose(np.min(freq_spacing), np.max(freq_spacing),
                              rtol=self._freq_array.tols[0], atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are not evenly spaced (probably '
                                 'because of a select operation). The uvfits format '
                                 'does not support unevenly spaced frequencies.')
            if not np.isclose(freq_spacing[0], self.channel_width,
                              rtol=self._freq_array.tols[0], atol=self._freq_array.tols[1]):
                raise ValueError('The frequencies are separated by more than their '
                                 'channel width (probably because of a select operation). '
                                 'The uvfits format does not support frequencies '
                                 'that are spaced by more than their channel width.')
            freq_spacing = freq_spacing[0]
        else:
            freq_spacing = self.channel_width

        if self.Npols > 1:
            pol_spacing = np.diff(self.polarization_array)
            if np.min(pol_spacing) < np.max(pol_spacing):
                raise ValueError('The polarization values are not evenly spaced (probably '
                                 'because of a select operation). The uvfits format '
                                 'does not support unevenly spaced polarizations.')
            pol_spacing = pol_spacing[0]
        else:
            pol_spacing = 1

        for p in self.extra():
            param = getattr(self, p)
            if param.name in self.uvfits_required_extra:
                if param.value is None:
                    if spoof_nonessential:
                        # spoof extra keywords required for uvfits
                        if isinstance(param, uvp.AntPositionParameter):
                            param.apply_spoof(self, 'Nants_telescope')
                        else:
                            param.apply_spoof()
                        setattr(self, p, param)
                    else:
                        raise ValueError('Required attribute {attribute} '
                                         'for uvfits not defined. Define or '
                                         'set spoof_nonessential to True to '
                                         'spoof this attribute.'
                                         .format(attribute=p))

        # check for unflagged data with nsample = 0. Warn if any found
        wh_nsample0 = np.where(self.nsample_array == 0)
        if np.any(~self.flag_array[wh_nsample0]):
            warnings.warn('Some unflagged data has nsample = 0. Flags and '
                          'nsamples are combined in uvfits files such that '
                          'these data will appear to be flagged.')

        weights_array = self.nsample_array * \
            np.where(self.flag_array, -1, 1)
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        data_array = np.conj(self.data_array[:, np.newaxis,
                                             np.newaxis, :, :, :, np.newaxis])
        weights_array = weights_array[:, np.newaxis, np.newaxis, :, :, :,
                                      np.newaxis]
        # uvfits_array_data shape will be  (Nblts,1,1,[Nspws],Nfreqs,Npols,3)
        uvfits_array_data = np.concatenate([data_array.real,
                                            data_array.imag,
                                            weights_array], axis=6)

        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        uvw_array_sec = -1 * self.uvw_array / const.c.to('m/s').value
        # jd_midnight = np.floor(self.time_array[0] - 0.5) + 0.5
        tzero = np.float32(self.time_array[0])

        # uvfits convention is that time_array + relevant PZERO = actual JD
        # We are setting PZERO4 = float32(first time of observation)
        time_array = np.float32(self.time_array - np.float64(tzero))

        int_time_array = self.integration_time

        baselines_use = self.antnums_to_baseline(self.ant_1_array,
                                                 self.ant_2_array,
                                                 attempt256=True)
        # Set up dictionaries for populating hdu
        # Note that uvfits antenna arrays are 1-indexed so we add 1
        # to our 0-indexed arrays
        group_parameter_dict = {'UU      ': uvw_array_sec[:, 0],
                                'VV      ': uvw_array_sec[:, 1],
                                'WW      ': uvw_array_sec[:, 2],
                                'DATE    ': time_array,
                                'BASELINE': baselines_use,
                                'ANTENNA1': self.ant_1_array + 1,
                                'ANTENNA2': self.ant_2_array + 1,
                                'SUBARRAY': np.ones_like(self.ant_1_array),
                                'INTTIM  ': int_time_array}

        pscal_dict = {'UU      ': 1.0, 'VV      ': 1.0, 'WW      ': 1.0,
                      'DATE    ': 1.0, 'BASELINE': 1.0, 'ANTENNA1': 1.0,
                      'ANTENNA2': 1.0, 'SUBARRAY': 1.0, 'INTTIM  ': 1.0}
        pzero_dict = {'UU      ': 0.0, 'VV      ': 0.0, 'WW      ': 0.0,
                      'DATE    ': tzero, 'BASELINE': 0.0, 'ANTENNA1': 0.0,
                      'ANTENNA2': 0.0, 'SUBARRAY': 0.0, 'INTTIM  ': 0.0}

        if write_lst:
            # lst is a non-standard entry (it's not in the AIPS memo)
            # but storing it can be useful (e.g. can avoid recalculating it on read)
            # need to store it in 2 parts to get enough accuracy
            # angles in uvfits files are stored in degrees, so first convert to degrees
            lst_array_deg = np.rad2deg(self.lst_array)
            lst_array_1 = np.float32(lst_array_deg)
            lst_array_2 = np.float32(lst_array_deg - np.float64(lst_array_1))
            group_parameter_dict['LST     '] = lst_array_1
            pscal_dict['LST     '] = 1.0
            pzero_dict['LST     '] = 0.0

        # list contains arrays of [u,v,w,date,baseline];
        # each array has shape (Nblts)
        parnames_use = ['UU      ', 'VV      ', 'WW      ', 'DATE    ']
        if (np.max(self.ant_1_array) < 255
                and np.max(self.ant_2_array) < 255):
            # if the number of antennas is less than 256 then include both the
            # baseline array and the antenna arrays in the group parameters.
            # Otherwise just use the antenna arrays
            parnames_use.append('BASELINE')

        parnames_use += ['ANTENNA1', 'ANTENNA2', 'SUBARRAY', 'INTTIM  ']

        if write_lst:
            parnames_use.append('LST     ')

        group_parameter_list = [group_parameter_dict[parname] for
                                parname in parnames_use]

        if write_lst:
            # add second LST array part
            parnames_use.append('LST     ')
            group_parameter_list.append(lst_array_2)

        hdu = fits.GroupData(uvfits_array_data, parnames=parnames_use,
                             pardata=group_parameter_list, bitpix=-32)
        hdu = fits.GroupsHDU(hdu)

        for i, key in enumerate(parnames_use):
            hdu.header['PSCAL' + str(i + 1) + '  '] = pscal_dict[key]
            hdu.header['PZERO' + str(i + 1) + '  '] = pzero_dict[key]

        # ISO string of first time in self.time_array
        hdu.header['DATE-OBS'] = Time(self.time_array[0], scale='utc',
                                      format='jd').isot

        hdu.header['CTYPE2  '] = 'COMPLEX '
        hdu.header['CRVAL2  '] = 1.0
        hdu.header['CRPIX2  '] = 1.0
        hdu.header['CDELT2  '] = 1.0

        # Note: This axis is called STOKES to comply with the AIPS memo 117
        # However, this confusing because it is NOT a true Stokes axis,
        #   it is really the polarization axis.
        hdu.header['CTYPE3  '] = 'STOKES  '
        hdu.header['CRVAL3  '] = self.polarization_array[0]
        hdu.header['CRPIX3  '] = 1.0
        hdu.header['CDELT3  '] = pol_spacing

        hdu.header['CTYPE4  '] = 'FREQ    '
        hdu.header['CRVAL4  '] = self.freq_array[0, 0]
        hdu.header['CRPIX4  '] = 1.0
        hdu.header['CDELT4  '] = freq_spacing

        hdu.header['CTYPE5  '] = 'IF      '
        hdu.header['CRVAL5  '] = 1.0
        hdu.header['CRPIX5  '] = 1.0
        hdu.header['CDELT5  '] = 1.0

        hdu.header['CTYPE6  '] = 'RA'
        hdu.header['CRVAL6  '] = self.phase_center_ra_degrees

        hdu.header['CTYPE7  '] = 'DEC'
        hdu.header['CRVAL7  '] = self.phase_center_dec_degrees

        hdu.header['BUNIT   '] = self.vis_units
        hdu.header['BSCALE  '] = 1.0
        hdu.header['BZERO   '] = 0.0

        hdu.header['OBJECT  '] = self.object_name
        hdu.header['TELESCOP'] = self.telescope_name
        hdu.header['LAT     '] = self.telescope_location_lat_lon_alt_degrees[0]
        hdu.header['LON     '] = self.telescope_location_lat_lon_alt_degrees[1]
        hdu.header['ALT     '] = self.telescope_location_lat_lon_alt[2]
        hdu.header['INSTRUME'] = self.instrument
        hdu.header['EPOCH   '] = float(self.phase_center_epoch)
        if self.phase_center_frame is not None:
            hdu.header['PHSFRAME'] = self.phase_center_frame

        if self.x_orientation is not None:
            hdu.header['XORIENT'] = self.x_orientation

        if self.blt_order is not None:
            blt_order_str = ', '.join(self.blt_order)
            hdu.header['BLTORDER'] = blt_order_str

        for line in self.history.splitlines():
            hdu.header.add_history(line)

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.items():
            # header keywords have to be 8 characters or less
            if len(str(key)) > 8:
                warnings.warn('key {key} in extra_keywords is longer than 8 '
                              'characters. It will be truncated to 8 as required '
                              'by the uvfits file format.'.format(key=key))
            keyword = key[:8].upper()
            if isinstance(value, (dict, list, np.ndarray)):
                raise TypeError('Extra keyword {keyword} is of {keytype}. '
                                'Only strings and numbers are '
                                'supported in uvfits.'.format(keyword=key,
                                                              keytype=type(value)))

            if keyword == 'COMMENT':
                for line in value.splitlines():
                    hdu.header.add_comment(line)
            else:
                hdu.header[keyword] = value

        # ADD the ANTENNA table
        staxof = np.zeros(self.Nants_telescope)

        # 0 specifies alt-az, 6 would specify a phased array
        mntsta = np.zeros(self.Nants_telescope)

        # beware, X can mean just about anything
        poltya = np.full((self.Nants_telescope), 'X', dtype=np.object_)
        polaa = [90.0] + np.zeros(self.Nants_telescope)
        poltyb = np.full((self.Nants_telescope), 'Y', dtype=np.object_)
        polab = [0.0] + np.zeros(self.Nants_telescope)

        col1 = fits.Column(name='ANNAME', format='8A',
                           array=self.antenna_names)
        # AIPS memo #117 says that antenna_positions should be relative to
        # the array center, but in a rotated ECEF frame so that the x-axis
        # goes through the local meridian.
        longitude = self.telescope_location_lat_lon_alt[1]
        rot_ecef_positions = uvutils.rotECEF_from_ECEF(self.antenna_positions,
                                                       longitude)
        col2 = fits.Column(name='STABXYZ', format='3D',
                           array=rot_ecef_positions)
        # convert to 1-indexed from 0-indexed indicies
        col3 = fits.Column(name='NOSTA', format='1J',
                           array=self.antenna_numbers + 1)
        col4 = fits.Column(name='MNTSTA', format='1J', array=mntsta)
        col5 = fits.Column(name='STAXOF', format='1E', array=staxof)
        col6 = fits.Column(name='POLTYA', format='1A', array=poltya)
        col7 = fits.Column(name='POLAA', format='1E', array=polaa)
        # col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
        col9 = fits.Column(name='POLTYB', format='1A', array=poltyb)
        col10 = fits.Column(name='POLAB', format='1E', array=polab)
        # col11 = fits.Column(name='POLCALB', format='3E', array=polcalb)
        # note ORBPARM is technically required, but we didn't put it in
        col_list = [col1, col2, col3, col4, col5, col6, col7, col9, col10]

        if self.antenna_diameters is not None:
            col12 = fits.Column(name='DIAMETER', format='1E', array=self.antenna_diameters)
            col_list.append(col12)

        cols = fits.ColDefs(col_list)

        ant_hdu = fits.BinTableHDU.from_columns(cols)

        ant_hdu.header['EXTNAME'] = 'AIPS AN'
        ant_hdu.header['EXTVER'] = 1

        # write XYZ coordinates if not already defined
        ant_hdu.header['ARRAYX'] = self.telescope_location[0]
        ant_hdu.header['ARRAYY'] = self.telescope_location[1]
        ant_hdu.header['ARRAYZ'] = self.telescope_location[2]
        ant_hdu.header['FRAME'] = 'ITRF'
        ant_hdu.header['GSTIA0'] = self.gst0
        ant_hdu.header['FREQ'] = self.freq_array[0, 0]
        ant_hdu.header['RDATE'] = self.rdate
        ant_hdu.header['UT1UTC'] = self.dut1

        ant_hdu.header['TIMSYS'] = self.timesys
        if self.timesys != 'UTC':
            raise ValueError('This file has a time system {tsys}. '
                             'Only "UTC" time system files are supported'.format(tsys=self.timesys))
        ant_hdu.header['ARRNAM'] = self.telescope_name
        ant_hdu.header['NO_IF'] = self.Nspws
        ant_hdu.header['DEGPDY'] = self.earth_omega
        # ant_hdu.header['IATUTC'] = 35.

        # set mandatory parameters which are not supported by this object
        # (or that we just don't understand)
        ant_hdu.header['NUMORB'] = 0

        # note: Bart had this set to 3. We've set it 0 after aips 117. -jph
        ant_hdu.header['NOPCAL'] = 0

        ant_hdu.header['POLTYPE'] = 'X-Y LIN'

        # note: we do not support the concept of "frequency setups"
        # -- lists of spws given in a SU table.
        ant_hdu.header['FREQID'] = -1

        # if there are offsets in images, this could be the culprit
        ant_hdu.header['POLARX'] = 0.0
        ant_hdu.header['POLARY'] = 0.0

        ant_hdu.header['DATUTC'] = 0  # ONLY UTC SUPPORTED

        # we always output right handed coordinates
        ant_hdu.header['XYZHAND'] = 'RIGHT'

        # ADD the FQ table
        # skipping for now and limiting to a single spw

        # write the file
        hdulist = fits.HDUList(hdus=[hdu, ant_hdu])
        hdulist.writeto(filename, overwrite=True)
