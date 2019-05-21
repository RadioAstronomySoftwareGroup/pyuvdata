# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing UVH5 files.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import warnings
import six

from . import UVData
from . import utils as uvutils

try:
    import h5py
except ImportError:  # pragma: no cover
    uvutils._reraise_context('h5py is not installed but is required for '
                             'uvh5 functionality')

# define HDF5 type for interpreting HERA correlator outputs (integers) as complex numbers
_hera_corr_dtype = np.dtype([('r', '<i4'), ('i', '<i4')])


def _read_uvh5_string(dataset, filename):
    """
    Handle backwards compatibility of string types for legacy uvh5 files.

    Args:
        dataset: HDF5 dataset containing string-like data
        filename: name of uvh5 file

    Returns:
        string: string of type <str> corresponding to data saved in dataset

    Notes:
        This function is only designed to work on scalar datasets. Arrays of strings should be
        handled differently. (See how antenna_names are handled below for an example.)
    """
    if dataset.dtype.type is np.object_:
        warnings.warn("Strings in metadata of {file} are not the correct type; rewrite with "
                      "write_uvh5 to ensure future compatibility. Suppoort for reading these "
                      "files will be removed in version 1.5.".format(file=filename),
                      DeprecationWarning)
        try:
            return uvutils._bytes_to_str(dataset[()])
        except AttributeError:
            # dataset[()] is already <str> type, and doesn't need to be decoded
            return dataset[()]
    else:
        return uvutils._bytes_to_str(dataset[()].tostring())


def _check_uvh5_dtype(dtype):
    """
    Check that a specified custom datatype conforms to uvh5 standards.

    Arguments:
        dtype: numpy datatype with an 'r' field and an 'i' field

    Returns:
        None
    """
    if not isinstance(dtype, np.dtype):
        raise ValueError("dtype in a uvh5 must be a numpy dtype")
    if 'r' not in dtype.names or 'i' not in dtype.names:
        raise ValueError("datatype must be be a compound datatype with an 'r' field and an 'i' field")
    rkind = dtype['r'].kind
    ikind = dtype['i'].kind
    if rkind != ikind:
        raise ValueError("datatype must have the same kind ('i4', 'r8', etc.) for both real and imaginary fields")
    return


def _read_complex_astype(dset, indices, dtype_out=np.complex64):
    """
    Read the given data set of a specified type to floating point complex data.

    Arguments:
        dset: reference to an HDF5 dataset on disk.
        indices: tuple representing indices to extract. Should be either lists of indices or
            numpy slice objects.
        dtype_out: the datatype of the output array. One of (complex, np.complex64, np.complex128).
            Default is np.complex64 (single-precision real and imaginary floats).

    Returns:
        output_array: array referenced in the dataset cast to complex values
    """
    if dtype_out not in (complex, np.complex64, np.complex128):
        raise ValueError("output datatype must be one of (complex, np.complex64, np.complex128)")
    dset_shape = [0, 0, 0, 0]
    for i in range(len(dset_shape)):
        if indices[i] == np.s_[:]:
            dset_shape[i] = dset.shape[i]
        else:
            dset_shape[i] = len(indices[i])
    output_array = np.empty(dset_shape, dtype=dtype_out)
    dtype_in = dset.dtype
    with dset.astype(dtype_in):
        output_array.real = dset['r'][indices]
        output_array.imag = dset['i'][indices]

    return output_array


def _write_complex_astype(data, dset, indices):
    """
    Write floating point complex data as a specified type.

    Arguments:
        data: data array to write out. Should be a complex-valued array that supports
            the .real and .imag attributes for accessing real and imaginary components.
        dset: reference to an HDF5 dataset on disk.
        indices: 4-tuple representing indices to write data to. Should be either lists of
            indices or numpy slice objects.

    Returns:
        None
    """
    # get datatype from dataset
    dtype_out = dset.dtype
    # make doubly sure dtype is valid; should be unless user is pathological
    _check_uvh5_dtype(dtype_out)
    with dset.astype(dtype_out):
        dset[indices[0], indices[1], indices[2], indices[3], 'r'] = data.real
        dset[indices[0], indices[1], indices[2], indices[3], 'i'] = data.imag
    return


class UVH5(UVData):
    """
    Defines an HDF5-specific subclass of UVData for reading and writing uvh5 files.
    This class should not be interacted with directly, instead use the read_uvh5
    and write_uvh5 methods on the UVData class.
    """

    def _read_header(self, header, filename, run_check_acceptability=True):
        """
        Internal function to read header information from a UVH5 file.

        Args:
            header: reference to an h5py data group that contains the header information.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.

        Returns:
            None
        """
        # get telescope information
        latitude = header['latitude'][()]
        longitude = header['longitude'][()]
        altitude = header['altitude'][()]
        if np.abs(latitude) <= np.pi and np.abs(longitude) <= np.pi:
            warnings.warn("It seems that the latitude and longitude are in radians; "
                          "support for interpreting these quantities in radians will "
                          "be removed in version 1.5. Rewrite with write_uvh5 "
                          "to ensure future compatibility.", DeprecationWarning)
            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
        else:
            self.telescope_location_lat_lon_alt_degrees = (latitude, longitude, altitude)
        self.instrument = _read_uvh5_string(header['instrument'], filename)
        self.telescope_name = _read_uvh5_string(header['telescope_name'], filename)

        # get source information
        self.object_name = _read_uvh5_string(header['object_name'], filename)

        # set history appropriately
        self.history = _read_uvh5_string(header['history'], filename)
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        # check for vis_units
        if 'vis_units' in header:
            self.vis_units = _read_uvh5_string(header['vis_units'], filename)
        else:
            # default to uncalibrated data
            self.vis_units = 'UNCALIB'

        # check for optional values
        if 'dut1' in header:
            self.dut1 = float(header['dut1'][()])
        if 'earth_omega' in header:
            self.earth_omega = float(header['earth_omega'][()])
        if 'gst0' in header:
            self.gst0 = float(header['gst0'][()])
        if 'rdate' in header:
            self.rdate = _read_uvh5_string(header['rdate'], filename)
        if 'timesys' in header:
            self.timesys = _read_uvh5_string(header['timesys'], filename)
        if 'x_orientation' in header:
            self.x_orientation = _read_uvh5_string(header['x_orientation'], filename)
        if 'blt_order' in header:
            blt_order_str = _read_uvh5_string(header['blt_order'], filename)
            self.blt_order = tuple(blt_order_str.split(', '))
            if self.blt_order == ('bda',):
                self._blt_order.form = (1,)

        if 'antenna_diameters' in header:
            self.antenna_diameters = header['antenna_diameters'][()]
        if 'uvplane_reference_time' in header:
            self.uvplane_reference_time = int(header['uvplane_reference_time'][()])

        # check for phasing information
        self.phase_type = _read_uvh5_string(header['phase_type'], filename)
        if self.phase_type == 'phased':
            self.set_phased()
            self.phase_center_ra = float(header['phase_center_ra'][()])
            self.phase_center_dec = float(header['phase_center_dec'][()])
            self.phase_center_epoch = float(header['phase_center_epoch'][()])
            if 'phase_center_frame' in header:
                self.phase_center_frame = _read_uvh5_string(header['phase_center_frame'], filename)
        elif self.phase_type == 'drift':
            self.set_drift()
        else:
            self.set_unknown_phase_type()

        # get antenna arrays
        # cast to native python int type
        self.Nants_data = int(header['Nants_data'][()])
        self.Nants_telescope = int(header['Nants_telescope'][()])
        self.ant_1_array = header['ant_1_array'][:]
        self.ant_2_array = header['ant_2_array'][:]
        self.antenna_names = [uvutils._bytes_to_str(n.tostring()) for n in header['antenna_names'][:]]
        self.antenna_numbers = header['antenna_numbers'][:]
        self.antenna_positions = header['antenna_positions'][:]

        # set telescope params
        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # get baseline array
        self.baseline_array = self.antnums_to_baseline(self.ant_1_array,
                                                       self.ant_2_array)
        self.Nbls = len(np.unique(self.baseline_array))

        # get uvw array
        self.uvw_array = header['uvw_array'][:, :]

        # get time information
        self.time_array = header['time_array'][:]
        integration_time = header['integration_time']
        if integration_time.size == 1 and int(header['Nblts'][()]) > 1:
            warnings.warn('{file} appears to be an old uvh5 format '
                          'with a single valued integration_time which has been deprecated. '
                          'Rewrite this file with write_uvh5 to ensure '
                          'future compatibility. Support for reading these files will be removed '
                          'in version 1.5.'.format(file=filename), DeprecationWarning)
            self.integration_time = np.ones_like(self.time_array, dtype=np.float64) * integration_time[()]
        else:
            self.integration_time = integration_time[:]
        if 'lst_array' in header:
            self.lst_array = header['lst_array'][:]
            # check that lst_array in file is self-consistent
            if run_check_acceptability:
                latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
                lst_array = uvutils.get_lst_for_time(self.time_array, latitude, longitude,
                                                     altitude)
                if not np.all(np.isclose(self.lst_array, lst_array, rtol=self._lst_array.tols[0],
                                         atol=self._lst_array.tols[1])):
                    warnings.warn("LST values stored in {file} are not self-consistent with time_array "
                                  "and telescope location. Consider recomputing with "
                                  "utils.get_lst_for_time.".format(file=filename))
        else:
            # compute lst_array from time_array and telescope location
            latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
            self.lst_array = uvutils.get_lst_for_time(self.time_array, latitude, longitude,
                                                      altitude)

        # get frequency information
        self.freq_array = header['freq_array'][:, :]
        self.channel_width = float(header['channel_width'][()])
        self.spw_array = header['spw_array'][:]

        # get polarization information
        self.polarization_array = header['polarization_array'][:]

        # get data shapes
        self.Nfreqs = int(header['Nfreqs'][()])
        self.Npols = int(header['Npols'][()])
        self.Ntimes = int(header['Ntimes'][()])
        self.Nblts = int(header['Nblts'][()])
        self.Nspws = int(header['Nspws'][()])

        # get extra_keywords
        if "extra_keywords" in header:
            self.extra_keywords = {}
            for key in header["extra_keywords"].keys():
                if header["extra_keywords"][key].dtype.type in (np.string_, np.object_):
                    self.extra_keywords[key] = _read_uvh5_string(header["extra_keywords"][key], filename)
                else:
                    self.extra_keywords[key] = header["extra_keywords"][key][()]

        return

    def _get_data(self, dgrp, antenna_nums, antenna_names, ant_str,
                  bls, frequencies, freq_chans, times, polarizations,
                  blt_inds, run_check, check_extra, run_check_acceptability,
                  data_array_dtype, keep_all_metadata):
        """
        Internal function to read just the visibility, flag, and nsample data of the uvh5 file.
        Separated from full read so that header/metadata and data can be read independently.
        """
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

        # get the fundamental datatype of the visdata; if integers, we need to cast to floats
        visdata_dtype = dgrp['visdata'].dtype
        if visdata_dtype not in ('complex64', 'complex128'):
            _check_uvh5_dtype(visdata_dtype)
            if data_array_dtype not in (np.complex64, np.complex128):
                raise ValueError("data_array_dtype must be np.complex64 or np.complex128")
            custom_dtype = True
        else:
            custom_dtype = False

        if min_frac == 1:
            # no select, read in all the data
            if custom_dtype:
                inds = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
                self.data_array = _read_complex_astype(dgrp['visdata'], inds, data_array_dtype)
            else:
                self.data_array = dgrp['visdata'][:, :, :, :]
            self.flag_array = dgrp['flags'][:, :, :, :]
            self.nsample_array = dgrp['nsamples'][:, :, :, :]
        else:
            # do select operations on everything except data_array, flag_array and nsample_array
            self._select_metadata(blt_inds, freq_inds, pol_inds, history_update_string,
                                  keep_all_metadata)

            # open references to datasets
            visdata_dset = dgrp['visdata']
            flags_dset = dgrp['flags']
            nsamples_dset = dgrp['nsamples']

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                if custom_dtype:
                    inds = (blt_inds, np.s_[:], np.s_[:], np.s_[:])
                    visdata = _read_complex_astype(visdata_dset, inds, data_array_dtype)
                else:
                    visdata = visdata_dset[blt_inds, :, :, :]
                flags = flags_dset[blt_inds, :, :, :]
                nsamples = nsamples_dset[blt_inds, :, :, :]

                assert(self.Nspws == visdata.shape[1])

                if freq_frac < 1:
                    visdata = visdata[:, :, freq_inds, :]
                    flags = flags[:, :, freq_inds, :]
                    nsamples = nsamples[:, :, freq_inds, :]
                if pol_frac < 1:
                    visdata = visdata[:, :, :, pol_inds]
                    flags = flags[:, :, :, pol_inds]
                    nsamples = nsamples[:, :, :, pol_inds]
            elif freq_frac == min_frac:
                if custom_dtype:
                    inds = (np.s_[:], np.s_[:], freq_inds, np.s_[:])
                    visdata = _read_complex_astype(visdata_dset, inds, data_array_dtype)
                else:
                    visdata = visdata_dset[:, :, freq_inds, :]
                flags = flags_dset[:, :, freq_inds, :]
                nsamples = nsamples_dset[:, :, freq_inds, :]

                if blt_frac < 1:
                    visdata = visdata[blt_inds, :, :, :]
                    flags = flags[blt_inds, :, :, :]
                    nsamples = nsamples[blt_inds, :, :, :]
                if pol_frac < 1:
                    visdata = visdata[:, :, :, pol_inds]
                    flags = flags[:, :, :, pol_inds]
                    nsamples = nsamples[:, :, :, pol_inds]
            else:
                if custom_dtype:
                    inds = (np.s_[:], np.s_[:], np.s_[:], pol_inds)
                    visdata = _read_complex_astype(visdata_dset, inds, data_array_dtype)
                else:
                    visdata = visdata_dset[:, :, :, pol_inds]
                flags = flags_dset[:, :, :, pol_inds]
                nsamples = nsamples_dset[:, :, :, pol_inds]

                if blt_frac < 1:
                    visdata = visdata[blt_inds, :, :, :]
                    flags = flags[blt_inds, :, :, :]
                    nsamples = nsamples[blt_inds, :, :, :]
                if freq_frac < 1:
                    visdata = visdata[:, :, freq_inds, :]
                    flags = flags[:, :, freq_inds, :]
                    nsamples = nsamples[:, :, freq_inds, :]

            # save arrays in object
            self.data_array = visdata
            self.flag_array = flags
            self.nsample_array = nsamples

        # check if object has all required UVParameters set
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        return

    def read_uvh5(self, filename, antenna_nums=None, antenna_names=None,
                  ant_str=None, bls=None, frequencies=None, freq_chans=None,
                  times=None, polarizations=None, blt_inds=None, read_data=True,
                  run_check=True, check_extra=True, run_check_acceptability=True,
                  data_array_dtype=np.complex128, keep_all_metadata=True):
        """
        Read in data from a UVH5 file.

        Args:
            filename: The file name to read.
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
                only the header info and metadata will be read in. Results in an
                incompletely defined object (check will not pass). Default True.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
            data_array_dtype: Datatype to store the output data_array as. Must be either
                np.complex64 (single-precision real and imaginary) or np.complex128 (double-
                precision real and imaginary). Only used if the datatype of the visibility
                data on-disk is not 'c8' or 'c16'. Default is np.complex128.
            keep_all_metadata: Option to keep all the metadata associated with antennas,
                even those that do not remain after the select option. Default is True.

        Returns:
            None
        """
        if not os.path.exists(filename):
            raise IOError(filename + ' not found')

        if not read_data:
            run_check = False

        # open hdf5 file for reading
        with h5py.File(filename, 'r') as f:
            # extract header information
            header = f['/Header']
            self._read_header(header, filename, run_check_acceptability=run_check_acceptability)

            if not read_data:
                # don't read in the data. This means the object is incomplete,
                # but that may not matter for many purposes.
                return

            # Now read in the data
            dgrp = f['/Data']
            self._get_data(dgrp, antenna_nums, antenna_names, ant_str,
                           bls, frequencies, freq_chans, times, polarizations,
                           blt_inds, run_check, check_extra, run_check_acceptability,
                           data_array_dtype, keep_all_metadata)

        return

    def _write_header(self, header):
        """Internal function to write uvh5 header information.
        """
        # write out telescope and source information
        header['latitude'] = self.telescope_location_lat_lon_alt_degrees[0]
        header['longitude'] = self.telescope_location_lat_lon_alt_degrees[1]
        header['altitude'] = self.telescope_location_lat_lon_alt_degrees[2]
        header['telescope_name'] = np.string_(self.telescope_name)
        header['instrument'] = np.string_(self.instrument)
        header['object_name'] = np.string_(self.object_name)

        # write out required UVParameters
        header['Nants_data'] = self.Nants_data
        header['Nants_telescope'] = self.Nants_telescope
        header['Nbls'] = self.Nbls
        header['Nblts'] = self.Nblts
        header['Nfreqs'] = self.Nfreqs
        header['Npols'] = self.Npols
        header['Nspws'] = self.Nspws
        header['Ntimes'] = self.Ntimes
        header['antenna_numbers'] = self.antenna_numbers
        header['uvw_array'] = self.uvw_array
        header['vis_units'] = np.string_(self.vis_units)
        header['channel_width'] = self.channel_width
        header['time_array'] = self.time_array
        header['freq_array'] = self.freq_array
        header['integration_time'] = self.integration_time
        header['lst_array'] = self.lst_array
        header['polarization_array'] = self.polarization_array
        header['spw_array'] = self.spw_array
        header['ant_1_array'] = self.ant_1_array
        header['ant_2_array'] = self.ant_2_array
        header['antenna_positions'] = self.antenna_positions

        # handle antenna_names
        if six.PY2:
            n_names = len(self.antenna_names)
            max_len_names = np.amax([len(n) for n in self.antenna_names])
            dtype = "S{:d}".format(max_len_names)
            header.create_dataset('antenna_names', (n_names,), dtype=dtype, data=self.antenna_names)
        else:
            header['antenna_names'] = np.string_(self.antenna_names, keepdims=True)

        # write out phasing information
        header['phase_type'] = np.string_(self.phase_type)
        if self.phase_center_ra is not None:
            header['phase_center_ra'] = self.phase_center_ra
        if self.phase_center_dec is not None:
            header['phase_center_dec'] = self.phase_center_dec
        if self.phase_center_epoch is not None:
            header['phase_center_epoch'] = self.phase_center_epoch
        if self.phase_center_frame is not None:
            header['phase_center_frame'] = np.string_(self.phase_center_frame)

        # write out optional parameters
        if self.dut1 is not None:
            header['dut1'] = self.dut1
        if self.earth_omega is not None:
            header['earth_omega'] = self.earth_omega
        if self.gst0 is not None:
            header['gst0'] = self.gst0
        if self.rdate is not None:
            header['rdate'] = np.string_(self.rdate)
        if self.timesys is not None:
            header['timesys'] = np.string_(self.timesys)
        if self.x_orientation is not None:
            header['x_orientation'] = np.string_(self.x_orientation)
        if self.blt_order is not None:
            header['blt_order'] = np.string_(', '.join(self.blt_order))
        if self.antenna_diameters is not None:
            header['antenna_diameters'] = self.antenna_diameters
        if self.uvplane_reference_time is not None:
            header['uvplane_reference_time'] = self.uvplane_reference_time

        # write out extra keywords if it exists and has elements
        if self.extra_keywords:
            extra_keywords = header.create_group("extra_keywords")
            for k in self.extra_keywords.keys():
                if isinstance(self.extra_keywords[k], str):
                    extra_keywords[k] = np.string_(self.extra_keywords[k])
                else:
                    extra_keywords[k] = self.extra_keywords[k]

        # write out history
        header['history'] = np.string_(self.history)

        return

    def write_uvh5(self, filename, run_check=True, check_extra=True,
                   run_check_acceptability=True, clobber=False,
                   data_compression=None, flags_compression="lzf", nsample_compression="lzf",
                   data_write_dtype=None, add_to_history=None):
        """
        Write an in-memory UVData object to a UVH5 file.

        Args:
            filename: The UVH5 file to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the file if it already exists.
                Default is False. If False and file exists, raises an IOError.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                 None (no filter/compression).
            flags_compression: HDF5 filter to apply when writing the flags_array. Default is
                 the LZF filter.
            nsample_compression: HDF5 filter to apply when writing the nsample_array. Default is
                 the LZF filter.
            data_write_dtype: datatype of output visibility data. If 'None', then the same datatype
                as data_array will be used. Otherwise, a numpy dtype object must be specified with
                an 'r' field and an 'i' field for real and imaginary parts, respectively. See
                uvh5.py for an example of defining such a datatype. Default is None.

        Returns:
            None

        Notes:
            The HDF5 library allows for the application of "filters" when writing data, which can
            provide moderate to significant levels of compression for the datasets in question.
            Testing has shown that for some typical cases of UVData objects (empty/sparse flag_array
            objects, and/or uniform nsample_arrays), the built-in LZF filter provides significant
            compression for minimal computational overhead.

            Note that for typical HERA data files written after mid-2018, the bitshuffle filter was
            applied to the data_array. Because of the lack of portability, it is not included as an
            option here; in the future, it may be added. Note that as long as bitshuffle is installed
            on the system in a way that h5py can find it, no action needs to be taken to _read_ a
            data_array encoded with bitshuffle (or an error will be raised).
        """
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # open file for writing
        with h5py.File(filename, 'w') as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # write out data, flags, and nsample arrays
            dgrp = f.create_group("Data")
            if data_write_dtype is None:
                if self.data_array.dtype == 'complex64':
                    data_write_dtype = 'c8'
                else:
                    data_write_dtype = 'c16'
            if data_write_dtype not in ('c8', 'c16'):
                _check_uvh5_dtype(data_write_dtype)
                visdata = dgrp.create_dataset("visdata", self.data_array.shape, chunks=True,
                                              compression=data_compression, dtype=data_write_dtype)
                indices = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
                _write_complex_astype(self.data_array, visdata, indices)
            else:
                visdata = dgrp.create_dataset("visdata", chunks=True,
                                              data=self.data_array,
                                              compression=data_compression,
                                              dtype=data_write_dtype)
            flags = dgrp.create_dataset("flags", chunks=True,
                                        data=self.flag_array,
                                        compression=flags_compression)
            nsample_array = dgrp.create_dataset("nsamples", chunks=True,
                                                data=self.nsample_array.astype(np.float32),
                                                compression=nsample_compression)

        return

    def initialize_uvh5_file(self, filename, clobber=False, data_compression=None,
                             flags_compression="lzf", nsample_compression="lzf",
                             data_write_dtype=None):
        """Initialize a UVH5 file on disk to be written to in parts.

        Args:
            filename: The UVH5 file to write to.
            clobber: Option to overwrite the file if it already exists.
                Default is False. If False and file exists, raises an IOError.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                 None (no filter/compression).
            flags_compression: HDF5 filter to apply when writing the flags_array. Default is
                 the LZF filter.
            nsample_compression: HDF5 filter to apply when writing the nsample_array. Default is
                 the LZF filter.
            data_write_dtype: datatype of output visibility data. If 'None', then double-precision
                floats will be used. The user may specify 'c8' for single-precision floats or 'c16'
                for double-presicion. Otherwise, a numpy dtype object must be specified with
                an 'r' field and an 'i' field for real and imaginary parts, respectively. See
                uvh5.py for an example of defining such a datatype. Default is None.

        Returns:
            None

        Notes:
            When partially writing out data, this function should be called first to initialize the
            file on disk. The data is then actually written by calling the write_uvh5_part method,
            with the same filename as the one specified in this function. See the tutorial for a
            worked example.

            The HDF5 library allows for the application of "filters" when writing data, which can
            provide moderate to significant levels of compression for the datasets in question.
            Testing has shown that for some typical cases of UVData objects (empty/sparse flag_array
            objects, and/or uniform nsample_arrays), the built-in LZF filter provides significant
            compression for minimal computational overhead.

            Note that for typical HERA data files written after mid-2018, the bitshuffle filter was
            applied to the data_array. Because of the lack of portability, it is not included as an
            option here; in the future, it may be added. Note that as long as bitshuffle is installed
            on the system in a way that h5py can find it, no action needs to be taken to _read_ a
            data_array encoded with bitshuffle (or an error will be raised).
        """
        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise IOError("File exists; skipping")

        # write header and empty arrays to file
        with h5py.File(filename, 'w') as f:
            # write header
            header = f.create_group("Header")
            self._write_header(header)

            # initialize the data groups on disk
            data_size = (self.Nblts, self.Nspws, self.Nfreqs, self.Npols)
            dgrp = f.create_group("Data")
            if data_write_dtype is None:
                # we don't know what kind of data we'll get--default to double-precision
                data_write_dtype = 'c16'
            if data_write_dtype not in ('c8', 'c16'):
                # make sure the data type is correct
                _check_uvh5_dtype(data_write_dtype)
            visdata = dgrp.create_dataset("visdata", data_size, chunks=True,
                                          dtype=data_write_dtype, compression=data_compression)
            flags = dgrp.create_dataset("flags", data_size, chunks=True,
                                        dtype='b1', compression=flags_compression)
            nsample_array = dgrp.create_dataset("nsamples", data_size, chunks=True,
                                                dtype='f4', compression=nsample_compression)

        return

    def _check_header(self, filename, run_check_acceptability=True):
        """
        Check that the metadata present in a file header matches the object's metadata.

        Args:
            header: reference to an h5py data group that contains the header information.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.

        Returns:
            None

        Notes:
            This function creates a new UVData object an reads in the header information saved
            on disk to compare with the object in memory. Note that this adds some small
            memory overhead, but this amount is typically much smaller than the size of the data.
        """
        uvd_file = UVH5()
        with h5py.File(filename, 'r') as f:
            header = f['/Header']
            uvd_file._read_header(header, filename, run_check_acceptability=run_check_acceptability)

        # temporarily remove data, flag, and nsample arrays, so we only check metadata
        if self.data_array is not None:
            data_array = self.data_array
            self.data_array = None
            replace_data = True
        else:
            replace_data = False
        if self.flag_array is not None:
            flag_array = self.flag_array
            self.flag_array = None
            replace_flags = True
        else:
            replace_flags = False
        if self.nsample_array is not None:
            nsample_array = self.nsample_array
            self.nsample_array = None
            replace_nsamples = True
        else:
            replace_nsamples = False

        if self != uvd_file:
            raise AssertionError("The object metadata in memory and metadata on disk are different")
        else:
            # clean up after ourselves
            if replace_data:
                self.data_array = data_array
            if replace_flags:
                self.flag_array = flag_array
            if replace_nsamples:
                self.nsample_array = nsample_array
            del uvd_file
        return

    def write_uvh5_part(self, filename, data_array, flags_array, nsample_array, check_header=True,
                        antenna_nums=None, antenna_names=None, ant_str=None, bls=None,
                        frequencies=None, freq_chans=None, times=None, polarizations=None,
                        blt_inds=None, run_check_acceptability=True, add_to_history=None):
        """
        Write out a part of a UVH5 file that has been previously initialized.

        Args:
            filename: the file on disk to write data to. It must already exist,
                and is assumed to have been initialized with initialize_uvh5_file.
            data_array: the data to write to disk. A check is done to ensure that
                the dimensions of the data passed in conform to the ones specified by
                the "selection" arguments.
            flags_array: the flags array to write to disk. A check is done to ensure
                that the dimensions of the data passed in conform to the ones specified
                by the "selection" arguments.
            nsample_array: the nsample array to write to disk. A check is done to ensure
                that the dimensions fo the data passed in conform to the ones specified
                by the "selection" arguments.
            check_header: option to check that the metadata present in the header
                on disk matches that in the object. Default is True.
            run_check_acceptability: If check_header, additional option to check
                acceptable range of the values of parameters after reading in the file.
                Default is True.
            antenna_nums: The antennas numbers to include when writing data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided.
            antenna_names: The antennas names to include when writing data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to write to the file. For length-2 tuples, the ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when writing data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y').
                See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be written for both baselines (1,2) and (2,3) to reflect a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.
            frequencies: The frequencies to include when writing data to the file.
            freq_chans: The frequency channel numbers to include when writing data to the file.
            times: The times to include when writing data to the file.
            polarizations: The polarizations to include when writing data to the file.
            blt_inds: The baseline-time indices to include when writing data to the file.
                This is not commonly used.
            add_to_history: String to append to history before write out. Default is no appending.

        Returns:
            None

        Notes:
            When partially writing out data, this function should be called after calling
            initialize_uvh5_file. The same filename is passed in, with an optional check to ensure
            that the object's metadata in-memory matches the header on-disk. See the tutorial for a
            worked example.
        """
        # check that the file already exists
        if not os.path.exists(filename):
            raise AssertionError("{0} does not exists; please first initialize it with initialize_uvh5_file".format(
                filename))

        if check_header:
            self._check_header(filename, run_check_acceptability=run_check_acceptability)

        # figure out which "full file" indices to write data to
        blt_inds, freq_inds, pol_inds, _ = self._select_preprocess(
            antenna_nums, antenna_names, ant_str, bls, frequencies, freq_chans, times,
            polarizations, blt_inds)

        # make sure that the dimensions of the data to write are correct
        if data_array.shape != flags_array.shape:
            raise AssertionError("data_array and flags_array must have the same shape")
        if data_array.shape != nsample_array.shape:
            raise AssertionError("data_array and nsample_array must have the same shape")

        # check what part of each dimension to grab
        # we can use numpy slice objects to index the h5py indices
        if blt_inds is not None:
            Nblts = len(blt_inds)

            # test if blts are regularly spaced
            if len(set(np.ediff1d(blt_inds))) <= 1:
                blt_reg_spaced = True
                blt_start = blt_inds[0]
                blt_end = blt_inds[-1] + 1
                if len(blt_inds) == 1:
                    d_blt = 1
                else:
                    d_blt = blt_inds[1] - blt_inds[0]
                blt_inds = np.s_[blt_start:blt_end:d_blt]
            else:
                blt_reg_spaced = False
        else:
            Nblts = self.Nblts
            blt_reg_spaced = True
            blt_inds = np.s_[:]
        if freq_inds is not None:
            Nfreqs = len(freq_inds)

            # test if frequencies are regularly spaced
            if len(set(np.ediff1d(freq_inds))) <= 1:
                freq_reg_spaced = True
                freq_start = freq_inds[0]
                freq_end = freq_inds[-1] + 1
                if len(freq_inds) == 1:
                    d_freq = 1
                else:
                    d_freq = freq_inds[1] - freq_inds[0]
                freq_inds = np.s_[freq_start:freq_end:d_freq]
            else:
                freq_reg_spaced = False
        else:
            Nfreqs = self.Nfreqs
            freq_reg_spaced = True
            freq_inds = np.s_[:]
        if pol_inds is not None:
            Npols = len(pol_inds)

            # test if pols are regularly spaced
            if len(set(np.ediff1d(pol_inds))) <= 1:
                pol_reg_spaced = True
                pol_start = pol_inds[0]
                pol_end = pol_inds[-1] + 1
                if len(pol_inds) == 1:
                    d_pol = 1
                else:
                    d_pol = pol_inds[1] - pol_inds[0]
                pol_inds = np.s_[pol_start:pol_end:d_pol]
            else:
                pol_reg_spaced = False
        else:
            Npols = self.Npols
            pol_reg_spaced = True
            pol_inds = np.s_[:]

        # check for proper size of input arrays
        proper_shape = (Nblts, 1, Nfreqs, Npols)
        if data_array.shape != proper_shape:
            raise AssertionError("data_array has shape {0}; was expecting {1}".format(data_array.shape,
                                                                                      proper_shape))

        # actually write the data
        with h5py.File(filename, 'r+') as f:
            dgrp = f['/Data']
            visdata_dset = dgrp['visdata']
            flags_dset = dgrp['flags']
            nsamples_dset = dgrp['nsamples']
            visdata_dtype = visdata_dset.dtype
            if visdata_dtype not in ('complex64', 'complex128'):
                custom_dtype = True
            else:
                custom_dtype = False

            # check if we can do fancy indexing
            # as long as at least 2 out of 3 axes can be written as slices, we can be fancy
            n_reg_spaced = np.count_nonzero([blt_reg_spaced, freq_reg_spaced, pol_reg_spaced])
            if n_reg_spaced >= 2:
                if custom_dtype:
                    indices = (blt_inds, np.s_[:], freq_inds, pol_inds)
                    _write_complex_astype(data_array, visdata_dset, indices)
                else:
                    visdata_dset[blt_inds, :, freq_inds, pol_inds] = data_array
                flags_dset[blt_inds, :, freq_inds, pol_inds] = flags_array
                nsamples_dset[blt_inds, :, freq_inds, pol_inds] = nsample_array
            elif n_reg_spaced == 1:
                # figure out which axis is regularly spaced
                if blt_reg_spaced:
                    for ifreq, freq_idx in enumerate(freq_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_inds, np.s_[:], freq_idx, pol_idx)
                                _write_complex_astype(data_array[:, :, ifreq, ipol], visdata_dset, indices)
                            else:
                                visdata_dset[blt_inds, :, freq_idx, pol_idx] = data_array[:, :, ifreq, ipol]
                            flags_dset[blt_inds, :, freq_idx, pol_idx] = flags_array[:, :, ifreq, ipol]
                            nsamples_dset[blt_inds, :, freq_idx, pol_idx] = nsample_array[:, :, ifreq, ipol]
                elif freq_reg_spaced:
                    for iblt, blt_idx in enumerate(blt_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_idx, np.s_[:], freq_inds, pol_idx)
                                _write_complex_astype(data_array[iblt, :, :, ipol], visdata_dset, indices)
                            else:
                                visdata_dset[blt_idx, :, freq_inds, pol_idx] = data_array[iblt, :, :, ipol]
                            flags_dset[blt_idx, :, freq_inds, pol_idx] = flags_array[iblt, :, :, ipol]
                            nsamples_dset[blt_idx, :, freq_inds, pol_idx] = nsample_array[iblt, :, :, ipol]
                else:  # pol_reg_spaced
                    for iblt, blt_idx in enumerate(blt_inds):
                        for ifreq, freq_idx in enumerate(freq_inds):
                            if custom_dtype:
                                indices = (blt_idx, np.s_[:], freq_idx, pol_inds)
                                _write_complex_astype(data_array[iblt, :, ifreq, :], visdata_dset, indices)
                            else:
                                visdata_dset[blt_idx, :, freq_idx, pol_inds] = data_array[iblt, :, ifreq, :]
                            flags_dset[blt_idx, :, freq_idx, pol_inds] = flags_array[iblt, :, ifreq, :]
                            nsamples_dset[blt_idx, :, freq_idx, pol_inds] = nsample_array[iblt, :, ifreq, :]
            else:
                # all axes irregularly spaced
                # perform a triple loop -- probably very slow!
                for iblt, blt_idx in enumerate(blt_inds):
                    for ifreq, freq_idx in enumerate(freq_inds):
                        for ipol, pol_idx in enumerate(pol_inds):
                            if custom_dtype:
                                indices = (blt_idx, np.s_[:], freq_idx, pol_idx)
                                _write_complex_astype(data_array[iblt, :, ifreq, ipol], visdata_dset, indices)
                            else:
                                visdata_dset[blt_idx, :, freq_idx, pol_idx] = data_array[iblt, :, ifreq, ipol]
                            flags_dset[blt_idx, :, freq_idx, pol_idx] = flags_array[iblt, :, ifreq, ipol]
                            nsamples_dset[blt_idx, :, freq_idx, pol_idx] = nsample_array[iblt, :, ifreq, ipol]

            # append to history if desired
            if add_to_history is not None:
                history = np.string_(self.history) + np.string_(add_to_history)
                if 'history' in f['Header']:
                    # erase dataset first b/c it has fixed-length string datatype
                    del f['Header']['history']
                f['Header']['history'] = np.string_(history)

        return
