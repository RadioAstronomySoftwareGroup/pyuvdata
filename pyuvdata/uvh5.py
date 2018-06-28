# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 The HERA Collaboration
# Licensed under the 2-clause BSD License

"""Class for reading and writing HDF5 files.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os
from .uvdata import UVData
from . import utils as uvutils


class UVH5(UVData):
    """
    Defines an HDF5-specific subclass of UVData for reading and writing uvh5 files.
    This class should not be interacted with directly, instead use the read_uvh5
    and write_uvh5 methods on the UVData class.
    """

    def read_uvh5(self, filename, antenna_nums=None, antenna_names=None,
                  ant_str=None, bls=None, frequencies=None, freq_channels=None,
                  times=None, polarizations=None, blt_inds=None, read_data=True,
                  run_check=True, check_extra=True,
                  run_check_acceptability=True):
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

        Returns:
            None
        """
        import h5py
        if not os.path.exists(filename):
            raise IOError(filename + ' not found')

        if not read_data:
            run_check = False

        # open hdf5 file for reading
        with h5py.File(filename, 'r') as f:
            # extract header information
            header = f['/Header']

            # get telescope information
            latitude = header['latitude'].value
            longitude = header['longitude'].value
            altitude = header['altitude'].value
            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
            self.instrument = header['instrument'].value

            # get source information
            self.object_name = header['object_name'].value

            # set history appropriately
            self.history = header['history'].value
            if not uvutils.check_history_version(self.history, self.pyuvdata_version_str):
                self.history += self.pyuvdata_version_str

            # check for vis_units
            if 'vis_units' in header:
                self.vis_units = header['vis_units'].value
            else:
                # default to uncalibrated data
                self.vis_units = 'UNCALIB'

            # check for optional values
            if 'dut1' in header:
                self.dut1 = float(header['dut1'].value)
            if 'earth_omega' in header:
                self.earth_omega = float(header['earth_omega'].value)
            if 'gst0' in header:
                self.gst0 = float(header['gst0'].value)
            if 'rdate' in header:
                self.rdate = header['rdate'].value
            if 'timesys' in header:
                self.timesys = header['timesys'].value
            if 'x_orientation' in header:
                self.x_orientation = header['x_orientation'].value
            if 'telescope_name' in header:
                self.telescope_name = header['telescope_name'].value
            if 'antenna_positions' in header:
                self.antenna_positions = header['antenna_positions'].value
            if 'antenna_diameters' in header:
                self.antenna_diameters = header['antenna_diameters'].value
            if 'uvplane_reference_time' in header:
                self.uvplane_reference_time = int(header['uvplane_reference_time'].value)

            # check for phasing information
            self.phase_type = header['phase_type'].value
            if self.phase_type == 'phased':
                self.set_phased()
                self.phase_center_ra = float(header['phase_center_ra'].value)
                self.phase_center_dec = float(header['phase_center_dec'].value)
                self.phase_center_epoch = float(header['phase_center_epoch'].value)
            elif self.phase_type == 'drift':
                self.set_drift()
                self.zenith_dec = header['zenith_dec'].value
                self.zenith_ra = header['zenith_ra'].value
            else:
                self.set_unknown_phase_type()

            # get antenna arrays
            # cast to native python int type
            self.Nants_data = int(header['Nants_data'].value)
            self.Nants_telescope = int(header['Nants_telescope'].value)
            self.ant_1_array = header['ant_1_array'].value
            self.ant_2_array = header['ant_2_array'].value
            self.antenna_names = list(header['antenna_names'].value)
            self.antenna_numbers = header['antenna_numbers'].value

            # get baseline array
            self.baseline_array = self.antnums_to_baseline(self.ant_1_array,
                                                           self.ant_2_array)
            self.Nbls = len(np.unique(self.baseline_array))

            # get uvw array
            self.uvw_array = header['uvw_array'].value

            # get time information
            self.time_array = header['time_array'].value
            self.integration_time = float(header['integration_time'].value)
            self.lst_array = header['lst_array'].value

            # get frequency information
            self.freq_array = header['freq_array'].value
            self.channel_width = float(header['channel_width'].value)
            self.spw_array = header['spw_array'].value

            # get polarization information
            self.polarization_array = header['polarization_array'].value

            # get data shapes
            self.Nfreqs = int(header['Nfreqs'].value)
            self.Npols = int(header['Npols'].value)
            self.Ntimes = int(header['Ntimes'].value)
            self.Nblts = int(header['Nblts'].value)
            self.Nspws = int(header['Nspws'].value)

            # get extra_keywords
            if "extra_keywords" in header:
                self.extra_keywords = {}
                for key in header["extra_keywords"].keys():
                    self.extra_keywords[key] = header["extra_keywords"][key].value

            if not read_data:
                # don't read in the data. This means the object is incomplete,
                # but that may not matter for many purposes.
                return

            # Now read in the data
            dgrp = f['/Data']
            self._get_data(dgrp, antenna_nums, antenna_names, ant_str,
                           bls, frequencies, freq_chans, times, polarizations,
                           blt_inds, run_check, check_extra, run_check_acceptability)

            return

    def _get_data(self, dgrp, antenna_nums, antenna_names, ant_str,
                  bls, frequencies, freq_chans, times, polarizations,
                  blt_inds, run_check, check_extra, run_check_acceptability):
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

        if min_frac == 1:
            # no select, read in all the data
            self.data_array = dgrp['visdata'].value
            self.flag_array = dgrp['flags'].value
            self.nsample_array = dgrp['nsample_array'].value
        else:
            # do select operations on everything except data_array, flag_array and nsample_array
            self._select_metadata(blt_inds, freq_inds, pol_inds, history_update_string)

            # open references to datasets
            visdata_dset = dgrp['visdata']
            flags_dset = dgrp['flags']
            nsample_array_dset = dgrp['nsample_array']

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                visdata = visdata_dset[blt_inds, :, :, :]
                flags = flags_dset[blt_inds, :, :, :]
                nsample_array = nsample_array_dset[blt_inds, :, :, :]

                assert(self.Nspws == visdata.shape[1])

                if freq_frac < 1:
                    visdata = visdata[:, :, freq_inds, :]
                    flags = flags[:, :, freq_inds, :]
                    nsample_array = nsample_array[:, :, freq_inds, :]
                if pol_frac < 1:
                    visdata = visdata[:, :, :, pol_inds]
                    flags = flags[:, :, :, pol_inds]
                    nsample_array = nsample_array[:, :, :, pol_inds]
            elif freq_frac == min_frac:
                visdata = visdata_dset[:, :, freq_inds, :]
                flags = flags_dset[:, :, freq_inds, :]
                nsample_array = nsample_array_dset[:, :, freq_inds, :]

                if blt_frac < 1:
                    visdata = visdata[blt_inds, :, :, :]
                    flags = flags[blt_inds, :, :, :]
                    nsample_array = nsample_array[blt_inds, :, :, :]
                if pol_frac < 1:
                    visdata = visdata[:, :, :, pol_inds]
                    flags = flags[:, :, :, pol_inds]
                    nsample_array = nsample_array[:, :, :, pol_inds]
            else:
                visdata = visdata_dset[:, :, :, pol_inds]
                flags = flags_dset[:, :, :, pol_inds]
                nsample_array = nsample_array_dset[:, :, :, pol_inds]

                if blt_frac < 1:
                    visdata = visdata[blt_inds, :, :, :]
                    flags = flags[blt_inds, :, :, :]
                    nsample_array = nsample_array[blt_inds, :, :, :]
                if freq_frac < 1:
                    visdata = visdata[:, :, freq_inds, :]
                    flags = flags[:, :, freq_inds, :]
                    nsample_array = nsample_array[:, :, freq_inds, :]

            # save arrays in object
            self.data_array = visdata
            self.flag_array = flags
            self.nsample_array = nsample_array

        # check if object has all required UVParameters set
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        return

    def write_uvh5(self, filename, run_check=True, check_extra=True,
                   run_check_acceptability=True, clobber=False,
                   data_compression=None, flags_compression="lzf", nsample_compression="lzf"):
        """
        Write a UVData object to a UVH5 file.

        Args:
            filename: The UVH5 file to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the file if it already exists. Default is False.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                 None (no filter/compression).
            flags_compression: HDF5 filter to apply when writing the flags_array. Default is
                 the LZF filter.
            nsample_compression: HDF5 filter to apply when writing the nsample_array. Default is
                 the LZF filter.

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
        import h5py
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        if os.path.exists(filename):
            if clobber:
                print("File exists; clobbering")
            else:
                raise ValueError("File exists; skipping")

        f = h5py.File(filename, 'w')
        header = f.create_group("Header")

        # write out telescope and source information
        header['latitude'] = self.telescope_location_lat_lon_alt[0]
        header['longitude'] = self.telescope_location_lat_lon_alt[1]
        header['altitude'] = self.telescope_location_lat_lon_alt[2]
        header['telescope_name'] = self.telescope_name
        header['instrument'] = self.instrument
        header['object_name'] = self.object_name

        # write out required UVParameters
        header['Nants_data'] = self.Nants_data
        header['Nants_telescope'] = self.Nants_telescope
        header['Nbls'] = self.Nbls
        header['Nblts'] = self.Nblts
        header['Nfreqs'] = self.Nfreqs
        header['Npols'] = self.Npols
        header['Nspws'] = self.Nspws
        header['Ntimes'] = self.Ntimes
        header['antenna_names'] = [uvutils.str_to_bytes(n) for n in self.antenna_names]
        header['antenna_numbers'] = self.antenna_numbers
        header['uvw_array'] = self.uvw_array
        header['vis_units'] = self.vis_units
        header['channel_width'] = self.channel_width
        header['time_array'] = self.time_array
        header['freq_array'] = self.freq_array
        header['integration_time'] = self.integration_time
        header['lst_array'] = self.lst_array
        header['polarization_array'] = self.polarization_array
        header['spw_array'] = self.spw_array
        header['ant_1_array'] = self.ant_1_array
        header['ant_2_array'] = self.ant_2_array

        # write out phasing information
        header['phase_type'] = self.phase_type
        if self.zenith_dec is not None:
            header['zenith_dec'] = self.zenith_dec
        if self.zenith_ra is not None:
            header['zenith_ra'] = self.zenith_ra
        if self.phase_center_ra is not None:
            header['phase_center_ra'] = self.phase_center_ra
        if self.phase_center_dec is not None:
            header['phase_center_dec'] = self.phase_center_dec
        if self.phase_center_epoch is not None:
            header['phase_center_epoch'] = self.phase_center_epoch

        # write out optional parameters
        if self.antenna_positions is not None:
            header['antenna_positions'] = self.antenna_positions
        if self.dut1 is not None:
            header['dut1'] = self.dut1
        if self.earth_omega is not None:
            header['earth_omega'] = self.earth_omega
        if self.gst0 is not None:
            header['gst0'] = self.gst0
        if self.rdate is not None:
            header['rdate'] = self.rdate
        if self.timesys is not None:
            header['timesys'] = self.timesys
        if self.x_orientation is not None:
            header['x_orientation'] = self.x_orientation
        if self.antenna_diameters is not None:
            header['antenna_diameters'] = self.antenna_diameters
        if self.uvplane_reference_time is not None:
            header['uvplane_reference_time'] = self.uvplane_reference_time

        # write out extra keywords if it exists and has elements
        if self.extra_keywords:
            extra_keywords = header.create_group("extra_keywords")
            for k in self.extra_keywords.keys():
                extra_keywords[k] = self.extra_keywords[k]

        # write out history
        header['history'] = self.history

        # write out data, flags, and nsample arrays
        dgrp = f.create_group("Data")
        if data_compression is not None:
            visdata = dgrp.create_dataset("visdata", chunks=True,
                                          data=self.data_array.astype(np.complex64),
                                          compression=data_compression)
        else:
            visdata = dgrp.create_dataset("visdata", chunks=True,
                                          data=self.data_array.astype(np.complex64))
        if flags_compression is not None:
            flags = dgrp.create_dataset("flags", chunks=True,
                                        data=self.flag_array,
                                        compression=flags_compression)
        else:
            flags = dgrp.create_dataset("flags", chunks=True,
                                        data=self.flag_array)
        if nsample_compression is not None:
            nsample_array = dgrp.create_dataset("nsample_array", chunks=True,
                                                data=self.nsample_array.astype(np.float32),
                                                compression=nsample_compression)
        else:
            nsample_array = dgrp.create_dataset("nsample_array", chunks=True,
                                                data=self.nsample_array.astype(np.float32))

        return
