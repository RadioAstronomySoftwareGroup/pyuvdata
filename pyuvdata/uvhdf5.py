"""Class for reading and writing HDF5 files."""
import numpy as np
import h5py
import os
from uvdata import UVData
import utils as uvutils


class UVHDF5(UVData):
    """
    Defines an HDF5-specific subclass of UVData for reading and writing uvhdf5 files.
    This class should not be interacted with directly, instead use the read_uvhdf5
    and write_uvhdf5 methods on the UVData class.
    """

    def read_uvhdf5(self, filename, run_check=True,
                    check_extra=True, run_check_acceptability=True, phase_type=None):
        if not os.path.exists(filename):
            raise(IOError, filename + ' not found')

        # open hdf5 file for reading
        f = h5py.File(filename, 'r')

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
            self.dut1 = header['dut1'].value
        if 'earth_omega' in header:
            self.earth_omega = header['earth_omega'].value
        if 'gst0' in header:
            self.gst0 = header['gst0'].value
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
        if 'instrument' in header:
            self.instrument = header['instrument'].value
        else:
            self.instrument = None

        # check for phasing information
        self.phase_type = header['phase_type'].value
        if self.phase_type == 'phased':
            self.set_phased()
            self.phase_center_ra = header['phase_center_ra'].value
            self.phase_center_dec = header['phase_center_dec'].value
            self.phase_center_epoch = header['phase_center_epoch'].value
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

        # get sample information
        self.nsample_array = header['nsample_array'].value

        # get data shapes
        self.Nfreqs = int(header['Nfreqs'].value)
        self.Npols = int(header['Npols'].value)
        self.Ntimes = int(header['Ntimes'].value)
        self.Nblts = int(header['Nblts'].value)
        self.Nspws = int(header['Nspws'].value)

        # read data array
        dgrp = f['/Data']
        self.data_array = dgrp['visdata'].value

        # read the flag array
        self.flag_array = dgrp['flags'].value

        # check if the object has all required UVParameters
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        return

    def write_uvhdf5(self, filename, run_check=True, check_extra=True,
                     run_check_acceptability=True, clobber=False):
        """
        Write data
        """
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        if os.path.exists(filename):
            if clobber:
                print "File exists; clobbering"
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

        # write out UVParameters
        header['Nants_data'] = self.Nants_data
        header['Nants_telescope'] = self.Nants_telescope
        header['Nbls'] = self.Nbls
        header['Nblts'] = self.Nblts
        header['Nfreqs'] = self.Nfreqs
        header['Npols'] = self.Npols
        header['Nspws'] = self.Nspws
        header['Ntimes'] = self.Ntimes

        # write out arrays
        header['antenna_names'] = self.antenna_names
        header['antenna_numbers'] = self.antenna_numbers
        header['uvw_array'] = self.uvw_array
        header['vis_units'] = self.vis_units
        header['channel_width'] = self.channel_width
        header['time_array'] = self.time_array
        header['freq_array'] = self.freq_array
        header['integration_time'] = self.integration_time
        header['lst_array'] = self.lst_array
        header['nsample_array'] = self.nsample_array
        header['polarization_array'] = self.polarization_array
        header['spw_array'] = self.spw_array

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
            header['earth_omega'] = self.gst0
        if self.rdate is not None:
            header['rdate'] = self.rdate
        if self.timesys is not None:
            header['timesys'] = self.timesys
        if self.x_orientation is not None:
            header['x_orientation'] = self.x_orientation

        # write out extra parameters
        for p in self.extra():
            param = getattr(self, p)
            if param.name == 'extra_keywords':
                for k in param.value.keys():
                    header[k] = param.value[k]
            elif param.name == 'antenna_positions':
                # we already dealt with these values above
                pass
            elif param.value is not None:
                header[param.name] = param.value

        # write out history
        header['history'] = self.history

        # write out antenna arrays
        header['ant_1_array'] = self.ant_1_array
        header['ant_2_array'] = self.ant_2_array

        # write out data and flags
        dgrp = f.create_group("Data")
        dgrp['visdata'] = self.data_array
        dgrp['flags'] = self.flag_array

        return
