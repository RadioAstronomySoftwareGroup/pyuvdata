"""Class for reading and writing HDF5 files."""
import numpy as np
import h5py
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
        latitude = header['latitude']
        longitude = header['longitude']
        altitude = header['altitude']
        self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)

        self.history = header['history']
        if not uvutils.check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str
        self.channel_width = header['channel_width']

        # check for vis_units
        if 'vis_units' in header:
            self.vis_units = header['vis_units']
        else:
            # default to uncalibrated data
            self.vis_units = 'UNCALIB'

        # check for optional values
        if 'dut1' in header:
            self.dut1 = header['dut1']
        if 'earth_omega' in header:
            self.earth_omega = header['earth_omega']
        if 'gst0' in header:
            self.gst0 = header['gst0']
        if 'rdate' in header:
            self.rdate = header['rdate']
        if 'timesys' in header:
            self.timesys = header['timesys']
        if 'x_orientation' in header:
            self.x_orientation = header['x_orientation']

        # get antenna arrays
        self.ant_1_array = header['ant_1_array']
        self.ant_2_array = header['ant_2_array']

        # get baseline array
        self.baseline_array = self.antnums_to_baseline(self.ant_1_array,
                                                       self.ant_2_array)
        self.Nbls = len(np.unique(self.baseline_array))

        # get time array
        self.time_array = header['time_array']

        # get shapes
        self.Nfreqs = header['Nfreqs']
        self.Npols = header['Npols']
        self.Ntimes = header['Ntimes']

        # read data array
        dgrp = f['/Data']
        self.data_array = dgrp['visdata']

        # read the flag array
        self.flag_array = dgrp['flags']

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
        header['latitude'] = self.telescope_location_lat_lon_alt[0]
        header['longitude'] = self.telescope_location_lat_lon_alt[1]
        header['altitude'] = self.telescope_location_lat_lon_alt[2c]

        # write out UVParameters
        header['Nants_data'] = self.Nants_data
        header['Nants_telescope'] = self.Nants_telescope
        header['Nbls'] = self.Nbls
        header['Nblts'] = self.Nblts
        header['Nfreqs'] = self.Nfreqs
        header['Npols'] = self.Npols
        header['Nspws'] = self.Nspws
        header['Ntimes'] = self.Ntimes
        header['antenna_names'] = self.antenna_names
        header['antenna_numbers'] = self.antenna_numbers
        header['uvw_array'] = self.uvw_array
        header['vis_units'] = self.vis_units

        # write out optional parameters
        for p in self.extra():
            param = getattr(self, p)
            header[param.name] = param.value

        # write out history
        header['history'] = self.history

        # write out antenna arrays
        header['ant_1_array'] = self.ant_1_array
        header['ant_2_array'] = self.ant_2_array

        # write out data and flags
        dgrp = f['/Data']
        dgrp['visdata'] = self.data_array
        dgrp['flags'] = self.flag_array

        return
