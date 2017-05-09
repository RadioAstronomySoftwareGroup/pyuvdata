"""Primary container for radio interferometer datasets."""
from astropy import constants as const
from astropy.time import Time
import os
import numpy as np
import warnings
import ephem
from uvbase import UVBase
import parameter as uvp
import telescopes as uvtel
import utils as uvutils
import version as uvversion


class UVData(UVBase):
    """
    A class for defining a radio interferometer dataset.

    Currently supported file types: uvfits, miriad, fhd.
    Provides phasing functions.

    Attributes:
        UVParameter objects: For full list see Parameters (docs/parameters.rst or https://pyuvdata.readthedocs.io/en/latest/parameters.html).
            Some are always required, some are required for certain phase_types
            and others are always optional.
    """

    def __init__(self):
        """Create a new UVData object."""
        # add the UVParameters to the class

        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)

        self._Ntimes = uvp.UVParameter('Ntimes', description='Number of times',
                                       expected_type=int)
        self._Nbls = uvp.UVParameter('Nbls', description='Number of baselines',
                                     expected_type=int)
        self._Nblts = uvp.UVParameter('Nblts', description='Number of baseline-times '
                                      '(i.e. number of spectra). Not necessarily '
                                      'equal to Nbls * Ntimes', expected_type=int)
        self._Nfreqs = uvp.UVParameter('Nfreqs', description='Number of frequency channels',
                                       expected_type=int)
        self._Npols = uvp.UVParameter('Npols', description='Number of polarizations',
                                      expected_type=int)

        desc = ('Array of the visibility data, shape: (Nblts, Nspws, Nfreqs, '
                'Npols), type = complex float, in units of self.vis_units')
        self._data_array = uvp.UVParameter('data_array', description=desc,
                                           form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                           expected_type=np.complex)

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = uvp.UVParameter('vis_units', description=desc,
                                          form='str', expected_type=str,
                                          acceptable_vals=["uncalib", "Jy", "K str"])

        desc = ('Number of data points averaged into each data element, '
                'NOT required to be an integer. type = float, same shape as data_array')
        self._nsample_array = uvp.UVParameter('nsample_array', description=desc,
                                              form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                              expected_type=(np.float))

        desc = 'Boolean flag, True is flagged, same shape as data_array.'
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                           expected_type=np.bool)

        self._Nspws = uvp.UVParameter('Nspws', description='Number of spectral windows '
                                      '(ie non-contiguous spectral chunks). '
                                      'More than one spectral window is not '
                                      'currently supported.', expected_type=int)

        self._spw_array = uvp.UVParameter('spw_array',
                                          description='Array of spectral window '
                                          'Numbers, shape (Nspws)', form=('Nspws',),
                                          expected_type=int)

        desc = ('Projected baseline vectors relative to phase center, ' +
                'shape (Nblts, 3), units meters')
        self._uvw_array = uvp.UVParameter('uvw_array', description=desc,
                                          form=('Nblts', 3),
                                          expected_type=np.float,
                                          acceptable_range=(1e-3, 1e8), tols=.001)

        desc = ('Array of times, center of integration, shape (Nblts), ' +
                'units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Nblts',),
                                           expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in days

        desc = ('Array of lsts, center of integration, shape (Nblts), ' +
                'units radians')
        self._lst_array = uvp.UVParameter('lst_array', description=desc,
                                          form=('Nblts',),
                                          expected_type=np.float,
                                          tols=radian_tol)

        desc = ('Array of first antenna indices, shape (Nblts), '
                'type = int, 0 indexed')
        self._ant_1_array = uvp.UVParameter('ant_1_array', description=desc,
                                            expected_type=int, form=('Nblts',))
        desc = ('Array of second antenna indices, shape (Nblts), '
                'type = int, 0 indexed')
        self._ant_2_array = uvp.UVParameter('ant_2_array', description=desc,
                                            expected_type=int, form=('Nblts',))

        desc = ('Array of baseline indices, shape (Nblts), '
                'type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16')
        self._baseline_array = uvp.UVParameter('baseline_array',
                                               description=desc,
                                               expected_type=int, form=('Nblts',))

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        desc = 'Array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        desc = ('Array of polarization integers, shape (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)')
        self._polarization_array = uvp.UVParameter('polarization_array',
                                                   description=desc,
                                                   expected_type=int,
                                                   acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
                                                   form=('Npols',))

        self._integration_time = uvp.UVParameter('integration_time',
                                                 description='Length of the integration (s)',
                                                 expected_type=np.float, tols=1e-3)  # 1 ms
        self._channel_width = uvp.UVParameter('channel_width',
                                              description='Width of frequency channels (Hz)',
                                              expected_type=np.float,
                                              tols=1e-3)  # 1 mHz

        # --- observation information ---
        self._object_name = uvp.UVParameter('object_name',
                                            description='Source or field '
                                            'observed (string)', form='str',
                                            expected_type=str)
        self._telescope_name = uvp.UVParameter('telescope_name',
                                               description='Name of telescope '
                                               '(string)', form='str',
                                               expected_type=str)
        self._instrument = uvp.UVParameter('instrument', description='Receiver or backend. '
                                           'Sometimes identical to telescope_name',
                                           form='str', expected_type=str)

        desc = ('Telescope location: xyz in ITRF (earth-centered frame). '
                'Can also be accessed using telescope_location_lat_lon_alt or '
                'telescope_location_lat_lon_alt_degrees properties')
        self._telescope_location = uvp.LocationParameter('telescope_location',
                                                         description=desc,
                                                         acceptable_range=(6.35e6, 6.39e6),
                                                         tols=1e-3)

        self._history = uvp.UVParameter('history', description='String of history, units English',
                                        form='str', expected_type=str)

        # --- phasing information ---
        desc = ('String indicating phasing type. Allowed values are "drift", '
                '"phased" and "unknown"')
        self._phase_type = uvp.UVParameter('phase_type', form='str', expected_type=str,
                                           description=desc, value='unknown',
                                           acceptable_vals=['drift', 'phased', 'unknown'])

        desc = ('Required if phase_type = "drift". Right ascension of zenith. '
                'units: radians, shape (Nblts). Can also be accessed using zenith_ra_degrees.')
        self._zenith_ra = uvp.AngleParameter('zenith_ra', required=False,
                                             description=desc,
                                             expected_type=np.float,
                                             form=('Nblts',),
                                             tols=radian_tol)

        desc = ('Required if phase_type = "drift". Declination of zenith. '
                'units: radians, shape (Nblts). Can also be accessed using zenith_dec_degrees.')
        # in practice, dec of zenith will never change; does not need to be shape Nblts
        self._zenith_dec = uvp.AngleParameter('zenith_dec', required=False,
                                              description=desc,
                                              expected_type=np.float,
                                              form=('Nblts',),
                                              tols=radian_tol)

        desc = ('Required if phase_type = "phased". Epoch year of the phase '
                'applied to the data (eg 2000.)')
        self._phase_center_epoch = uvp.UVParameter('phase_center_epoch',
                                                   required=False,
                                                   description=desc,
                                                   expected_type=np.float)

        desc = ('Required if phase_type = "phased". Right ascension of phase '
                'center (see uvw_array), units radians. Can also be accessed using phase_center_ra_degrees.')
        self._phase_center_ra = uvp.AngleParameter('phase_center_ra',
                                                   required=False,
                                                   description=desc,
                                                   expected_type=np.float,
                                                   tols=radian_tol)

        desc = ('Required if phase_type = "phased". Declination of phase center '
                '(see uvw_array), units radians. Can also be accessed using phase_center_dec_degrees.')
        self._phase_center_dec = uvp.AngleParameter('phase_center_dec',
                                                    required=False,
                                                    description=desc,
                                                    expected_type=np.float,
                                                    tols=radian_tol)

        # --- antenna information ----
        desc = ('Number of antennas with data present (i.e. number of unique '
                'entries in ant_1_array and ant_2_array). May be smaller ' +
                'than the number of antennas in the array')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc,
                                           expected_type=int)

        desc = ('Number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope',
                                                description=desc, expected_type=int)

        desc = ('List of antenna names, shape (Nants_telescope), '
                'with numbers given by antenna_numbers (which can be matched '
                'to ant_1_array and ant_2_array). There must be one entry '
                'here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_names = uvp.UVParameter('antenna_names', description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('List of integer antenna numbers corresponding to antenna_names, '
                'shape (Nants_telescope). There must be one '
                'entry here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers', description=desc,
                                                form=('Nants_telescope',),
                                                expected_type=int)

        # -------- extra, non-required parameters ----------
        desc = ('Any user supplied extra keywords, type=dict')
        self._extra_keywords = uvp.UVParameter('extra_keywords', required=False,
                                               description=desc, value={},
                                               spoof_val={}, expected_type=dict)

        desc = ('Array giving coordinates of antennas relative to '
                'telescope_location (ITRF frame), shape (Nants_telescope, 3)')
        self._antenna_positions = uvp.AntPositionParameter('antenna_positions',
                                                           required=False,
                                                           description=desc,
                                                           form=('Nants_telescope', 3),
                                                           expected_type=np.float,
                                                           tols=1e-3)  # 1 mm

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._gst0 = uvp.UVParameter('gst0', required=False,
                                     description='Greenwich sidereal time at '
                                                 'midnight on reference date',
                                     spoof_val=0.0)
        self._rdate = uvp.UVParameter('rdate', required=False,
                                      description='Date for which the GST0 or '
                                                  'whatever... applies',
                                      spoof_val='')
        self._earth_omega = uvp.UVParameter('earth_omega', required=False,
                                            description='Earth\'s rotation rate '
                                                        'in degrees per day',
                                            spoof_val=360.985)
        self._dut1 = uvp.UVParameter('dut1', required=False,
                                     description='DUT1 (google it) AIPS 117 '
                                                 'calls it UT1UTC',
                                     spoof_val=0.0)
        self._timesys = uvp.UVParameter('timesys', required=False,
                                        description='We only support UTC',
                                        spoof_val='UTC', form='str')

        desc = ('FHD thing we do not understand, something about the time '
                'at which the phase center is normal to the chosen UV plane '
                'for phasing')
        self._uvplane_reference_time = uvp.UVParameter('uvplane_reference_time',
                                                       required=False,
                                                       description=desc,
                                                       spoof_val=0)

        # String to add to history of any files written with this version of pyuvdata
        self.pyuvdata_version_str = ('  Read/written with pyuvdata version: ' +
                                     uvversion.version + '.')
        if uvversion.git_hash is not '':
            self.pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin +
                                          '.  Git hash: ' + uvversion.git_hash +
                                          '.  Git branch: ' + uvversion.git_branch +
                                          '.  Git description: ' + uvversion.git_description)

        super(UVData, self).__init__()

    def check(self, run_check_acceptability=True):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Args:
            run_check_acceptability: Option to check if values in required parameters
                are acceptable. Default is True.
        """
        # first run the basic check from UVBase
        if np.all(self.ant_1_array == self.ant_2_array):
            # Special case of only containing auto correlations
            self._uvw_array.acceptable_range = (0.0, 0.0)
        super(UVData, self).check(run_check_acceptability=run_check_acceptability)

        # Check internal consistency of numbers which don't explicitly correspond
        # to the shape of another array.
        nants_data_calc = int(len(np.unique(self.ant_1_array.tolist() +
                                            self.ant_2_array.tolist())))
        if self.Nants_data != nants_data_calc:
            raise ValueError('Nants_data must be equal to the number of unique '
                             'values in ant_1_array and ant_2_array')

        if self.Nbls != len(np.unique(self.baseline_array)):
            raise ValueError('Nbls must be equal to the number of unique '
                             'baselines in the data_array')

        if self.Ntimes != len(np.unique(self.time_array)):
            raise ValueError('Ntimes must be equal to the number of unique '
                             'times in the time_array')

        return True

    def set_drift(self):
        """Set phase_type to 'drift' and adjust required parameters."""
        self.phase_type = 'drift'
        self._zenith_ra.required = True
        self._zenith_dec.required = True
        self._phase_center_epoch.required = False
        self._phase_center_ra.required = False
        self._phase_center_dec.required = False

    def set_phased(self):
        """Set phase_type to 'phased' and adjust required parameters."""
        self.phase_type = 'phased'
        self._zenith_ra.required = False
        self._zenith_dec.required = False
        self._phase_center_epoch.required = True
        self._phase_center_ra.required = True
        self._phase_center_dec.required = True

    def set_unknown_phase_type(self):
        """Set phase_type to 'unknown' and adjust required parameters."""
        self.phase_type = 'unknown'
        self._zenith_ra.required = False
        self._zenith_dec.required = False
        self._phase_center_epoch.required = False
        self._phase_center_ra.required = False
        self._phase_center_dec.required = False

    def known_telescopes(self):
        """
        Retun a list of telescopes known to pyuvdata.

        This is just a shortcut to uvdata.telescopes.known_telescopes()
        """
        return uvtel.known_telescopes()

    def set_telescope_params(self, overwrite=False):
        """
        Set telescope related parameters.

        If the telescope_name is in the known_telescopes, set any missing
        telescope-associated parameters (e.g. telescope location) to the value
        for the known telescope.

        Args:
            overwrite: Option to overwrite existing telescope-associated
                parameters with the values from the known telescope.
                Default is False.
        """
        telescope_obj = uvtel.get_telescope(self.telescope_name)
        if telescope_obj is not False:
            params_set = []
            for p in telescope_obj:
                self_param = getattr(self, p)
                if overwrite is True or self_param.value is None:
                    params_set.append(self_param.name)
                    prop_name = self_param.name
                    setattr(self, prop_name, getattr(telescope_obj, prop_name))
            if len(params_set) == 1:
                params_set_str = params_set[0]
                warnings.warn('{param} is not set. Using known values '
                              'for {telescope_name}.'.format(param=params_set_str,
                                                             telescope_name=telescope_obj.telescope_name))
            elif len(params_set) > 1:
                params_set_str = ', '.join(params_set)
                warnings.warn('{params} are not set. Using known values '
                              'for {telescope_name}.'.format(params=params_set_str,
                                                             telescope_name=telescope_obj.telescope_name))
        else:
            raise ValueError('Telescope {telescope_name} is not in '
                             'known_telescopes.'.format(telescope_name=self.telescope_name))

    def baseline_to_antnums(self, baseline):
        """
        Get the antenna numbers corresponding to a given baseline number.

        Args:
            baseline: integer baseline number

        Returns:
            tuple with the two antenna numbers corresponding to the baseline.
        """
        if self.Nants_telescope > 2048:
            raise StandardError('error Nants={Nants}>2048 not '
                                'supported'.format(Nants=self.Nants_telescope))
        if np.min(baseline) > 2**16:
            ant2 = (baseline - 2**16) % 2048 - 1
            ant1 = (baseline - 2**16 - (ant2 + 1)) / 2048 - 1
        else:
            ant2 = (baseline) % 256 - 1
            ant1 = (baseline - (ant2 + 1)) / 256 - 1
        return np.int32(ant1), np.int32(ant2)

    def antnums_to_baseline(self, ant1, ant2, attempt256=False):
        """
        Get the baseline number corresponding to two given antenna numbers.

        Args:
            ant1: first antenna number (integer)
            ant2: second antenna number (integer)
            attempt256: Option to try to use the older 256 standard used in
                many uvfits files (will use 2048 standard if there are more
                than 256 antennas). Default is False.

        Returns:
            integer baseline number corresponding to the two antenna numbers.
        """
        ant1, ant2 = np.int64((ant1, ant2))
        if self.Nants_telescope > 2048:
            raise StandardError('cannot convert ant1, ant2 to a baseline index '
                                'with Nants={Nants}>2048.'
                                .format(Nants=self.Nants_telescope))
        if attempt256:
            if (np.max(ant1) < 255 and np.max(ant2) < 255):
                return 256 * (ant1 + 1) + (ant2 + 1)
            else:
                print('Max antnums are {} and {}'.format(np.max(ant1), np.max(ant2)))
                message = 'antnums_to_baseline: found > 256 antennas, using ' \
                          '2048 baseline indexing. Beware compatibility ' \
                          'with CASA etc'
                warnings.warn(message)

        return np.int64(2048 * (ant2 + 1) + (ant1 + 1) + 2**16)

    def set_lsts_from_time_array(self):
        """Set the lst_array based from the time_array."""
        lsts = []
        curtime = self.time_array[0]
        self.lst_array = np.zeros(self.Nblts)
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        for ind, jd in enumerate(np.unique(self.time_array)):
            t = Time(jd, format='jd', location=(longitude, latitude))
            self.lst_array[np.where(np.isclose(jd, self.time_array, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').radian

    def juldate2ephem(self, num):
        """
        Convert Julian date to ephem date, measured from noon, Dec. 31, 1899.

        Args:
            num: Julian date

        Returns:
            ephem date, measured from noon, Dec. 31, 1899.
        """
        return ephem.date(num - 2415020.)

    def unphase_to_drift(self):
        """Convert from a phased dataset to a drift dataset."""
        if self.phase_type == 'phased':
            pass
        elif self.phase_type == 'drift':
            raise ValueError('The data is already drift scanning; can only ' +
                             'unphase phased data.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to drift or phased to '
                             'reflect the phasing status of the data')

        latitude, longitude, altitude = self.telescope_location_lat_lon_alt

        obs = ephem.Observer()
        # obs inits with default values for parameters -- be sure to replace them
        obs.lat = latitude
        obs.lon = longitude

        phase_center = ephem.FixedBody()
        epoch = (self.phase_center_epoch - 2000.) * 365.2422 + ephem.J2000  # convert years to ephemtime
        phase_center._epoch = epoch
        phase_center._ra = self.phase_center_ra
        phase_center._dec = self.phase_center_dec

        self.zenith_ra = np.zeros_like(self.time_array)
        self.zenith_dec = np.zeros_like(self.time_array)

        # apply -w phasor
        w_lambda = (self.uvw_array[:, 2].reshape(self.Nblts, 1) /
                    const.c.to('m/s').value * self.freq_array.reshape(1, self.Nfreqs))
        phs = np.exp(-1j * 2 * np.pi * (-1) * w_lambda[:, None, :, None])
        self.data_array *= phs

        unique_times = np.unique(self.time_array)
        for jd in unique_times:
            inds = np.where(self.time_array == jd)[0]
            obs.date, obs.epoch = self.juldate2ephem(jd), self.juldate2ephem(jd)
            phase_center.compute(obs)
            phase_center_ra, phase_center_dec = phase_center.a_ra, phase_center.a_dec
            zenith_ra = obs.sidereal_time()
            zenith_dec = latitude
            self.zenith_ra[inds] = zenith_ra
            self.zenith_dec[inds] = zenith_dec

            # generate rotation matrices
            m0 = uvutils.top2eq_m(0., phase_center_dec)
            m1 = uvutils.eq2top_m(phase_center_ra - zenith_ra, zenith_dec)

            # rotate and write uvws
            uvw = self.uvw_array[inds, :]
            uvw = np.dot(m0, uvw.T).T
            uvw = np.dot(m1, uvw.T).T
            self.uvw_array[inds, :] = uvw

        # remove phase center
        self.phase_center_ra = None
        self.phase_center_dec = None
        self.phase_center_epoch = None
        self.set_drift()

    def phase_to_time(self, time):
        """
        Phase a drift scan dataset to the ra/dec of zenith at a particular time.

        Args:
            time: The time to phase to.
        """
        if self.phase_type == 'drift':
            pass
        elif self.phase_type == 'phased':
            raise ValueError('The data is already phased; can only phase ' +
                             'drift scanning data.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to drift or phased to '
                             'reflect the phasing status of the data')

        obs = ephem.Observer()
        # obs inits with default values for parameters -- be sure to replace them
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt
        obs.lat = latitude
        obs.lon = longitude

        obs.date, obs.epoch = self.juldate2ephem(time), self.juldate2ephem(time)

        ra = obs.sidereal_time()
        dec = latitude
        epoch = self.juldate2ephem(time)
        self.phase(ra, dec, epoch)

    def phase(self, ra, dec, epoch):
        """
        Phase a drift scan dataset to a single ra/dec at a particular epoch.

        Will not phase already phased data.

        Args:
            ra: The ra to phase to in radians.
            dec: The dec to phase to in radians.
            epoch: The epoch to use for phasing. Should be an ephem date,
                measured from noon Dec. 31, 1899.
        """
        if self.phase_type == 'drift':
            pass
        elif self.phase_type == 'phased':
            raise ValueError('The data is already phased; can only phase '
                             'drift scan data. Use unphase_to_drift to '
                             'convert to a drift scan.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to "drift" or "phased" to '
                             'reflect the phasing status of the data')

        obs = ephem.Observer()
        # obs inits with default values for parameters -- be sure to replace them
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt
        obs.lat = latitude
        obs.lon = longitude

        # create a pyephem object for the phasing position
        precess_pos = ephem.FixedBody()
        precess_pos._ra = ra
        precess_pos._dec = dec
        precess_pos._epoch = epoch

        # calculate RA/DEC in J2000 and write to object
        obs.date, obs.epoch = ephem.J2000, ephem.J2000
        precess_pos.compute(obs)

        self.phase_center_ra = precess_pos.a_ra + 0.0  # force to be a float not ephem.Angle
        self.phase_center_dec = precess_pos.a_dec + 0.0  # force to be a float not ephem.Angle
        # explicitly set epoch to J2000
        self.phase_center_epoch = 2000.0

        unique_times, unique_inds = np.unique(self.time_array, return_index=True)
        for ind, jd in enumerate(unique_times):
            inds = np.where(self.time_array == jd)[0]
            lst = self.lst_array[unique_inds[ind]]
            # calculate ra/dec of phase center in current epoch
            obs.date, obs.epoch = self.juldate2ephem(jd), self.juldate2ephem(jd)
            precess_pos.compute(obs)
            ra, dec = precess_pos.a_ra, precess_pos.a_dec

            # generate rotation matrices
            m0 = uvutils.top2eq_m(lst - obs.sidereal_time(), latitude)
            m1 = uvutils.eq2top_m(lst - ra, dec)

            # rotate and write uvws
            uvw = self.uvw_array[inds, :]
            uvw = np.dot(m0, uvw.T).T
            uvw = np.dot(m1, uvw.T).T
            self.uvw_array[inds, :] = uvw

        # calculate data and apply phasor
        w_lambda = (self.uvw_array[:, 2].reshape(self.Nblts, 1) /
                    const.c.to('m/s').value * self.freq_array.reshape(1, self.Nfreqs))
        phs = np.exp(-1j * 2 * np.pi * w_lambda[:, None, :, None])
        self.data_array *= phs

        del(obs)
        self.set_phased()

    def select(self, antenna_nums=None, antenna_names=None, ant_pairs_nums=None,
               frequencies=None, freq_chans=None,
               times=None, polarizations=None, blt_inds=None, run_check=True,
               run_check_acceptability=True):
        """
        Select specific antennas, antenna pairs, frequencies, times and
        polarizations to keep in the object while discarding others.

        Also supports selecting specific baseline-time indices to keep while
        discarding others, but this is not commonly used. The history attribute
        on the object will be updated to identify the operations performed.

        Args:
            antenna_nums: The antennas numbers to keep in the object (antenna
                positions and names for the removed antennas will be retained).
                This cannot be provided if antenna_names is also provided.
            antenna_names: The antennas names to keep in the object (antenna
                positions and names for the removed antennas will be retained).
                This cannot be provided if antenna_nums is also provided.
            ant_pairs_nums: A list of antenna number tuples (e.g. [(0,1), (3,2)])
                specifying baselines to keep in the object. Ordering of the
                numbers within the tuple does not matter.
            frequencies: The frequencies to keep in the object.
            freq_chans: The frequency channel numbers to keep in the object.
            times: The times to keep in the object.
            polarizations: The polarizations to keep in the object.
            blt_inds: The baseline-time indices to keep in the object. This is
                not commonly used.
            run_check: Option to check for the existence and proper shapes of
                required parameters after downselecting data on this object. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after  downselecting data on this object. Default is True.
        """
        # build up history string as we go
        history_update_string = '  Downselected to specific '
        n_selects = 0

        # Antennas, times and blt_inds all need to be combined into a set of
        # blts indices to keep.

        # test for blt_inds presence before adding inds from antennas & times
        if blt_inds is not None:
            blt_inds = uvutils.get_iterable(blt_inds)
            history_update_string += 'baseline-times'
            n_selects += 1

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError('Only one of antenna_nums and antenna_names can be provided.')

            antenna_names = uvutils.get_iterable(antenna_names)
            antenna_nums = []
            for s in antenna_names:
                if s not in self.antenna_names:
                    raise ValueError('Antenna name {a} is not present in the antenna_names array'.format(a=s))
                antenna_nums.append(self.antenna_numbers[np.where(np.array(self.antenna_names) == s)[0]])

        if antenna_nums is not None:
            antenna_nums = uvutils.get_iterable(antenna_nums)
            if n_selects > 0:
                history_update_string += ', antennas'
            else:
                history_update_string += 'antennas'
            n_selects += 1
            inds1 = np.zeros(0, dtype=np.int)
            inds2 = np.zeros(0, dtype=np.int)
            for ant in antenna_nums:
                if ant in self.ant_1_array or ant in self.ant_2_array:
                    wh1 = np.where(self.ant_1_array == ant)[0]
                    wh2 = np.where(self.ant_2_array == ant)[0]
                    if len(wh1) > 0:
                        inds1 = np.append(inds1, list(wh1))
                    if len(wh2) > 0:
                        inds2 = np.append(inds2, list(wh2))
                else:
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=ant))

            ant_blt_inds = np.array(list(set(inds1).intersection(inds2)), dtype=np.int)
            if blt_inds is not None:
                blt_inds = np.array(list(set(blt_inds).intersection(ant_blt_inds)), dtype=np.int)
            else:
                blt_inds = ant_blt_inds

        if ant_pairs_nums is not None:
            if isinstance(ant_pairs_nums, tuple) and len(ant_pairs_nums) == 2:
                ant_pairs_nums = [ant_pairs_nums]
            if not all(isinstance(item, tuple) for item in ant_pairs_nums):
                raise ValueError('ant_pairs_nums must be a list of tuples of antenna numbers.')
            if not all([isinstance(item[0], (int, long)) for item in ant_pairs_nums] +
                       [isinstance(item[1], (int, long)) for item in ant_pairs_nums]):
                raise ValueError('ant_pairs_nums must be a list of tuples of antenna numbers.')
            if n_selects > 0:
                history_update_string += ', antenna pairs'
            else:
                history_update_string += 'antenna pairs'
            n_selects += 1
            ant_pair_blt_inds = np.zeros(0, dtype=np.int)
            for pair in ant_pairs_nums:
                if not (pair[0] in self.ant_1_array or pair[0] in self.ant_2_array):
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=pair[0]))
                if not (pair[1] in self.ant_1_array or pair[1] in self.ant_2_array):
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=pair[1]))
                wh1 = np.where(np.logical_and(self.ant_1_array == pair[0], self.ant_2_array == pair[1]))[0]
                wh2 = np.where(np.logical_and(self.ant_1_array == pair[1], self.ant_2_array == pair[0]))[0]
                if len(wh1) > 0:
                    ant_pair_blt_inds = np.append(ant_pair_blt_inds, list(wh1))
                if len(wh2) > 0:
                    ant_pair_blt_inds = np.append(ant_pair_blt_inds, list(wh2))
                if len(wh1) == 0 and len(wh2) == 0:
                    raise ValueError('Antenna pair {p} does not have any data '
                                     'associated with it.'.format(p=pair))

            if blt_inds is not None:
                blt_inds = np.array(list(set(blt_inds).intersection(ant_pair_blt_inds)), dtype=np.int)
            else:
                blt_inds = ant_pair_blt_inds

        if times is not None:
            times = uvutils.get_iterable(times)
            if n_selects > 0:
                history_update_string += ', times'
            else:
                history_update_string += 'times'
            n_selects += 1

            time_blt_inds = np.zeros(0, dtype=np.int)
            for jd in times:
                if jd in self.time_array:
                    time_blt_inds = np.append(time_blt_inds, np.where(self.time_array == jd)[0])
                else:
                    raise ValueError('Time {t} is not present in the time_array'.format(t=jd))

            if blt_inds is not None:
                blt_inds = np.array(list(set(blt_inds).intersection(time_blt_inds)), dtype=np.int)
            else:
                blt_inds = time_blt_inds

        if blt_inds is not None:

            if len(blt_inds) == 0:
                raise ValueError('No baseline-times were found that match criteria')
            if max(blt_inds) >= self.Nblts:
                raise ValueError('blt_inds contains indices that are too large')
            if min(blt_inds) < 0:
                raise ValueError('blt_inds contains indices that are negative')

            blt_inds = list(sorted(set(list(blt_inds))))
            self.Nblts = len(blt_inds)
            self.baseline_array = self.baseline_array[blt_inds]
            self.Nbls = len(np.unique(self.baseline_array))
            self.time_array = self.time_array[blt_inds]
            self.lst_array = self.lst_array[blt_inds]
            self.data_array = self.data_array[blt_inds, :, :, :]
            self.flag_array = self.flag_array[blt_inds, :, :, :]
            self.nsample_array = self.nsample_array[blt_inds, :, :, :]
            self.uvw_array = self.uvw_array[blt_inds, :]

            self.ant_1_array = self.ant_1_array[blt_inds]
            self.ant_2_array = self.ant_2_array[blt_inds]
            self.Nants_data = int(len(set(self.ant_1_array.tolist() + self.ant_2_array.tolist())))

            self.Ntimes = len(np.unique(self.time_array))

            if self.phase_type == 'drift':
                self.zenith_ra = self.zenith_ra[blt_inds]
                self.zenith_dec = self.zenith_dec[blt_inds]

        if freq_chans is not None:
            freq_chans = uvutils.get_iterable(freq_chans)
            if frequencies is None:
                frequencies = self.freq_array[0, freq_chans]
            else:
                frequencies = uvutils.get_iterable(frequencies)
                frequencies = np.sort(list(set(frequencies) |
                                      set(self.freq_array[0, freq_chans])))

        if frequencies is not None:
            frequencies = uvutils.get_iterable(frequencies)
            if n_selects > 0:
                history_update_string += ', frequencies'
            else:
                history_update_string += 'frequencies'
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int)
            # this works because we only allow one SPW. This will have to be reworked when we support more.
            freq_arr_use = self.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError('Frequency {f} is not present in the freq_array'.format(f=f))

            freq_ind_separation = freq_inds[1:] - freq_inds[:-1]
            if np.min(freq_ind_separation) < np.max(freq_ind_separation):
                warnings.warn('Selected frequencies are not evenly spaced. This '
                              'will make it impossible to write this data out to '
                              'some file types')
            elif np.max(freq_ind_separation) > 1:
                warnings.warn('Selected frequencies are not contiguous. This '
                              'will make it impossible to write this data out to '
                              'some file types.')

            freq_inds = list(sorted(set(list(freq_inds))))
            self.Nfreqs = len(freq_inds)
            self.freq_array = self.freq_array[:, freq_inds]
            self.data_array = self.data_array[:, :, freq_inds, :]
            self.flag_array = self.flag_array[:, :, freq_inds, :]
            self.nsample_array = self.nsample_array[:, :, freq_inds, :]

        if polarizations is not None:
            polarizations = uvutils.get_iterable(polarizations)
            if n_selects > 0:
                history_update_string += ', polarizations'
            else:
                history_update_string += 'polarizations'
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int)
            for p in polarizations:
                if p in self.polarization_array:
                    pol_inds = np.append(pol_inds, np.where(self.polarization_array == p)[0])
                else:
                    raise ValueError('Polarization {p} is not present in the polarization_array'.format(p=p))

            if len(pol_inds) > 2:
                pol_ind_separation = pol_inds[1:] - pol_inds[:-1]
                if np.min(pol_ind_separation) < np.max(pol_ind_separation):
                    warnings.warn('Selected polarization values are not evenly spaced. This '
                                  'will make it impossible to write this data out to '
                                  'some file types')

            pol_inds = list(sorted(set(list(pol_inds))))
            self.Npols = len(pol_inds)
            self.polarization_array = self.polarization_array[pol_inds]
            self.data_array = self.data_array[:, :, :, pol_inds]
            self.flag_array = self.flag_array[:, :, :, pol_inds]
            self.nsample_array = self.nsample_array[:, :, :, pol_inds]

        history_update_string += ' using pyuvdata.'
        self.history = self.history + history_update_string

        # check if object is self-consistent
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype is 'uvfits':
            import uvfits
            other_obj = uvfits.UVFITS()
        elif filetype is 'fhd':
            import fhd
            other_obj = fhd.FHD()
        elif filetype is 'miriad':
            import miriad
            other_obj = miriad.Miriad()
        else:
            raise ValueError('filetype must be uvfits, miriad, or fhd')
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_uvfits(self, filename, run_check=True, run_check_acceptability=True):
        """
        Read in data from a uvfits file.

        Args:
            filename: The uvfits file to read from.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        import uvfits
        uvfits_obj = uvfits.UVFITS()
        uvfits_obj.read_uvfits(filename, run_check=run_check,
                               run_check_acceptability=run_check_acceptability)
        self._convert_from_filetype(uvfits_obj)
        del(uvfits_obj)

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False, run_check=True, run_check_acceptability=True):
        """
        Write the data to a uvfits file.

        Args:
            filename: The uvfits file to write to.
            spoof_nonessential: Option to spoof the values of optional
                UVParameters that are not set but are required for uvfits files.
                Default is False.
            force_phase: Option to automatically phase drift scan data to
                zenith of the first timestamp. Default is False.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters before writing the file. Default is True.
        """
        uvfits_obj = self._convert_to_filetype('uvfits')
        uvfits_obj.write_uvfits(filename, spoof_nonessential=spoof_nonessential,
                                force_phase=force_phase, run_check=run_check,
                                run_check_acceptability=run_check_acceptability)
        del(uvfits_obj)

    def read_fhd(self, filelist, use_model=False, run_check=True,
                 run_check_acceptability=True):
        """
        Read in data from a list of FHD files.

        Args:
            filelist: The list of FHD save files to read from. Must include at
                least one polarization file, a params file and a flag file.
            use_model: Option to read in the model visibilities rather than the
                dirty visibilities. Default is False.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        import fhd
        fhd_obj = fhd.FHD()
        fhd_obj.read_fhd(filelist, use_model=use_model, run_check=run_check,
                         run_check_acceptability=run_check_acceptability)
        self._convert_from_filetype(fhd_obj)
        del(fhd_obj)

    def read_miriad(self, filepath, correct_lat_lon=True, run_check=True, run_check_acceptability=True):
        """
        Read in data from a miriad file.

        Args:
            filepath: The miriad file directory to read from.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        import miriad
        miriad_obj = miriad.Miriad()
        miriad_obj.read_miriad(filepath, correct_lat_lon=correct_lat_lon,
                               run_check=run_check, run_check_acceptability=run_check_acceptability)
        self._convert_from_filetype(miriad_obj)
        del(miriad_obj)

    def write_miriad(self, filepath, run_check=True, run_check_acceptability=True,
                     clobber=False, no_antnums=False):
        """
        Write the data to a miriad file.

        Args:
            filename: The miriad file directory to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
        """
        miriad_obj = self._convert_to_filetype('miriad')
        miriad_obj.write_miriad(filepath, run_check=run_check,
                                run_check_acceptability=run_check_acceptability,
                                clobber=clobber, no_antnums=no_antnums)
        del(miriad_obj)
