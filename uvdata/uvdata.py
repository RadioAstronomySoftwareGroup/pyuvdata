"""Primary container for radio interferometer datasets."""
from astropy import constants as const
from astropy.time import Time
import os
import numpy as np
import warnings
import aipy as a
import ephem
from uvbase import UVBase
import parameter as uvp
import telescopes as uvtel


def _warning(msg, *a):
    """Improve the printing of user warnings."""
    return str(msg) + '\n'


class UVData(UVBase):
    """
    A class for defining a radio interferometer dataset.

    Currently supported file types: uvfits, miriad, fhd
    Provides phasing functions.

    Attributes are all UVParameter objects. For full list see docs/parameters.rst
        Some are always required, some are required for certain phase_types
        and others are always optional.
    """

    def __init__(self):
        """Create a new UVData object."""
        # add the UVParameters to the class
        radian_tol = 2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0)  # 1 mas in radians

        self._Ntimes = uvp.UVParameter('Ntimes', description='Number of times')
        self._Nbls = uvp.UVParameter('Nbls', description='number of baselines')
        self._Nblts = uvp.UVParameter('Nblts', description='Ntimes * Nbls')
        self._Nfreqs = uvp.UVParameter('Nfreqs', description='number of frequency channels')
        self._Npols = uvp.UVParameter('Npols', description='number of polarizations')

        desc = ('array of the visibility data, shape: (Nblts, Nspws, Nfreqs, '
                'Npols), type = complex float, in units of self.vis_units')
        self._data_array = uvp.UVParameter('data_array', description=desc,
                                           form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                           expected_type=np.complex)

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = uvp.UVParameter('vis_units', description=desc,
                                          form='str',
                                          sane_vals=["uncalib", "Jy", "K str"])

        desc = ('number of data points averaged into each data element, '
                'type = int, same shape as data_array')
        self._nsample_array = uvp.UVParameter('nsample_array', description=desc,
                                              form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                              expected_type=(np.float, np.int))

        desc = 'boolean flag, True is flagged, same shape as data_array.'
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                           expected_type=np.bool)

        self._Nspws = uvp.UVParameter('Nspws', description='number of spectral windows '
                                      '(ie non-contiguous spectral chunks)')

        self._spw_array = uvp.UVParameter('spw_array',
                                          description='array of spectral window '
                                          'numbers, shape (Nspws)', form=('Nspws',))

        desc = ('Projected baseline vectors relative to phase center, ' +
                'shape (3, Nblts), units meters')
        self._uvw_array = uvp.UVParameter('uvw_array', description=desc,
                                          form=('Nblts', 3),
                                          expected_type=np.float,
                                          sane_vals=(1e-3, 1e8), tols=.001)

        desc = ('array of times, center of integration, shape (Nblts), ' +
                'units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Nblts',),
                                           expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in days

        desc = ('array of lsts, center of integration, shape (Nblts), ' +
                'units radians')
        self._lst_array = uvp.UVParameter('lst_array', description=desc,
                                          form=('Nblts',),
                                          expected_type=np.float,
                                          tols=radian_tol)

        desc = ('array of first antenna indices, shape (Nblts), '
                'type = int, 0 indexed')
        self._ant_1_array = uvp.UVParameter('ant_1_array', description=desc,
                                            form=('Nblts',))
        desc = ('array of second antenna indices, shape (Nblts), '
                'type = int, 0 indexed')
        self._ant_2_array = uvp.UVParameter('ant_2_array', description=desc,
                                            form=('Nblts',))

        desc = ('array of baseline indices, shape (Nblts), '
                'type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16')
        self._baseline_array = uvp.UVParameter('baseline_array',
                                               description=desc,
                                               form=('Nblts',))

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        desc = 'array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        desc = ('array of polarization integers, shape (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)')
        self._polarization_array = uvp.UVParameter('polarization_array',
                                                   description=desc,
                                                   form=('Npols',))

        self._integration_time = uvp.UVParameter('integration_time',
                                                 description='length of the integration (s)',
                                                 expected_type=np.float, tols=1e-3)  # 1 ms
        self._channel_width = uvp.UVParameter('channel_width',
                                              description='width of channel (Hz)',
                                              expected_type=np.float,
                                              tols=1e-3)  # 1 mHz

        # --- observation information ---
        self._object_name = uvp.UVParameter('object_name',
                                            description='source or field '
                                            'observed (string)', form='str')
        self._telescope_name = uvp.UVParameter('telescope_name',
                                               description='name of telescope '
                                               '(string)', form='str')
        self._instrument = uvp.UVParameter('instrument', description='receiver or backend.',
                                           form='str')

        desc = ('telescope location: xyz in ITRF (earth-centered frame). '
                'Can also be set using telescope_location_lat_lon_alt or '
                'telescope_location_lat_lon_alt_degrees properties')
        self._telescope_location = uvp.LocationParameter('telescope_location',
                                                         description=desc,
                                                         expected_type=np.float,
                                                         form=(3,), tols=1e-3)

        self._history = uvp.UVParameter('history', description='string of history, units English',
                                        form='str')

        # --- phasing information ---
        desc = ('string indicating phasing type. Allowed values are "drift", '
                '"phased" and "unknown"')
        self._phase_type = uvp.UVParameter('phase_type', form='str',
                                           description=desc, value='unknown',
                                           sane_vals=['drift', 'phased', 'unknown'])

        desc = ('Required if phase_type = "drift". Right ascension of zenith. '
                'units: radians, shape (Nblts)')
        self._zenith_ra = uvp.AngleParameter('zenith_ra', required=False,
                                             description=desc,
                                             expected_type=np.float,
                                             form=('Nblts',),
                                             tols=radian_tol)

        desc = ('Required if phase_type = "drift". Declination of zenith. '
                'units: radians, shape (Nblts)')
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
                'center (see uvw_array), units radians')
        self._phase_center_ra = uvp.AngleParameter('phase_center_ra',
                                                   required=False,
                                                   description=desc,
                                                   expected_type=np.float,
                                                   tols=radian_tol)

        desc = ('Required if phase_type = "phased". Declination of phase center '
                '(see uvw_array), units radians')
        self._phase_center_dec = uvp.AngleParameter('phase_center_dec',
                                                    required=False,
                                                    description=desc,
                                                    expected_type=np.float,
                                                    tols=radian_tol)

        # --- antenna information ----
        desc = ('number of antennas with data present. May be smaller ' +
                'than the number of antennas in the array')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc)

        desc = ('number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope', description=desc)

        desc = ('list of antenna names, shape (Nants_telescope), '
                'with numbers given by antenna_numbers (which can be matched '
                'to ant_1_array and ant_2_array). There must be one entry '
                'here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_names = uvp.UVParameter('antenna_names', description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('integer antenna number corresponding to antenna_names, '
                'shape (Nants_telescope). There must be one '
                'entry here for each unique entry in self.ant_1_array and '
                'self.ant_2_array, but there may be extras as well.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers', description=desc,
                                                form=('Nants_telescope',))

        # -------- extra, non-required parameters ----------
        desc = ('any user supplied extra keywords, type=dict')
        self._extra_keywords = uvp.UVParameter('extra_keywords', required=False,
                                               description=desc, value={},
                                               spoof_val={}, expected_type=dict)

        desc = ('array giving coordinates of antennas relative to '
                'telescope_location (ITRF frame), shape (Nants_telescope, 3)')
        self._antenna_positions = uvp.AntPositionParameter('antenna_positions',
                                                           required=False,
                                                           description=desc,
                                                           form=('Nants_telescope', 3),
                                                           tols=1e-3)  # 1 mm

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._gst0 = uvp.UVParameter('gst0', required=False,
                                     description='Greenwich sidereal time at '
                                                 'midnight on reference date',
                                     spoof_val=0.0)
        self._rdate = uvp.UVParameter('rdate', required=False,
                                      description='date for which the GST0 or '
                                                  'whatever... applies',
                                      spoof_val='')
        self._earth_omega = uvp.UVParameter('earth_omega', required=False,
                                            description='earth\'s rotation rate '
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

        super(UVData, self).__init__()
        warnings.formatwarning = _warning

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
            i = (baseline - 2**16) % 2048 - 1
            j = (baseline - 2**16 - (i + 1)) / 2048 - 1
        else:
            i = (baseline) % 256 - 1
            j = (baseline - (i + 1)) / 256 - 1
        return np.int32(i), np.int32(j)

    def antnums_to_baseline(self, i, j, attempt256=False):
        """
        Get the baseline number corresponding to two given antenna numbers.

        Args:
            i: first antenna number (integer)
            j: second antenna number (integer)
            attempt256: Option to try to use the older 256 standard used in
                many uvfits files (will use 2048 standard if there are more
                than 256 antennas). Default is False.
        Returns:
            integer baseline number corresponding to the two antenna numbers.
        """
        i, j = np.int64((i, j))
        if self.Nants_telescope > 2048:
            raise StandardError('cannot convert i,j to a baseline index '
                                'with Nants={Nants}>2048.'
                                .format(Nants=self.Nants_telescope))
        if attempt256:
            if (np.max(i) < 255 and np.max(j) < 255):
                return 256 * (j + 1) + (i + 1)
            else:
                print('Max antnums are {} and {}'.format(np.max(i), np.max(j)))
                message = 'antnums_to_baseline: found > 256 antennas, using ' \
                          '2048 baseline indexing. Beware compatibility ' \
                          'with CASA etc'
                warnings.warn(message)

        return np.int64(2048 * (j + 1) + (i + 1) + 2**16)

    def set_lsts_from_time_array(self):
        """Set the lst_array based from the time_array."""
        lsts = []
        curtime = self.time_array[0]
        for ind, jd in enumerate(self.time_array):
            if ind == 0 or not np.isclose(jd, curtime, atol=1e-6, rtol=1e-12):
                curtime = jd
                latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
                t = Time(jd, format='jd', location=(longitude, latitude))
            lsts.append(t.sidereal_time('apparent').radian)
        self.lst_array = np.array(lsts)

    def juldate2ephem(self, num):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
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

        for ind, jd in enumerate(self.time_array):

            # apply -w phasor
            w_lambda = self.uvw_array[ind, 2] / const.c.to('m/s').value * self.freq_array
            phs = np.exp(-1j * 2 * np.pi * (-1) * w_lambda)
            phs.shape += (1,)
            self.data_array[ind] *= phs

            # calculate ra/dec of phase center in current epoch
            obs.date, obs.epoch = self.juldate2ephem(jd), self.juldate2ephem(jd)
            phase_center.compute(obs)
            phase_center_ra, phase_center_dec = phase_center.a_ra, phase_center.a_dec

            zenith_ra = obs.sidereal_time()
            zenith_dec = latitude
            self.zenith_ra[ind] = zenith_ra
            self.zenith_dec[ind] = zenith_dec

            # generate rotation matrices
            m0 = a.coord.top2eq_m(0., phase_center_dec)
            m1 = a.coord.eq2top_m(phase_center_ra - zenith_ra, zenith_dec)

            # rotate and write uvws
            uvw = self.uvw_array[ind, :]
            uvw = np.dot(m0, uvw)
            uvw = np.dot(m1, uvw)
            self.uvw_array[ind, :] = uvw

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

        for ind, jd in enumerate(self.time_array):
            # calculate ra/dec of phase center in current epoch
            obs.date, obs.epoch = self.juldate2ephem(jd), self.juldate2ephem(jd)
            precess_pos.compute(obs)
            ra, dec = precess_pos.a_ra, precess_pos.a_dec

            # generate rotation matrices
            m0 = a.coord.top2eq_m(self.lst_array[ind] - obs.sidereal_time(), latitude)
            m1 = a.coord.eq2top_m(self.lst_array[ind] - ra, dec)

            # rotate and write uvws
            uvw = self.uvw_array[ind, :]
            uvw = np.dot(m0, uvw)
            uvw = np.dot(m1, uvw)
            self.uvw_array[ind, :] = uvw

            # calculate data and apply phasor
            w_lambda = uvw[2] / const.c.to('m/s').value * self.freq_array
            phs = np.exp(-1j * 2 * np.pi * w_lambda)
            phs.shape += (1,)
            self.data_array[ind] *= phs

        del(obs)
        self.set_phased()

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

    def read_uvfits(self, filename, run_check=True, run_sanity_check=True):
        """
        Read in data from a uvfits file.

        Args:
            filename: The uvfits file to read from.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_sanity_check: Option to sanity check the values of
                required parameters after reading in the file. Default is True.
        """
        import uvfits
        uvfits_obj = uvfits.UVFITS()
        uvfits_obj.read_uvfits(filename, run_check=True, run_sanity_check=True)
        self._convert_from_filetype(uvfits_obj)
        del(uvfits_obj)

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False, run_check=True, run_sanity_check=True):
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
            run_sanity_check: Option to sanity check the values of
                required parameters before writing the file. Default is True.
        """
        uvfits_obj = self._convert_to_filetype('uvfits')
        uvfits_obj.write_uvfits(filename, spoof_nonessential=spoof_nonessential,
                                force_phase=force_phase, run_check=True,
                                run_sanity_check=True)
        del(uvfits_obj)

    def read_fhd(self, filelist, use_model=False, run_check=True,
                 run_sanity_check=True):
        """
        Read in data from a set of FHD files.

        Args:
            filelist: The list of FHD save files to read from.
            use_model: Option to read in the model visibilities rather than the
                dirty visibilities. Default is False.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_sanity_check: Option to sanity check the values of
                required parameters after reading in the file. Default is True.
        """
        import fhd
        fhd_obj = fhd.FHD()
        fhd_obj.read_fhd(filelist, use_model=use_model, run_check=True,
                         run_sanity_check=True)
        self._convert_from_filetype(fhd_obj)
        del(fhd_obj)

    def read_miriad(self, filepath, run_check=True, run_sanity_check=True):
        """
        Read in data from a uvfits file.

        Args:
            filepath: The miriad file directory to read from.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_sanity_check: Option to sanity check the values of
                required parameters after reading in the file. Default is True.
        """
        import miriad
        miriad_obj = miriad.Miriad()
        miriad_obj.read_miriad(filepath, run_check=True, run_sanity_check=True)
        self._convert_from_filetype(miriad_obj)
        del(miriad_obj)

    def write_miriad(self, filename, run_check=True, run_sanity_check=True,
                     clobber=False):
        """
        Write the data to a uvfits file.

        Args:
            filename: The uvfits file to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_sanity_check: Option to sanity check the values of
                required parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
        """
        miriad_obj = self._convert_to_filetype('miriad')
        miriad_obj.write_miriad(filename, run_check=True, run_sanity_check=True,
                                clobber=clobber)
        del(miriad_obj)
