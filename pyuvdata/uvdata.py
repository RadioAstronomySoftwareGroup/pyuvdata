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
import copy
import collections


class UVData(UVBase):
    """
    A class for defining a radio interferometer dataset.

    Currently supported file types: uvfits, miriad, fhd.
    Provides phasing functions.

    Attributes:
        UVParameter objects: For full list see UVData Parameters
            (http://pyuvdata.readthedocs.io/en/latest/uvdata_parameters.html).
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
                                           form=('Nblts', 'Nspws',
                                                 'Nfreqs', 'Npols'),
                                           expected_type=np.complex)

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = uvp.UVParameter('vis_units', description=desc,
                                          form='str', expected_type=str,
                                          acceptable_vals=["uncalib", "Jy", "K str"])

        desc = ('Number of data points averaged into each data element, '
                'NOT required to be an integer. type = float, same shape as data_array')
        self._nsample_array = uvp.UVParameter('nsample_array', description=desc,
                                              form=('Nblts', 'Nspws',
                                                    'Nfreqs', 'Npols'),
                                              expected_type=(np.float))

        desc = 'Boolean flag, True is flagged, same shape as data_array.'
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nblts', 'Nspws',
                                                 'Nfreqs', 'Npols'),
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
                                                   acceptable_vals=list(
                                                       np.arange(-8, 0)) + list(np.arange(1, 5)),
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
                                                         acceptable_range=(
                                                             6.35e6, 6.39e6),
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
        desc = ('Orientation of the physical dipole corresponding to what is '
                'labelled as the x polarization. Examples include "east" '
                '(indicating east/west orientation) and "north" (indicating '
                'north/south orientation)')
        self._x_orientation = uvp.UVParameter('x_orientation', description=desc,
                                              required=False, expected_type=str)

        desc = ('Any user supplied extra keywords, type=dict')
        self._extra_keywords = uvp.UVParameter('extra_keywords', required=False,
                                               description=desc, value={},
                                               spoof_val={}, expected_type=dict)

        desc = ('Array giving coordinates of antennas relative to '
                'telescope_location (ITRF frame), shape (Nants_telescope, 3), '
                'units meters. See the tutorial page in the documentation '
                'for an example of how to convert this to topocentric frame.')
        self._antenna_positions = uvp.AntPositionParameter('antenna_positions',
                                                           required=False,
                                                           description=desc,
                                                           form=(
                                                               'Nants_telescope', 3),
                                                           expected_type=np.float,
                                                           tols=1e-3)  # 1 mm

        desc = ('Array of antenna diameters in meters. Used by CASA to '
                'construct a default beam if no beam is supplied.')
        self._antenna_diameters = uvp.UVParameter('antenna_diameters',
                                                  required=False,
                                                  description=desc,
                                                  form=('Nants_telescope',),
                                                  expected_type=np.float,
                                                  tols=1e-3)  # 1 mm

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._gst0 = uvp.UVParameter('gst0', required=False,
                                     description='Greenwich sidereal time at '
                                                 'midnight on reference date',
                                     spoof_val=0.0, expected_type=np.float)
        self._rdate = uvp.UVParameter('rdate', required=False,
                                      description='Date for which the GST0 or '
                                                  'whatever... applies',
                                      spoof_val='', form='str')
        self._earth_omega = uvp.UVParameter('earth_omega', required=False,
                                            description='Earth\'s rotation rate '
                                                        'in degrees per day',
                                            spoof_val=360.985, expected_type=np.float)
        self._dut1 = uvp.UVParameter('dut1', required=False,
                                     description='DUT1 (google it) AIPS 117 '
                                                 'calls it UT1UTC',
                                     spoof_val=0.0, expected_type=np.float)
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

    def check(self, check_extra=True, run_check_acceptability=True):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Args:
            check_extra: If true, check all parameters, otherwise only check
                required parameters.
            run_check_acceptability: Option to check if values in parameters
                are acceptable. Default is True.
        """
        # first run the basic check from UVBase
        if np.all(self.ant_1_array == self.ant_2_array):
            # Special case of only containing auto correlations, adjust uvw acceptable_range
            self._uvw_array.acceptable_range = (0.0, 0.0)

        super(UVData, self).check(check_extra=check_extra,
                                  run_check_acceptability=run_check_acceptability)

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

        # issue warning if extra_keywords keys are longer than 8 characters
        for key in self.extra_keywords.keys():
            if len(key) > 8:
                warnings.warn('key {key} in extra_keywords is longer than 8 '
                              'characters. It will be truncated to 8 if written '
                              'to uvfits or miriad file formats.'.format(key=key))

        # issue warning if extra_keywords values are lists, arrays or dicts
        for key, value in self.extra_keywords.iteritems():
            if isinstance(value, (list, dict, np.ndarray)):
                warnings.warn('{key} in extra_keywords is a list, array or dict, '
                              'which will raise an error when writing uvfits or '
                              'miriad file types'.format(key=key))

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
                telescope_param = getattr(telescope_obj, p)
                self_param = getattr(self, p)
                if telescope_param.value is not None and (overwrite is True or
                                                          self_param.value is None):
                    telescope_shape = telescope_param.expected_shape(telescope_obj)
                    self_shape = self_param.expected_shape(self)
                    if telescope_shape == self_shape:
                        params_set.append(self_param.name)
                        prop_name = self_param.name
                        setattr(self, prop_name, getattr(telescope_obj, prop_name))
                    else:
                        # expected shapes aren't equal. This can happen e.g. with diameters,
                        # which is a single value on the telescope object but is
                        # an array of length Nants_telescope on the UVData object
                        if telescope_shape == () and self_shape != 'str':
                            array_val = np.zeros(self_shape,
                                                 dtype=telescope_param.expected_type) + telescope_param.value
                            params_set.append(self_param.name)
                            prop_name = self_param.name
                            setattr(self, prop_name, array_val)
                        else:
                            raise ValueError('parameter {p} on the telescope '
                                             'object does not have a compatible '
                                             'expected shape.')
            if len(params_set) > 0:
                params_set_str = ', '.join(params_set)
                warnings.warn('{params} is not set. Using known values '
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
                print('Max antnums are {} and {}'.format(
                    np.max(ant1), np.max(ant2)))
                message = 'antnums_to_baseline: found > 256 antennas, using ' \
                          '2048 baseline indexing. Beware compatibility ' \
                          'with CASA etc'
                warnings.warn(message)

        return np.int64(2048 * (ant1 + 1) + (ant2 + 1) + 2**16)

    def order_pols(self, order='AIPS'):
        '''
        Arranges polarizations in orders corresponding to either AIPS or CASA convention.
        CASA stokes types are ordered with cross-pols in between (e.g. XX,XY,YX,YY) while
        AIPS orders pols with auto-pols followed by cross-pols (e.g. XX,YY,XY,YX)
        Args:
        order: string, either 'CASA' or 'AIPS'. Default='AIPS'
        '''
        if(order == 'AIPS'):
            pol_inds = np.argsort(self.polarization_array)
            pol_inds = pol_inds[::-1]
        elif(order == 'CASA'):  # sandwich
            casa_order = np.array([1, 2, 3, 4, -1, -3, -4, -2, -5, -7, -8, -6])
            pol_inds = []
            for pol in self.polarization_array:
                pol_inds.append(np.where(casa_order == pol)[0][0])
            pol_inds = np.argsort(pol_inds)
        else:
            warnings.warn('Invalid order supplied. No sorting performed.')
            pol_inds = range(self.Npols)
        # Generate a map from original indices to new indices
        if not np.array_equal(pol_inds, self.Npols):
            self.reorder_pols(order=pol_inds)

    def set_lsts_from_time_array(self):
        """Set the lst_array based from the time_array."""
        lsts = []
        curtime = self.time_array[0]
        self.lst_array = np.zeros(self.Nblts)
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        for ind, jd in enumerate(np.unique(self.time_array)):
            t = Time(jd, format='jd', location=(longitude, latitude))
            self.lst_array[np.where(np.isclose(
                jd, self.time_array, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').radian

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
        epoch = (self.phase_center_epoch - 2000.) * 365.2422 + \
            ephem.J2000  # convert years to ephemtime
        phase_center._epoch = epoch
        phase_center._ra = self.phase_center_ra
        phase_center._dec = self.phase_center_dec

        self.zenith_ra = np.zeros_like(self.time_array)
        self.zenith_dec = np.zeros_like(self.time_array)

        # apply -w phasor
        w_lambda = (self.uvw_array[:, 2].reshape(self.Nblts, 1).astype(np.float64) /
                    const.c.to('m/s').value * self.freq_array.reshape(1, self.Nfreqs))
        phs = np.exp(-1j * 2 * np.pi * (-1) * w_lambda[:, None, :, None])
        self.data_array *= phs

        unique_times = np.unique(self.time_array)
        for jd in unique_times:
            inds = np.where(self.time_array == jd)[0]
            obs.date, obs.epoch = self.juldate2ephem(
                jd), self.juldate2ephem(jd)
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

        obs.date, obs.epoch = self.juldate2ephem(
            time), self.juldate2ephem(time)

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

        self.phase_center_ra = precess_pos.a_ra + \
            0.0  # force to be a float not ephem.Angle
        self.phase_center_dec = precess_pos.a_dec + \
            0.0  # force to be a float not ephem.Angle
        # explicitly set epoch to J2000
        self.phase_center_epoch = 2000.0

        unique_times, unique_inds = np.unique(
            self.time_array, return_index=True)
        uvws = np.zeros(self.uvw_array.shape, dtype=np.float64)
        for ind, jd in enumerate(unique_times):
            inds = np.where(self.time_array == jd)[0]
            lst = self.lst_array[unique_inds[ind]]
            # calculate ra/dec of phase center in current epoch
            obs.date, obs.epoch = self.juldate2ephem(
                jd), self.juldate2ephem(jd)
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
            uvws[inds, :] = uvw

        # calculate data and apply phasor
        w_lambda = (uvws[:, 2].reshape(self.Nblts, 1) /
                    const.c.to('m/s').value * self.freq_array.reshape(1, self.Nfreqs))
        phs = np.exp(-1j * 2 * np.pi * w_lambda[:, None, :, None])
        self.data_array *= phs

        del(obs)
        self.set_phased()

    def __add__(self, other, run_check=True, check_extra=True,
                run_check_acceptability=True, inplace=False):
        """
        Combine two UVData objects. Objects can be added along frequency,
        polarization, and/or baseline-time axis.

        Args:
            other: Another UVData object which will be added to self.
            run_check: Option to check for the existence and proper shapes of
                parameters after combining objects. Default is True.
            check_extra: Option to check optional parameters as well as
                required ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after combining objects. Default is True.
            inplace: Overwrite self as we go, otherwise create a third object
                as the sum of the two (default).
        """
        if inplace:
            this = self
        else:
            this = copy.deepcopy(self)
        # Check that both objects are UVData and valid
        this.check(check_extra=check_extra, run_check_acceptability=run_check_acceptability)
        if not isinstance(other, this.__class__):
            raise(ValueError('Only UVData objects can be added to a UVData object'))
        other.check(check_extra=check_extra, run_check_acceptability=run_check_acceptability)

        # Check objects are compatible
        # Note zenith_ra will not necessarily be the same if times are different.
        # But phase_center should be the same, even if in drift (empty parameters)
        compatibility_params = ['_vis_units', '_integration_time', '_channel_width',
                                '_object_name', '_telescope_name', '_instrument',
                                '_telescope_location', '_phase_type',
                                '_Nants_telescope', '_antenna_names',
                                '_antenna_numbers', '_antenna_positions',
                                '_phase_center_ra', '_phase_center_dec',
                                '_phase_center_epoch']
        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = 'UVParameter ' + \
                    a[1:] + ' does not match. Cannot combine objects.'
                raise(ValueError(msg))

        # Build up history string
        history_update_string = ' Combined data along '
        n_axes = 0

        # Create blt arrays for convenience
        prec_t = - 2 * \
            np.floor(np.log10(this._time_array.tols[-1])).astype(int)
        prec_b = 8
        this_blts = np.array(["_".join(["{1:.{0}f}".format(prec_t, blt[0]),
                                        str(blt[1]).zfill(prec_b)]) for blt in
                              zip(this.time_array, this.baseline_array)])
        other_blts = np.array(["_".join(["{1:.{0}f}".format(prec_t, blt[0]),
                                         str(blt[1]).zfill(prec_b)]) for blt in
                               zip(other.time_array, other.baseline_array)])
        # Check we don't have overlapping data
        both_pol = np.intersect1d(
            this.polarization_array, other.polarization_array)
        both_freq = np.intersect1d(
            this.freq_array[0, :], other.freq_array[0, :])
        both_blts = np.intersect1d(this_blts, other_blts)
        if len(both_pol) > 0:
            if len(both_freq) > 0:
                if len(both_blts) > 0:
                    raise(ValueError('These objects have overlapping data and'
                                     ' cannot be combined.'))

        temp = np.nonzero(~np.in1d(other_blts, this_blts))[0]
        if len(temp) > 0:
            bnew_inds = temp
            new_blts = other_blts[temp]
            history_update_string += 'baseline-time'
            n_axes += 1
        else:
            bnew_inds, new_blts = ([], [])

        temp = np.nonzero(
            ~np.in1d(other.freq_array[0, :], this.freq_array[0, :]))[0]
        if len(temp) > 0:
            fnew_inds = temp
            new_freqs = other.freq_array[0, temp]
            if n_axes > 0:
                history_update_string += ', frequency'
            else:
                history_update_string += 'frequency'
            n_axes += 1
        else:
            fnew_inds, new_freqs = ([], [])

        temp = np.nonzero(~np.in1d(other.polarization_array,
                                   this.polarization_array))[0]
        if len(temp) > 0:
            pnew_inds = temp
            new_pols = other.polarization_array[temp]
            if n_axes > 0:
                history_update_string += ', polarization'
            else:
                history_update_string += 'polarization'
            n_axes += 1
        else:
            pnew_inds, new_pols = ([], [])
        # Pad out self to accommodate new data
        if len(bnew_inds) > 0:
            this_blts = np.concatenate((this_blts, new_blts))
            blt_order = np.argsort(this_blts)
            zero_pad = np.zeros(
                (len(bnew_inds), this.Nspws, this.Nfreqs, this.Npols))
            this.data_array = np.concatenate([this.data_array, zero_pad], axis=0)
            this.nsample_array = np.concatenate([this.nsample_array, zero_pad], axis=0)
            this.flag_array = np.concatenate([this.flag_array,
                                              1 - zero_pad], axis=0).astype(np.bool)
            this.uvw_array = np.concatenate([this.uvw_array,
                                             other.uvw_array[bnew_inds, :]], axis=0)[blt_order, :]
            this.time_array = np.concatenate([this.time_array,
                                              other.time_array[bnew_inds]])[blt_order]
            this.lst_array = np.concatenate(
                [this.lst_array, other.lst_array[bnew_inds]])[blt_order]
            this.ant_1_array = np.concatenate([this.ant_1_array,
                                               other.ant_1_array[bnew_inds]])[blt_order]
            this.ant_2_array = np.concatenate([this.ant_2_array,
                                               other.ant_2_array[bnew_inds]])[blt_order]
            this.baseline_array = np.concatenate([this.baseline_array,
                                                  other.baseline_array[bnew_inds]])[blt_order]
            if this.phase_type == 'drift':
                this.zenith_ra = np.concatenate([this.zenith_ra,
                                                 other.zenith_ra[bnew_inds]])[blt_order]
                this.zenith_dec = np.concatenate([this.zenith_dec,
                                                 other.zenith_dec[bnew_inds]])[blt_order]
        if len(fnew_inds) > 0:
            zero_pad = np.zeros((this.data_array.shape[0], this.Nspws, len(fnew_inds),
                                 this.Npols))
            this.freq_array = np.concatenate([this.freq_array,
                                              other.freq_array[:, fnew_inds]], axis=1)
            f_order = np.argsort(this.freq_array[0, :])
            this.freq_array = this.freq_array[:, f_order]
            this.data_array = np.concatenate([this.data_array, zero_pad], axis=2)
            this.nsample_array = np.concatenate([this.nsample_array, zero_pad], axis=2)
            this.flag_array = np.concatenate([this.flag_array, 1 - zero_pad],
                                             axis=2).astype(np.bool)
        if len(pnew_inds) > 0:
            zero_pad = np.zeros((this.data_array.shape[0], this.Nspws,
                                 this.data_array.shape[2], len(pnew_inds)))
            this.polarization_array = np.concatenate([this.polarization_array,
                                                      other.polarization_array[pnew_inds]])
            p_order = np.argsort(np.abs(this.polarization_array))
            this.polarization_array = this.polarization_array[p_order]
            this.data_array = np.concatenate([this.data_array, zero_pad], axis=3)
            this.nsample_array = np.concatenate([this.nsample_array, zero_pad], axis=3)
            this.flag_array = np.concatenate([this.flag_array, 1 - zero_pad],
                                             axis=3).astype(np.bool)

        # Now populate the data
        pol_t2o = np.nonzero(
            np.in1d(this.polarization_array, other.polarization_array))[0]
        freq_t2o = np.nonzero(
            np.in1d(this.freq_array[0, :], other.freq_array[0, :]))[0]
        blt_t2o = np.nonzero(np.in1d(this_blts, other_blts))[0]
        this.data_array[np.ix_(blt_t2o, [0], freq_t2o,
                               pol_t2o)] = other.data_array
        this.nsample_array[np.ix_(
            blt_t2o, [0], freq_t2o, pol_t2o)] = other.nsample_array
        this.flag_array[np.ix_(blt_t2o, [0], freq_t2o,
                               pol_t2o)] = other.flag_array
        if len(bnew_inds) > 0:
            this.data_array = this.data_array[blt_order, :, :, :]
            this.nsample_array = this.nsample_array[blt_order, :, :, :]
            this.flag_array = this.flag_array[blt_order, :, :, :]
        if len(fnew_inds) > 0:
            this.data_array = this.data_array[:, :, f_order, :]
            this.nsample_array = this.nsample_array[:, :, f_order, :]
            this.flag_array = this.flag_array[:, :, f_order, :]
        if len(pnew_inds) > 0:
            this.data_array = this.data_array[:, :, :, p_order]
            this.nsample_array = this.nsample_array[:, :, :, p_order]
            this.flag_array = this.flag_array[:, :, :, p_order]

        # Update N parameters (e.g. Npols)
        this.Ntimes = len(np.unique(this.time_array))
        this.Nbls = len(np.unique(this.baseline_array))
        this.Nblts = this.uvw_array.shape[0]
        this.Nfreqs = this.freq_array.shape[1]
        this.Npols = this.polarization_array.shape[0]
        this.Nants_data = len(
            np.unique(this.ant_1_array.tolist() + this.ant_2_array.tolist()))

        # Check specific requirements
        if this.Nfreqs > 1:
            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                              rtol=this._freq_array.tols[0], atol=this._freq_array.tols[1]):
                warnings.warn('Combined frequencies are not evenly spaced. This will '
                              'make it impossible to write this data out to some file types.')
            elif np.max(freq_separation) > this.channel_width:
                warnings.warn('Combined frequencies are not contiguous. This will make '
                              'it impossible to write this data out to some file types.')

        if this.Npols > 2:
            pol_separation = np.diff(this.polarization_array)
            if np.min(pol_separation) < np.max(pol_separation):
                warnings.warn('Combined polarizations are not evenly spaced. This will '
                              'make it impossible to write this data out to some file types.')

        if n_axes > 0:
            history_update_string += ' axis using pyuvdata.'
            this.history += history_update_string

        other_hist_words = other.history.split(' ')
        add_hist = ''
        for i, word in enumerate(other_hist_words):
            if word not in this.history:
                add_hist += ' ' + word
                keep_going = (i + 1 < len(other_hist_words))
                while keep_going:
                    if ((other_hist_words[i + 1] is ' ') or
                            (other_hist_words[i + 1] not in this.history)):
                        add_hist += ' ' + other_hist_words[i + 1]
                        del(other_hist_words[i + 1])
                        keep_going = (i + 1 < len(other_hist_words))
                    else:
                        keep_going = False
        this.history += add_hist

        # Check final object is self-consistent
        if run_check:
            this.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

        if not inplace:
            return this

    def __iadd__(self, other):
        """
        In place add.

        Args:
            other: Another UVData object which will be added to self.
        """
        self.__add__(other, inplace=True)
        return self

    def select(self, antenna_nums=None, antenna_names=None, ant_pairs_nums=None,
               frequencies=None, freq_chans=None,
               times=None, polarizations=None, blt_inds=None, run_check=True,
               check_extra=True, run_check_acceptability=True, inplace=True):
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
                parameters after downselecting data on this object. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after downselecting data on this object. Default is True.
            inplace: Option to perform the select directly on self (True, default) or return
                a new UVData object, which is a subselection of self (False)
        """
        if inplace:
            uv_object = self
        else:
            uv_object = copy.deepcopy(self)
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
                raise ValueError(
                    'Only one of antenna_nums and antenna_names can be provided.')

            antenna_names = uvutils.get_iterable(antenna_names)
            antenna_nums = []
            for s in antenna_names:
                if s not in uv_object.antenna_names:
                    raise ValueError(
                        'Antenna name {a} is not present in the antenna_names array'.format(a=s))
                antenna_nums.append(uv_object.antenna_numbers[np.where(
                    np.array(uv_object.antenna_names) == s)[0]])

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
                if ant in uv_object.ant_1_array or ant in uv_object.ant_2_array:
                    wh1 = np.where(uv_object.ant_1_array == ant)[0]
                    wh2 = np.where(uv_object.ant_2_array == ant)[0]
                    if len(wh1) > 0:
                        inds1 = np.append(inds1, list(wh1))
                    if len(wh2) > 0:
                        inds2 = np.append(inds2, list(wh2))
                else:
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=ant))

            ant_blt_inds = np.array(
                list(set(inds1).intersection(inds2)), dtype=np.int)
            if blt_inds is not None:
                blt_inds = np.array(
                    list(set(blt_inds).intersection(ant_blt_inds)), dtype=np.int)
            else:
                blt_inds = ant_blt_inds

        if ant_pairs_nums is not None:
            if isinstance(ant_pairs_nums, tuple) and len(ant_pairs_nums) == 2:
                ant_pairs_nums = [ant_pairs_nums]
            if not all(isinstance(item, tuple) for item in ant_pairs_nums):
                raise ValueError(
                    'ant_pairs_nums must be a list of tuples of antenna numbers.')
            if not all([isinstance(item[0], (int, long, np.integer)) for item in ant_pairs_nums] +
                       [isinstance(item[1], (int, long, np.integer)) for item in ant_pairs_nums]):
                raise ValueError(
                    'ant_pairs_nums must be a list of tuples of antenna numbers.')
            if n_selects > 0:
                history_update_string += ', antenna pairs'
            else:
                history_update_string += 'antenna pairs'
            n_selects += 1
            ant_pair_blt_inds = np.zeros(0, dtype=np.int)
            for pair in ant_pairs_nums:
                if not (pair[0] in uv_object.ant_1_array or pair[0] in uv_object.ant_2_array):
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=pair[0]))
                if not (pair[1] in uv_object.ant_1_array or pair[1] in uv_object.ant_2_array):
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=pair[1]))
                wh1 = np.where(np.logical_and(
                    uv_object.ant_1_array == pair[0], uv_object.ant_2_array == pair[1]))[0]
                wh2 = np.where(np.logical_and(
                    uv_object.ant_1_array == pair[1], uv_object.ant_2_array == pair[0]))[0]
                if len(wh1) > 0:
                    ant_pair_blt_inds = np.append(ant_pair_blt_inds, list(wh1))
                if len(wh2) > 0:
                    ant_pair_blt_inds = np.append(ant_pair_blt_inds, list(wh2))
                if len(wh1) == 0 and len(wh2) == 0:
                    raise ValueError('Antenna pair {p} does not have any data '
                                     'associated with it.'.format(p=pair))

            if blt_inds is not None:
                blt_inds = np.array(
                    list(set(blt_inds).intersection(ant_pair_blt_inds)), dtype=np.int)
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
                if jd in uv_object.time_array:
                    time_blt_inds = np.append(
                        time_blt_inds, np.where(uv_object.time_array == jd)[0])
                else:
                    raise ValueError(
                        'Time {t} is not present in the time_array'.format(t=jd))

            if blt_inds is not None:
                blt_inds = np.array(
                    list(set(blt_inds).intersection(time_blt_inds)), dtype=np.int)
            else:
                blt_inds = time_blt_inds

        if blt_inds is not None:

            if len(blt_inds) == 0:
                raise ValueError(
                    'No baseline-times were found that match criteria')
            if max(blt_inds) >= uv_object.Nblts:
                raise ValueError(
                    'blt_inds contains indices that are too large')
            if min(blt_inds) < 0:
                raise ValueError('blt_inds contains indices that are negative')

            blt_inds = list(sorted(set(list(blt_inds))))
            uv_object.Nblts = len(blt_inds)
            uv_object.baseline_array = uv_object.baseline_array[blt_inds]
            uv_object.Nbls = len(np.unique(uv_object.baseline_array))
            uv_object.time_array = uv_object.time_array[blt_inds]
            uv_object.lst_array = uv_object.lst_array[blt_inds]
            uv_object.data_array = uv_object.data_array[blt_inds, :, :, :]
            uv_object.flag_array = uv_object.flag_array[blt_inds, :, :, :]
            uv_object.nsample_array = uv_object.nsample_array[blt_inds, :, :, :]
            uv_object.uvw_array = uv_object.uvw_array[blt_inds, :]

            uv_object.ant_1_array = uv_object.ant_1_array[blt_inds]
            uv_object.ant_2_array = uv_object.ant_2_array[blt_inds]
            uv_object.Nants_data = int(
                len(set(uv_object.ant_1_array.tolist() + uv_object.ant_2_array.tolist())))

            uv_object.Ntimes = len(np.unique(uv_object.time_array))

            if uv_object.phase_type == 'drift':
                uv_object.zenith_ra = uv_object.zenith_ra[blt_inds]
                uv_object.zenith_dec = uv_object.zenith_dec[blt_inds]

        if freq_chans is not None:
            freq_chans = uvutils.get_iterable(freq_chans)
            if frequencies is None:
                frequencies = uv_object.freq_array[0, freq_chans]
            else:
                frequencies = uvutils.get_iterable(frequencies)
                frequencies = np.sort(list(set(frequencies) |
                                           set(uv_object.freq_array[0, freq_chans])))

        if frequencies is not None:
            frequencies = uvutils.get_iterable(frequencies)
            if n_selects > 0:
                history_update_string += ', frequencies'
            else:
                history_update_string += 'frequencies'
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int)
            # this works because we only allow one SPW. This will have to be reworked when we support more.
            freq_arr_use = uv_object.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(
                        freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        'Frequency {f} is not present in the freq_array'.format(f=f))

            if len(frequencies) > 1:
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
            uv_object.Nfreqs = len(freq_inds)
            uv_object.freq_array = uv_object.freq_array[:, freq_inds]
            uv_object.data_array = uv_object.data_array[:, :, freq_inds, :]
            uv_object.flag_array = uv_object.flag_array[:, :, freq_inds, :]
            uv_object.nsample_array = uv_object.nsample_array[:,
                                                              :, freq_inds, :]

        if polarizations is not None:
            polarizations = uvutils.get_iterable(polarizations)
            if n_selects > 0:
                history_update_string += ', polarizations'
            else:
                history_update_string += 'polarizations'
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int)
            for p in polarizations:
                if p in uv_object.polarization_array:
                    pol_inds = np.append(pol_inds, np.where(
                        uv_object.polarization_array == p)[0])
                else:
                    raise ValueError(
                        'Polarization {p} is not present in the polarization_array'.format(p=p))

            if len(pol_inds) > 2:
                pol_ind_separation = pol_inds[1:] - pol_inds[:-1]
                if np.min(pol_ind_separation) < np.max(pol_ind_separation):
                    warnings.warn('Selected polarization values are not evenly spaced. This '
                                  'will make it impossible to write this data out to '
                                  'some file types')

            pol_inds = list(sorted(set(list(pol_inds))))
            uv_object.Npols = len(pol_inds)
            uv_object.polarization_array = uv_object.polarization_array[pol_inds]
            uv_object.data_array = uv_object.data_array[:, :, :, pol_inds]
            uv_object.flag_array = uv_object.flag_array[:, :, :, pol_inds]
            uv_object.nsample_array = uv_object.nsample_array[:,
                                                              :, :, pol_inds]

        history_update_string += ' using pyuvdata.'
        uv_object.history = uv_object.history + history_update_string

        # check if object is uv_object-consistent
        if run_check:
            uv_object.check(check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)

        if not inplace:
            return uv_object

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

    def read_uvfits(self, filename, run_check=True, check_extra=True,
                    run_check_acceptability=True):
        """
        Read in data from a uvfits file.

        Args:
            filename: The uvfits file or list of files to read from.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        import uvfits
        if isinstance(filename, (list, tuple)):
            self.read_uvfits(filename[0], run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
            if len(filename) > 1:
                for f in filename[1:]:
                    uv2 = UVData()
                    uv2.read_uvfits(f, run_check=run_check, check_extra=check_extra,
                                    run_check_acceptability=run_check_acceptability)
                    self += uv2
                del(uv2)
        else:
            uvfits_obj = uvfits.UVFITS()
            uvfits_obj.read_uvfits(filename, run_check=run_check, check_extra=check_extra,
                                   run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(uvfits_obj)
            del(uvfits_obj)

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False, run_check=True, check_extra=True,
                     run_check_acceptability=True):
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
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
        """
        uvfits_obj = self._convert_to_filetype('uvfits')
        uvfits_obj.write_uvfits(filename, spoof_nonessential=spoof_nonessential,
                                force_phase=force_phase, run_check=run_check,
                                check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability)
        del(uvfits_obj)

    def read_ms(self, filepath, run_check=True, check_extra=True,
                run_check_acceptability=True, data_column='DATA', pol_order='AIPS'):
        """
        Read in data from a measurement set

        Args:
            filepath: The measurement set file directory or list of directories to read from.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check the values of parameters
                after reading in the file. Default is True.
            data_column: name of CASA data column to read into data_array.
                'DATA', 'MODEL', or 'CORRECTED_DATA'
            pol_order: specify whether you want polarizations ordered by
                'CASA' or 'AIPS' conventions.
        """

        # check if casacore is installed
        try:
            import casacore
        except(ImportError):
            # only import skip if importerror raised
            # this way, nose import errors aren't raised
            # unless this errors is already encountered.
            from nose.plugins.skip import SkipTest
            raise SkipTest('casacore not detected. casacore'
                           ' must be installed for measurement set functionality')
        import ms

        if isinstance(filepath, (list, tuple)):
            self.read_ms(filepath[0], run_check=run_check, check_extra=check_extra,
                         run_check_acceptability=run_check_acceptability,
                         data_column=data_column, pol_order=pol_order)
            if len(filepath) > 1:
                for f in filepath[1:]:
                    uv2 = UVData()
                    uv2.read_ms(f, run_check=run_check, check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability,
                                data_column=data_column, pol_order=pol_order)
                    self += uv2
                del(uv2)
        else:
            ms_obj = ms.MS()
            ms_obj.read_ms(filepath, run_check=run_check, check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability,
                           data_column=data_column, pol_order=pol_order)
            self._convert_from_filetype(ms_obj)
            del(ms_obj)

    def read_fhd(self, filelist, use_model=False, run_check=True, check_extra=True,
                 run_check_acceptability=True):
        """
        Read in data from a list of FHD files.

        Args:
            filelist: The list of FHD save files to read from. Must include at
                least one polarization file, a params file and a flag file.
                Can also be a list of lists to read multiple data sets.
            use_model: Option to read in the model visibilities rather than the
                dirty visibilities. Default is False.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        import fhd
        if isinstance(filelist[0], (list, tuple)):
            self.read_fhd(filelist[0], use_model=use_model, run_check=run_check,
                          check_extra=check_extra,
                          run_check_acceptability=run_check_acceptability)
            if len(filelist) > 1:
                for f in filelist[1:]:
                    uv2 = UVData()
                    uv2.read_fhd(f, use_model=use_model, run_check=run_check,
                                 check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)
                    self += uv2
                del(uv2)
        else:
            fhd_obj = fhd.FHD()
            fhd_obj.read_fhd(filelist, use_model=use_model, run_check=run_check,
                             check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(fhd_obj)
            del(fhd_obj)

    def read_miriad(self, filepath, correct_lat_lon=True, run_check=True,
                    check_extra=True,
                    run_check_acceptability=True, phase_type=None):
        """
        Read in data from a miriad file.

        Args:
            filepath: The miriad file directory or list of directories to read from.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        import miriad
        if isinstance(filepath, (list, tuple)):
            self.read_miriad(filepath[0], correct_lat_lon=correct_lat_lon,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability,
                             phase_type=phase_type)
            if len(filepath) > 1:
                for f in filepath[1:]:
                    uv2 = UVData()
                    uv2.read_miriad(f, correct_lat_lon=correct_lat_lon,
                                    run_check=run_check, check_extra=check_extra,
                                    run_check_acceptability=run_check_acceptability,
                                    phase_type=phase_type)
                    self += uv2
                del(uv2)
        else:
            miriad_obj = miriad.Miriad()
            miriad_obj.read_miriad(filepath, correct_lat_lon=correct_lat_lon,
                                   run_check=run_check, check_extra=check_extra,
                                   run_check_acceptability=run_check_acceptability,
                                   phase_type=phase_type)
            self._convert_from_filetype(miriad_obj)
            del(miriad_obj)

    def write_miriad(self, filepath, run_check=True, check_extra=True,
                     run_check_acceptability=True, clobber=False, no_antnums=False):
        """
        Write the data to a miriad file.

        Args:
            filename: The miriad file directory to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
            no_antnums: Option to not write the antnums variable to the file.
                Should only be used for testing purposes.
        """
        miriad_obj = self._convert_to_filetype('miriad')
        miriad_obj.write_miriad(filepath, run_check=run_check, check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability,
                                clobber=clobber, no_antnums=no_antnums)
        del(miriad_obj)

    def reorder_pols(self, order=None, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Rearrange polarizations in the event they are not uvfits compatible.

        Args:
            order: Provide the order which to shuffle the data. Default will
                sort by absolute value of pol values.
            run_check: Option to check for the existence and proper shapes of
                parameters after reordering. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reordering. Default is True.
        """
        if order is None:
            order = np.argsort(np.abs(self.polarization_array))
        self.polarization_array = self.polarization_array[order]
        self.data_array = self.data_array[:, :, :, order]
        self.nsample_array = self.nsample_array[:, :, :, order]
        self.flag_array = self.flag_array[:, :, :, order]

        # check if object is self-consistent
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

    def get_ants(self):
        """
        Returns numpy array of unique antennas in the data.
        """
        return np.unique(np.append(self.ant_1_array, self.ant_2_array))

    def get_baseline_nums(self):
        """
        Returns numpy array of unique baseline numbers in data.
        """
        return np.unique(self.baseline_array)

    def get_antpairs(self):
        """
        Returns list of unique antpair tuples (ant1, ant2) in data.
        """
        return [self.baseline_to_antnums(bl) for bl in self.get_baseline_nums()]

    def get_pols(self):
        """
        Returns list of pols in string format.
        """
        return uvutils.polnum2str(self.polarization_array)

    def get_antpairpols(self):
        """
        Returns list of unique antpair + pol tuples (ant1, ant2, pol) in data.
        """
        bli = 0
        pols = self.get_pols()
        bls = self.get_antpairs()
        return [(bl) + (pol,) for bl in bls for pol in pols]

    def get_feedpols(self):
        """
        Returns list of unique antenna feed polarizations (e.g. ['X', 'Y']) in data.
        NOTE: Will return ValueError if any stokes visibilities are present.
        """
        if np.any(self.polarization_array > 0):
            raise ValueError('Stokes visibilities cannot be interpreted as feed polarizations')
        else:
            return list(set(''.join(self.get_pols())))

    def antpair2ind(self, ant1, ant2):
        """
        Get blt indices for given (ordered) antenna pair.
        """
        return np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]

    def _key2inds(self, key):
        """
        Interpret user specified key as a combination of antenna pair and/or polarization.

        Args:
            key: Identifier of data. Key can be 1, 2, or 3 numbers:
                if len(key) == 1:
                    if (key < 5) or (type(key) is str):  interpreted as a
                                 polarization number/name, return all blts for that pol.
                    else: interpreted as a baseline number. Return all times and
                          polarizations for that baseline.
                if len(key) == 2: interpreted as an antenna pair. Return all
                    times and pols for that baseline.
                if len(key) == 3: interpreted as antenna pair and pol (ant1, ant2, pol).
                    Return all times for that baseline, pol. pol may be a string.

        Returns:
            blt_ind1: numpy array with blt indices for antenna pair.
            blt_ind2: numpy array with blt indices for conjugate antenna pair.
            pol_ind: numpy array with polarization indices
        """
        key = uvutils.get_iterable(key)
        if type(key) is str:
            # Single string given, assume it is polarization
            pol_ind = np.where(self.polarization_array == uvutils.polstr2num(key))[0]
            if len(pol_ind) == 0:
                raise KeyError('Polarization {pol} not found in data.'.format(pol=key))
            blt_ind1 = np.arange(self.Nblts)
            blt_ind2 = np.array([], dtype=np.int64)
        elif len(key) == 1:
            key = key[0]  # For simplicity
            if isinstance(key, collections.Iterable):
                # Nested tuple. Call function again.
                blt_ind1, blt_ind2, pol_ind = self._key2inds(key)
            elif key < 5:
                # Small number, assume it is a polarization number a la AIPS memo
                pol_ind = np.where(self.polarization_array == key)[0]
                if len(pol_ind) == 0:
                    raise KeyError('Polarization {pol} not found in data.'.format(pol=key))
                blt_ind1 = np.arange(self.Nblts)
                blt_ind2 = np.array([], dtype=np.int64)
            else:
                # Larger number, assume it is a baseline number
                inv_bl = self.antnums_to_baseline(self.baseline_to_antnums(key)[1],
                                                  self.baseline_to_antnums(key)[0])
                blt_ind1 = np.where(self.baseline_array == key)[0]
                blt_ind2 = np.where(self.baseline_array == inv_bl)[0]
                if len(blt_ind1) + len(blt_ind2) == 0:
                    raise KeyError('Baseline {bl} not found in data.'.format(bl=key))
                pol_ind = np.arange(self.Npols)
        elif len(key) == 2:
            # Key is an antenna pair
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError('Antenna pair {pair} not found in data'.format(pair=key))
            pol_ind = np.arange(self.Npols)
        elif len(key) == 3:
            # Key is an antenna pair + pol
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError('Antenna pair {pair} not found in '
                               'data'.format(pair=(key[0], key[1])))
            if type(key[2]) is str:
                # pol is str
                pol_ind = np.where(self.polarization_array == uvutils.polstr2num(key[2]))[0]
            else:
                # polarization number a la AIPS memo
                pol_ind = np.where(self.polarization_array == key[2])[0]
            if len(pol_ind) == 0:
                raise KeyError('Polarization {pol} not found in data.'.format(pol=key[2]))
        # Catch autos
        if np.array_equal(blt_ind1, blt_ind2):
            blt_ind2 = np.array([])
        return (blt_ind1, blt_ind2, pol_ind)

    def _smart_slicing(self, data, ind1, ind2, indp, **kwargs):
        """
        Method for quickly picking out the relevant section of data for get_data or get_flags

        Args:
            data: 4-dimensional array in the format of self.data_array
            ind1: list with blt indices for antenna pair (e.g. from self._key2inds)
            ind2: list with blt indices for conjugate antenna pair. (e.g. from self._key2inds)
            indp: list with polarization indices (e.g. from self._key2inds)
        kwargs:
            squeeze: 'default': squeeze pol and spw dimensions if possible (default)
                     'none': no squeezing of resulting numpy array
                     'full': squeeze all length 1 dimensions
            force_copy: Option to explicitly make a copy of the data. Default is False.

        Returns:
            out: numpy array copy (or if possible, a read-only view) of relevant section of data
        """
        force_copy = kwargs.pop('force_copy', False)
        squeeze = kwargs.pop('squeeze', 'default')

        if len(set(np.ediff1d(indp))) <= 1:
            p_reg_spaced = True
            p_start = indp[0]
            p_stop = indp[-1] + 1
            if len(indp) == 1:
                dp = 1
            else:
                dp = indp[1] - indp[0]
        else:
            p_reg_spaced = False

        if len(ind2) == 0:
            # only unconjugated baselines
            if len(set(np.ediff1d(ind1))) <= 1:
                blt_start = ind1[0]
                blt_stop = ind1[-1] + 1
                if len(ind1) == 1:
                    dblt = 1
                else:
                    dblt = ind1[1] - ind1[0]
                if p_reg_spaced:
                    out = data[blt_start:blt_stop:dblt, :, :, p_start:p_stop:dp]
                else:
                    out = data[blt_start:blt_stop:dblt, :, :, indp]
            else:
                out = data[ind1, :, :, :]
                if p_reg_spaced:
                    out = out[:, :, :, p_start:p_stop:dp]
                else:
                    out = out[:, :, :, indp]
        elif len(ind1) == 0:
            # only conjugated baselines
            if len(set(np.ediff1d(ind2))) <= 1:
                blt_start = ind2[0]
                blt_stop = ind2[-1] + 1
                if len(ind2) == 1:
                    dblt = 1
                else:
                    dblt = ind2[1] - ind2[0]
                if p_reg_spaced:
                    out = np.conj(data[blt_start:blt_stop:dblt, :, :, p_start:p_stop:dp])
                else:
                    out = np.conj(data[blt_start:blt_stop:dblt, :, :, indp])
            else:
                out = data[ind2, :, :, :]
                if p_reg_spaced:
                    out = np.conj(out[:, :, :, p_start:p_stop:dp])
                else:
                    out = np.conj(out[:, :, :, indp])
        else:
            # both conjugated and unconjugated baselines
            out = np.append(data[ind1, :, :, :], np.conj(data[ind2, :, :, :]), axis=0)
            if p_reg_spaced:
                out = out[:, :, :, p_start:p_stop:dp]
            else:
                out = out[:, :, :, indp]

        if squeeze is 'full':
            out = np.squeeze(out)
        elif squeeze is 'default':
            if out.shape[3] is 1:
                # one polarization dimension
                out = np.squeeze(out, axis=3)
            if out.shape[1] is 1:
                # one spw dimension
                out = np.squeeze(out, axis=1)
        elif squeeze is not 'none':
            raise ValueError('"' + str(squeeze) + '" is not a valid option for squeeze.'
                             'Only "default", "none", or "full" are allowed.')

        if force_copy:
            out = np.array(out)
        elif out.base is not None:
            # if out is a view rather than a copy, make it read-only
            out.flags.writeable = False

        return out

    def get_data(self, *args, **kwargs):
        """
        Function for quick access to numpy array with data corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.

        Returns:
            Numpy array of data corresponding to key.
            If data exists conjugate to requested antenna pair, it will be conjugated
            before returning.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.data_array, ind1, ind2, indp, **kwargs)
        return out

    def get_flags(self, *args, **kwargs):
        """
        Function for quick access to numpy array with flags corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.

        Returns:
            Numpy array of flags corresponding to key.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.flag_array, ind1, ind2, indp, **kwargs)
        return out

    def get_nsamples(self, *args, **kwargs):
        """
        Function for quick access to numpy array with nsamples corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.

        Returns:
            Numpy array of nsamples corresponding to key.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.nsample_array, ind1, ind2, indp, **kwargs)
        return out

    def get_times(self, *args):
        """
        Find the time_array entries for a given antpair or baseline number.
        Meant to be used in conjunction with get_data function.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.

        Returns:
            Numpy array of times corresonding to key.
        """
        ind1, ind2, indp = self._key2inds(args)
        return np.append(self.time_array[ind1], self.time_array[ind2])

    def antpairpol_iter(self, squeeze='default'):
        """
        Generates numpy arrays of data for each antpair, pol combination.

        Args:
            squeeze: 'default': squeeze pol and spw dimensions if possible
                     'none': no squeezing of resulting numpy array
                     'full': squeeze all length 1 dimensions

        Returns (for each iteration):
            key: tuple with antenna1, antenna2, and polarization string
            data: Numpy array with data which is the result of self[key]
        """
        antpairpols = self.get_antpairpols()
        for key in antpairpols:
            yield (key, self.get_data(key, squeeze=squeeze))
