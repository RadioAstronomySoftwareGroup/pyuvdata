from astropy import constants as const
from astropy.time import Time
import os.path as op
import numpy as np
import warnings
import aipy as a
import ephem
from astropy.utils import iers
import uvdata

data_path = op.join(uvdata.__path__[0], 'data')

iers_a = iers.IERS_A.open(op.join(data_path, 'finals.all'))


class UVProperty:
    def __init__(self, required=True, value=None, spoof_val=None,
                 form=(), description='', expected_type=np.int, sane_vals=None,
                 tols=(1e-05, 1e-08)):
        self.required = required
        # cannot set a spoof_val for required properties
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description
        self.form = form
        if self.form == 'str':
            self.expected_type = str
        else:
            self.expected_type = expected_type
        self.sane_vals = sane_vals
        if np.size(tols) == 1:
            # Only one tolerance given, assume absolute, set relative to zero
            self.tols = (0, tols)
        else:
            self.tols = tols  # relative and absolute tolerances to be used in np.isclose

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # only check that value is identical
            isequal = True
            if not isinstance(self.value, other.value.__class__):
                isequal = False
            if isinstance(self.value, np.ndarray):
                if self.value.shape != other.value.shape:
                    isequal = False
                elif not np.allclose(self.value, other.value,
                                     rtol=self.tols[0], atol=self.tols[1]):
                    isequal = False
            else:
                str_type = False
                if isinstance(self.value, (str, unicode)):
                    str_type = True
                if isinstance(self.value, list):
                    if isinstance(self.value[0], str):
                        str_type = True

                if not str_type:
                    try:
                        if not np.isclose(np.array(self.value),
                                          np.array(other.value),
                                          rtol=self.tols[0], atol=self.tols[1]):
                            isequal = False
                    except:
                        print self.value, other.value
                        isequal = False
                else:
                    if self.value != other.value:
                        if not isinstance(self.value, list):
                            if self.value.replace('\n', '') != other.value.replace('\n', ''):
                                isequal = False
                        else:
                            isequal = False

            return isequal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_spoof(self, *args):
        self.value = self.spoof_val

    def expected_size(self, dataobj):
        # Takes the form of the property and returns the size
        # expected, given values in the UVData object
        if self.form == 'str':
            return self.form
        elif isinstance(self.form, np.int):
            # Fixed size, just return the form
            return self.form
        else:
            # Given by other attributes, look up values
            esize = ()
            for p in self.form:
                if isinstance(p, np.int):
                    esize = esize + (p,)
                else:
                    prop = getattr(dataobj, p)
                    if prop.value is None:
                        raise ValueError('Missing UVData property {p} needed to '
                                         'calculate expected size of property'.format(p=p))
                    esize = esize + (prop.value,)
            return esize

    def sanity_check(self):
        # A quick method for checking that values are sane
        # This needs development
        sane = False  # Default to insanity
        if self.sane_vals is None:
            sane = True
        else:
            testval = np.mean(np.abs(self.value))
            if (testval >= self.sane_vals[0]) and (testval <= self.sane_vals[1]):
                sane = True
        return sane


class AntPositionUVProperty(UVProperty):
    def apply_spoof(self, uvdata):
        self.value = np.zeros((len(uvdata.antenna_indices.value), 3))


class ExtraKeywordUVProperty(UVProperty):
    def __init__(self, required=False, value={}, spoof_val={},
                 description=''):
        self.required = required
        # cannot set a spoof_val for required properties
        if not self.required:
            self.spoof_val = spoof_val
        self.value = value
        self.description = description


class AngleUVProperty(UVProperty):
    def degrees(self):
        if self.value is None:
            return None
        else:
            return self.value * 180. / np.pi

    def set_degrees(self, degree_val):
        if degree_val is None:
            self.value = None
        else:
            self.value = degree_val * np.pi / 180.


class UVData:
    supported_file_types = ['uvfits', 'miriad', 'fhd']

    def __init__(self):
        # add the default_required_attributes to the class?
        # dimension definitions
        self.Ntimes = UVProperty(description='Number of times')
        self.Nbls = UVProperty(description='number of baselines')
        self.Nblts = UVProperty(description='Ntimes * Nbls')
        self.Nfreqs = UVProperty(description='number of frequency channels')
        self.Npols = UVProperty(description='number of polarizations')

        desc = ('array of the visibility data, size: (Nblts, Nspws, Nfreqs, '
                'Npols), type = complex float, in units of self.vis_units')
        self.data_array = UVProperty(description=desc,
                                     form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                     expected_type=np.complex)

        self.vis_units = UVProperty(description='Visibility units, options '
                                    '["uncalib","Jy","K str"]', form='str')

        desc = ('number of data points averaged into each data element, '
                'type = int, same shape as data_array')
        self.nsample_array = UVProperty(description=desc,
                                        form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                        expected_type=(np.float, np.int))

        self.flag_array = UVProperty(description='boolean flag, True is '
                                     'flagged, same shape as data_array.',
                                     form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                     expected_type=np.bool)

        self.Nspws = UVProperty(description='number of spectral windows '
                                '(ie non-contiguous spectral chunks)')

        self.spw_array = UVProperty(description='array of spectral window '
                                    'numbers', form=('Nspws',))

        desc = ('Projected baseline vectors relative to phase center, ' +
                '(3,Nblts), units meters')
        self.uvw_array = UVProperty(description=desc, form=(3, 'Nblts'),
                                    expected_type=np.float, sane_vals=(1e-3, 1e8),
                                    tols=.001)

        self.time_array = UVProperty(description='array of times, center '
                                     'of integration, dimension (Nblts), '
                                     'units Julian Date', form=('Nblts',),
                                     expected_type=np.float,
                                     tols=1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in days

        self.lst_array = UVProperty(description='array of lsts, center '
                                    'of integration, dimension (Nblts), '
                                    'units radians', form=('Nblts',),
                                    expected_type=np.float,
                                    tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in radians

        desc = ('array of first antenna indices, dimensions (Nblts), '
                'type = int, 0 indexed')
        self.ant_1_array = UVProperty(description=desc, form=('Nblts',))
        desc = ('array of second antenna indices, dimensions (Nblts), '
                'type = int, 0 indexed')
        self.ant_2_array = UVProperty(description=desc, form=('Nblts',))

        desc = ('array of baseline indices, dimensions (Nblts), '
                'type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16 '
                '(may this break casa?)')
        self.baseline_array = UVProperty(description=desc, form=('Nblts',))

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        self.freq_array = UVProperty(description='array of frequencies, '
                                     'dimensions (Nspws,Nfreqs), units Hz',
                                     form=('Nspws', 'Nfreqs'),
                                     expected_type=np.float,
                                     tols=1e-3)  # mHz

        desc = ('array of polarization integers (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)')
        self.polarization_array = UVProperty(description=desc, form=('Npols',))

        self.integration_time = UVProperty(description='length of the '
                                           'integration (s)',
                                           expected_type=np.float,
                                           tols=1e-3)  # 1 ms
        self.channel_width = UVProperty(description='width of channel (Hz)',
                                        expected_type=np.float,
                                        tols=1e-3)  # 1 mHz

        # --- observation information ---
        self.object_name = UVProperty(description='source or field '
                                      'observed (string)', form='str')
        self.telescope_name = UVProperty(description='name of telescope '
                                         '(string)', form='str')
        self.instrument = UVProperty(description='receiver or backend.', form='str')
        self.latitude = AngleUVProperty(description='latitude of telescope, '
                                        'units radians', expected_type=np.float,
                                        tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians
        self.longitude = AngleUVProperty(description='longitude of telescope, '
                                         'units degrees', expected_type=np.float,
                                         tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians
        self.altitude = UVProperty(description='altitude of telescope, '
                                   'units meters', expected_type=np.float,
                                   tols=1e-3)  # 1 mm
        self.history = UVProperty(description='string of history, units '
                                  'English', form='str')

        desc = ('epoch year of the phase applied to the data (eg 2000)')
        self.phase_center_epoch = UVProperty(description=desc, expected_type=np.float)

        # --- antenna information ----
        desc = ('number of antennas with data present. May be smaller ' +
                'than the number of antennas in the array')
        self.Nants_data = UVProperty(description=desc)
        desc = ('number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self.Nants_telescope = UVProperty(description=desc)
        desc = ('list of antenna names, dimensions (Nants_telescope), '
                'indexed by self.ant_1_array, self.ant_2_array, '
                'self.antenna_indices. There must be one '
                'entry here for each unique entry in self.ant_1_array and '
                'self.ant_2_array, but there may be extras as well.')
        self.antenna_names = UVProperty(description=desc, form=('Nants_telescope',),
                                        expected_type=str)

        desc = ('integer index into antenna_names, dimensions '
                '(Nants_telescope). There must be one '
                'entry here for each unique entry in self.ant_1_array and '
                'self.ant_2_array, but there may be extras as well.')
        self.antenna_indices = UVProperty(description=desc, form=('Nants_telescope',))

        # -------- extra, non-required properties ----------
        desc = ('any user supplied extra keywords, type=dict')
        self.extra_keywords = ExtraKeywordUVProperty(description=desc)

        self.dateobs = UVProperty(required=False,
                                  description='date of observation')

        desc = ('coordinate frame for antenna positions '
                '(eg "ITRF" -also google ECEF). NB: ECEF has x running '
                'through long=0 and z through the north pole')
        self.xyz_telescope_frame = UVProperty(required=False, description=desc,
                                              spoof_val='ITRF', form='str')

        self.x_telescope = UVProperty(required=False,
                                      description='x coordinates of array '
                                      'center in meters in coordinate frame',
                                      spoof_val=0,
                                      tols=1e-3)  # 1 mm
        self.y_telescope = UVProperty(required=False,
                                      description='y coordinates of array '
                                      'center in meters in coordinate frame',
                                      spoof_val=0,
                                      tols=1e-3)  # 1 mm
        self.z_telescope = UVProperty(required=False,
                                      description='z coordinates of array '
                                      'center in meters in coordinate frame',
                                      spoof_val=0,
                                      tols=1e-3)  # 1 mm
        desc = ('array giving coordinates of antennas relative to '
                '{x,y,z}_telescope in the same frame, (Nants_telescope, 3)')
        self.antenna_positions = AntPositionUVProperty(required=False,
                                                       description=desc,
                                                       form=('Nants_telescope', 3),
                                                       tols=1e-3)  # 1 mm

        desc = ('ra of zenith. units: radians, shape (Nblts)')
        self.zenith_ra = AngleUVProperty(required=False, description=desc,
                                         form=('Nblts',),
                                         tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('dec of zenith. units: radians, shape (Nblts)')
        # in practice, dec of zenith will never change; does not need to
        #  be shape Nblts
        self.zenith_dec = AngleUVProperty(required=False, description=desc,
                                          form=('Nblts',),
                                          tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('right ascension of phase center (see uvw_array), '
                'units radians')
        self.phase_center_ra = AngleUVProperty(required=False,
                                               description=desc,
                                               tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('declination of phase center (see uvw_array), '
                'units radians')
        self.phase_center_dec = AngleUVProperty(required=False,
                                                description=desc,
                                                tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self.GST0 = UVProperty(required=False,
                               description='Greenwich sidereal time at '
                               'midnight on reference date', spoof_val=0.0)
        self.RDate = UVProperty(required=False,
                                description='date for which the GST0 or '
                                'whatever... applies', spoof_val='')
        self.earth_omega = UVProperty(required=False,
                                      description='earth\'s rotation rate '
                                      'in degrees per day', spoof_val=360.985)
        self.DUT1 = UVProperty(required=False,
                               description='DUT1 (google it) AIPS 117 '
                               'calls it UT1UTC', spoof_val=0.0)
        self.TIMESYS = UVProperty(required=False,
                                  description='We only support UTC',
                                  spoof_val='UTC', form='str')

        desc = ('FHD thing we do not understand, something about the time '
                'at which the phase center is normal to the chosen UV plane '
                'for phasing')
        self.uvplane_reference_time = UVProperty(required=False,
                                                 description=desc,
                                                 spoof_val=0)

    def property_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        prop_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, UVProperty):
                prop_list.append(a)
        for a in prop_list:
            yield a

    def required_property_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        required_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, UVProperty):
                if attr.required:
                    required_list.append(a)
        for a in required_list:
            yield a

    def extra_property_iter(self):
        attribute_list = [a for a in dir(self) if not a.startswith('__') and
                          not callable(getattr(self, a))]
        extra_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, UVProperty):
                if not attr.required:
                    extra_list.append(a)
        for a in extra_list:
            yield a

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # only check that required properties are identical
            isequal = True
            for p in self.required_property_iter():
                self_prop = getattr(self, p)
                other_prop = getattr(other, p)
                if self_prop != other_prop:
                    print('property {pname} does not match. Left is {lval} '
                          'and right is {rval}'.
                          format(pname=p, lval=str(self_prop.value),
                                 rval=str(other_prop.value)))
                    isequal = False
            return isequal
        else:
            print('Classes do not match')
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def baseline_to_antnums(self, baseline):
        if self.Nants_telescope.value > 2048:
            raise StandardError('error Nants={Nants}>2048 not '
                                'supported'.format(Nants=self.Nants_telescope.value))
        if np.min(baseline) > 2**16:
            i = (baseline - 2**16) % 2048 - 1
            j = (baseline - 2**16 - (i + 1)) / 2048 - 1
        else:
            i = (baseline) % 256 - 1
            j = (baseline - (i + 1)) / 256 - 1
        return np.int32(i), np.int32(j)

    def antnums_to_baseline(self, i, j, attempt256=False):
        # set the attempt256 keyword to True to (try to) use the older
        # 256 standard used in many uvfits files
        # (will use 2048 standard if there are more than 256 antennas)
        i, j = np.int64((i, j))
        if self.Nants_telescope.value > 2048:
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

    def set_LatLonAlt_from_XYZ(self, overwrite=False):
        if (self.xyz_telescope_frame.value == "ITRF" and
            None not in (self.x_telescope.value,
                         self.y_telescope.value,
                         self.z_telescope.value)):
            # see wikipedia geodetic_datum and Datum transformations of
            # GPS positions PDF in docs folder
            gps_b = 6356752.31424518
            gps_a = 6378137
            e_squared = 6.69437999014e-3
            e_prime_squared = 6.73949674228e-3
            gps_p = np.sqrt(self.x_telescope.value**2 +
                            self.y_telescope.value**2)
            gps_theta = np.arctan2(self.z_telescope.value * gps_a,
                                   gps_p * gps_b)
            if self.latitude.value is None or overwrite:
                self.latitude.value = np.arctan2(self.z_telescope.value +
                                                 e_prime_squared * gps_b *
                                                 np.sin(gps_theta)**3,
                                                 gps_p - e_squared * gps_a *
                                                 np.cos(gps_theta)**3)

            if self.longitude.value is None or overwrite:
                self.longitude.value = np.arctan2(self.y_telescope.value,
                                                  self.x_telescope.value)
            gps_N = gps_a / np.sqrt(1 - e_squared *
                                    np.sin(self.latitude.value)**2)
            if self.altitude.value is None or overwrite:
                self.altitude.value = ((gps_p / np.cos(self.latitude.value)) -
                                       gps_N)
        else:
            raise ValueError('No x, y or z_telescope value assigned or '
                             'xyz_telescope_frame is not "ITRF"')

    def set_XYZ_from_LatLonAlt(self, overwrite=False):
        # check that the coordinates we need actually exist
        if None not in (self.latitude.value, self.longitude.value,
                        self.altitude.value):
            # see wikipedia geodetic_datum and Datum transformations of
            # GPS positions PDF in docs folder
            gps_b = 6356752.31424518
            gps_a = 6378137
            e_squared = 6.69437999014e-3
            e_prime_squared = 6.73949674228e-3
            gps_N = gps_a / np.sqrt(1 - e_squared *
                                    np.sin(self.latitude.value)**2)
            if self.x_telescope.value is None or overwrite:
                self.x_telescope.value = ((gps_N + self.altitude.value) *
                                          np.cos(self.latitude.value) *
                                          np.cos(self.longitude.value))
            if self.y_telescope.value is None or overwrite:
                self.y_telescope.value = ((gps_N + self.altitude.value) *
                                          np.cos(self.latitude.value) *
                                          np.sin(self.longitude.value))
            if self.z_telescope.value is None or overwrite:
                self.z_telescope.value = ((gps_b**2 / gps_a**2 * gps_N +
                                          self.altitude.value) *
                                          np.sin(self.latitude.value))
        else:
            raise ValueError('lat, lon or altitude not found')

    def set_lsts_from_time_array(self):
        lsts = []
        curtime = self.time_array.value[0]
        for ind, jd in enumerate(self.time_array.value):
            if ind == 0 or not np.isclose(jd, curtime, atol=1e-6, rtol=1e-12):
                curtime = jd
                t = Time(jd, format='jd', location=(self.longitude.degrees(),
                                                    self.latitude.degrees()))
                t.delta_ut1_utc = iers_a.ut1_utc(t)
            lsts.append(t.sidereal_time('apparent').radian)
        self.lst_array.value = np.array(lsts)
        return True

    def juldate2ephem(self, num):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
        return ephem.date(num - 2415020.)

    def phase(self, ra=None, dec=None, epoch=ephem.J2000, time=None):
        # phase drift scan data to a single ra/dec at the set epoch
        # or time in jd (i.e. ra/dec of zenith at that time in current epoch).
        # ra/dec should be in radians.
        # epoch should be an ephem date, measured from noon Dec. 31, 1899.
        # will not phase already phased data.
        if (self.phase_center_ra.value is not None or
                self.phase_center_dec.value is not None):
            raise ValueError('The data is already phased; can only phase ' +
                             'drift scanning data.')

        obs = ephem.Observer()
        # obs inits with default values for parameters -- be sure to replace them
        obs.lat = self.latitude.value
        obs.lon = self.longitude.value
        if ra is not None and dec is not None and epoch is not None and time is None:
            pass

        elif ra is None and dec is None and time is not None:
            # NB if phasing to a time, epoch does not need to be None, but it is ignored
            obs.date, obs.epoch = self.juldate2ephem(time), self.juldate2ephem(time)

            ra = self.longitude.value - obs.sidereal_time()
            dec = self.latitude.value
            epoch = time

        else:
            raise ValueError('Need to define either ra/dec/epoch or time ' +
                             '(but not both).')

        # create a pyephem object for the phasing position
        precess_pos = ephem.FixedBody()
        precess_pos._ra = ra
        precess_pos._dec = dec
        precess_pos._epoch = epoch

        # calculate RA/DEC in J2000 and write to object
        obs.date, obs.epoch = ephem.J2000, ephem.J2000
        precess_pos.compute(obs)
        self.phase_center_ra.value = precess_pos.ra
        self.phase_center_dec.value = precess_pos.dec

        for ind, jd in enumerate(self.time_array.value):
            # calculate ra/dec of phase center in current epoch
            obs.date, obs.epoch = self.juldate2ephem(jd), self.juldate2ephem(jd)
            precess_pos.compute(obs)
            ra, dec = precess_pos.ra, precess_pos.dec

            # generate rotation matrices
            m0 = a.coord.top2eq_m(self.lst_array.value[ind] - obs.sidereal_time(), self.latitude.value)
            m1 = a.coord.eq2top_m(self.lst_array.value[ind] - ra, dec)

            # rotate and write uvws
            uvw = self.uvw_array.value[:, ind]
            uvw = np.dot(m0, uvw)
            uvw = np.dot(m1, uvw)
            self.uvw_array.value[:, ind] = uvw

            # calculate data and apply phasor
            w_lambda = uvw[2] / const.c.to('m/s').value * self.freq_array.value
            phs = np.exp(-1j * 2 * np.pi * w_lambda)
            phs.shape += (1,)
            self.data_array.value[ind] *= phs

        del(obs)
        return True

    def check(self, run_sanity_check=True):
        # loop through all required properties, make sure that they are filled
        for p in self.required_property_iter():
            prop = getattr(self, p)
            # Check required property exists
            if prop.value is None:
                raise ValueError('Required UVProperty ' + p +
                                 ' has not been set.')

            # Check required property size
            esize = prop.expected_size(self)
            if esize is None:
                raise ValueError('Required UVProperty ' + p +
                                 ' expected size is not defined.')
            elif esize == 'str':
                # Check that it's a string
                if not isinstance(prop.value, str):
                    raise ValueError('UVProperty ' + p + 'expected to be '
                                     'string, but is not')
            else:
                # Check the size of the property value. Note that np.shape
                # returns an empty tuple for single numbers. esize should do the same.
                if not np.shape(prop.value) == esize:
                    raise ValueError('UVProperty ' + p + 'is not expected size.')
                if esize == ():
                    # Single element
                    if not isinstance(prop.value, prop.expected_type):
                        raise ValueError('UVProperty ' + p + ' is not the appropriate'
                                         ' type. Is: ' + str(type(prop.value)) +
                                         '. Should be: ' + str(prop.expected_type))
                else:
                    if isinstance(prop.value, list):
                        # List needs to be handled differently than array (I think)
                        if not isinstance(prop.value[0], prop.expected_type):
                            raise ValueError('UVProperty ' + p + ' is not the'
                                             ' appropriate type. Is: ' +
                                             str(type(prop.value[0])) + '. Should'
                                             ' be: ' + str(prop.expected_type))
                    else:
                        # Array
                        if not isinstance(prop.value.item(0), prop.expected_type):
                            raise ValueError('UVProperty ' + p + ' is not the appropriate'
                                             ' type. Is: ' + str(prop.value.dtype) +
                                             '. Should be: ' + str(prop.expected_type))

            if run_sanity_check:
                if not prop.sanity_check():
                    raise ValueError('UVProperty ' + p + ' has insane values.')

        return True

    def write(self, filename, spoof_nonessential=False, force_phase=False,
              run_check=True, run_sanity_check=True):
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        status = False
        # filename ending in .uvfits gets written as a uvfits
        if filename.endswith('.uvfits'):
            status = self.write_uvfits(filename, spoof_nonessential=spoof_nonessential, force_phase=force_phase)
        return status

    def read(self, filename, file_type, use_model=False, run_check=True,
             run_sanity_check=True):
        """
        General read function which calls file_type specific read functions
        Inputs:
            filename: string or list of strings
                May be a file name, directory name or a list of file names
                depending on file_type
            file_type: string
                Must be a supported type, see self.supported_file_types
        """
        if file_type not in self.supported_file_types:
            raise ValueError('file_type must be one of ' +
                             ' '.join(self.supported_file_types))
        if file_type == 'uvfits':
            status = self.read_uvfits(filename, run_check=run_check,
                                      run_sanity_check=run_sanity_check)
        elif file_type == 'miriad':
            status = self.read_miriad(filename, run_check=run_check,
                                      run_sanity_check=run_sanity_check)
        elif file_type == 'fhd':
            status = self.read_fhd(filename, use_model=use_model, run_check=run_check,
                                   run_sanity_check=run_sanity_check)
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
        return status

    def convert_from_filetype(self, other):
        for p in other.property_iter():
            prop = getattr(other, p)
            setattr(self, p, prop)

    def convert_to_filetype(self, filetype):
        if filetype is 'uvfits':
            other_obj = uvdata.uvfits.UVFITS()
        elif filetype is 'fhd':
            other_obj = uvdata.fhd.FHD()
        elif filetype is 'miriad':
            other_obj = uvdata.miriad.Miriad()
        else:
            raise ValueError('filetype must be uvfits or fhd')
        for p in self.property_iter():
            prop = getattr(self, p)
            setattr(other_obj, p, prop)
        return other_obj

    def read_uvfits(self, filename):
        uvfits_obj = uvdata.uvfits.UVFITS()
        ret_val = uvfits_obj.read_uvfits(filename)
        self.convert_from_filetype(uvfits_obj)
        return ret_val

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False):
        uvfits_obj = self.convert_to_filetype('uvfits')
        ret_val = uvfits_obj.write_uvfits(filename,
                                          spoof_nonessential=spoof_nonessential,
                                          force_phase=force_phase)
        return ret_val

    def read_fhd(self, filelist, use_model=False):
        fhd_obj = uvdata.fhd.FHD()
        ret_val = fhd_obj.read_fhd(filelist, use_model=use_model)
        self.convert_from_filetype(fhd_obj)
        return ret_val

    def read_miriad(self, filepath, FLEXIBLE_OPTION=True):
        miriad_obj = uvdata.miriad.Miriad()
        ret_val = miriad_obj.read_miriad(filepath, FLEXIBLE_OPTION=FLEXIBLE_OPTION)
        self.convert_from_filetype(miriad_obj)
        return ret_val
