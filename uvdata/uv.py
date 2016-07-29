from astropy import constants as const
from astropy.time import Time
import os.path as op
import os
import numpy as np
import warnings
import aipy as a
import ephem
from astropy.utils import iers
from uvdata.uvbase import UVBase
import uvdata.parameter as uvp
import uvdata.utils as utils
import uvdata

data_path = op.join(uvdata.__path__[0], 'data')

iers_a = iers.IERS_A.open(op.join(data_path, 'finals.all'))


def _warning(message, category=UserWarning, filename='', lineno=-1):
    print(message)


class UVData(UVBase):
    supported_read_file_types = ['uvfits', 'miriad', 'fhd']
    supported_write_file_types = ['uvfits', 'miriad', 'fhd']

    def __init__(self):
        # add the UVParameters to the class
        self._Ntimes = uvp.UVParameter('Ntimes', description='Number of times')
        self._Nbls = uvp.UVParameter('Nbls', description='number of baselines')
        self._Nblts = uvp.UVParameter('Nblts', description='Ntimes * Nbls')
        self._Nfreqs = uvp.UVParameter('Nfreqs', description='number of frequency channels')
        self._Npols = uvp.UVParameter('Npols', description='number of polarizations')

        desc = ('array of the visibility data, size: (Nblts, Nspws, Nfreqs, '
                'Npols), type = complex float, in units of self.vis_units')
        self._data_array = uvp.UVParameter('data_array', description=desc,
                                           form=('Nblts', 'Nspws', 'Nfreqs', 'Npols'),
                                           expected_type=np.complex)

        self._vis_units = uvp.UVParameter('vis_units',
                                          description='Visibility units, options '
                                                      '["uncalib","Jy","K str"]',
                                          form='str')

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

        self._spw_array = uvp.UVParameter('spw_array', description='array of spectral window '
                                          'numbers', form=('Nspws',))

        desc = ('Projected baseline vectors relative to phase center, ' +
                '(3,Nblts), units meters')
        self._uvw_array = uvp.UVParameter('uvw_array', description=desc,
                                          form=(3, 'Nblts'), expected_type=np.float,
                                          sane_vals=(1e-3, 1e8), tols=.001)

        desc = ('array of times, center of integration, dimension (Nblts), ' +
                'units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Nblts',), expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in days

        desc = ('array of lsts, center of integration, dimension (Nblts), ' +
                'units radians')
        self._lst_array = uvp.UVParameter('lst_array', description=desc, form=('Nblts',),
                                          expected_type=np.float,
                                          tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in radians

        desc = ('array of first antenna indices, dimensions (Nblts), '
                'type = int, 0 indexed')
        self._ant_1_array = uvp.UVParameter('ant_1_array', description=desc, form=('Nblts',))
        desc = ('array of second antenna indices, dimensions (Nblts), '
                'type = int, 0 indexed')
        self._ant_2_array = uvp.UVParameter('ant_2_array', description=desc, form=('Nblts',))

        desc = ('array of baseline indices, dimensions (Nblts), '
                'type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16 '
                '(may this break casa?)')
        self._baseline_array = uvp.UVParameter('baseline_array', description=desc, form=('Nblts',))

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        desc = 'array of frequencies, dimensions (Nspws,Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        desc = ('array of polarization integers (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)')
        self._polarization_array = uvp.UVParameter('polarization_array',
                                                   description=desc, form=('Npols',))

        self._integration_time = uvp.UVParameter('integration_time',
                                                 description='length of the integration (s)',
                                                 expected_type=np.float, tols=1e-3)  # 1 ms
        self._channel_width = uvp.UVParameter('channel_width',
                                              description='width of channel (Hz)',
                                              expected_type=np.float,
                                              tols=1e-3)  # 1 mHz

        # --- observation information ---
        self._object_name = uvp.UVParameter('object_name', description='source or field '
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

        desc = ('epoch year of the phase applied to the data (eg 2000.)')
        self._phase_center_epoch = uvp.UVParameter('phase_center_epoch', description=desc,
                                                   expected_type=np.float)

        self._is_phased = uvp.UVParameter('is_phased', required=True,
                                          expected_type=bool,
                                          description='true/false whether data is '
                                                      'phased (true) or drift scanning (false)')

        # --- antenna information ----
        desc = ('number of antennas with data present. May be smaller ' +
                'than the number of antennas in the array')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc)
        desc = ('number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope', description=desc)
        desc = ('list of antenna names, dimensions (Nants_telescope), '
                'with numbers given by antenna_numbers (which can be matched '
                'to ant_1_array and ant_2_array). There must be one entry '
                'here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_names = uvp.UVParameter('antenna_names', description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('integer antenna number corresponding to antenna_names, '
                'dimensions (Nants_telescope). There must be one '
                'entry here for each unique entry in self.ant_1_array and '
                'self.ant_2_array, but there may be extras as well.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers', description=desc,
                                                form=('Nants_telescope',))

        # -------- extra, non-required parameters ----------
        desc = ('any user supplied extra keywords, type=dict')
        self._extra_keywords = uvp.ExtraKeywordParameter('extra_keywords',
                                                         description=desc)

        self._dateobs = uvp.UVParameter('dateobs', required=False,
                                        description='date of observation')

        desc = ('array giving coordinates of antennas relative to '
                'telescope_location (ITRF frame), (Nants_telescope, 3)')
        self._antenna_positions = uvp.AntPositionParameter('antenna_positions',
                                                           required=False,
                                                           description=desc,
                                                           form=('Nants_telescope', 3),
                                                           tols=1e-3)  # 1 mm

        desc = ('ra of zenith. units: radians, shape (Nblts)')
        self._zenith_ra = uvp.AngleParameter('zenith_ra', required=False,
                                             description=desc,
                                             form=('Nblts',),
                                             tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('dec of zenith. units: radians, shape (Nblts)')
        # in practice, dec of zenith will never change; does not need to be shape Nblts
        self._zenith_dec = uvp.AngleParameter('zenith_dec', required=False,
                                              description=desc,
                                              form=('Nblts',),
                                              tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('right ascension of phase center (see uvw_array), '
                'units radians')
        self._phase_center_ra = uvp.AngleParameter('phase_center_ra', required=False,
                                                   description=desc,
                                                   tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        desc = ('declination of phase center (see uvw_array), '
                'units radians')
        self._phase_center_dec = uvp.AngleParameter('phase_center_dec', required=False,
                                                    description=desc,
                                                    tols=2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0))  # 1 mas in radians

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._GST0 = uvp.UVParameter('GST0', required=False,
                                     description='Greenwich sidereal time at '
                                                 'midnight on reference date',
                                     spoof_val=0.0)
        self._RDate = uvp.UVParameter('RDate', required=False,
                                      description='date for which the GST0 or '
                                                  'whatever... applies',
                                      spoof_val='')
        self._earth_omega = uvp.UVParameter('earth_omega', required=False,
                                            description='earth\'s rotation rate '
                                                        'in degrees per day',
                                            spoof_val=360.985)
        self._DUT1 = uvp.UVParameter('DUT1', required=False,
                                     description='DUT1 (google it) AIPS 117 '
                                                 'calls it UT1UTC',
                                     spoof_val=0.0)
        self._TIMESYS = uvp.UVParameter('TIMESYS', required=False,
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
        # warnings.showwarning = _warning

    def known_telescopes(self):
        return uvdata.telescopes.known_telescopes()

    def set_telescope_params(self, overwrite=False):
        telescope_obj = uvdata.telescopes.get_telescope(self.telescope_name)
        if telescope_obj is not False:
            params_set = []
            for p in telescope_obj.parameter_iter():
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
        # set the attempt256 keyword to True to (try to) use the older
        # 256 standard used in many uvfits files
        # (will use 2048 standard if there are more than 256 antennas)
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
        lsts = []
        curtime = self.time_array[0]
        for ind, jd in enumerate(self.time_array):
            if ind == 0 or not np.isclose(jd, curtime, atol=1e-6, rtol=1e-12):
                curtime = jd
                latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
                t = Time(jd, format='jd', location=(longitude, latitude))
                t.delta_ut1_utc = iers_a.ut1_utc(t)
            lsts.append(t.sidereal_time('apparent').radian)
        self.lst_array = np.array(lsts)
        return True

    def juldate2ephem(self, num):
        """Convert Julian date to ephem date, measured from noon, Dec. 31, 1899."""
        return ephem.date(num - 2415020.)

    def unphase_to_drift(self):
        if not self.is_phased:
            raise ValueError('The data is already drift scanning; can only ' +
                             'unphase phased data.')

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
            w_lambda = self.uvw_array[2, ind] / const.c.to('m/s').value * self.freq_array
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
            uvw = self.uvw_array[:, ind]
            uvw = np.dot(m0, uvw)
            uvw = np.dot(m1, uvw)
            self.uvw_array[:, ind] = uvw

        # remove phase center
        self.phase_center_ra = None
        self.phase_center_dec = None
        self.is_phased = False
        return True

    def phase(self, ra=None, dec=None, epoch=ephem.J2000, time=None):
        # phase drift scan data to a single ra/dec at the set epoch
        # or time in jd (i.e. ra/dec of zenith at that time in current epoch).
        # ra/dec should be in radians.
        # epoch should be an ephem date, measured from noon Dec. 31, 1899.
        # will not phase already phased data.
        if self.is_phased:
            raise ValueError('The data is already phased; can only phase ' +
                             'drift scanning data.')

        obs = ephem.Observer()
        # obs inits with default values for parameters -- be sure to replace them
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt
        obs.lat = latitude
        obs.lon = longitude
        if ra is not None and dec is not None and epoch is not None and time is None:
            pass

        elif ra is None and dec is None and time is not None:
            # NB if phasing to a time, epoch does not need to be None, but it is ignored
            obs.date, obs.epoch = self.juldate2ephem(time), self.juldate2ephem(time)

            ra = obs.sidereal_time()
            dec = latitude
            epoch = self.juldate2ephem(time)

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

        self.phase_center_ra = precess_pos.a_ra
        self.phase_center_dec = precess_pos.a_dec
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
            uvw = self.uvw_array[:, ind]
            uvw = np.dot(m0, uvw)
            uvw = np.dot(m1, uvw)
            self.uvw_array[:, ind] = uvw

            # calculate data and apply phasor
            w_lambda = uvw[2] / const.c.to('m/s').value * self.freq_array
            phs = np.exp(-1j * 2 * np.pi * w_lambda)
            phs.shape += (1,)
            self.data_array[ind] *= phs

        del(obs)
        self.is_phased = True
        return True

    def write(self, filename, file_type, spoof_nonessential=False, force_phase=False,
              run_check=True, run_sanity_check=True, clobber=False):
        if run_check:
            self.check(run_sanity_check=run_sanity_check)

        status = False
        if file_type not in self.supported_write_file_types:
            raise ValueError('file_type must be one of ' +
                             ' '.join(self.supported_write_file_types))

        file_path = op.dirname(filename)
        if not op.exists(file_path):
            os.mkdir(file_path)

        if file_type == 'uvfits':
            status = self.write_uvfits(filename,
                                       spoof_nonessential=spoof_nonessential,
                                       force_phase=force_phase, run_check=False)
        elif file_type == 'miriad':
            status = self.write_miriad(filename, run_check=False,
                                       clobber=clobber)
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
        if file_type not in self.supported_read_file_types:
            raise ValueError('file_type must be one of ' +
                             ' '.join(self.supported_read_file_types))
        if file_type == 'uvfits':
            # Note we will run check later, not in specific read functions.
            status = self.read_uvfits(filename, run_check=run_check,
                                      run_sanity_check=run_sanity_check)
        elif file_type == 'miriad':
            status = self.read_miriad(filename, run_check=run_check,
                                      run_sanity_check=run_sanity_check)
        elif file_type == 'fhd':
            status = self.read_fhd(filename, use_model=use_model, run_check=run_check,
                                   run_sanity_check=run_sanity_check)
        return status

    def convert_from_filetype(self, other):
        for p in other.parameter_iter():
            param = getattr(other, p)
            setattr(self, p, param)

    def convert_to_filetype(self, filetype):
        if filetype is 'uvfits':
            other_obj = uvdata.uvfits.UVFITS()
        elif filetype is 'fhd':
            other_obj = uvdata.fhd.FHD()
        elif filetype is 'miriad':
            other_obj = uvdata.miriad.Miriad()
        else:
            raise ValueError('filetype must be uvfits, miriad, or fhd')
        for p in self.parameter_iter():
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_uvfits(self, filename, run_check=True, run_sanity_check=True):
        uvfits_obj = uvdata.uvfits.UVFITS()
        ret_val = uvfits_obj.read_uvfits(filename, run_check=True, run_sanity_check=True)
        self.convert_from_filetype(uvfits_obj)
        return ret_val

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False, run_check=True, run_sanity_check=True):
        uvfits_obj = self.convert_to_filetype('uvfits')
        ret_val = uvfits_obj.write_uvfits(filename,
                                          spoof_nonessential=spoof_nonessential,
                                          force_phase=force_phase,
                                          run_check=True, run_sanity_check=True)
        return ret_val

    def read_fhd(self, filelist, use_model=False, run_check=True,
                 run_sanity_check=True):
        fhd_obj = uvdata.fhd.FHD()
        ret_val = fhd_obj.read_fhd(filelist, use_model=use_model,
                                   run_check=True, run_sanity_check=True)
        self.convert_from_filetype(fhd_obj)
        return ret_val

    def read_miriad(self, filepath, run_check=True, run_sanity_check=True):
        miriad_obj = uvdata.miriad.Miriad()
        ret_val = miriad_obj.read_miriad(filepath, run_check=True,
                                         run_sanity_check=True)
        self.convert_from_filetype(miriad_obj)
        return ret_val

    def write_miriad(self, filename, run_check=True, run_sanity_check=True,
                     clobber=False):
        miriad_obj = self.convert_to_filetype('miriad')
        ret_val = miriad_obj.write_miriad(filename,
                                          run_check=True, run_sanity_check=True,
                                          clobber=clobber)
        return ret_val
