import numpy as np
import warnings
from uvbase import UVBase
import parameter as uvp
import utils as uvutils
import version as uvversion


class UVCal(UVBase):
    """ A class defining calibration solutions

        Currently supported file types: calfits

        Attributes:
            UVParameter objects: For full list see UVCal Parameters
                (http://pyuvdata.readthedocs.io/en/latest/uvcal.html).
                Some are always required, some are required for certain cal_types
                and others are always optional.
    """
    def __init__(self):
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)
        self._Nfreqs = uvp.UVParameter('Nfreqs',
                                       description='Number of frequency channels',
                                       expected_type=int)
        self._Njones = uvp.UVParameter('Njones',
                                       description='Number of polarizations calibration'
                                       'parameters (Number of jones matrix elements.).',
                                       expected_type=int)
        self._Ntimes = uvp.UVParameter('Ntimes',
                                       description='Number of times',
                                       expected_type=int)
        self._history = uvp.UVParameter('history',
                                        description='String of history, units English',
                                        form='str', expected_type=str)
        self._Nspws = uvp.UVParameter('Nspws', description='Number of spectral windows '
                                      '(ie non-contiguous spectral chunks). '
                                      'More than one spectral window is not '
                                      'currently supported.', expected_type=int)

        desc = ('Time range (in JD) that gain solutions are valid for.',
                'list: (start_time, end_time) in JD.')
        self._time_range = uvp.UVParameter('time_range',
                                           description=desc,
                                           form=(2,),
                                           expected_type=float)

        desc = ('Name of telescope. e.g. HERA. String.')
        self._telescope_name = uvp.UVParameter('telescope_name',
                                               description=desc,
                                               form='str',
                                               expected_type=str)

        desc = ('Number of antennas with data present (i.e. number of unique '
                'entries in ant_array). May be smaller ' +
                'than the number of antennas in the telescope')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc,
                                           expected_type=int)

        desc = ('Number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope',
                                                description=desc,
                                                expected_type=int)

        desc = ('Array of antenna indices for data arrays, shape (Nants_data), '
                'type = int, 0 indexed')
        self._ant_array = uvp.UVParameter('ant_array', description=desc,
                                          expected_type=int, form=('Nants_data',))

        desc = ('List of antenna names, shape (Nants_telescope), '
                'with numbers given by antenna_numbers (which can be matched '
                'to ant_array). There must be one entry here for each unique '
                'entry in ant_array, but there may be extras as well.')
        self._antenna_names = uvp.UVParameter('antenna_names',
                                              description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('List of integer antenna numbers corresponding to antenna_names,'
                'shape (Nants_telescope). There must be one entry here for each unique '
                'entry in ant_array, but there may be extras as well.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers',
                                                description=desc,
                                                form=('Nants_telescope',),
                                                expected_type=int)

        desc = 'Array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        desc = ('Channel width of of a frequency bin. Units Hz.')
        self._channel_width = uvp.UVParameter('channel_width',
                                              description=desc,
                                              expected_type=np.float,
                                              tols=1e-3)

        desc = ('Array of antenna polarization integers, shape (Njones). '
                'linear pols -5:-8 (jxx, jyy, jxy, jyx).'
                'circular pols -1:-4 (jrr, jll. jrl, jlr).')

        self._jones_array = uvp.UVParameter('jones_array',
                                            description=desc,
                                            expected_type=int,
                                            acceptable_vals=list(np.arange(-8, 0)),
                                            form=('Njones',))

        desc = ('Array of times, center of integration, shape (Ntimes), ' +
                'units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Ntimes',),
                                           expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))

        desc = ('Integration time of a time bin (s).')
        self._integration_time = uvp.UVParameter('integration_time',
                                                 description=desc,
                                                 expected_type=np.float,
                                                 tols=1e-3)  # 1ms

        desc = ('The convention for applying he calibration solutions to data.'
                'Indicates that to calibrate one should divide or multiply '
                'uncalibrated data by gains.')
        self._gain_convention = uvp.UVParameter('gain_convention', form='str',
                                                expected_type=str,
                                                description=desc,
                                                acceptable_vals=['divide', 'multiply'])

        desc = ('Array of flags to be applied to calibrated data (logical OR '
                'of input and flag generated by calibration). True is flagged.'
                'shape: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), type = bool.')
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nants_data', 'Nspws', 'Nfreqs',
                                                 'Ntimes', 'Njones'),
                                           expected_type=np.bool)

        desc = ('Array of qualities of calibration solutions. '
                'shape depends on the cal_type, if cal_type is gain or unknown, '
                'shape is: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), '
                'if cal_type is delay, shape is (Nants_data, Nspws, Ntimes, Njones), '
                'type = float.')
        self._quality_array = uvp.UVParameter('quality_array', description=desc,
                                              form=('Nants_data', 'Nspws', 'Nfreqs',
                                                    'Ntimes', 'Njones'),
                                              expected_type=np.float)

        desc = ('Orientation of the physical dipole corresponding to what is '
                'labelled as the x polarization. Values are east '
                '(east/west orientation),  north (north/south orientation) or '
                'unknown.')
        self._x_orientation = uvp.UVParameter('x_orientation', description=desc,
                                              expected_type=str,
                                              acceptable_vals=['east', 'north', 'unknown'])
        # --- cal_type parameters ---
        desc = ('cal type parameter. Values are delay, gain or unknown.')
        self._cal_type = uvp.UVParameter('cal_type', form='str',
                                         expected_type=str, value='unknown',
                                         description=desc,
                                         acceptable_vals=['delay', 'gain', 'unknown'])

        desc = ('Required if cal_type = "gain". Array of gains, '
                'shape: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), type = complex float.')
        self._gain_array = uvp.UVParameter('gain_array', description=desc,
                                           required=False,
                                           form=('Nants_data', 'Nspws', 'Nfreqs',
                                                 'Ntimes', 'Njones'),
                                           expected_type=np.complex)

        desc = ('Required if cal_type = "delay". Array of delays with units of seconds.'
                'shape: (Nants_data, Nspws, Ntimes, Njones), type = float')
        self._delay_array = uvp.UVParameter('delay_array', description=desc,
                                            required=False,
                                            form=('Nants_data', 'Nspws', 'Ntimes', 'Njones'),
                                            expected_type=np.float)

        desc = ('Required if cal_type = "delay". Frequency range that solutions are valid for.',
                'list: (start_frequency, end_frequency) in Hz.')
        self._freq_range = uvp.UVParameter('freq_range',
                                           description=desc, form=(2,),
                                           expected_type=float, tols=1e-3)

        # --- truly optional parameters ---
        desc = ('Array of input flags, True is flagged. shape: (Nants_data, Nspws, '
                'Nfreqs, Ntimes, Njones), type = bool.')
        self._input_flag_array = uvp.UVParameter('input_flag_array',
                                                 description=desc,
                                                 required=False,
                                                 form=('Nants_data', 'Nspws', 'Nfreqs',
                                                       'Ntimes', 'Njones'),
                                                 expected_type=np.bool)

        desc = ('Origin (on github for e.g) of calibration software. Url and branch.')
        self._git_origin_cal = uvp.UVParameter('git_origin_cal', form='str',
                                               expected_type=str,
                                               description=desc,
                                               required=False)

        desc = ('Commit hash of calibration software(from git_origin_cal) used'
                'to generate solutions.')
        self._git_hash_cal = uvp.UVParameter('git_hash_cal', form='str',
                                             expected_type=str,
                                             description=desc,
                                             required=False)

        desc = ('Name of observer who calculated solutions in this file.')
        self._observer = uvp.UVParameter('observer', form='str',
                                         description=desc,
                                         expected_type=str,
                                         required=False)

        desc = ('Array of qualities of calibration for entire arrays. '
                'shape depends on the cal_type, if cal_type is gain or unknown, '
                'shape is: (Nspws, Nfreqs, Ntimes, Njones), '
                'if cal_type is delay, shape is (Nspws, Ntimes, Njones), '
                'type = float.')
        self._total_quality_array = uvp.UVParameter('total_quality_array', description=desc,
                                                    form=('Nspws', 'Nfreqs',
                                                          'Ntimes', 'Njones'),
                                                    expected_type=np.float,
                                                    required=False)

        # String to add to history of any files written with this version of pyuvdata
        self.pyuvdata_version_str = ('  Read/written with pyuvdata version: ' +
                                     uvversion.version + '.')
        if uvversion.git_hash is not '':
            self.pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin +
                                          '.  Git hash: ' + uvversion.git_hash +
                                          '.  Git branch: ' + uvversion.git_branch +
                                          '.  Git description: ' + uvversion.git_description)

        super(UVCal, self).__init__()

    def set_gain(self):
        """Set cal_type to 'gain' and adjust required parameters."""
        self.cal_type = 'gain'
        self._gain_array.required = True
        self._delay_array.required = False
        self._freq_range.required = False
        self._quality_array.form = self._gain_array.form

    def set_delay(self):
        """Set cal_type to 'delay' and adjust required parameters."""
        self.cal_type = 'delay'
        self._gain_array.required = False
        self._delay_array.required = True
        self._freq_range.required = True
        self._quality_array.form = self._delay_array.form

    def set_unknown_cal_type(self):
        """Set cal_type to 'unknown' and adjust required parameters."""
        self.cal_type = 'unknown'
        self._gain_array.required = False
        self._delay_array.required = False
        self._freq_range.required = False
        self._quality_array.form = self._gain_array.form

    def select(self, antenna_nums=None, antenna_names=None,
               frequencies=None, freq_chans=None,
               times=None, jones=None, run_check=True,
               run_check_acceptability=True):
        """
        Select specific antennas, frequencies, times and
        jones polarization terms to keep in the object while discarding others.

        The history attribute on the object will be updated to identify the
        operations performed.

        Args:
            antenna_nums: The antennas numbers to keep in the object (antenna
                positions and names for the removed antennas will be retained).
                This cannot be provided if antenna_names is also provided.
            antenna_names: The antennas names to keep in the object (antenna
                positions and names for the removed antennas will be retained).
                This cannot be provided if antenna_nums is also provided.
            frequencies: The frequencies to keep in the object.
            freq_chans: The frequency channel numbers to keep in the object.
            times: The times to keep in the object.
            jones: The jones polarization terms to keep in the object.
            run_check: Option to check for the existence and proper shapes of
                required parameters after downselecting data on this object. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after  downselecting data on this object. Default is True.
        """
        # build up history string as we go
        history_update_string = '  Downselected to specific '
        n_selects = 0

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError('Only one of antenna_nums and antenna_names can be provided.')

            antenna_names = uvutils.get_iterable(antenna_names)
            antenna_nums = []
            for s in antenna_names:
                if s not in self.antenna_names:
                    raise ValueError('Antenna name {a} is not present in the antenna_names array'.format(a=s))
                ind = np.where(np.array(self.antenna_names) == s)[0][0]
                antenna_nums.append(self.antenna_numbers[ind])

        if antenna_nums is not None:
            antenna_nums = uvutils.get_iterable(antenna_nums)
            history_update_string += 'antennas'
            n_selects += 1

            ant_inds = np.zeros(0, dtype=np.int)
            for ant in antenna_nums:
                if ant in self.ant_array:
                    ant_inds = np.append(ant_inds, np.where(self.ant_array == ant)[0])
                else:
                    raise ValueError('Antenna number {a} is not present in the '
                                     ' array'.format(a=ant))

            ant_inds = list(sorted(set(list(ant_inds))))
            self.Nants_data = len(ant_inds)
            self.ant_array = self.ant_array[ant_inds]
            self.flag_array = self.flag_array[ant_inds, :, :, :, :]
            if self.cal_type == 'delay':
                self.quality_array = self.quality_array[ant_inds, :, :, :]
                self.delay_array = self.delay_array[ant_inds, :, :, :]
            else:
                self.quality_array = self.quality_array[ant_inds, :, :, :, :]
                self.gain_array = self.gain_array[ant_inds, :, :, :, :]

            if self.input_flag_array is not None:
                self.input_flag_array = self.input_flag_array[ant_inds, :, :, :, :]

        if times is not None:
            times = uvutils.get_iterable(times)
            if n_selects > 0:
                history_update_string += ', times'
            else:
                history_update_string += 'times'
            n_selects += 1

            time_inds = np.zeros(0, dtype=np.int)
            for jd in times:
                if jd in self.time_array:
                    time_inds = np.append(time_inds, np.where(self.time_array == jd)[0])
                else:
                    raise ValueError('Time {t} is not present in the time_array'.format(t=jd))

            time_inds = list(sorted(set(list(time_inds))))
            self.Ntimes = len(time_inds)
            self.time_array = self.time_array[time_inds]

            if self.Ntimes > 1:
                time_separation = np.diff(self.time_array)
                if not np.isclose(np.min(time_separation), np.max(time_separation),
                                  rtol=self._time_array.tols[0],
                                  atol=self._time_array.tols[1]):
                    warnings.warn('Selected times are not evenly spaced. This '
                                  'is not supported by the calfits format.')

            self.flag_array = self.flag_array[:, :, :, time_inds, :]
            if self.cal_type == 'delay':
                self.quality_array = self.quality_array[:, :, time_inds, :]
                self.delay_array = self.delay_array[:, :, time_inds, :]
            else:
                self.quality_array = self.quality_array[:, :, :, time_inds, :]
                self.gain_array = self.gain_array[:, :, :, time_inds, :]

            if self.input_flag_array is not None:
                self.input_flag_array = self.input_flag_array[:, :, :, time_inds, :]

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

            freq_inds = list(sorted(set(list(freq_inds))))
            self.Nfreqs = len(freq_inds)
            self.freq_array = self.freq_array[:, freq_inds]

            if self.Nfreqs > 1:
                freq_separation = self.freq_array[0, 1:] - self.freq_array[0, :-1]
                if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                                  rtol=self._freq_array.tols[0],
                                  atol=self._freq_array.tols[1]):
                    warnings.warn('Selected frequencies are not evenly spaced. This '
                                  'is not supported by the calfits format')

            self.flag_array = self.flag_array[:, :, freq_inds, :, :]
            if self.cal_type == 'delay':
                pass
            else:
                self.quality_array = self.quality_array[:, :, freq_inds, :, :]
                self.gain_array = self.gain_array[:, :, freq_inds, :, :]

            if self.input_flag_array is not None:
                self.input_flag_array = self.input_flag_array[:, :, freq_inds, :, :]

        if jones is not None:
            jones = uvutils.get_iterable(jones)
            if n_selects > 0:
                history_update_string += ', jones polarization terms'
            else:
                history_update_string += 'jones polarization terms'
            n_selects += 1

            jones_inds = np.zeros(0, dtype=np.int)
            for j in jones:
                if j in self.jones_array:
                    jones_inds = np.append(jones_inds, np.where(self.jones_array == j)[0])
                else:
                    raise ValueError('Jones term {j} is not present in the jones_array'.format(j=j))

            jones_inds = list(sorted(set(list(jones_inds))))
            self.Njones = len(jones_inds)
            self.jones_array = self.jones_array[jones_inds]
            if len(jones_inds) > 2:
                jones_separation = self.jones_array[1:] - self.jones_array[:-1]
                if np.min(jones_separation) < np.max(jones_separation):
                    warnings.warn('Selected jones polarization terms are not evenly spaced. This '
                                  'is not supported by the calfits format')
            self.flag_array = self.flag_array[:, :, :, :, jones_inds]
            if self.cal_type == 'delay':
                self.quality_array = self.quality_array[:, :, :, jones_inds]
                self.delay_array = self.delay_array[:, :, :, jones_inds]
            else:
                self.quality_array = self.quality_array[:, :, :, :, jones_inds]
                self.gain_array = self.gain_array[:, :, :, :, jones_inds]

            if self.input_flag_array is not None:
                self.input_flag_array = self.input_flag_array[:, :, :, :, jones_inds]

        history_update_string += ' using pyuvdata.'
        self.history = self.history + history_update_string

        # check if object is self-consistent
        if run_check:
            self.check(run_check_acceptability=run_check_acceptability)

    def convert_to_gain(self, delay_convention='minus'):
        """
        Convert non-gain cal_types to gains.

        For the delay cal_type the gain is calculated as:
            gain = 1 * exp((+/-) * 2 * pi * j * delay * frequency)
            where the (+/-) is dictated by the delay_convention

        Args:
            delay_convention: exponent sign to use in the conversion. Defaults to minus.
        """
        if self.cal_type == 'gain':
            raise ValueError('The data is already a gain cal_type.')
        elif self.cal_type == 'delay':
            if delay_convention == 'minus':
                conv = -1
            elif delay_convention == 'plus':
                conv = 1
            else:
                raise ValueError('delay_convention can only be "minus" or "plus"')

            phase_array = np.zeros((self.Nants_data, self.Nspws, self.Nfreqs, self.Ntimes, self.Njones))
            for si in range(self.Nspws):
                temp = conv * 2 * np.pi * np.dot(self.delay_array[:, si, :, :, np.newaxis],
                                                 self.freq_array[si, np.newaxis, :])
                temp = np.transpose(temp, (0, 3, 1, 2))
                phase_array[:, si, :, :, :] = temp

            gain_array = np.exp(1j * phase_array)
            new_quality = np.repeat(self.quality_array[:, :, np.newaxis, :, :], self.Nfreqs, axis=2)
            self.set_gain()
            self.gain_array = gain_array
            self.quality_array = new_quality
            self.delay_array = None
            self.check()
        else:
            raise(ValueError, 'cal_type is unknown, cannot convert to gain')

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype is 'calfits':
            import calfits
            other_obj = calfits.CALFITS()
        else:
            raise ValueError('filetype must be calfits.')
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_calfits(self, filename, run_check=True, run_check_acceptability=True):
        """
        Read in data from a calfits file.

        Args:
            filename: The uvfits file to read from.
        """
        import calfits
        uvfits_obj = calfits.CALFITS()
        uvfits_obj.read_calfits(filename, run_check=run_check,
                                run_check_acceptability=run_check_acceptability)
        self._convert_from_filetype(uvfits_obj)
        del(uvfits_obj)

    def write_calfits(self, filename,  run_check=True, run_check_acceptability=True,
                      clobber=False):
        """Write data to a calfits file.

        Args:
            filename: The calfits filename to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptability of the values of
                required parameters before writing the file. Default is True.
            clobber: Overwrite file.
        """
        calfits_obj = self._convert_to_filetype('calfits')
        calfits_obj.write_calfits(filename,
                                  run_check=run_check,
                                  run_check_acceptability=run_check_acceptability,
                                  clobber=clobber)
        del(calfits_obj)
