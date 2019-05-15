# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import numpy as np
import warnings
import copy

from .uvbase import UVBase
from . import parameter as uvp
from . import utils as uvutils


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
                                       description='Number of Jones calibration'
                                       'parameters (Number of Jones matrix elements '
                                       'calculated in calibration).',
                                       expected_type=int)
        desc = ('Number of times with different calibrations calculated '
                '(if a calibration is calculated over a range of integrations, '
                'this gives the number of separate calibrations along the time axis).')
        self._Ntimes = uvp.UVParameter('Ntimes', description=desc,
                                       expected_type=int)
        self._history = uvp.UVParameter('history',
                                        description='String of history, units English',
                                        form='str', expected_type=str)
        self._Nspws = uvp.UVParameter('Nspws', description='Number of spectral windows '
                                      '(ie non-contiguous spectral chunks). '
                                      'More than one spectral window is not '
                                      'currently supported.', expected_type=int)

        desc = ('Time range (in JD) that cal solutions are valid for.'
                'list: [start_time, end_time] in JD.')
        self._time_range = uvp.UVParameter('time_range', description=desc,
                                           form=2, expected_type=float)

        desc = ('Name of telescope. e.g. HERA. String.')
        self._telescope_name = uvp.UVParameter('telescope_name',
                                               description=desc,
                                               form='str',
                                               expected_type=str)

        desc = ('Number of antennas that have data associated with them '
                '(i.e. length of ant_array), which may be smaller than the number'
                'of antennas in the telescope (i.e. length of antenna_numbers).')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc,
                                           expected_type=int)

        desc = ('Number of antennas in the antenna_numbers array. May be larger '
                'than the number of antennas with gains associated with them.')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope',
                                                description=desc,
                                                expected_type=int)

        desc = ('Array of integer antenna numbers that appear in self.gain_array, with shape (Nants_data,). '
                'This array is ordered to match the inherent ordering of the zeroth axis of self.gain_array.')
        self._ant_array = uvp.UVParameter('ant_array', description=desc,
                                          expected_type=int, form=('Nants_data',))

        desc = ('Array of antenna names with shape (Nants_telescope,). '
                'Ordering of elements matches ordering of antenna_numbers.')
        self._antenna_names = uvp.UVParameter('antenna_names',
                                              description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('Array of all integer-valued antenna numbers in the telescope with shape (Nants_telescope,). '
                'Ordering of elements matches that of antenna_names. This array is not necessarily identical '
                'to ant_array, in that this array holds all antenna numbers associated with the telescope, not '
                'just antennas with data, and has an in principle non-specific ordering.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers',
                                                description=desc,
                                                form=('Nants_telescope',),
                                                expected_type=int)

        self._spw_array = uvp.UVParameter('spw_array',
                                          description='Array of spectral window '
                                          'numbers, shape (Nspws).', form=('Nspws',),
                                          expected_type=int)

        desc = 'Array of frequencies, center of the channel, shape (Nspws, Nfreqs), units Hz.'
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

        desc = ('Array of calibration solution times, center of integration, '
                'shape (Ntimes), units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Ntimes',),
                                           expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))

        desc = ('Integration time of a time bin, units seconds.')
        self._integration_time = uvp.UVParameter('integration_time',
                                                 description=desc,
                                                 expected_type=np.float,
                                                 tols=1e-3)  # 1ms

        desc = ('The convention for applying the calibration solutions to data.'
                'Values are "divide" or "multiply", indicating that to calibrate '
                'one should divide or multiply uncalibrated data by gains. '
                'Mathematically this indicates the alpha exponent in the equation: '
                'calibrated data = gain^alpha * uncalibrated data. A value of '
                '"divide" represents alpha=-1 and "multiply" represents alpha=1.')
        self._gain_convention = uvp.UVParameter('gain_convention', form='str',
                                                expected_type=str,
                                                description=desc,
                                                acceptable_vals=['divide', 'multiply'])

        desc = ('Array of flags to be applied to calibrated data (logical OR '
                'of input and flag generated by calibration). True is flagged. '
                'Shape: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), type = bool.')
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nants_data', 'Nspws', 'Nfreqs',
                                                 'Ntimes', 'Njones'),
                                           expected_type=np.bool)

        desc = ('Array of qualities of calibration solutions. '
                'The shape depends on cal_type, if the cal_type is "gain" or '
                '"unknown", the shape is: (Nants_data, Nspws, Nfreqs, Ntimes, Njones), '
                'if the cal_type is "delay", the shape is (Nants_data, Nspws, 1, Ntimes, Njones), '
                'type = float.')
        self._quality_array = uvp.UVParameter('quality_array', description=desc,
                                              form=('Nants_data', 'Nspws', 'Nfreqs',
                                                    'Ntimes', 'Njones'),
                                              expected_type=np.float)

        desc = ('Orientation of the physical dipole corresponding to what is '
                'labelled as the x polarization. Options are "east" '
                '(indicating east/west orientation) and "north" (indicating '
                'north/south orientation)')
        self._x_orientation = uvp.UVParameter('x_orientation', description=desc,
                                              expected_type=str,
                                              acceptable_vals=['east', 'north'])

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

        desc = ('Required if cal_type = "delay". Array of delays with units of seconds. '
                'Shape: (Nants_data, Nspws, 1, Ntimes, Njones), type = float.')
        self._delay_array = uvp.UVParameter('delay_array', description=desc,
                                            required=False,
                                            form=('Nants_data', 'Nspws', 1, 'Ntimes', 'Njones'),
                                            expected_type=np.float)

        desc = ('Required if cal_type = "delay". Frequency range that solutions are valid for.'
                'list: [start_frequency, end_frequency] in Hz.')
        self._freq_range = uvp.UVParameter('freq_range', required=False,
                                           description=desc, form=2,
                                           expected_type=float, tols=1e-3)

        # --- cal_style parameters ---
        desc = ('Style of calibration. Values are sky or redundant.')
        self._cal_style = uvp.UVParameter('cal_style', form='str',
                                          expected_type=str,
                                          description=desc,
                                          acceptable_vals=['sky', 'redundant'])

        desc = ('Required if cal_style = "sky". Short string describing field '
                'center or dominant source.')
        self._sky_field = uvp.UVParameter('sky_field', form='str', required=False,
                                          expected_type=str, description=desc)

        desc = ('Required if cal_style = "sky". Name of calibration catalog.')
        self._sky_catalog = uvp.UVParameter('sky_catalog', form='str', required=False,
                                            expected_type=str, description=desc)

        desc = ('Required if cal_style = "sky". Phase reference antenna.')
        self._ref_antenna_name = uvp.UVParameter('ref_antenna_name', form='str',
                                                 required=False,
                                                 expected_type=str, description=desc)

        desc = ('Number of sources used.')
        self._Nsources = uvp.UVParameter('Nsources', required=False,
                                         expected_type=np.int, description=desc)

        desc = ('Range of baselines used for calibration.')
        self._baseline_range = uvp.UVParameter('baseline_range', form=2,
                                               required=False,
                                               expected_type=np.float, description=desc)

        desc = ('Name of diffuse model.')
        self._diffuse_model = uvp.UVParameter('diffuse_model', form='str',
                                              required=False,
                                              expected_type=str, description=desc)

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

        desc = ('Commit hash of calibration software (from git_origin_cal) used '
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

        desc = ('Array of qualities of the calibration for entire arrays. '
                'The shape depends on cal_type, if the cal_type is "gain" or '
                '"unknown", the shape is: (Nspws, Nfreqs, Ntimes, Njones), '
                'if the cal_type is "delay", the shape is (Nspws, 1, Ntimes, Njones), '
                'type = float.')
        self._total_quality_array = uvp.UVParameter('total_quality_array', description=desc,
                                                    form=('Nspws', 'Nfreqs',
                                                          'Ntimes', 'Njones'),
                                                    expected_type=np.float,
                                                    required=False)

        desc = ('Any user supplied extra keywords, type=dict. Keys should be '
                '8 character or less strings if writing to calfits files. '
                'Use the special key "comment" for long multi-line string comments.')
        self._extra_keywords = uvp.UVParameter('extra_keywords', required=False,
                                               description=desc, value={},
                                               spoof_val={}, expected_type=dict)

        super(UVCal, self).__init__()

    def check(self, check_extra=True, run_check_acceptability=True):
        """
        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Args:
            run_check_acceptability: Option to check if values in required parameters
                are acceptable. Default is True.
        """
        # Make sure requirements are set properly for cal_style
        if self.cal_style == 'sky':
            self.set_sky()
        elif self.cal_style == 'redundant':
            self.set_redundant()

        # check for deprecated x_orientation strings and convert to new values (if possible)
        if self.x_orientation is not None:
            if self.x_orientation not in self._x_orientation.acceptable_vals:
                warn_string = ('x_orientation {xval} is not one of [{vals}], '
                               .format(xval=self.x_orientation,
                                       vals=(', ').join(self._x_orientation.acceptable_vals)))
                if self.x_orientation.lower() == 'e':
                    self.x_orientation = 'east'
                    warn_string += 'converting to "east".'
                elif self.x_orientation.lower() == 'n':
                    self.x_orientation = 'north'
                    warn_string += 'converting to "north".'
                else:
                    warn_string += 'cannot be converted.'

                warnings.warn(warn_string + ' Only [{vals}] will be supported '
                              'starting in version 1.5'
                              .format(vals=(', ').join(self._x_orientation.acceptable_vals)),
                              DeprecationWarning)

        # first run the basic check from UVBase
        super(UVCal, self).check(check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)

        # require that all entries in ant_array exist in antenna_numbers
        if not all(ant in self.antenna_numbers for ant in self.ant_array):
            raise ValueError('All antennas in ant_array must be in antenna_numbers.')

        # issue warning if extra_keywords keys are longer than 8 characters
        for key in self.extra_keywords.keys():
            if len(key) > 8:
                warnings.warn('key {key} in extra_keywords is longer than 8 '
                              'characters. It will be truncated to 8 if written '
                              'to a calfits file format.'.format(key=key))

        # issue warning if extra_keywords values are lists, arrays or dicts
        for key, value in self.extra_keywords.items():
            if isinstance(value, (list, dict, np.ndarray)):
                warnings.warn('{key} in extra_keywords is a list, array or dict, '
                              'which will raise an error when writing calfits '
                              'files'.format(key=key))

        return True

    def set_gain(self):
        """Set cal_type to 'gain' and adjust required parameters."""
        self.cal_type = 'gain'
        self._gain_array.required = True
        self._delay_array.required = False
        self._freq_range.required = False
        self._quality_array.form = self._gain_array.form
        self._total_quality_array.form = self._gain_array.form[1:]

    def set_delay(self):
        """Set cal_type to 'delay' and adjust required parameters."""
        self.cal_type = 'delay'
        self._gain_array.required = False
        self._delay_array.required = True
        self._freq_range.required = True
        self._quality_array.form = self._delay_array.form
        self._total_quality_array.form = self._delay_array.form[1:]

    def set_unknown_cal_type(self):
        """Set cal_type to 'unknown' and adjust required parameters."""
        self.cal_type = 'unknown'
        self._gain_array.required = False
        self._delay_array.required = False
        self._freq_range.required = False
        self._quality_array.form = self._gain_array.form
        self._total_quality_array.form = self._gain_array.form[1:]

    def set_sky(self):
        """Set cal_style to 'sky' and adjust required parameters."""
        self.cal_style = 'sky'
        self._sky_field.required = True
        self._sky_catalog.required = True
        self._ref_antenna_name.required = True

    def set_redundant(self):
        """Set cal_style to 'redundant' and adjust required parameters."""
        self.cal_style = 'redundant'
        self._sky_field.required = False
        self._sky_catalog.required = False
        self._ref_antenna_name.required = False

    def select(self, antenna_nums=None, antenna_names=None,
               frequencies=None, freq_chans=None,
               times=None, jones=None, run_check=True, check_extra=True,
               run_check_acceptability=True, inplace=True):
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
            check_extra: Option to check shapes and types of optional parameters
                as well as required ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after  downselecting data on this object. Default is True.
            inplace: Option to perform the select directly on self (True, default) or return
                a new UVCal object, which is a subselection of self (False)
        """
        if inplace:
            cal_object = self
        else:
            cal_object = copy.deepcopy(self)

        # build up history string as we go
        history_update_string = '  Downselected to specific '
        n_selects = 0

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError('Only one of antenna_nums and antenna_names can be provided.')

            antenna_names = uvutils._get_iterable(antenna_names)
            antenna_nums = []
            for s in antenna_names:
                if s not in cal_object.antenna_names:
                    raise ValueError('Antenna name {a} is not present in the antenna_names array'.format(a=s))
                ind = np.where(np.array(cal_object.antenna_names) == s)[0][0]
                antenna_nums.append(cal_object.antenna_numbers[ind])

        if antenna_nums is not None:
            antenna_nums = uvutils._get_iterable(antenna_nums)
            history_update_string += 'antennas'
            n_selects += 1

            ant_inds = np.zeros(0, dtype=np.int)
            for ant in antenna_nums:
                if ant in cal_object.ant_array:
                    ant_inds = np.append(ant_inds, np.where(cal_object.ant_array == ant)[0])
                else:
                    raise ValueError('Antenna number {a} is not present in the '
                                     ' array'.format(a=ant))

            ant_inds = list(sorted(set(list(ant_inds))))
            cal_object.Nants_data = len(ant_inds)
            cal_object.ant_array = cal_object.ant_array[ant_inds]
            cal_object.flag_array = cal_object.flag_array[ant_inds, :, :, :, :]
            cal_object.quality_array = cal_object.quality_array[ant_inds, :, :, :, :]
            if cal_object.cal_type == 'delay':
                cal_object.delay_array = cal_object.delay_array[ant_inds, :, :, :, :]
            else:
                cal_object.gain_array = cal_object.gain_array[ant_inds, :, :, :, :]

            if cal_object.input_flag_array is not None:
                cal_object.input_flag_array = cal_object.input_flag_array[ant_inds, :, :, :, :]

            if cal_object.total_quality_array is not None:
                warnings.warn('Cannot preserve total_quality_array when changing '
                              'number of antennas; discarding')
                cal_object.total_quality_array = None

        if times is not None:
            times = uvutils._get_iterable(times)
            if n_selects > 0:
                history_update_string += ', times'
            else:
                history_update_string += 'times'
            n_selects += 1

            time_inds = np.zeros(0, dtype=np.int)
            for jd in times:
                if jd in cal_object.time_array:
                    time_inds = np.append(time_inds, np.where(cal_object.time_array == jd)[0])
                else:
                    raise ValueError('Time {t} is not present in the time_array'.format(t=jd))

            time_inds = list(sorted(set(list(time_inds))))
            cal_object.Ntimes = len(time_inds)
            cal_object.time_array = cal_object.time_array[time_inds]

            if cal_object.Ntimes > 1:
                time_separation = np.diff(cal_object.time_array)
                if not np.isclose(np.min(time_separation), np.max(time_separation),
                                  rtol=cal_object._time_array.tols[0],
                                  atol=cal_object._time_array.tols[1]):
                    warnings.warn('Selected times are not evenly spaced. This '
                                  'is not supported by the calfits format.')

            cal_object.flag_array = cal_object.flag_array[:, :, :, time_inds, :]
            cal_object.quality_array = cal_object.quality_array[:, :, :, time_inds, :]
            if cal_object.cal_type == 'delay':
                cal_object.delay_array = cal_object.delay_array[:, :, :, time_inds, :]
            else:
                cal_object.gain_array = cal_object.gain_array[:, :, :, time_inds, :]

            if cal_object.input_flag_array is not None:
                cal_object.input_flag_array = cal_object.input_flag_array[:, :, :, time_inds, :]

            if cal_object.total_quality_array is not None:
                cal_object.total_quality_array = cal_object.total_quality_array[:, :, time_inds, :]

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if frequencies is None:
                frequencies = cal_object.freq_array[0, freq_chans]
            else:
                frequencies = uvutils._get_iterable(frequencies)
                frequencies = np.sort(list(set(frequencies)
                                      | set(cal_object.freq_array[0, freq_chans])))

        if frequencies is not None:
            frequencies = uvutils._get_iterable(frequencies)
            if n_selects > 0:
                history_update_string += ', frequencies'
            else:
                history_update_string += 'frequencies'
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int)
            # this works because we only allow one SPW. This will have to be reworked when we support more.
            freq_arr_use = cal_object.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError('Frequency {f} is not present in the freq_array'.format(f=f))

            freq_inds = list(sorted(set(list(freq_inds))))
            cal_object.Nfreqs = len(freq_inds)
            cal_object.freq_array = cal_object.freq_array[:, freq_inds]

            if cal_object.Nfreqs > 1:
                freq_separation = cal_object.freq_array[0, 1:] - cal_object.freq_array[0, :-1]
                if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                                  rtol=cal_object._freq_array.tols[0],
                                  atol=cal_object._freq_array.tols[1]):
                    warnings.warn('Selected frequencies are not evenly spaced. This '
                                  'is not supported by the calfits format')

            cal_object.flag_array = cal_object.flag_array[:, :, freq_inds, :, :]
            if cal_object.cal_type == 'delay':
                pass
            else:
                cal_object.quality_array = cal_object.quality_array[:, :, freq_inds, :, :]
                cal_object.gain_array = cal_object.gain_array[:, :, freq_inds, :, :]

            if cal_object.input_flag_array is not None:
                cal_object.input_flag_array = cal_object.input_flag_array[:, :, freq_inds, :, :]

            if cal_object.cal_type == 'delay':
                pass
            else:
                if cal_object.total_quality_array is not None:
                    cal_object.total_quality_array = cal_object.total_quality_array[:, freq_inds, :, :]

        if jones is not None:
            jones = uvutils._get_iterable(jones)
            if n_selects > 0:
                history_update_string += ', jones polarization terms'
            else:
                history_update_string += 'jones polarization terms'
            n_selects += 1

            jones_inds = np.zeros(0, dtype=np.int)
            for j in jones:
                if j in cal_object.jones_array:
                    jones_inds = np.append(jones_inds, np.where(cal_object.jones_array == j)[0])
                else:
                    raise ValueError('Jones term {j} is not present in the jones_array'.format(j=j))

            jones_inds = list(sorted(set(list(jones_inds))))
            cal_object.Njones = len(jones_inds)
            cal_object.jones_array = cal_object.jones_array[jones_inds]
            if len(jones_inds) > 2:
                jones_separation = cal_object.jones_array[1:] - cal_object.jones_array[:-1]
                if np.min(jones_separation) < np.max(jones_separation):
                    warnings.warn('Selected jones polarization terms are not evenly spaced. This '
                                  'is not supported by the calfits format')

            cal_object.flag_array = cal_object.flag_array[:, :, :, :, jones_inds]
            cal_object.quality_array = cal_object.quality_array[:, :, :, :, jones_inds]
            if cal_object.cal_type == 'delay':
                cal_object.delay_array = cal_object.delay_array[:, :, :, :, jones_inds]
            else:
                cal_object.gain_array = cal_object.gain_array[:, :, :, :, jones_inds]

            if cal_object.input_flag_array is not None:
                cal_object.input_flag_array = cal_object.input_flag_array[:, :, :, :, jones_inds]

            if cal_object.total_quality_array is not None:
                cal_object.total_quality_array = cal_object.total_quality_array[:, :, :, jones_inds]

        history_update_string += ' using pyuvdata.'
        cal_object.history = cal_object.history + history_update_string

        # check if object is self-consistent
        if run_check:
            cal_object.check(check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)

        if not inplace:
            return cal_object

    def convert_to_gain(self, delay_convention='minus', run_check=True, check_extra=True,
                        run_check_acceptability=True):
        """
        Convert non-gain cal_types to gains.

        For the delay cal_type the gain is calculated as:
            gain = 1 * exp((+/-) * 2 * pi * j * delay * frequency)
            where the (+/-) is dictated by the delay_convention

        Args:
            delay_convention: exponent sign to use in the conversion. Defaults to minus.
            run_check: Option to check for the existence and proper shapes of
                parameters after converting this object. Default is True.
            check_extra: Option to check shapes and types of optional parameters
                as well as required ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after converting this object. Default is True.
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

            self.history += '  Converted from delays to gains using pyuvdata.'

            phase_array = np.zeros((self.Nants_data, self.Nspws, self.Nfreqs, self.Ntimes, self.Njones))
            for si in range(self.Nspws):
                temp = conv * 2 * np.pi * np.dot(self.delay_array[:, si, 0, :, :, np.newaxis],
                                                 self.freq_array[si, np.newaxis, :])
                temp = np.transpose(temp, (0, 3, 1, 2))
                phase_array[:, si, :, :, :] = temp

            gain_array = np.exp(1j * phase_array)
            new_quality = np.repeat(self.quality_array[:, :, :, :, :], self.Nfreqs, axis=2)
            self.set_gain()
            self.gain_array = gain_array
            self.quality_array = new_quality
            self.delay_array = None
            if self.total_quality_array is not None:
                new_total_quality_array = np.repeat(self.total_quality_array[:, :, :, :], self.Nfreqs, axis=1)
                self.total_quality_array = new_total_quality_array

            # check if object is self-consistent
            if run_check:
                self.check(check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability)
        else:
            raise ValueError('cal_type is unknown, cannot convert to gain')

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype is 'calfits':
            from . import calfits
            other_obj = calfits.CALFITS()
        else:
            raise ValueError('filetype must be calfits.')
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_calfits(self, filename, run_check=True, check_extra=True,
                     run_check_acceptability=True, strict_fits=False):
        """
        Read in data from a calfits file.

        Args:
            filename: The calfits file or list of files to read from.
                      string path, or list or tuple of string paths.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
            strict_fits: boolean
                If True, require that the data axes have cooresponding NAXIS, CRVAL,
                CDELT and CRPIX keywords. If False, allow CRPIX to be missing and
                set it equal to zero and allow the CRVAL for the spw directions to
                be missing and set it to zero. This keyword exists to support old
                calfits files that were missing many CRPIX and CRVAL keywords.
                Default is False.
        """
        from . import calfits
        if isinstance(filename, (list, tuple)):
            self.read_calfits(filename[0], run_check=run_check,
                              check_extra=check_extra,
                              run_check_acceptability=run_check_acceptability,
                              strict_fits=strict_fits)
            if len(filename) > 1:
                for f in filename[1:]:
                    uvcal2 = UVCal()
                    uvcal2.read_calfits(f, run_check=run_check,
                                        check_extra=check_extra,
                                        run_check_acceptability=run_check_acceptability,
                                        strict_fits=strict_fits)
                    self += uvcal2
                del(uvcal2)
        else:
            calfits_obj = calfits.CALFITS()
            calfits_obj.read_calfits(filename, run_check=run_check,
                                     check_extra=check_extra,
                                     run_check_acceptability=run_check_acceptability,
                                     strict_fits=strict_fits)
            self._convert_from_filetype(calfits_obj)
            del(calfits_obj)

    def write_calfits(self, filename, run_check=True, check_extra=True,
                      run_check_acceptability=True, clobber=False):
        """Write data to a calfits file.

        Args:
            filename: The calfits filename to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
        """
        calfits_obj = self._convert_to_filetype('calfits')
        calfits_obj.write_calfits(filename,
                                  run_check=run_check, check_extra=check_extra,
                                  run_check_acceptability=run_check_acceptability,
                                  clobber=clobber)
        del(calfits_obj)

    def read_fhd_cal(self, cal_file, obs_file, settings_file=None, raw=True,
                     extra_history=None, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Read data from an FHD cal.sav file.

        Args:
            cal_file: The cal.sav file to read from.
            obs_file: The obs.sav file to read from.
            settings_file: The settings_file to read from. Optional, but very
                useful for provenance.
            raw: Option to use the raw (per antenna, per frequency) solution or
                to use the fitted (polynomial over phase/amplitude) solution.
                Default is True (meaning use the raw solutions).
            extra_history: Optional string or list of strings to add to the
                object's history parameter. Default is None.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        from . import fhd_cal
        if isinstance(cal_file, (list, tuple)):
            if isinstance(obs_file, (list, tuple)):
                if len(obs_file) != len(cal_file):
                    raise ValueError('Number of obs_files must match number of cal_files')
            else:
                raise ValueError('Number of obs_files must match number of cal_files')

            if settings_file is not None:
                if isinstance(settings_file, (list, tuple)):
                    if len(settings_file) != len(cal_file):
                        raise ValueError('Number of settings_files must match number of cal_files')
                else:
                    raise ValueError('Number of settings_files must match number of cal_files')
                settings_file_use = settings_file[0]

            self.read_fhd_cal(cal_file[0], obs_file[0], settings_file=settings_file_use,
                              raw=raw, extra_history=extra_history,
                              run_check=run_check, check_extra=check_extra,
                              run_check_acceptability=run_check_acceptability)
            if len(cal_file) > 1:
                for ind, f in enumerate(cal_file[1:]):
                    uvcal2 = UVCal()
                    if settings_file is not None:
                        settings_file_use = settings_file[ind + 1]
                    uvcal2.read_fhd_cal(f, obs_file[ind + 1],
                                        settings_file=settings_file_use,
                                        raw=raw, extra_history=extra_history,
                                        run_check=run_check, check_extra=check_extra,
                                        run_check_acceptability=run_check_acceptability)

                    self += uvcal2
                del(uvcal2)
        else:
            if isinstance(obs_file, (list, tuple)):
                raise ValueError('Number of obs_files must match number of cal_files')
            if settings_file is not None:
                if isinstance(settings_file, (list, tuple)):
                    raise ValueError('Number of settings_files must match number of cal_files')

            fhd_cal_obj = fhd_cal.FHDCal()
            fhd_cal_obj.read_fhd_cal(cal_file, obs_file, settings_file=settings_file,
                                     raw=raw, extra_history=extra_history,
                                     run_check=run_check, check_extra=check_extra,
                                     run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(fhd_cal_obj)
            del(fhd_cal_obj)

    def __add__(self, other, run_check=True, check_extra=True,
                run_check_acceptability=True, inplace=False):
        """
        Combine two UVCal objects. Objects can be added along antenna, frequency,
        time, and/or Jones axis.

        Args:
            other: Another UVCal object which will be added to self.
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
        # Check that both objects are UVCal and valid
        this.check(check_extra=check_extra, run_check_acceptability=run_check_acceptability)
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError('Only UVCal (or subclass) objects can be added to '
                                 'a UVCal (or subclass) object')
        other.check(check_extra=check_extra, run_check_acceptability=run_check_acceptability)

        # Check objects are compatible
        compatibility_params = ['_cal_type', '_integration_time', '_channel_width',
                                '_telescope_name', '_gain_convention', '_x_orientation',
                                '_cal_style', '_ref_antenna_name']
        if this.cal_type == 'delay':
            compatibility_params.append('_freq_range')
        warning_params = ['_observer', '_git_hash_cal', '_sky_field',
                          '_sky_catalog', '_Nsources', '_baseline_range',
                          '_diffuse_model']

        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = 'UVParameter ' + \
                    a[1:] + ' does not match. Cannot combine objects.'
                raise ValueError(msg)
        for a in warning_params:
            if getattr(this, a) != getattr(other, a):
                msg = 'UVParameter ' + \
                    a[1:] + ' does not match. Combining anyway.'
                warnings.warn(msg)

        # Build up history string
        history_update_string = ' Combined data along '
        n_axes = 0

        # Check we don't have overlapping data
        both_jones = np.intersect1d(
            this.jones_array, other.jones_array)
        both_times = np.intersect1d(
            this.time_array, other.time_array)
        if this.cal_type != 'delay':
            both_freq = np.intersect1d(
                this.freq_array[0, :], other.freq_array[0, :])
        else:
            # Make a non-empty array so we raise an error if other data is duplicated
            both_freq = [0]
        both_ants = np.intersect1d(
            this.ant_array, other.ant_array)
        if len(both_jones) > 0:
            if len(both_times) > 0:
                if len(both_freq) > 0:
                    if len(both_ants) > 0:
                        raise ValueError('These objects have overlapping data and'
                                         ' cannot be combined.')

        temp = np.nonzero(~np.in1d(other.ant_array, this.ant_array))[0]
        if len(temp) > 0:
            anew_inds = temp
            new_ants = other.ant_array[temp]
            history_update_string += 'antenna'
            n_axes += 1
        else:
            anew_inds, new_ants = ([], [])

        temp = np.nonzero(~np.in1d(other.time_array, this.time_array))[0]
        if len(temp) > 0:
            tnew_inds = temp
            new_times = other.time_array[temp]
            if n_axes > 0:
                history_update_string += ', time'
            else:
                history_update_string += 'time'
            n_axes += 1
        else:
            tnew_inds, new_times = ([], [])

        # adding along frequency axis is not supported for delay-type cal files
        if this.cal_type == 'gain':
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
        else:
            fnew_inds, new_freqs = ([], [])

        temp = np.nonzero(~np.in1d(other.jones_array,
                                   this.jones_array))[0]
        if len(temp) > 0:
            jnew_inds = temp
            new_jones = other.jones_array[temp]
            if n_axes > 0:
                history_update_string += ', jones'
            else:
                history_update_string += 'jones'
            n_axes += 1
        else:
            jnew_inds, new_jones = ([], [])

        # Initialize tqa variables
        can_combine_tqa = True
        if this.cal_type == 'delay':
            Nf_tqa = 1
        else:
            Nf_tqa = this.Nfreqs

        # Pad out self to accommodate new data
        if len(anew_inds) > 0:
            this.ant_array = np.concatenate([this.ant_array, other.ant_array[anew_inds]])
            order = np.argsort(this.ant_array)
            this.ant_array = this.ant_array[order]
            zero_pad_data = np.zeros(
                (len(anew_inds), this.Nspws, this.quality_array.shape[2], this.Ntimes,
                 this.Njones))
            zero_pad_flags = np.zeros(
                (len(anew_inds), this.Nspws, this.Nfreqs, this.Ntimes, this.Njones))
            if this.cal_type == 'delay':
                this.delay_array = np.concatenate([this.delay_array, zero_pad_data], axis=0)[
                    order, :, :, :, :]
            else:
                this.gain_array = np.concatenate([this.gain_array, zero_pad_data], axis=0)[
                    order, :, :, :, :]
            this.flag_array = np.concatenate([this.flag_array,
                                              1 - zero_pad_flags], axis=0).astype(np.bool)[
                                                  order, :, :, :, :]
            this.quality_array = np.concatenate([this.quality_array, zero_pad_data], axis=0)[
                order, :, :, :, :]

            # If total_quality_array exists, we set it to None and warn the user
            if this.total_quality_array is not None or other.total_quality_array is not None:
                warnings.warn("Total quality array detected in at least one file; the "
                              "array in the new object will be set to 'None' because "
                              "whole-array values cannot be combined when adding antennas")
                this.total_quality_array = None
                can_combine_tqa = False

            if this.input_flag_array is not None:
                zero_pad = np.zeros(
                    (len(anew_inds), this.Nspws, this.Nfreqs, this.Ntimes, this.Njones))
                this.input_flag_array = np.concatenate(
                    [this.input_flag_array, 1 - zero_pad], axis=0).astype(np.bool)[
                        order, :, :, :, :]
            elif other.input_flag_array is not None:
                zero_pad = np.zeros(
                    (len(anew_inds), this.Nspws, this.Nfreqs, this.Ntimes, this.Njones))
                this.input_flag_array = np.array(1 - np.zeros(
                    (this.Nants_data, this.Nspws, this.Nfreqs, this.Ntimes,
                     this.Njones))).astype(np.bool)
                this.input_flag_array = np.concatenate([this.input_flag_array,
                                                        1 - zero_pad],
                                                       axis=0).astype(np.bool)[
                                                           order, :, :, :, :]

        if len(fnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as the main data
            # Also do not need to worry about different cases for gain v. delay type
            zero_pad = np.zeros((this.quality_array.shape[0], this.Nspws, len(fnew_inds),
                                 this.Ntimes, this.Njones))
            this.freq_array = np.concatenate([this.freq_array,
                                              other.freq_array[:, fnew_inds]], axis=1)
            order = np.argsort(this.freq_array[0, :])
            this.freq_array = this.freq_array[:, order]
            this.gain_array = np.concatenate([this.gain_array, zero_pad], axis=2)[
                :, :, order, :, :]
            this.flag_array = np.concatenate([this.flag_array,
                                              1 - zero_pad], axis=2).astype(np.bool)[
                                                  :, :, order, :, :]
            this.quality_array = np.concatenate([this.quality_array, zero_pad], axis=2)[
                :, :, order, :, :]

            if this.total_quality_array is not None and can_combine_tqa:
                zero_pad = np.zeros((this.Nspws, len(fnew_inds), this.Ntimes, this.Njones))
                this.total_quality_array = np.concatenate([this.total_quality_array, zero_pad],
                                                          axis=1)[:, order, :, :]
            elif other.total_quality_array is not None and can_combine_tqa:
                zero_pad = np.zeros((this.Nspws, len(fnew_inds), this.Ntimes, this.Njones))
                this.total_quality_array = np.zeros((this.Nspws, Nf_tqa, this.Ntimes, this.Njones))
                this.total_quality_array = np.concatenate([this.total_quality_array, zero_pad],
                                                          axis=1)[:, order, :, :]

        if len(tnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as the main data
            zero_pad_data = np.zeros(
                (this.quality_array.shape[0], this.Nspws, this.quality_array.shape[2],
                 len(tnew_inds), this.Njones))
            zero_pad_flags = np.zeros(
                (this.flag_array.shape[0], this.Nspws, this.flag_array.shape[2],
                 len(tnew_inds), this.Njones))
            this.time_array = np.concatenate([this.time_array, other.time_array[tnew_inds]])
            order = np.argsort(this.time_array)
            this.time_array = this.time_array[order]
            if this.cal_type == 'delay':
                this.delay_array = np.concatenate([this.delay_array, zero_pad_data], axis=3)[
                    :, :, :, order, :]
            else:
                this.gain_array = np.concatenate([this.gain_array, zero_pad_data], axis=3)[
                    :, :, :, order, :]
            this.flag_array = np.concatenate([this.flag_array,
                                              1 - zero_pad_flags], axis=3).astype(np.bool)[
                                                  :, :, :, order, :]
            this.quality_array = np.concatenate([this.quality_array, zero_pad_data], axis=3)[
                :, :, :, order, :]
            if this.total_quality_array is not None and can_combine_tqa:
                zero_pad = np.zeros((this.Nspws, this.quality_array.shape[2], len(tnew_inds),
                                     this.Njones))
                this.total_quality_array = np.concatenate([this.total_quality_array, zero_pad],
                                                          axis=2)[:, :, order, :]
            elif other.total_quality_array is not None and can_combine_tqa:
                zero_pad = np.zeros((this.Nspws, this.quality_array.shape[2], len(tnew_inds),
                                     this.Njones))
                this.total_quality_array = np.zeros((this.Nspws, Nf_tqa, this.Ntimes, this.Njones))
                this.total_quality_array = np.concatenate([this.total_quality_array, zero_pad],
                                                          axis=2)[:, :, order, :]

            if this.input_flag_array is not None:
                zero_pad = np.zeros(
                    (this.input_flag_array.shape[0], this.Nspws,
                     this.input_flag_array.shape[2], len(tnew_inds), this.Njones))
                this.input_flag_array = np.concatenate(
                    [this.input_flag_array, 1 - zero_pad], axis=3).astype(np.bool)[
                        :, :, :, order, :]
            elif other.input_flag_array is not None:
                zero_pad = np.zeros(
                    (this.flag_array.shape[0], this.Nspws,
                     this.flag_array.shape[2], len(tnew_inds), this.Njones))
                this.input_flag_array = np.array(1 - np.zeros(
                    (this.flag_array.shape[0], this.Nspws,
                     this.flag_array.shape[2], this.flag_array.shape[3],
                     this.Njones))).astype(np.bool)
                this.input_flag_array = np.concatenate([this.input_flag_array,
                                                        1 - zero_pad],
                                                       axis=3).astype(np.bool)[
                                                           :, :, :, order, :]

        if len(jnew_inds) > 0:
            # Exploit the fact that quality array has the same dimensions as the main data
            zero_pad_data = np.zeros(
                (this.quality_array.shape[0], this.Nspws, this.quality_array.shape[2],
                 this.quality_array.shape[3], len(jnew_inds)))
            zero_pad_flags = np.zeros(
                (this.flag_array.shape[0], this.Nspws, this.flag_array.shape[2],
                 this.flag_array.shape[3], len(jnew_inds)))
            this.jones_array = np.concatenate([this.jones_array, other.jones_array[jnew_inds]])
            order = np.argsort(np.abs(this.jones_array))
            this.jones_array = this.jones_array[order]
            if this.cal_type == 'delay':
                this.delay_array = np.concatenate([this.delay_array, zero_pad_data], axis=4)[
                    :, :, :, :, order]
            else:
                this.gain_array = np.concatenate([this.gain_array, zero_pad_data], axis=4)[
                    :, :, :, :, order]
            this.flag_array = np.concatenate([this.flag_array,
                                              1 - zero_pad_flags], axis=4).astype(np.bool)[
                                                  :, :, :, :, order]
            this.quality_array = np.concatenate([this.quality_array, zero_pad_data], axis=4)[
                :, :, :, :, order]

            if this.total_quality_array is not None and can_combine_tqa:
                zero_pad = np.zeros((this.Nspws, this.quality_array.shape[2],
                                     this.quality_array.shape[3], len(jnew_inds)))
                this.total_quality_array = np.concatenate([this.total_quality_array, zero_pad],
                                                          axis=3)[:, :, :, order]
            elif other.total_quality_array is not None and can_combine_tqa:
                zero_pad = np.zeros((this.Nspws, this.quality_array.shape[2],
                                     this.quality_array.shape[3], len(jnew_inds)))
                this.total_quality_array = np.zeros((this.Nspws, Nf_tqa, this.Ntimes, this.Njones))
                this.total_quality_array = np.concatenate([this.total_quality_array, zero_pad],
                                                          axis=3)[:, :, :, order]

            if this.input_flag_array is not None:
                zero_pad = np.zeros(
                    (this.input_flag_array.shape[0], this.Nspws,
                     this.input_flag_array.shape[2], this.input_flag_array.shape[3],
                     len(jnew_inds)))
                this.input_flag_array = np.concatenate(
                    [this.input_flag_array, 1 - zero_pad], axis=4).astype(np.bool)[
                        :, :, :, :, order]
            elif other.input_flag_array is not None:
                zero_pad = np.zeros(
                    (this.flag_array.shape[0], this.Nspws,
                     this.flag_array.shape[2], this.flag_array.shape[3],
                     len(jnew_inds)))
                this.input_flag_array = np.array(1 - np.zeros(
                    (this.flag_array.shape[0], this.Nspws,
                     this.flag_array.shape[2], this.flag_array.shape[3],
                     this.Njones))).astype(np.bool)
                this.input_flag_array = np.concatenate([this.input_flag_array,
                                                        1 - zero_pad],
                                                       axis=4).astype(np.bool)[
                                                           :, :, :, :, order]

        # Now populate the data
        jones_t2o = np.nonzero(
            np.in1d(this.jones_array, other.jones_array))[0]
        times_t2o = np.nonzero(
            np.in1d(this.time_array, other.time_array))[0]
        freqs_t2o = np.nonzero(
            np.in1d(this.freq_array[0, :], other.freq_array[0, :]))[0]
        ants_t2o = np.nonzero(
            np.in1d(this.ant_array, other.ant_array))[0]
        if this.cal_type == 'delay':
            this.delay_array[np.ix_(ants_t2o, [0], [0], times_t2o,
                                    jones_t2o)] = other.delay_array
            this.quality_array[np.ix_(ants_t2o, [0], [0], times_t2o,
                                      jones_t2o)] = other.quality_array
        else:
            this.gain_array[np.ix_(ants_t2o, [0], freqs_t2o, times_t2o,
                                   jones_t2o)] = other.gain_array
            this.quality_array[np.ix_(ants_t2o, [0], freqs_t2o, times_t2o,
                                      jones_t2o)] = other.quality_array
        this.flag_array[np.ix_(ants_t2o, [0], freqs_t2o, times_t2o,
                               jones_t2o)] = other.flag_array
        if this.total_quality_array is not None:
            if other.total_quality_array is not None:
                if this.cal_type == 'delay':
                    this.total_quality_array[np.ix_([0], [0], times_t2o,
                                                    jones_t2o)] = other.total_quality_array
                else:
                    this.total_quality_array[np.ix_([0], freqs_t2o, times_t2o,
                                                    jones_t2o)] = other.total_quality_array
        if this.input_flag_array is not None:
            if other.input_flag_array is not None:
                this.input_flag_array[np.ix_(ants_t2o, [0], freqs_t2o, times_t2o,
                                             jones_t2o)] = other.input_flag_array

        # Update N parameters (e.g. Npols)
        this.Njones = this.jones_array.shape[0]
        this.Ntimes = this.time_array.shape[0]
        if this.cal_type == 'gain':
            this.Nfreqs = this.freq_array.shape[1]
        this.Nants_data = len(
            np.unique(this.ant_array.tolist() + other.ant_array.tolist()))

        # Check specific requirements
        if this.cal_type == 'gain' and this.Nfreqs > 1:
            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                              rtol=this._freq_array.tols[0], atol=this._freq_array.tols[1]):
                warnings.warn('Combined frequencies are not evenly spaced. This will '
                              'make it impossible to write this data out to some file types.')
            elif np.max(freq_separation) > this.channel_width:
                warnings.warn('Combined frequencies are not contiguous. This will make '
                              'it impossible to write this data out to some file types.')

        if this.Njones > 2:
            jones_separation = np.diff(this.jones_array)
            if np.min(jones_separation) < np.max(jones_separation):
                warnings.warn('Combined Jones elements are not evenly spaced. This will '
                              'make it impossible to write this data out to some file types.')

        if n_axes > 0:
            history_update_string += ' axis using pyuvdata.'
            this.history += history_update_string

        this.history = uvutils._combine_histories(this.history, other.history)

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
            other: Another UVCal object which will be added to self.
        """
        self.__add__(other, inplace=True)
        return self
