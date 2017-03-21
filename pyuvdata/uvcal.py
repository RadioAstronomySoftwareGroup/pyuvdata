import numpy as np
from uvbase import UVBase
import parameter as uvp


class UVCal(UVBase):
    """ A class defining a calibration """
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

        desc = ('Frequency range that gain solutions are valid for.',
                'list: (start_frequency, end_frequency) in Hz.')
        self._freq_range = uvp.UVParameter('freq_range',
                                           description=desc,
                                           form=(2,),
                                           expected_type=float)

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
                'entries in ant_1_array and ant_2_array). May be smaller ' +
                'than the number of antennas in the array')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc,
                                           expected_type=int)

        desc = ('List of antenna names, shape (Nants_telescope), '
                'with numbers given by antenna_numbers (which can be matched '
                'to ant_1_array and ant_2_array). There must be one entry '
                'here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_names = uvp.UVParameter('antenna_names',
                                              description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('List of integer antenna numbers corresponding to antenna_names,'
                'shape (Nants_telescope). There must be one '
                'entry here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers',
                                                description=desc,
                                                form=('Nants_telescope',),
                                                expected_type=int)

        desc = ('Number of antennas in the array. May be larger ' +
                'than the number of antennas with data')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope',
                                                description=desc,
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

        desc = ('Array of flags to be applied to calibrated data (logical OR \
                 of input and flag generated by calibration). True is flagged.'
                'shape: (Nants_data, Nfreqs, Ntimes, Njones), type = bool.')
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nants_data', 'Nfreqs',
                                                 'Ntimes', 'Njones'),
                                           expected_type=np.bool)

        desc = ('Array of qualities of calibration solutions \
                shape: (Nants_data, Nfreqs, Ntimes, '
                'Njones), type = float.')
        self._quality_array = uvp.UVParameter('quality_array', description=desc,
                                              form=('Nants_data', 'Nfreqs',
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

        desc = ('Array of gains, shape: (Nants_data, Nfreqs, Ntimes, '
                'Njones), type = complex float.')
        self._gain_array = uvp.UVParameter('gain_array', description=desc,
                                           required=False,
                                           form=('Nants_data', 'Nfreqs',
                                                 'Ntimes', 'Njones'),
                                           expected_type=np.complex)

        desc = ('Array of delays. shape: (Nants_data, Ntimes, Njones), type = float')
        self._delay_array = uvp.UVParameter('delay_array', description=desc,
                                            required=False,
                                            form=('Nants_data', 1, 'Ntimes', 'Njones'),
                                            expected_type=np.float)

        # --- truly optional parameters ---
        desc = ('Array of input flags, True is flagged. shape: (Nants_data, '
                'Nfreqs, Ntimes, Njones), type = bool.')
        self._input_flag_array = uvp.UVParameter('input_flag_array',
                                                 description=desc,
                                                 required=False,
                                                 form=('Nants_data', 'Nfreqs',
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

        super(UVCal, self).__init__()

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
        super(UVCal, self).check(run_check_acceptability=run_check_acceptability)

        # then check some other things
        nants_data_calc = int(len(np.unique(self.antenna_numbers)))
        if self.Nants_data != nants_data_calc:
            raise ValueError('Nants_data must be equal to the number of unique '
                             'values antenna_numbers.')

        if self.Nants_data > self.Nants_telescope:
            raise ValueError('Nants_data must be less than or equal to Nants_telescope')
        return True

    def set_gain(self):
        """Set cal_type to 'gain' and adjust required parameters."""
        self.cal_type = 'gain'
        self._gain_array.required = True
        self._delay_array.required = False
        self._quality_array.form = self._gain_array.form

    def set_delay(self):
        """Set cal_type to 'delay' and adjust required parameters."""
        self.cal_type = 'delay'
        self._gain_array.required = False
        self._delay_array.required = True
        self._quality_array.form = self._delay_array.form

    def set_unknown_cal_type(self):
        """Set cal_type to 'unknown' and adjust required parameters."""
        self.cal_type = 'unknown'
        self._gain_array.required = False
        self._delay_array.required = False
        self._quality_array.form = self.gain_array.form

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

    def write_calfits(self, filename, spoof_nonessential=False,
                      run_check=True, run_check_acceptability=True, clobber=False):
        """Write data to a calfits file.

        Args:
            filename: The calfits filename to write to.
            spoof_nonessential: Option to spoof the values of optional
                UVParameters that are not set but are required for uvfits files.
                Default is False.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptability of the values of
                required parameters before writing the file. Default is True.
            clobber: Overwrite file.
        """
        calfits_obj = self._convert_to_filetype('calfits')
        calfits_obj.write_calfits(filename,
                                  spoof_nonessential=spoof_nonessential,
                                  run_check=run_check,
                                  run_check_acceptability=run_check_acceptability,
                                  clobber=clobber)
        del(calfits_obj)
