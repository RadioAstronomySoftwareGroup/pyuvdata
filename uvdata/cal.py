# comment notes
# 1. Currently supporting direction independent gains. Future versions may
#   support direction dependent gains with list of such objects.
# 2.
import numpy as np
from uvbase import UVBase
import parameter as uvp


class UVCal(UVBase):
    """ A class defining a calibration """
    def __init__(self):
        self._Nfreqs = uvp.UVParameter('Nfreqs',
                                       description='Number of frequency channels',
                                       expected_type=int)
        self._Npols = uvp.UVParameter('Npols',
                                      description='Number of polarizations',
                                      expected_type=int)
        self._Ntimes = uvp.UVParameter('Ntimes',
                                       description='Number of times',
                                       expected_type=int)
        self._history = uvp.UVParameter('history',
                                        description='String of history, units English',
                                        form='str', expected_type=str)

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

        desc = ('List of integer antenna numbers corresponding to'
                'antenna_names,'
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

        desc = ('Array of polarization integers, shape (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)')
        self._polarization_array = uvp.UVParameter('polarization_array',
                                                   description=desc,
                                                   expected_type=int,
                                                   sane_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
                                                   form=('Npols',))

        desc = ('Array of times, center of integration, shape (Nblts), ' +
                'units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Nblts',),
                                           expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))

        desc = ('The convention for applying he calibration solutions to data.'
                'Divide/multiply uncalibrated data by gains.')
        self._gain_convention = uvp.UVParameter('gain_convention', form='str',
                                                expected_type=str,
                                                description=desc,
                                                sane_vals=['divide', 'multiply'])

        desc = ('Array of flags, True is flagged.'
                'shape: (Nants_data, Nfreqs, Ntimes, Npols), type = bool.')
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nants_data', 'Nfreqs',
                                                 'Ntimes', 'Npols'),
                                           expected_type=np.bool)

        desc = ('Array of qualities, shape: (Nants_data, Nfreqs, Ntimes, '
                'Npols), type = float.')
        self._quality_array = uvp.UVParameter('quality_array', description=desc,
                                              form=('Nants_data', 'Nfreqs',
                                                    'Ntimes', 'Npols'),
                                              expected_type=np.float)
        desc = ('Delay or gain switch parameter. Values are delay/gain.')
        self._delay_gain_switch = uvp.UVParameter('delay_gain_switch', form='str',
                                                  expected_type=str,
                                                  description=desc,
                                                  sane_vals=['delay', 'gain'])

        desc = ('Polarization orientation. Values are E/N/Unknown.')
        self._x_orientation = uvp.UVParameter('x_orientation', description=desc,
                                              expected_type=str,
                                              sane_vals=['E', 'N', 'U'])
# --- optional parameters ---
        desc = ('Array of gains, shape: (Nants_data, Nfreqs, Ntimes, '
                'Npols), type = complex float.')
        self._gain_array = uvp.UVParameter('gain_array', description=desc,
                                           required=False,
                                           form=('Nants_data', 'Nfreqs',
                                                 'Ntimes', 'Npols'),
                                           expected_type=np.complex)

        desc = ('Array of delays. shape= (Nants_data, Ntimes, Npols), type = float')
        self._delay_array = uvp.UVParameter('delay_array', description=desc,
                                            required=False,
                                            form=('Nants_data', 'Ntimes',
                                                  'Npols'),
                                            expected_type=np.float)

# --- truly optional parameters ---
        desc = ('Array of input flags, True is flagged. shape: (Nants_data, '
                'Nfreqs, Ntimes, Npols), type = bool.')
        self._input_flag_array = uvp.UVParameter('input_flag_array',
                                                 description=desc,
                                                 required=False,
                                                 form=('Nants_data', 'Nfreqs',
                                                       'Ntimes', 'Npols'),
                                                 expected_type=np.bool)
        super(UVCal, self).__init__()
        # need to have either gain_array or delay_array
