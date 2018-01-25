from scipy.io.idl import readsav
import numpy as np
import warnings
from uvcal import UVCal


class FHD_cal(UVCal):
    """
    Defines a FHD-specific subclass of UVCal for reading FHD calibration save files.
    This class should not be interacted with directly, instead use the read_fhd_cal
    method on the UVCal class.
    """

    def read_fhd_cal(self, filename, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Read data from an FHD cal.sav file.

        Args:
            filename: The cal.save file to read to.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """

        this_dict = readsav(filename, python_dict=True)
        cal_data = this_dict['cal']

        self.Nfreqs = cal_data['N_FREQ'][0]
        self.freq_array = cal_data['FREQ'][0]
        self.channel_width = self.freq_array[1] - self.freq_array[0]
        # FHD only calculates one calibration over all the times.
        # cal_data.n_times gives the number of times that goes into that one
        # calibration, UVCal.Ntimes gives the number of separate calibrations
        # along the time axis.
        self.Ntimes = 1
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.Njones = cal_data['N_POL'][0]

        self.Nants_data = cal_data['N_TILE'][0]
        self.Nants_telescope = cal_data['N_TILE'][0]
        self.antenna_names = cal_data.['TILE_NAMES'][0]
        self.antenna_numbers = np.arange(self.Nants_telescope)
        self.ant_array = np.arange(self.Nants_data)

        self.gain_convention = 'divide'
        self.x_orientation = 'east'

        self.cal_type = 'gain'
        fit_gain_array_in = cal_data['GAIN'][0]
        fit_gain_array = np.zeros(self._gain_array.expected_shape, dtype=np.complex_)
        for jones_i, arr in enumerate(fit_gain_array_in):
            fit_gain_array[:, 0, :, 0, jones_i] = arr
        res_gain_array_in = cal_data['GAIN_RESIDUAL'][0]
        res_gain_array = np.zeros(self._gain_array.expected_shape, dtype=np.complex_)
        for jones_i, arr in enumerate(fit_gain_array_in):
            res_gain_array[:, 0, :, 0, jones_i] = arr
        self.gain_array = fit_gain_array + res_gain_array

        # I'm not sure how to get these yet
        # self.time_range  # list of [start time, stop time] in JD
        # self.history  # anything we should preserve?
        # self.telescope_name
        # self.jones_array  # this gives the ordering along the polarization axis of the gains
        # self.integration_time  # I can generate this if I have a time array...
        # self.flag_array  # this has axes for antenna, frequency and polarization and is meant to represent flags after calibration
        # self.quality_array  # this has axes for antenna, frequency and polarization and is usually set to a chi^2 like measure
        # --- these are optional ----
        # self.input_flag_array   # this has axes for antenna, frequency and polarization and is meant to represent flags before calibration
        # self.git_origin_cal  # (url & branch)
        # self.git_hash_cal
        # self.observer  # person who ran the code?
        # self.total_quality_array  # this has axes for frequency and polarization and represents global chi^2 stuff
        # self.extra_keywords  # anything we think is important and should be preserved
