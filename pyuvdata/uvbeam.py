"""Primary container for radio telescope antenna beams."""
import numpy as np
import warnings
import copy
from uvbase import UVBase
import parameter as uvp
import pyuvdata.utils as uvutils


class UVBeam(UVBase):
    """
    A class for defining a radio telescope antenna beam.

    Attributes:
        UVParameter objects: For full list see UVBeam Parameters
            (http://pyuvdata.readthedocs.io/en/latest/uvbeam_parameters.html).
            Some are always required, some are required for certain beam_types,
            antenna_types and pixel_coordinate_systems and others are always optional.
    """

    coordinate_system_dict = {
        # uniformly gridded az/za coordinate system, az runs from East to North
        'az_za': ['az', 'za'],
        # sine projection at zenith. y points North, x point East
        'sin_zenith': ['sin_x', 'sin_y'],
        # HEALPix map with zenith at north pole and
        # az, za coordinate axes (for efield) where az runs from East to North.
        'healpix': ['hpx_inds']}

    def __init__(self):
        """Create a new UVBeam object."""
        # add the UVParameters to the class
        self._Nfreqs = uvp.UVParameter('Nfreqs', description='Number of frequency channels',
                                       expected_type=int)

        self._Nspws = uvp.UVParameter('Nspws', description='Number of spectral windows '
                                      '(ie non-contiguous spectral chunks). '
                                      'More than one spectral window is not '
                                      'currently supported.', expected_type=int)

        desc = ('Number of directions in vector_coordinate_system, options '
                'are 2 or 3 (or 1 if beam_type is "power")')
        self._Naxes_vec = uvp.UVParameter('Naxes_vec', description=desc,
                                          expected_type=int, acceptable_vals=[2, 3])

        desc = ('Pixel coordinate system, options are: ' +
                ', '.join(self.coordinate_system_dict.keys()))
        self._pixel_coordinate_system = uvp.UVParameter('pixel_coordinate_system',
                                                        description=desc, form='str',
                                                        expected_type=str,
                                                        acceptable_vals=self.coordinate_system_dict.keys())

        desc = ('Number of elements along the first pixel axis. '
                'Not required if pixel_coordinate_system is "healpix".')
        self._Naxes1 = uvp.UVParameter('Naxes1', description=desc, expected_type=int,
                                       required=False)

        desc = ('Coordinates along first pixel axis. '
                'Not required if pixel_coordinate_system is "healpix".')
        self._axis1_array = uvp.UVParameter('axis1_array', description=desc,
                                            expected_type=np.float,
                                            required=False, form=('Naxes1',))

        desc = ('Number of elements along the second pixel axis. '
                'Not required if pixel_coordinate_system is "healpix".')
        self._Naxes2 = uvp.UVParameter('Naxes2', description=desc, expected_type=int,
                                       required=False)

        desc = ('Coordinates along second pixel axis. '
                'Not required if pixel_coordinate_system is "healpix".')
        self._axis2_array = uvp.UVParameter('axis2_array', description=desc,
                                            expected_type=np.float,
                                            required=False, form=('Naxes2',))

        desc = ('Healpix nside parameter. Only required if pixel_coordinate_system is "healpix".')
        self._nside = uvp.UVParameter('nside', description=desc, expected_type=int,
                                      required=False)

        desc = ('Healpix ordering parameter, allowed values are "ring" and "nested". '
                'Only required if pixel_coordinate_system is "healpix".')
        self._ordering = uvp.UVParameter('ordering', description=desc, expected_type=str,
                                         required=False, acceptable_vals=['ring', 'nested'])

        desc = ('Number of healpix pixels. Only required if pixel_coordinate_system is "healpix".')
        self._Npixels = uvp.UVParameter('Npixels', description=desc, expected_type=int,
                                        required=False)

        desc = ('Healpix pixel numbers. Only required if pixel_coordinate_system is "healpix".')
        self._pixel_array = uvp.UVParameter('pixel_array', description=desc, expected_type=int,
                                            required=False, form=('Npixels',))

        desc = 'String indicating beam type. Allowed values are "efield", and "power".'
        self._beam_type = uvp.UVParameter('beam_type', description=desc, form='str',
                                          expected_type=str,
                                          acceptable_vals=['efield', 'power'])

        desc = ('Beam basis vector components -- directions for which the '
                'electric field values are recorded in the pixel coordinate system. '
                'Not required if beam_type is "power". The shape depends on the '
                'pixel_coordinate_system, if it is "healpix", the shape is: '
                '(Naxes_vec, 2, Npixels), otherwise it is (Naxes_vec, 2, Naxes1, Naxes2)')
        self._basis_vector_array = uvp.UVParameter('basis_vector_array',
                                                   description=desc, required=False,
                                                   expected_type=np.float,
                                                   form=('Naxes_vec', 2, 'Naxes2', 'Naxes1'),
                                                   acceptable_range=(0, 1),
                                                   tols=1e-3)

        self._Nfeeds = uvp.UVParameter('Nfeeds', description='Number of feeds. '
                                       'Not required if beam_type is "power".',
                                       expected_type=int, acceptable_vals=[1, 2],
                                       required=False)

        desc = ('Array of feed orientations. shape (Nfeeds). '
                'options are: N/E or x/y or R/L. Not required if beam_type is "power".')
        self._feed_array = uvp.UVParameter('feed_array', description=desc, required=False,
                                           expected_type=str, form=('Nfeeds',),
                                           acceptable_vals=['N', 'E', 'x', 'y',
                                                            'R', 'L'])

        self._Npols = uvp.UVParameter('Npols', description='Number of polarizations. '
                                      'Only required if beam_type is "power".',
                                      expected_type=int, required=False)

        desc = ('Array of polarization integers, shape (Npols). '
                'AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  '
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX). '
                'Only required if beam_type is "power".')
        self._polarization_array = uvp.UVParameter('polarization_array',
                                                   description=desc, required=False,
                                                   expected_type=int, form=('Npols',),
                                                   acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)))

        desc = 'Array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        self._spw_array = uvp.UVParameter('spw_array',
                                          description='Array of spectral window '
                                          'Numbers, shape (Nspws)', form=('Nspws',),
                                          expected_type=int)

        desc = ('Normalization standard of data_array, options are: '
                '"physical", "peak" or "solid_angle". Physical normalization '
                'means that the frequency dependence of the antenna sensitivity '
                'is included in the data_array while the frequency dependence '
                'of the receiving chain is included in the bandpass_array. '
                'Peak normalized means that for each frequency the data_array'
                'is separately normalized such that the peak is 1 (so the beam '
                'is dimensionless) and all frequency dependence is moved to the '
                'bandpass_array. Solid angle normalized means the peak normalized '
                'beam is divided by the integral of the beam over the sphere, '
                'so the beam has dimensions of 1/stradian.')
        self._data_normalization = uvp.UVParameter('data_normalization', description=desc,
                                                   form='str', expected_type=str,
                                                   acceptable_vals=["peak", "solid_angle"])

        desc = ('Depending on beam type, either complex E-field values ("efield" beam type) '
                'or power values ("power" beam type) for beam model. units are linear '
                'normalized to either peak or solid angle as given by data_normalization. '
                'The shape depends on the beam_type and pixel_coordinate_system, if it is '
                '"healpix", the shape is: (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, Npixels), '
                'otherwise it is (Naxes_vec, Nspws, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)')
        self._data_array = uvp.UVParameter('data_array', description=desc,
                                           expected_type=np.complex,
                                           form=('Naxes_vec', 'Nspws', 'Nfeeds',
                                                 'Nfreqs', 'Naxes2', 'Naxes1'),
                                           acceptable_range=(0, 1),
                                           tols=1e-3)

        desc = ('Frequency dependence of the beam. Depending on the data_normalization, '
                'this may contain only the frequency dependence of the receiving '
                'chain ("physical" normalization) or all the frequency dependence '
                '("peak" normalization).')
        self._bandpass_array = uvp.UVParameter('bandpass_array', description=desc,
                                               expected_type=np.float,
                                               form=('Nspws', 'Nfreqs'),
                                               tols=1e-3)

        # --------- metadata -------------
        self._telescope_name = uvp.UVParameter('telescope_name',
                                               description='Name of telescope '
                                               '(string)', form='str',
                                               expected_type=str)

        self._feed_name = uvp.UVParameter('feed_name',
                                          description='Name of physical feed '
                                          '(string)', form='str',
                                          expected_type=str)

        self._feed_version = uvp.UVParameter('feed_version',
                                             description='Version of physical feed '
                                             '(string)', form='str',
                                             expected_type=str)

        self._model_name = uvp.UVParameter('model_name',
                                           description='Name of beam model '
                                           '(string)', form='str',
                                           expected_type=str)

        self._model_version = uvp.UVParameter('model_version',
                                              description='Version of beam model '
                                              '(string)', form='str',
                                              expected_type=str)

        self._history = uvp.UVParameter('history', description='String of history, units English',
                                        form='str', expected_type=str)

        # ---------- phased_array stuff -------------
        desc = ('String indicating antenna type. Allowed values are "simple", and '
                '"phased_array"')
        self._antenna_type = uvp.UVParameter('antenna_type', form='str', expected_type=str,
                                             description=desc,
                                             acceptable_vals=['simple', 'phased_array'])

        desc = ('Required if antenna_type = "phased_array". Number of elements '
                'in phased array')
        self._Nelements = uvp.UVParameter('Nelements', required=False,
                                          description=desc, expected_type=int)

        desc = ('Required if antenna_type = "phased_array". Element coordinate '
                'system, options are: N-E or x-y')
        self._element_coordinate_system = \
            uvp.UVParameter('element_coordinate_system', required=False,
                            description=desc, expected_type=str,
                            acceptable_vals=['N-E', 'x-y'])

        desc = ('Required if antenna_type = "phased_array". Array of element '
                'locations in element coordinate system,  shape: (2, Nelements)')
        self._element_location_array = uvp.UVParameter('element_location_array',
                                                       required=False,
                                                       description=desc,
                                                       form=('2', 'Nelements'),
                                                       expected_type=np.float)

        desc = ('Required if antenna_type = "phased_array". Array of element '
                'delays, units: seconds, shape: (Nelements)')
        self._delay_array = uvp.UVParameter('delay_array', required=False,
                                            description=desc,
                                            form=('Nelements',),
                                            expected_type=np.float)

        desc = ('Required if antenna_type = "phased_array". Array of element '
                'gains, units: dB, shape: (Nelements)')
        self._gain_array = uvp.UVParameter('gain_array', required=False,
                                           description=desc,
                                           form=('Nelements',),
                                           expected_type=np.float)

        desc = ('Required if antenna_type = "phased_array". Matrix of complex '
                'element couplings, units: dB, '
                'shape: (Nelements, Nelements, Nfeed, Nfeed, Nspws, Nfreqs)')
        self._coupling_matrix = uvp.UVParameter('coupling_matrix', required=False,
                                                description=desc,
                                                form=('Nelements', 'Nelements',
                                                      'Nfeed', 'Nfeed', 'Nspws', 'Nfreqs'),
                                                expected_type=np.complex)

        # -------- extra, non-required parameters ----------
        desc = ('Any user supplied extra keywords, type=dict')
        self._extra_keywords = uvp.UVParameter('extra_keywords', required=False,
                                               description=desc, value={},
                                               spoof_val={}, expected_type=dict)

        desc = 'Input impedence of receiving chain, units: Ohms'
        self._input_impedence = uvp.UVParameter('input_impedence', required=False,
                                                description=desc,
                                                expected_type=np.float, tols=1e-3)

        desc = 'Output impedence of receiving chain, units: Ohms'
        self._output_impedence = uvp.UVParameter('output_impedence', required=False,
                                                 description=desc,
                                                 expected_type=np.float, tols=1e-3)

        desc = 'Array of receiver temperatures, shape (Nspws, Nfreqs), units K'
        self._receiver_temperature_array = \
            uvp.UVParameter('receiver_temperature_array', required=False,
                            description=desc, form=('Nspws', 'Nfreqs'),
                            expected_type=np.float, tols=1e-3)

        desc = 'Array of antenna losses, shape (Nspws, Nfreqs), units dB?'
        self._loss_array = uvp.UVParameter('loss_array', required=False,
                                           description=desc, form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)

        desc = 'Array of antenna-amplifier mismatches, shape (Nspws, Nfreqs), units ?'
        self._mismatch_array = uvp.UVParameter('mismatch_array', required=False,
                                               description=desc,
                                               form=('Nspws', 'Nfreqs'),
                                               expected_type=np.float,
                                               tols=1e-3)

        desc = ('S parameters of receiving chain, shape (4, Nspws, Nfreqs), '
                'ordering: s11, s12, s21, s22. units ?')
        self._s_parameters = uvp.UVParameter('s_parameters', required=False,
                                             description=desc,
                                             form=(4, 'Nspws', 'Nfreqs'),
                                             expected_type=np.float,
                                             tols=1e-3)

        super(UVBeam, self).__init__()

    def check(self, run_check_acceptability=True):
        """
        Check that all required parameters are set reasonably.

        Check that required parameters exist and have appropriate shapes.
        Optionally check if the values are acceptable.

        Args:
            run_check_acceptability: Option to check if values in required parameters
                are acceptable. Default is True.
        """
        # first make sure the required parameters and forms are set properly
        # for the pixel_coordinate_system
        self.set_cs_params()

        # first run the basic check from UVBase
        super(UVBeam, self).check(run_check_acceptability=run_check_acceptability)

        return True

    def set_cs_params(self):
        """
        Set various forms and required parameters depending on pixel_coordinate_system.
        """
        if self.pixel_coordinate_system == 'healpix':
            self._Naxes1.required = False
            self._axis1_array.required = False
            self._Naxes2.required = False
            self._axis2_array.required = False
            self._nside.required = True
            self._ordering.required = True
            self._Npixels.required = True
            self._pixel_array.required = True
            self._basis_vector_array.form = ('Naxes_vec', 2, 'Npixels')
            if self.beam_type == "power":
                self._data_array.form = ('Naxes_vec', 'Nspws', 'Npols', 'Nfreqs',
                                         'Npixels')
            else:
                self._data_array.form = ('Naxes_vec', 'Nspws', 'Nfeeds', 'Nfreqs',
                                         'Npixels')
        else:
            self._Naxes1.required = True
            self._axis1_array.required = True
            self._Naxes2.required = True
            self._axis2_array.required = True
            self._nside.required = False
            self._ordering.required = False
            self._Npixels.required = False
            self._pixel_array.required = False
            self._basis_vector_array.form = ('Naxes_vec', 2, 'Naxes2', 'Naxes1')
            if self.beam_type == "power":
                self._data_array.form = ('Naxes_vec', 'Nspws', 'Npols', 'Nfreqs',
                                         'Naxes2', 'Naxes1')
            else:
                self._data_array.form = ('Naxes_vec', 'Nspws', 'Nfeeds', 'Nfreqs',
                                         'Naxes2', 'Naxes1')

    def set_efield(self):
        """Set beam_type to 'efield' and adjust required parameters."""
        self.beam_type = 'efield'
        self._Naxes_vec.acceptable_vals = [2, 3]
        self._basis_vector_array.required = True
        self._Nfeeds.required = True
        self._feed_array.required = True
        self._Npols.required = False
        self._polarization_array.required = False
        self._data_array.expected_type = np.complex
        # call set_cs_params to fix data_array form
        self.set_cs_params()

    def set_power(self):
        """Set beam_type to 'power' and adjust required parameters."""
        self.beam_type = 'power'
        self._Naxes_vec.acceptable_vals = [1, 2, 3]
        self._basis_vector_array.required = False
        self._Nfeeds.required = False
        self._feed_array.required = False
        self._Npols.required = True
        self._polarization_array.required = True
        self._data_array.expected_type = np.float
        # call set_cs_params to fix data_array form
        self.set_cs_params()

    def set_simple(self):
        """Set antenna_type to 'simple' and adjust required parameters."""
        self.antenna_type = 'simple'
        self._Nelements.required = False
        self._element_coordinate_system.required = False
        self._element_location_array.required = False
        self._delay_array.required = False
        self._gain_array.required = False
        self._coupling_matrix.required = False

    def set_phased_array(self):
        """Set antenna_type to 'phased_array' and adjust required parameters."""
        self.antenna_type = 'phased_array'
        self._Nelements.required = True
        self._element_coordinate_system.required = True
        self._element_location_array.required = True
        self._delay_array.required = True
        self._gain_array.required = True
        self._coupling_matrix.required = True

    def __add__(self, other, run_check=True, run_check_acceptability=True, inplace=False):
        """
        Combine two UVBeam objects. Objects can be added along frequency,
        feed or polarization (for efield or power beams), and/or pixel axes.

        Args:
            other: Another UVBeam object which will be added to self.
            run_check: Option to check for the existence and proper shapes of
                required parameters after combining objects. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after combining objects. Default is True.
            inplace: Overwrite self as we go, otherwise create a third object
                as the sum of the two (default).
        """
        if inplace:
            this = self
        else:
            this = copy.deepcopy(self)
        # Check that both objects are UVBeam and valid
        this.check(run_check_acceptability=False)
        if not isinstance(other, this.__class__):
            raise(ValueError('Only UVBeam objects can be added to a UVBeam object'))
        other.check(run_check_acceptability=False)

        # Check objects are compatible
        compatibility_params = ['_beam_type', '_data_normalization', '_telescope_name',
                                '_feed_name', '_feed_version', '_model_name',
                                '_model_version', '_pixel_coordinate_system',
                                '_Naxes_vec', '_nside', '_ordering']
        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = 'UVParameter ' + \
                    a[1:] + ' does not match. Cannot combine objects.'
                raise(ValueError(msg))

        # check for presence of optional parameters with a frequency axis in both objects
        optional_freq_params = ['_receiver_temperature_array', '_loss_array',
                                '_mismatch_array', '_s_parameters']
        for attr in optional_freq_params:
            this_attr = getattr(this, attr)
            other_attr = getattr(other, attr)
            if (this_attr.value is None or other_attr.value is None) and this_attr != other_attr:
                warnings.warn('Only one of the UVBeam objects being combined '
                              'has optional parameter {attr}. After the sum the '
                              'final object will not have {attr}'.format(attr=attr))
                if this_attr.value is not None:
                    this_attr.value = None
                    setattr(this, attr, this_attr)

        # Build up history string
        history_update_string = ' Combined data along '
        n_axes = 0

        # Check we don't have overlapping data
        if this.beam_type == 'power':
            both_pol = np.intersect1d(this.polarization_array,
                                      other.polarization_array)
        else:
            both_pol = np.intersect1d(this.feed_array, other.feed_array)

        both_freq = np.intersect1d(this.freq_array[0, :], other.freq_array[0, :])

        if this.pixel_coordinate_system == 'healpix':
            both_pixels = np.intersect1d(this.pixel_array, other.pixel_array)
        else:
            both_axis1 = np.intersect1d(this.axis1_array, other.axis1_array)
            both_axis2 = np.intersect1d(this.axis2_array, other.axis2_array)

        if len(both_pol) > 0:
            if len(both_freq) > 0:
                if self.pixel_coordinate_system == 'healpix':
                    if len(both_pixels) > 0:
                        raise(ValueError('These objects have overlapping data and'
                                         ' cannot be combined.'))
                else:
                    if len(both_axis1) > 0:
                        if len(both_axis2) > 0:
                            raise(ValueError('These objects have overlapping data and'
                                             ' cannot be combined.'))

        if this.pixel_coordinate_system == 'healpix':
            temp = np.nonzero(~np.in1d(other.pixel_array, this.pixel_array))[0]
            if len(temp) > 0:
                pix_new_inds = temp
                new_pixels = other.pixel_array[temp]
                history_update_string += 'healpix pixel'
                n_axes += 1
            else:
                pix_new_inds, new_pixels = ([], [])
        else:
            temp = np.nonzero(~np.in1d(other.axis1_array, this.axis1_array))[0]
            if len(temp) > 0:
                ax1_new_inds = temp
                new_ax1 = other.axis1_array[temp]
                history_update_string += 'first image'
                n_axes += 1
            else:
                ax1_new_inds, new_ax1 = ([], [])

            temp = np.nonzero(~np.in1d(other.axis2_array, this.axis2_array))[0]
            if len(temp) > 0:
                ax2_new_inds = temp
                new_ax2 = other.axis2_array[temp]
                if n_axes > 0:
                    history_update_string += ', second image'
                else:
                    history_update_string += 'second image'
                n_axes += 1
            else:
                ax2_new_inds, new_ax2 = ([], [])

        temp = np.nonzero(~np.in1d(other.freq_array[0, :],
                                   this.freq_array[0, :]))[0]
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

        if this.beam_type == 'power':
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
        else:
            temp = np.nonzero(~np.in1d(other.feed_array,
                                       this.feed_array))[0]
            if len(temp) > 0:
                pnew_inds = temp
                new_pols = other.feed_array[temp]
                if n_axes > 0:
                    history_update_string += ', feed'
                else:
                    history_update_string += 'feed'
                n_axes += 1
            else:
                pnew_inds, new_pols = ([], [])

        # Pad out self to accommodate new data
        if this.pixel_coordinate_system == 'healpix':
            if len(pix_new_inds) > 0:
                data_pix_axis = 4
                data_pad_dims = tuple(list(this.data_array.shape[0:data_pix_axis]) +
                                      [len(pix_new_inds)] +
                                      list(this.data_array.shape[data_pix_axis + 1:]))
                data_zero_pad = np.zeros(data_pad_dims)

                this.pixel_array = np.concatenate([this.pixel_array,
                                                  other.pixel_array[pix_new_inds]])
                order = np.argsort(this.pixel_array)
                this.pixel_array = this.pixel_array[order]

                this.data_array = np.concatenate([this.data_array, data_zero_pad],
                                                 axis=data_pix_axis)[:, :, :, :, order]

                if this.beam_type == 'efield':
                    basisvec_pix_axis = 2
                    basisvec_pad_dims = tuple(list(this.basis_vector_array.shape[0:basisvec_pix_axis]) +
                                              [len(pix_new_inds)] +
                                              list(this.basis_vector_array.shape[basisvec_pix_axis + 1:]))
                    basisvec_zero_pad = np.zeros(basisvec_pad_dims)

                    this.basis_vector_array = np.concatenate([this.basis_vector_array,
                                                             basisvec_zero_pad],
                                                             axis=basisvec_pix_axis)[:, :, order]
        else:
            if len(ax1_new_inds) > 0:
                data_ax1_axis = 5
                data_pad_dims = tuple(list(this.data_array.shape[0:data_ax1_axis]) +
                                      [len(ax1_new_inds)] +
                                      list(this.data_array.shape[data_ax1_axis + 1:]))
                data_zero_pad = np.zeros(data_pad_dims)

                this.axis1_array = np.concatenate([this.axis1_array,
                                                   other.axis1_array[ax1_new_inds]])
                order = np.argsort(this.axis1_array)
                this.axis1_array = this.axis1_array[order]
                this.data_array = np.concatenate([this.data_array, data_zero_pad],
                                                 axis=data_ax1_axis)[:, :, :, :, :, order]

                if this.beam_type == 'efield':
                    basisvec_ax1_axis = 3
                    basisvec_pad_dims = tuple(list(this.basis_vector_array.shape[0:basisvec_ax1_axis]) +
                                              [len(ax1_new_inds)] +
                                              list(this.basis_vector_array.shape[basisvec_ax1_axis + 1:]))
                    basisvec_zero_pad = np.zeros(basisvec_pad_dims)

                    this.basis_vector_array = np.concatenate([this.basis_vector_array, basisvec_zero_pad],
                                                             axis=basisvec_ax1_axis)[:, :, :, order]

            if len(ax2_new_inds) > 0:
                data_ax2_axis = 4
                data_pad_dims = tuple(list(this.data_array.shape[0:data_ax2_axis]) +
                                      [len(ax2_new_inds)] +
                                      list(this.data_array.shape[data_ax2_axis + 1:]))
                data_zero_pad = np.zeros(data_pad_dims)

                this.axis2_array = np.concatenate([this.axis2_array,
                                                   other.axis2_array[ax2_new_inds]])
                order = np.argsort(this.axis2_array)
                this.axis2_array = this.axis2_array[order]

                this.data_array = np.concatenate([this.data_array, data_zero_pad],
                                                 axis=data_ax2_axis)[:, :, :, :, order, ...]

                if this.beam_type == 'efield':
                    basisvec_ax2_axis = 2
                    basisvec_pad_dims = tuple(list(this.basis_vector_array.shape[0:basisvec_ax2_axis]) +
                                              [len(ax2_new_inds)] +
                                              list(this.basis_vector_array.shape[basisvec_ax2_axis + 1:]))
                    basisvec_zero_pad = np.zeros(basisvec_pad_dims)

                    this.basis_vector_array = np.concatenate([this.basis_vector_array, basisvec_zero_pad],
                                                             axis=basisvec_ax2_axis)[:, :, order, ...]

        if len(fnew_inds) > 0:
            faxis = 3
            data_pad_dims = tuple(list(this.data_array.shape[0:faxis]) +
                                  [len(fnew_inds)] +
                                  list(this.data_array.shape[faxis + 1:]))
            data_zero_pad = np.zeros(data_pad_dims)

            this.freq_array = np.concatenate([this.freq_array,
                                              other.freq_array[:, fnew_inds]], axis=1)
            order = np.argsort(this.freq_array[0, :])
            this.freq_array = this.freq_array[:, order]

            this.bandpass_array = np.concatenate([this.bandpass_array,
                                                  np.zeros((1, len(fnew_inds)))],
                                                 axis=1)[:, order]

            this.data_array = np.concatenate([this.data_array, data_zero_pad],
                                             axis=faxis)[:, :, :, order, ...]
            if this.receiver_temperature_array is not None:
                this.receiver_temperature_array = np.concatenate([this.receiver_temperature_array,
                                                                  np.zeros((1, len(fnew_inds)))],
                                                                 axis=1)[:, order]
            if this.loss_array is not None:
                this.loss_array = np.concatenate([this.loss_array,
                                                  np.zeros((1, len(fnew_inds)))],
                                                 axis=1)[:, order]
            if this.mismatch_array is not None:
                this.mismatch_array = np.concatenate([this.mismatch_array,
                                                      np.zeros((1, len(fnew_inds)))],
                                                     axis=1)[:, order]
            if this.s_parameters is not None:
                this.s_parameters = np.concatenate([this.s_parameters,
                                                    np.zeros((4, 1, len(fnew_inds)))],
                                                   axis=2)[:, :, order]

        if len(pnew_inds) > 0:
            paxis = 2
            data_pad_dims = tuple(list(this.data_array.shape[0:paxis]) +
                                  [len(pnew_inds)] +
                                  list(this.data_array.shape[paxis + 1:]))
            data_zero_pad = np.zeros(data_pad_dims)

            if this.beam_type == 'power':
                this.polarization_array = np.concatenate([this.polarization_array,
                                                          other.polarization_array[pnew_inds]])
                order = np.argsort(np.abs(this.polarization_array))
                this.polarization_array = this.polarization_array[order]
            else:
                this.feed_array = np.concatenate([this.feed_array,
                                                  other.feed_array[pnew_inds]])
                order = np.argsort(this.feed_array)
                this.feed_array = this.feed_array[order]

            this.data_array = np.concatenate([this.data_array, data_zero_pad], axis=paxis)[
                :, :, order, ...]

        # Now populate the data
        if this.beam_type == 'power':
            this.Npols = this.polarization_array.shape[0]
            pol_t2o = np.nonzero(np.in1d(this.polarization_array,
                                 other.polarization_array))[0]
        else:
            this.Nfeeds = this.feed_array.shape[0]
            pol_t2o = np.nonzero(np.in1d(this.feed_array, other.feed_array))[0]

        freq_t2o = np.nonzero(np.in1d(this.freq_array[0, :],
                              other.freq_array[0, :]))[0]

        if this.pixel_coordinate_system == 'healpix':
            this.Npixels = this.pixel_array.shape[0]
            pix_t2o = np.nonzero(np.in1d(this.pixel_array, other.pixel_array))[0]
            this.data_array[np.ix_(np.arange(this.Naxes_vec), [0], pol_t2o, freq_t2o,
                                   pix_t2o)] = other.data_array
            if this.beam_type == 'efield':
                this.basis_vector_array[np.ix_(np.arange(this.Naxes_vec), np.arange(2),
                                               pix_t2o)] = other.basis_vector_array
        else:
            this.Naxes1 = this.axis1_array.shape[0]
            this.Naxes2 = this.axis2_array.shape[0]
            ax1_t2o = np.nonzero(np.in1d(this.axis1_array, other.axis1_array))[0]
            ax2_t2o = np.nonzero(np.in1d(this.axis2_array, other.axis2_array))[0]
            this.data_array[np.ix_(np.arange(this.Naxes_vec), [0], pol_t2o, freq_t2o,
                                   ax2_t2o, ax1_t2o)] = other.data_array
            if this.beam_type == 'efield':
                this.basis_vector_array[np.ix_(np.arange(this.Naxes_vec), np.arange(2),
                                               ax2_t2o, ax1_t2o)] = other.basis_vector_array

        this.bandpass_array[np.ix_([0], freq_t2o)] = other.bandpass_array

        if this.receiver_temperature_array is not None:
            this.receiver_temperature_array[np.ix_([0], freq_t2o)] = other.receiver_temperature_array
        if this.loss_array is not None:
            this.loss_array[np.ix_([0], freq_t2o)] = other.loss_array
        if this.mismatch_array is not None:
            this.mismatch_array[np.ix_([0], freq_t2o)] = other.mismatch_array
        if this.s_parameters is not None:
            this.s_parameters[np.ix_(np.arange(4), [0], freq_t2o)] = other.s_parameters

        this.Nfreqs = this.freq_array.shape[1]

        # Check specific requirements
        if this.Nfreqs > 1:
            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                              rtol=this._freq_array.tols[0], atol=this._freq_array.tols[1]):
                warnings.warn('Combined frequencies are not evenly spaced. This will '
                              'make it impossible to write this data out to some file types.')

        if self.beam_type == 'power' and this.Npols > 2:
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
            this.check(run_check_acceptability=run_check_acceptability)

        if not inplace:
            return this

    def __iadd__(self, other):
        """
        In place add.

        Args:
            other: Another UVBeam object which will be added to self.
        """
        self.__add__(other, inplace=True)
        return self

    def select(self, axis1_inds=None, axis2_inds=None, pixels=None,
               frequencies=None, freq_chans=None,
               feeds=None, polarizations=None, run_check=True,
               run_check_acceptability=True, inplace=True):
        """
        Select specific image axis indices or pixels (if healpix), frequencies and
        feeds or polarizations (if power) to keep in the object while discarding others.

        The history attribute on the object will be updated to identify the
        operations performed.

        Args:
            axis1_inds: The indices along the first image axis to keep in the object.
                Cannot be set if pixel_coordinate_system is "healpix".
            axis2_inds: The indices along the second image axis to keep in the object.
                Cannot be set if pixel_coordinate_system is "healpix".
            pixels: The healpix pixels to keep in the object.
                Cannot be set if pixel_coordinate_system is not "healpix".
            frequencies: The frequencies to keep in the object.
            freq_chans: The frequency channel numbers to keep in the object.
            feeds: The feeds to keep in the object. Cannot be set if the beam_type is "power".
            polarizations: The polarizations to keep in the object.
                Cannot be set if the beam_type is "efield".
            run_check: Option to check for the existence and proper shapes of
                required parameters after downselecting data on this object. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after  downselecting data on this object. Default is True.
            inplace: Option to perform the select directly on self (True, default) or return
                a new UVBeam object, which is a subselection of self (False)
        """
        if inplace:
            beam_object = self
        else:
            beam_object = copy.deepcopy(self)

        # build up history string as we go
        history_update_string = '  Downselected to specific '
        n_selects = 0

        if axis1_inds is not None:
            if beam_object.pixel_coordinate_system == 'healpix':
                raise ValueError('axis1_inds cannot be used with healpix coordinate system')

            history_update_string += 'parts of first image axis'
            n_selects += 1

            axis1_inds = list(sorted(set(list(axis1_inds))))
            if min(axis1_inds) < 0 or max(axis1_inds) > beam_object.Naxes1 - 1:
                raise ValueError('axis1_inds must be > 0 and < Naxes1')
            beam_object.Naxes1 = len(axis1_inds)
            beam_object.axis1_array = beam_object.axis1_array[axis1_inds]

            if beam_object.Naxes1 > 1:
                axis1_spacing = np.diff(beam_object.axis1_array)
                if not np.isclose(np.min(axis1_spacing), np.max(axis1_spacing),
                                  rtol=beam_object._axis1_array.tols[0],
                                  atol=beam_object._axis1_array.tols[1]):
                    warnings.warn('Selected values along first image axis are '
                                  'not evenly spaced. This is not supported by '
                                  'the regularly gridded beam fits format')

            beam_object.data_array = beam_object.data_array[:, :, :, :, :, axis1_inds]
            if beam_object.beam_type == 'efield':
                beam_object.basis_vector_array = beam_object.basis_vector_array[:, :, :, axis1_inds]

        if axis2_inds is not None:
            if beam_object.pixel_coordinate_system == 'healpix':
                raise ValueError('axis2_inds cannot be used with healpix coordinate system')

            if n_selects > 0:
                history_update_string += ', parts of second image axis'
            else:
                history_update_string += 'parts of second image axis'
            n_selects += 1

            axis2_inds = list(sorted(set(list(axis2_inds))))
            if min(axis2_inds) < 0 or max(axis2_inds) > beam_object.Naxes2 - 1:
                raise ValueError('axis2_inds must be > 0 and < Naxes2')
            beam_object.Naxes2 = len(axis2_inds)
            beam_object.axis2_array = beam_object.axis2_array[axis2_inds]

            if beam_object.Naxes2 > 1:
                axis2_spacing = np.diff(beam_object.axis2_array)
                if not np.isclose(np.min(axis2_spacing), np.max(axis2_spacing),
                                  rtol=beam_object._axis2_array.tols[0],
                                  atol=beam_object._axis2_array.tols[1]):
                    warnings.warn('Selected values along second image axis are '
                                  'not evenly spaced. This is not supported by '
                                  'the regularly gridded beam fits format')

            beam_object.data_array = beam_object.data_array[:, :, :, :, axis2_inds, :]
            if beam_object.beam_type == 'efield':
                beam_object.basis_vector_array = beam_object.basis_vector_array[:, :, axis2_inds, :]

        if pixels is not None:
            if beam_object.pixel_coordinate_system != 'healpix':
                raise ValueError('pixels can only be used with healpix coordinate system')

            history_update_string += 'healpix pixels'
            n_selects += 1

            pix_inds = np.zeros(0, dtype=np.int)
            for p in pixels:
                if p in beam_object.pixel_array:
                    pix_inds = np.append(pix_inds, np.where(beam_object.pixel_array == p)[0])
                else:
                    raise ValueError('Pixel {p} is not present in the pixel_array'.format(p=p))

            pix_inds = list(sorted(set(list(pix_inds))))
            beam_object.Npixels = len(pix_inds)
            beam_object.pixel_array = beam_object.pixel_array[pix_inds]

            beam_object.data_array = beam_object.data_array[:, :, :, :, pix_inds]
            if beam_object.beam_type == 'efield':
                beam_object.basis_vector_array = beam_object.basis_vector_array[:, :, pix_inds]

        if freq_chans is not None:
            freq_chans = uvutils.get_iterable(freq_chans)
            if frequencies is None:
                frequencies = beam_object.freq_array[0, freq_chans]
            else:
                frequencies = uvutils.get_iterable(frequencies)
                frequencies = np.sort(list(set(frequencies) |
                                      set(beam_object.freq_array[0, freq_chans])))

        if frequencies is not None:
            frequencies = uvutils.get_iterable(frequencies)
            if n_selects > 0:
                history_update_string += ', frequencies'
            else:
                history_update_string += 'frequencies'
            n_selects += 1

            freq_inds = np.zeros(0, dtype=np.int)
            # this works because we only allow one SPW. This will have to be reworked when we support more.
            freq_arr_use = beam_object.freq_array[0, :]
            for f in frequencies:
                if f in freq_arr_use:
                    freq_inds = np.append(freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError('Frequency {f} is not present in the freq_array'.format(f=f))

            freq_inds = list(sorted(set(list(freq_inds))))
            beam_object.Nfreqs = len(freq_inds)
            beam_object.freq_array = beam_object.freq_array[:, freq_inds]
            beam_object.bandpass_array = beam_object.bandpass_array[:, freq_inds]

            if beam_object.Nfreqs > 1:
                freq_separation = beam_object.freq_array[0, 1:] - beam_object.freq_array[0, :-1]
                if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                                  rtol=beam_object._freq_array.tols[0],
                                  atol=beam_object._freq_array.tols[1]):
                    warnings.warn('Selected frequencies are not evenly spaced. This '
                                  'is not supported by the regularly gridded beam fits format')

            if beam_object.pixel_coordinate_system == 'healpix':
                beam_object.data_array = beam_object.data_array[:, :, :, freq_inds, :]
            else:
                beam_object.data_array = beam_object.data_array[:, :, :, freq_inds, :, :]

            if beam_object.antenna_type == 'phased_array':
                beam_object.coupling_matrix = beam_object.coupling_matrix[:, :, :, :, :, freq_inds]

            if beam_object.receiver_temperature_array is not None:
                beam_object.receiver_temperature_array = beam_object.receiver_temperature_array[:, freq_inds]

            if beam_object.loss_array is not None:
                beam_object.loss_array = beam_object.loss_array[:, freq_inds]

            if beam_object.mismatch_array is not None:
                beam_object.mismatch_array = beam_object.mismatch_array[:, freq_inds]

            if beam_object.s_parameters is not None:
                beam_object.s_parameters = beam_object.s_parameters[:, :, freq_inds]

        if feeds is not None:
            if beam_object.beam_type == 'power':
                raise ValueError('feeds cannot be used with power beams')

            feeds = uvutils.get_iterable(feeds)
            if n_selects > 0:
                history_update_string += ', feeds'
            else:
                history_update_string += 'feeds'
            n_selects += 1

            feed_inds = np.zeros(0, dtype=np.int)
            for f in feeds:
                if f in beam_object.feed_array:
                    feed_inds = np.append(feed_inds, np.where(beam_object.feed_array == f)[0])
                else:
                    raise ValueError('Feed {f} is not present in the feed_array'.format(f=f))

            feed_inds = list(sorted(set(list(feed_inds))))
            beam_object.Nfeeds = len(feed_inds)
            beam_object.feed_array = beam_object.feed_array[feed_inds]

            if beam_object.pixel_coordinate_system == 'healpix':
                beam_object.data_array = beam_object.data_array[:, :, feed_inds, :, :]
            else:
                beam_object.data_array = beam_object.data_array[:, :, feed_inds, :, :, :]

        if polarizations is not None:
            if beam_object.beam_type == 'efield':
                raise ValueError('polarizations cannot be used with efield beams')

            polarizations = uvutils.get_iterable(polarizations)
            if n_selects > 0:
                history_update_string += ', polarizations'
            else:
                history_update_string += 'polarizations'
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int)
            for p in polarizations:
                if p in beam_object.polarization_array:
                    pol_inds = np.append(pol_inds, np.where(beam_object.polarization_array == p)[0])
                else:
                    raise ValueError('polarization {p} is not present in the polarization_array'.format(p=p))

            pol_inds = list(sorted(set(list(pol_inds))))
            beam_object.Npols = len(pol_inds)
            beam_object.polarization_array = beam_object.polarization_array[pol_inds]

            if len(pol_inds) > 2:
                pol_separation = (beam_object.polarization_array[1:] -
                                  beam_object.polarization_array[:-1])
                if np.min(pol_separation) < np.max(pol_separation):
                    warnings.warn('Selected polarizations are not evenly spaced. This '
                                  'is not supported by the regularly gridded beam fits format')

            if beam_object.pixel_coordinate_system == 'healpix':
                beam_object.data_array = beam_object.data_array[:, :, pol_inds, :, :]
            else:
                beam_object.data_array = beam_object.data_array[:, :, pol_inds, :, :, :]

        history_update_string += ' using pyuvdata.'
        beam_object.history = beam_object.history + history_update_string

        # check if object is self-consistent
        if run_check:
            beam_object.check(run_check_acceptability=run_check_acceptability)

        if not inplace:
            return beam_object

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype is 'beamfits':
            import beamfits
            other_obj = beamfits.BeamFITS()
        else:
            raise ValueError('filetype must be beamfits')
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_beamfits(self, filename, run_check=True, run_check_acceptability=True):
        """
        Read in data from a beamfits file.

        Args:
            filename: The beamfits file or list of files to read from.
            run_check: Option to check for the existence and proper shapes of
                required parameters after reading in the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters after reading in the file. Default is True.
        """
        import beamfits
        if isinstance(filename, (list, tuple)):
            self.read_beamfits(filename[0], run_check=run_check,
                               run_check_acceptability=run_check_acceptability)
            if len(filename) > 1:
                for f in filename[1:]:
                    beam2 = UVBeam()
                    beam2.read_beamfits(f, run_check=run_check,
                                        run_check_acceptability=run_check_acceptability)
                    self += beam2
                del(beam2)
        else:
            beamfits_obj = beamfits.BeamFITS()
            beamfits_obj.read_beamfits(filename, run_check=run_check,
                                       run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(beamfits_obj)
            del(beamfits_obj)

    def write_beamfits(self, filename, run_check=True,
                       run_check_acceptability=True, clobber=False):
        """
        Write the data to a beamfits file.

        Args:
            filename: The beamfits file to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
        """
        beamfits_obj = self._convert_to_filetype('beamfits')
        beamfits_obj.write_beamfits(filename, run_check=run_check,
                                    run_check_acceptability=run_check_acceptability,
                                    clobber=clobber)
        del(beamfits_obj)
