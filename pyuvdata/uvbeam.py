"""Primary container for radio telescope antenna beams."""
import numpy as np
from uvbase import UVBase
import parameter as uvp


class UVBeam(UVBase):
    """
    A class for defining a radio telescope antenna beam.

    Attributes:
        UVParameter objects: For full list see UVBeam Parameters
            (http://pyuvdata.readthedocs.io/en/latest/uvbeam_parameters.html).
            Some are always required, some are required for certain phase_types
            and others are always optional.
    """

    coordinate_system_dict = {
        # uniformly gridded az/za coordinate system, az runs from East to North
        'az_za': ['az', 'za'],
        # sine projection at zenith. y points North, x point East
        'sin_zenith': ['sin_x', 'sin_y'],
        # HEALPix map with zenith at north pole.
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

        desc = ('Pixel coordinate system, options are: '
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
        self._ordering = uvp.UVParameter('ordering', description=desc, expected_type=int,
                                         required=False)

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
                'circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)'
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

        desc = 'Normalization standard of data_array, options are: "peak", "solid_angle"'
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

        desc = 'Array of system temperatures, shape (Nspws, Nfreqs), units K'
        self._system_temperature_array = \
            uvp.UVParameter('system_temperature_array', required=False,
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

        desc = 'S parameters of receiving chain, shape (Nspws, Nfreqs), units ?'
        self._s_parameters = uvp.UVParameter('s_parameters', required=False,
                                             description=desc,
                                             form=(4, 'Nspws', 'Nfreqs'),
                                             expected_type=np.float,
                                             tols=1e-3)

        super(UVBeam, self).__init__()

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
