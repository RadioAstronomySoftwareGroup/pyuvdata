"""Primary container for radio telescope antenna beams."""
import numpy as np
from uvbase import UVBase
import parameter as uvp
import version as uvversion


class UVBeam(UVBase):
    """
    A class for defining a radio telescope antenna beam.

    Attributes:
        UVParameter objects: For full list see UVBeam Parameters
            (http://pyuvdata.readthedocs.io/en/latest/uvbeam_parameters.html).
            Some are always required, some are required for certain phase_types
            and others are always optional.
    """

    def __init__(self):
        """Create a new UVBeam object."""
        # add the UVParameters to the class

        self._Nfreqs = uvp.UVParameter('Nfreqs', description='Number of frequency channels',
                                       expected_type=int)

        self._Npixels = uvp.UVParameter('Npixels', description='Number of pixels '
                                        'with beam model values',
                                        expected_type=int)

        self._Nfeeds = uvp.UVParameter('Nfeeds', description='Number of feeds',
                                       expected_type=int, acceptable_vals=[1, 2])

        self._Naxes = uvp.UVParameter('Naxes', description='Number of directions '
                                      'in coordinate system, options are 2 or 3',
                                      expected_type=int, acceptable_vals=[2, 3])

        self._Nspws = uvp.UVParameter('Nspws', description='Number of spectral windows '
                                      '(ie non-contiguous spectral chunks). '
                                      'More than one spectral window is not '
                                      'currently supported.', expected_type=int)

        coordinate_system_dict = {'az_el': {'naxes': 2, 'axis_list': ['az', 'el']},
                                  'spherical': {'naxes': 3, 'axis_list': ['x', 'y', 'z']}}
        desc = 'Pixel coordinate system, options are: ' + ', '.join(coordinate_system_dict.keys())
        self._coordinate_system = uvp.UVParameter('coordinate_system',
                                                  description=desc, form='str',
                                                  expected_type=str,
                                                  acceptable_vals=coordinate_system_dict.keys())

        desc = 'Array of pixel locations, shape: (Naxes, Npixels)'
        self._pixel_location_array = uvp.UVParameter('pixel_location_array',
                                                     description=desc,
                                                     form=('Naxes', 'Npixels'),
                                                     expected_type=np.float)

        desc = ('Array of feed orientations. shape (Nfeeds). '
                'options are: N/E or x/y or R/L')
        self._feed_array = uvp.UVParameter('feed_array', description=desc,
                                           expected_type=str, form=('Nfeeds',),
                                           acceptable_vals=['N', 'E', 'x', 'y',
                                                            'R', 'L'])

        desc = 'Array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        self._spw_array = uvp.UVParameter('spw_array',
                                          description='Array of spectral window '
                                          'Numbers, shape (Nspws)', form=('Nspws',),
                                          expected_type=int)

        desc = ('Beam basis vector components -- directions for which the '
                'electric field values are recorded in the pixel coordinate system. '
                'If the components are aligned with the pixel coordinate system '
                'all the values will be 1 or 0. shape: (Naxes, Npixels).')
        self._basis_vector_array = uvp.UVParameter('basis_vector_array',
                                                   description=desc,
                                                   expected_type=np.float,
                                                   form=('Naxes', 'Npixels'),
                                                   acceptable_range=(0, 1),
                                                   tols=1e-3)

        desc = ('Complex E-field values for beam model, units V/m. '
                'shape = (Nfeeds, Naxes, Npixels, Nspws, Nfreq)')
        self._efield_array = uvp.UVParameter('efield_array',
                                             description=desc,
                                             expected_type=np.complex,
                                             form=('Nfeeds', 'Naxes', 'Npixels', 'Nspws', 'Nfreq'),
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
                                             description=desc, value='simple',
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
                'shape: (Nelements, Nelements, Nfeed, Nfeed, Nspws, Nfreq)')
        self._coupling_matrix = uvp.UVParameter('coupling_matrix', required=False,
                                                description=desc,
                                                form=('Nelements', 'Nelements',
                                                      'Nfeed', 'Nfeed', 'Nspws', 'Nfreq'),
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

        # String to add to history of any files written with this version of pyuvdata
        self.pyuvdata_version_str = ('  Read/written with pyuvdata version: ' +
                                     uvversion.version + '.')
        if uvversion.git_hash is not '':
            self.pyuvdata_version_str += ('  Git origin: ' + uvversion.git_origin +
                                          '.  Git hash: ' + uvversion.git_hash +
                                          '.  Git branch: ' + uvversion.git_branch +
                                          '.  Git description: ' + uvversion.git_description + '.')

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
        # first run the basic check from UVBase
        super(UVBeam, self).check(run_check_acceptability=run_check_acceptability)

        # Check consistency of coordinate_system and Naxes
        if self.Naxes != self.coordinate_system_dict[self.coordinate_system]['naxes']:
            raise ValueError('Number of coordinate axes is not consistent with coordinate_system.')

        return True

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
        if isinstance(filename, (list, tuple)):
            self.read_beamfits(filename[0], run_check=run_check,
                               run_check_acceptability=run_check_acceptability)
            if len(filename) > 1:
                for f in filename[1:]:
                    beam2 = UVBeam()
                    beam2.read_beamfits(f, run_check=run_check,
                                        run_check_acceptability=run_check_acceptability)
                    self += beam2
                del(uv2)
        else:
            beamfits_obj = beamfits.BeamFITS()
            beamfits_obj.read_beamfits(filename, run_check=run_check,
                                       run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(beamfits_obj)
            del(beamfits_obj)

    def write_beamfits(self, filename, run_check=True, run_check_acceptability=True):
        """
        Write the data to a beamfits file.

        Args:
            filename: The beamfits file to write to.
            run_check: Option to check for the existence and proper shapes of
                required parameters before writing the file. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                required parameters before writing the file. Default is True.
        """
        beamfits_obj = self._convert_to_filetype('beamfits')
        beamfits_obj.write_beamfits(filename, run_check=run_check,
                                    run_check_acceptability=run_check_acceptability)
        del(beamfits_obj)
