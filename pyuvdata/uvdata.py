# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio interferometer datasets.

"""
from __future__ import absolute_import, division, print_function

import os
import copy
import collections
import re
import numpy as np
import six
import warnings
from astropy import constants as const
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, FK5, Angle

from .uvbase import UVBase
from . import parameter as uvp
from . import telescopes as uvtel
from . import utils as uvutils


class UVData(UVBase):
    """
    A class for defining a radio interferometer dataset.

    Currently supported file types: uvfits, miriad, fhd.
    Provides phasing functions.

    Attributes:
        UVParameter objects: For full list see UVData Parameters
            (http://pyuvdata.readthedocs.io/en/latest/uvdata_parameters.html).
            Some are always required, some are required for certain phase_types
            and others are always optional.
    """

    def __init__(self):
        """Create a new UVData object."""
        # add the UVParameters to the class

        # standard angle tolerance: 10 mas in radians.
        # Should perhaps be decreased to 1 mas in the future
        radian_tol = 10 * 2 * np.pi * 1e-3 / (60.0 * 60.0 * 360.0)

        self._Ntimes = uvp.UVParameter('Ntimes', description='Number of times',
                                       expected_type=int)
        self._Nbls = uvp.UVParameter('Nbls', description='Number of baselines',
                                     expected_type=int)
        self._Nblts = uvp.UVParameter('Nblts', description='Number of baseline-times '
                                      '(i.e. number of spectra). Not necessarily '
                                      'equal to Nbls * Ntimes', expected_type=int)
        self._Nfreqs = uvp.UVParameter('Nfreqs', description='Number of frequency channels',
                                       expected_type=int)
        self._Npols = uvp.UVParameter('Npols', description='Number of polarizations',
                                      expected_type=int)

        desc = ('Array of the visibility data, shape: (Nblts, Nspws, Nfreqs, '
                'Npols), type = complex float, in units of self.vis_units')
        self._data_array = uvp.UVParameter('data_array', description=desc,
                                           form=('Nblts', 'Nspws',
                                                 'Nfreqs', 'Npols'),
                                           expected_type=np.complex)

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str"'
        self._vis_units = uvp.UVParameter('vis_units', description=desc,
                                          form='str', expected_type=str,
                                          acceptable_vals=["uncalib", "Jy", "K str"])

        desc = ('Number of data points averaged into each data element, '
                'NOT required to be an integer, type = float, same shape as data_array.'
                'The product of the integration_time and the nsample_array '
                'value for a visibility reflects the total amount of time '
                'that went into the visibility. Best practice is for the '
                'nsample_array to be used to track flagging within an integration_time '
                '(leading to a decrease of the nsample array value below 1) and '
                'LST averaging (leading to an increase in the nsample array '
                'value). So datasets that have not been LST averaged should '
                'have nsample array values less than or equal to 1.'
                'Note that many files do not follow this convention, but it is '
                'safe to assume that the product of the integration_time and '
                'the nsample_array is the total amount of time included in a visibility.')
        self._nsample_array = uvp.UVParameter('nsample_array', description=desc,
                                              form=('Nblts', 'Nspws',
                                                    'Nfreqs', 'Npols'),
                                              expected_type=(np.float))

        desc = 'Boolean flag, True is flagged, same shape as data_array.'
        self._flag_array = uvp.UVParameter('flag_array', description=desc,
                                           form=('Nblts', 'Nspws',
                                                 'Nfreqs', 'Npols'),
                                           expected_type=np.bool)

        self._Nspws = uvp.UVParameter('Nspws', description='Number of spectral windows '
                                      '(ie non-contiguous spectral chunks). '
                                      'More than one spectral window is not '
                                      'currently supported.', expected_type=int)

        self._spw_array = uvp.UVParameter('spw_array',
                                          description='Array of spectral window '
                                          'Numbers, shape (Nspws)', form=('Nspws',),
                                          expected_type=int)

        desc = ('Projected baseline vectors relative to phase center, '
                'shape (Nblts, 3), units meters. Convention is: uvw = xyz(ant2) - xyz(ant1).'
                'Note that this is the Miriad convention but it is different '
                'from the AIPS/FITS convention (where uvw = xyz(ant1) - xyz(ant2)).')
        self._uvw_array = uvp.UVParameter('uvw_array', description=desc,
                                          form=('Nblts', 3),
                                          expected_type=np.float,
                                          acceptable_range=(0, 1e8), tols=1e-3)

        desc = ('Array of times, center of integration, shape (Nblts), '
                'units Julian Date')
        self._time_array = uvp.UVParameter('time_array', description=desc,
                                           form=('Nblts',),
                                           expected_type=np.float,
                                           tols=1e-3 / (60.0 * 60.0 * 24.0))  # 1 ms in days

        desc = ('Array of lsts, center of integration, shape (Nblts), '
                'units radians')
        self._lst_array = uvp.UVParameter('lst_array', description=desc,
                                          form=('Nblts',),
                                          expected_type=np.float,
                                          tols=radian_tol)

        desc = ('Array of first antenna indices, shape (Nblts), '
                'type = int, 0 indexed')
        self._ant_1_array = uvp.UVParameter('ant_1_array', description=desc,
                                            expected_type=int, form=('Nblts',))
        desc = ('Array of second antenna indices, shape (Nblts), '
                'type = int, 0 indexed')
        self._ant_2_array = uvp.UVParameter('ant_2_array', description=desc,
                                            expected_type=int, form=('Nblts',))

        desc = ('Array of baseline indices, shape (Nblts), '
                'type = int; baseline = 2048 * (ant1+1) + (ant2+1) + 2^16')
        self._baseline_array = uvp.UVParameter('baseline_array',
                                               description=desc,
                                               expected_type=int, form=('Nblts',))

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        desc = 'Array of frequencies, shape (Nspws, Nfreqs), units Hz'
        self._freq_array = uvp.UVParameter('freq_array', description=desc,
                                           form=('Nspws', 'Nfreqs'),
                                           expected_type=np.float,
                                           tols=1e-3)  # mHz

        desc = ('Array of polarization integers, shape (Npols). '
                'AIPS Memo 117 says: pseudo-stokes 1:4 (pI, pQ, pU, pV);  '
                'circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX). '
                'NOTE: AIPS Memo 117 actually calls the pseudo-Stokes polarizations '
                '"Stokes", but this is inaccurate as visibilities cannot be in '
                'true Stokes polarizations for physical antennas. We adopt the '
                'term pseudo-Stokes to refer to linear combinations of instrumental '
                'visibility polarizations (e.g. pI = xx + yy).')
        self._polarization_array = uvp.UVParameter('polarization_array',
                                                   description=desc,
                                                   expected_type=int,
                                                   acceptable_vals=list(
                                                       np.arange(-8, 0)) + list(np.arange(1, 5)),
                                                   form=('Npols',))

        desc = ('Length of the integration in seconds, shape (NBlts). '
                'The product of the integration_time and the nsample_array '
                'value for a visibility reflects the total amount of time '
                'that went into the visibility. Best practice is for the '
                'integration_time to reflect the length of time a visibility '
                'was integrated over (so it should vary in the case of '
                'baseline-dependent averaging and be a way to do selections '
                'for differently integrated baselines).'
                'Note that many files do not follow this convention, but it is '
                'safe to assume that the product of the integration_time and '
                'the nsample_array is the total amount of time included in a visibility.')
        self._integration_time = uvp.UVParameter('integration_time',
                                                 description=desc,
                                                 form=('Nblts',),
                                                 expected_type=np.float, tols=1e-3)  # 1 ms
        self._channel_width = uvp.UVParameter('channel_width',
                                              description='Width of frequency channels (Hz)',
                                              expected_type=np.float,
                                              tols=1e-3)  # 1 mHz

        # --- observation information ---
        self._object_name = uvp.UVParameter('object_name',
                                            description='Source or field '
                                            'observed (string)', form='str',
                                            expected_type=str)
        self._telescope_name = uvp.UVParameter('telescope_name',
                                               description='Name of telescope '
                                               '(string)', form='str',
                                               expected_type=str)
        self._instrument = uvp.UVParameter('instrument', description='Receiver or backend. '
                                           'Sometimes identical to telescope_name',
                                           form='str', expected_type=str)

        desc = ('Telescope location: xyz in ITRF (earth-centered frame). '
                'Can also be accessed using telescope_location_lat_lon_alt or '
                'telescope_location_lat_lon_alt_degrees properties')
        self._telescope_location = uvp.LocationParameter('telescope_location',
                                                         description=desc,
                                                         acceptable_range=(
                                                             6.35e6, 6.39e6),
                                                         tols=1e-3)

        self._history = uvp.UVParameter('history', description='String of history, units English',
                                        form='str', expected_type=str)

        # --- phasing information ---
        desc = ('String indicating phasing type. Allowed values are "drift", '
                '"phased" and "unknown"')
        self._phase_type = uvp.UVParameter('phase_type', form='str', expected_type=str,
                                           description=desc, value='unknown',
                                           acceptable_vals=['drift', 'phased', 'unknown'])

        desc = ('Required if phase_type = "phased". Epoch year of the phase '
                'applied to the data (eg 2000.)')
        self._phase_center_epoch = uvp.UVParameter('phase_center_epoch',
                                                   required=False,
                                                   description=desc,
                                                   expected_type=np.float)

        desc = ('Required if phase_type = "phased". Right ascension of phase '
                'center (see uvw_array), units radians. Can also be accessed using phase_center_ra_degrees.')
        self._phase_center_ra = uvp.AngleParameter('phase_center_ra',
                                                   required=False,
                                                   description=desc,
                                                   expected_type=np.float,
                                                   tols=radian_tol)

        desc = ('Required if phase_type = "phased". Declination of phase center '
                '(see uvw_array), units radians. Can also be accessed using phase_center_dec_degrees.')
        self._phase_center_dec = uvp.AngleParameter('phase_center_dec',
                                                    required=False,
                                                    description=desc,
                                                    expected_type=np.float,
                                                    tols=radian_tol)

        desc = ('Only relevant if phase_type = "phased". Specifies the frame the'
                ' data and uvw_array are phased to. Options are "gcrs" and "icrs",'
                ' default is "icrs"')
        self._phase_center_frame = uvp.UVParameter('phase_center_frame',
                                                   required=False,
                                                   description=desc,
                                                   expected_type=str,
                                                   acceptable_vals=['icrs', 'gcrs'])

        # --- antenna information ----
        desc = ('Number of antennas with data present (i.e. number of unique '
                'entries in ant_1_array and ant_2_array). May be smaller '
                'than the number of antennas in the array')
        self._Nants_data = uvp.UVParameter('Nants_data', description=desc,
                                           expected_type=int)

        desc = ('Number of antennas in the array. May be larger '
                'than the number of antennas with data')
        self._Nants_telescope = uvp.UVParameter('Nants_telescope',
                                                description=desc, expected_type=int)

        desc = ('List of antenna names, shape (Nants_telescope), '
                'with numbers given by antenna_numbers (which can be matched '
                'to ant_1_array and ant_2_array). There must be one entry '
                'here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_names = uvp.UVParameter('antenna_names', description=desc,
                                              form=('Nants_telescope',),
                                              expected_type=str)

        desc = ('List of integer antenna numbers corresponding to antenna_names, '
                'shape (Nants_telescope). There must be one '
                'entry here for each unique entry in ant_1_array and '
                'ant_2_array, but there may be extras as well.')
        self._antenna_numbers = uvp.UVParameter('antenna_numbers', description=desc,
                                                form=('Nants_telescope',),
                                                expected_type=int)

        # -------- extra, non-required parameters ----------
        desc = ('Orientation of the physical dipole corresponding to what is '
                'labelled as the x polarization. Examples include "east" '
                '(indicating east/west orientation) and "north" (indicating '
                'north/south orientation)')
        self._x_orientation = uvp.UVParameter('x_orientation', description=desc,
                                              required=False, expected_type=str)

        desc = ('Any user supplied extra keywords, type=dict. Keys should be '
                '8 character or less strings if writing to uvfits or miriad files. '
                'Use the special key "comment" for long multi-line string comments.')
        self._extra_keywords = uvp.UVParameter('extra_keywords', required=False,
                                               description=desc, value={},
                                               spoof_val={}, expected_type=dict)

        desc = ('Array giving coordinates of antennas relative to '
                'telescope_location (ITRF frame), shape (Nants_telescope, 3), '
                'units meters. See the tutorial page in the documentation '
                'for an example of how to convert this to topocentric frame.'
                'Will be a required parameter in a future version.')
        self._antenna_positions = uvp.AntPositionParameter('antenna_positions',
                                                           required=False,
                                                           description=desc,
                                                           form=(
                                                               'Nants_telescope', 3),
                                                           expected_type=np.float,
                                                           tols=1e-3)  # 1 mm

        desc = ('Array of antenna diameters in meters. Used by CASA to '
                'construct a default beam if no beam is supplied.')
        self._antenna_diameters = uvp.UVParameter('antenna_diameters',
                                                  required=False,
                                                  description=desc,
                                                  form=('Nants_telescope',),
                                                  expected_type=np.float,
                                                  tols=1e-3)  # 1 mm

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._gst0 = uvp.UVParameter('gst0', required=False,
                                     description='Greenwich sidereal time at '
                                                 'midnight on reference date',
                                     spoof_val=0.0, expected_type=np.float)
        self._rdate = uvp.UVParameter('rdate', required=False,
                                      description='Date for which the GST0 or '
                                                  'whatever... applies',
                                      spoof_val='', form='str')
        self._earth_omega = uvp.UVParameter('earth_omega', required=False,
                                            description='Earth\'s rotation rate '
                                                        'in degrees per day',
                                            spoof_val=360.985, expected_type=np.float)
        self._dut1 = uvp.UVParameter('dut1', required=False,
                                     description='DUT1 (google it) AIPS 117 '
                                                 'calls it UT1UTC',
                                     spoof_val=0.0, expected_type=np.float)
        self._timesys = uvp.UVParameter('timesys', required=False,
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

    def check(self, check_extra=True, run_check_acceptability=True):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Args:
            check_extra: If true, check all parameters, otherwise only check
                required parameters.
            run_check_acceptability: Option to check if values in parameters
                are acceptable. Default is True.
        """
        # first run the basic check from UVBase
        # set the phase type based on object's value
        if self.phase_type == 'phased':
            self.set_phased()
        elif self.phase_type == 'drift':
            self.set_drift()
        else:
            self.set_unknown_phase_type()

        super(UVData, self).check(check_extra=check_extra,
                                  run_check_acceptability=run_check_acceptability)

        # Check internal consistency of numbers which don't explicitly correspond
        # to the shape of another array.
        nants_data_calc = int(len(np.unique(self.ant_1_array.tolist()
                                            + self.ant_2_array.tolist())))
        if self.Nants_data != nants_data_calc:
            raise ValueError('Nants_data must be equal to the number of unique '
                             'values in ant_1_array and ant_2_array')

        if self.Nbls != len(np.unique(self.baseline_array)):
            raise ValueError('Nbls must be equal to the number of unique '
                             'baselines in the data_array')

        if self.Ntimes != len(np.unique(self.time_array)):
            raise ValueError('Ntimes must be equal to the number of unique '
                             'times in the time_array')

        # issue warning if extra_keywords keys are longer than 8 characters
        for key in self.extra_keywords.keys():
            if len(key) > 8:
                warnings.warn('key {key} in extra_keywords is longer than 8 '
                              'characters. It will be truncated to 8 if written '
                              'to uvfits or miriad file formats.'.format(key=key))

        # issue warning if extra_keywords values are lists, arrays or dicts
        for key, value in self.extra_keywords.items():
            if isinstance(value, (list, dict, np.ndarray)):
                warnings.warn('{key} in extra_keywords is a list, array or dict, '
                              'which will raise an error when writing uvfits or '
                              'miriad file types'.format(key=key))

        # issue deprecation warning if antenna positions are not set
        if self.antenna_positions is None:
            warnings.warn('antenna_positions are not defined. '
                          'antenna_positions will be a required parameter in '
                          'future versions.', PendingDeprecationWarning)

        # check auto and cross-corrs have sensible uvws
        autos = np.isclose(self.ant_1_array - self.ant_2_array, 0.0)
        if not np.all(np.isclose(self.uvw_array[autos], 0.0,
                                 rtol=self._uvw_array.tols[0],
                                 atol=self._uvw_array.tols[1])):
            raise ValueError("Some auto-correlations have non-zero "
                             "uvw_array coordinates.")
        if np.any(np.isclose([np.linalg.norm(uvw) for uvw in self.uvw_array[~autos]], 0.0,
                             rtol=self._uvw_array.tols[0],
                             atol=self._uvw_array.tols[1])):
            raise ValueError("Some cross-correlations have near-zero "
                             "uvw_array magnitudes.")

        return True

    def set_drift(self):
        """Set phase_type to 'drift' and adjust required parameters."""
        self.phase_type = 'drift'
        self._phase_center_epoch.required = False
        self._phase_center_ra.required = False
        self._phase_center_dec.required = False

    def set_phased(self):
        """Set phase_type to 'phased' and adjust required parameters."""
        self.phase_type = 'phased'
        self._phase_center_epoch.required = True
        self._phase_center_ra.required = True
        self._phase_center_dec.required = True

    def set_unknown_phase_type(self):
        """Set phase_type to 'unknown' and adjust required parameters."""
        self.phase_type = 'unknown'
        self._phase_center_epoch.required = False
        self._phase_center_ra.required = False
        self._phase_center_dec.required = False

    def known_telescopes(self):
        """
        Retun a list of telescopes known to pyuvdata.

        This is just a shortcut to uvdata.telescopes.known_telescopes()
        """
        return uvtel.known_telescopes()

    def set_telescope_params(self, overwrite=False):
        """
        Set telescope related parameters.

        If the telescope_name is in the known_telescopes, set any missing
        telescope-associated parameters (e.g. telescope location) to the value
        for the known telescope.

        Args:
            overwrite: Option to overwrite existing telescope-associated
                parameters with the values from the known telescope.
                Default is False.
        """
        telescope_obj = uvtel.get_telescope(self.telescope_name)
        if telescope_obj is not False:
            params_set = []
            for p in telescope_obj:
                telescope_param = getattr(telescope_obj, p)
                self_param = getattr(self, p)
                if telescope_param.value is not None and (overwrite is True
                                                          or self_param.value is None):
                    telescope_shape = telescope_param.expected_shape(telescope_obj)
                    self_shape = self_param.expected_shape(self)
                    if telescope_shape == self_shape:
                        params_set.append(self_param.name)
                        prop_name = self_param.name
                        setattr(self, prop_name, getattr(telescope_obj, prop_name))
                    else:
                        # expected shapes aren't equal. This can happen e.g. with diameters,
                        # which is a single value on the telescope object but is
                        # an array of length Nants_telescope on the UVData object
                        if telescope_shape == () and self_shape != 'str':
                            array_val = np.zeros(self_shape,
                                                 dtype=telescope_param.expected_type) + telescope_param.value
                            params_set.append(self_param.name)
                            prop_name = self_param.name
                            setattr(self, prop_name, array_val)
                        else:
                            raise ValueError('parameter {p} on the telescope '
                                             'object does not have a compatible '
                                             'expected shape.')
            if len(params_set) > 0:
                params_set_str = ', '.join(params_set)
                warnings.warn('{params} is not set. Using known values '
                              'for {telescope_name}.'.format(params=params_set_str,
                                                             telescope_name=telescope_obj.telescope_name))
        else:
            raise ValueError('Telescope {telescope_name} is not in '
                             'known_telescopes.'.format(telescope_name=self.telescope_name))

    def baseline_to_antnums(self, baseline):
        """
        Get the antenna numbers corresponding to a given baseline number.

        Args:
            baseline: integer baseline number

        Returns:
            tuple with the two antenna numbers corresponding to the baseline.
        """
        return uvutils.baseline_to_antnums(baseline, self.Nants_telescope)

    def antnums_to_baseline(self, ant1, ant2, attempt256=False):
        """
        Get the baseline number corresponding to two given antenna numbers.

        Args:
            ant1: first antenna number (integer)
            ant2: second antenna number (integer)
            attempt256: Option to try to use the older 256 standard used in
                many uvfits files (will use 2048 standard if there are more
                than 256 antennas). Default is False.

        Returns:
            integer baseline number corresponding to the two antenna numbers.
        """
        return uvutils.antnums_to_baseline(ant1, ant2, self.Nants_telescope, attempt256=attempt256)

    def order_pols(self, order='AIPS'):
        '''
        Arranges polarizations in orders corresponding to either AIPS or CASA convention.
        CASA stokes types are ordered with cross-pols in between (e.g. XX,XY,YX,YY) while
        AIPS orders pols with auto-pols followed by cross-pols (e.g. XX,YY,XY,YX)
        Args:
        order: string, either 'CASA' or 'AIPS'. Default='AIPS'
        '''
        if(order == 'AIPS'):
            pol_inds = np.argsort(self.polarization_array)
            pol_inds = pol_inds[::-1]
        elif(order == 'CASA'):  # sandwich
            casa_order = np.array([1, 2, 3, 4, -1, -3, -4, -2, -5, -7, -8, -6])
            pol_inds = []
            for pol in self.polarization_array:
                pol_inds.append(np.where(casa_order == pol)[0][0])
            pol_inds = np.argsort(pol_inds)
        else:
            warnings.warn('Invalid order supplied. No sorting performed.')
            pol_inds = list(range(self.Npols))
        # Generate a map from original indices to new indices
        if not np.array_equal(pol_inds, self.Npols):
            self.reorder_pols(order=pol_inds)

    def set_lsts_from_time_array(self):
        """Set the lst_array based from the time_array."""
        latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
        self.lst_array = uvutils.get_lst_for_time(self.time_array, latitude, longitude, altitude)

    def unphase_to_drift(self, phase_frame=None, use_ant_pos=False):
        """
        Convert from a phased dataset to a drift dataset.
        See the phasing memo under docs/references for more documentation.

        Args:
            phase_frame: the astropy frame to phase from. Either 'icrs' or 'gcrs'.
                'gcrs' accounts for precession & nutation, 'icrs' also includes abberation.
                Defaults to using the 'phase_center_frame' attribute or 'icrs'
                if that attribute is None
            use_ant_pos: If True, calculate the uvws directly from the
                antenna positions rather than from the existing uvws.
        """
        if self.phase_type == 'phased':
            pass
        elif self.phase_type == 'drift':
            raise ValueError('The data is already drift scanning; can only '
                             'unphase phased data.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to drift or phased to '
                             'reflect the phasing status of the data')

        if phase_frame is None:
            if self.phase_center_frame is not None:
                phase_frame = self.phase_center_frame
            else:
                phase_frame = 'icrs'

        icrs_coord = SkyCoord(ra=self.phase_center_ra, dec=self.phase_center_dec,
                              unit='radian', frame='icrs')
        if phase_frame == 'icrs':
            frame_phase_center = icrs_coord
        else:
            # use center of observation for obstime for gcrs
            center_time = np.mean([np.max(self.time_array), np.min(self.time_array)])
            icrs_coord.obstime = Time(center_time, format='jd')
            frame_phase_center = icrs_coord.transform_to('gcrs')

        # This promotion is REQUIRED to get the right answer when we
        # add in the telescope location for ICRS
        # In some cases, the uvws are already float64, but sometimes they're not
        self.uvw_array = np.float64(self.uvw_array)

        # apply -w phasor
        w_lambda = (self.uvw_array[:, 2].reshape(self.Nblts, 1)
                    / const.c.to('m/s').value * self.freq_array.reshape(1, self.Nfreqs))
        phs = np.exp(-1j * 2 * np.pi * (-1) * w_lambda[:, None, :, None])
        self.data_array *= phs

        unique_times, unique_inds = np.unique(self.time_array, return_index=True)
        for ind, jd in enumerate(unique_times):
            inds = np.where(self.time_array == jd)[0]

            obs_time = Time(jd, format='jd')

            itrs_telescope_location = SkyCoord(x=self.telescope_location[0] * units.m,
                                               y=self.telescope_location[1] * units.m,
                                               z=self.telescope_location[2] * units.m,
                                               representation='cartesian',
                                               frame='itrs', obstime=obs_time)
            frame_telescope_location = itrs_telescope_location.transform_to(phase_frame)
            itrs_lat_lon_alt = self.telescope_location_lat_lon_alt

            if use_ant_pos:
                ant_uvw = uvutils.phase_uvw(self.telescope_location_lat_lon_alt[1],
                                            self.telescope_location_lat_lon_alt[0],
                                            self.antenna_positions)

                for bl_ind in inds:
                    ant1_index = np.where(self.antenna_numbers == self.ant_1_array[bl_ind])[0][0]
                    ant2_index = np.where(self.antenna_numbers == self.ant_2_array[bl_ind])[0][0]
                    self.uvw_array[bl_ind, :] = ant_uvw[ant2_index, :] - ant_uvw[ant1_index, :]

            else:
                uvws_use = self.uvw_array[inds, :]

                uvw_rel_positions = uvutils.unphase_uvw(frame_phase_center.ra.rad,
                                                        frame_phase_center.dec.rad,
                                                        uvws_use)

                frame_telescope_location.representation = 'cartesian'

                frame_uvw_coord = SkyCoord(x=uvw_rel_positions[:, 0] * units.m + frame_telescope_location.x,
                                           y=uvw_rel_positions[:, 1] * units.m + frame_telescope_location.y,
                                           z=uvw_rel_positions[:, 2] * units.m + frame_telescope_location.z,
                                           representation='cartesian',
                                           frame=phase_frame, obstime=obs_time)

                itrs_uvw_coord = frame_uvw_coord.transform_to('itrs')

                # now convert them to ENU, which is the space uvws are in
                self.uvw_array[inds, :] = uvutils.ENU_from_ECEF(itrs_uvw_coord.cartesian.get_xyz().value.T,
                                                                *itrs_lat_lon_alt)

        # remove phase center
        self.phase_center_frame = None
        self.phase_center_ra = None
        self.phase_center_dec = None
        self.phase_center_epoch = None
        self.set_drift()

    def phase(self, ra, dec, epoch='J2000', phase_frame='icrs', use_ant_pos=False):
        """"
        Phase a drift scan dataset to a single ra/dec at a particular epoch.
        See the phasing memo under docs/references for more documentation.

        Tested against MWA_Tools/CONV2UVFITS/convutils.
        Will not phase already phased data.

        Args:
            ra: The ra to phase to in radians.
            dec: The dec to phase to in radians.
            epoch: The epoch to use for phasing.
                Either an astropy Time object or the string "J2000" (which is the default).
                Note that the epoch is only used to evaluate the ra & dec values,
                if the epoch is not J2000, the ra & dec values are interpreted
                as FK5 ra/dec values and translated to J2000, the data are then
                phased to the J2000 ra/dec values.
            phase_frame: the astropy frame to phase to. Either 'icrs' or 'gcrs'.
                'gcrs' accounts for precession & nutation,
                'icrs' accounts for precession, nutation & abberation.
                Default is 'icrs'.
            use_ant_pos: If True, calculate the uvws directly from the
                antenna positions rather than from the existing uvws.
        """
        if self.phase_type == 'drift':
            pass
        elif self.phase_type == 'phased':
            raise ValueError('The data is already phased; can only phase '
                             'drift scan data. Use unphase_to_drift to '
                             'convert to a drift scan.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to "drift" or "phased" to '
                             'reflect the phasing status of the data')

        if phase_frame not in ['icrs', 'gcrs']:
            raise ValueError('phase_frame can only be set to icrs or gcrs.')

        if epoch == "J2000" or epoch == 2000:
            icrs_coord = SkyCoord(ra=ra, dec=dec, unit='radian', frame='icrs')
        else:
            assert(isinstance(epoch, Time))
            phase_center_coord = SkyCoord(ra=ra, dec=dec, unit='radian',
                                          equinox=epoch, frame=FK5)
            # convert to icrs (i.e. J2000) to write to object
            icrs_coord = phase_center_coord.transform_to('icrs')

        self.phase_center_ra = icrs_coord.ra.radian
        self.phase_center_dec = icrs_coord.dec.radian
        self.phase_center_epoch = 2000.0

        if phase_frame == 'icrs':
            frame_phase_center = icrs_coord
        else:
            # use center of observation for obstime for gcrs
            center_time = np.mean([np.max(self.time_array), np.min(self.time_array)])
            icrs_coord.obstime = Time(center_time, format='jd')
            frame_phase_center = icrs_coord.transform_to('gcrs')

        # This promotion is REQUIRED to get the right answer when we
        # add in the telescope location for ICRS
        self.uvw_array = np.float64(self.uvw_array)

        unique_times, unique_inds = np.unique(self.time_array, return_index=True)
        for ind, jd in enumerate(unique_times):
            inds = np.where(self.time_array == jd)[0]

            obs_time = Time(jd, format='jd')

            itrs_telescope_location = SkyCoord(x=self.telescope_location[0] * units.m,
                                               y=self.telescope_location[1] * units.m,
                                               z=self.telescope_location[2] * units.m,
                                               representation='cartesian',
                                               frame='itrs', obstime=obs_time)
            itrs_lat_lon_alt = self.telescope_location_lat_lon_alt

            frame_telescope_location = itrs_telescope_location.transform_to(phase_frame)

            frame_telescope_location.representation = 'cartesian'

            if use_ant_pos:
                # This promotion is REQUIRED to get the right answer when we
                # add in the telescope location for ICRS
                ecef_ant_pos = np.float64(self.antenna_positions) + self.telescope_location

                itrs_ant_coord = SkyCoord(x=ecef_ant_pos[:, 0] * units.m,
                                          y=ecef_ant_pos[:, 1] * units.m,
                                          z=ecef_ant_pos[:, 2] * units.m,
                                          representation='cartesian',
                                          frame='itrs', obstime=obs_time)

                frame_ant_coord = itrs_ant_coord.transform_to(phase_frame)

                frame_ant_rel = (frame_ant_coord.cartesian
                                 - frame_telescope_location.cartesian).get_xyz().T.value

                frame_ant_uvw = uvutils.phase_uvw(frame_phase_center.ra.rad,
                                                  frame_phase_center.dec.rad,
                                                  frame_ant_rel)

                for bl_ind in inds:
                    ant1_index = np.where(self.antenna_numbers == self.ant_1_array[bl_ind])[0][0]
                    ant2_index = np.where(self.antenna_numbers == self.ant_2_array[bl_ind])[0][0]
                    self.uvw_array[bl_ind, :] = frame_ant_uvw[ant2_index, :] - frame_ant_uvw[ant1_index, :]
            else:
                # Also, uvws should be thought of like ENU, not ECEF (or rotated ECEF)
                # convert them to ECEF to transform between frames
                uvws_use = self.uvw_array[inds, :]

                uvw_ecef = uvutils.ECEF_from_ENU(uvws_use, *itrs_lat_lon_alt)

                itrs_uvw_coord = SkyCoord(x=uvw_ecef[:, 0] * units.m,
                                          y=uvw_ecef[:, 1] * units.m,
                                          z=uvw_ecef[:, 2] * units.m,
                                          representation='cartesian',
                                          frame='itrs', obstime=obs_time)
                frame_uvw_coord = itrs_uvw_coord.transform_to(phase_frame)

                # this takes out the telescope location in the new frame,
                # so these are vectors again
                frame_rel_uvw = (frame_uvw_coord.cartesian.get_xyz().value.T
                                 - frame_telescope_location.cartesian.get_xyz().value)

                self.uvw_array[inds, :] = uvutils.phase_uvw(frame_phase_center.ra.rad,
                                                            frame_phase_center.dec.rad,
                                                            frame_rel_uvw)

        # calculate data and apply phasor
        w_lambda = (self.uvw_array[:, 2].reshape(self.Nblts, 1)
                    / const.c.to('m/s').value * self.freq_array.reshape(1, self.Nfreqs))
        phs = np.exp(-1j * 2 * np.pi * w_lambda[:, None, :, None])
        self.data_array *= phs

        self.phase_center_frame = phase_frame
        self.set_phased()

    def phase_to_time(self, time, phase_frame='icrs', use_ant_pos=False):
        """
        Phase a drift scan dataset to the ra/dec of zenith at a particular time.
        See the phasing memo under docs/references for more documentation.

        Args:
            time: The time to phase to, an astropy Time object.
            phase_frame: the astropy frame to phase to. Either 'icrs' or 'gcrs'.
                'gcrs' accounts for precession & nutation,
                'icrs' accounts for precession, nutation & abberation.
                Default is 'icrs'.
            use_ant_pos: If True, calculate the uvws directly from the
                antenna positions rather than from the existing uvws.
        """
        if self.phase_type == 'drift':
            pass
        elif self.phase_type == 'phased':
            raise ValueError('The data is already phased; can only phase '
                             'drift scanning data.')
        else:
            raise ValueError('The phasing type of the data is unknown. '
                             'Set the phase_type to drift or phased to '
                             'reflect the phasing status of the data')

        if not isinstance(time, Time):
            raise(TypeError, "time must be an astropy.time.Time object")

        # Generate ra/dec of zenith at time in the phase_frame coordinate system
        # to use for phasing
        telescope_location = EarthLocation.from_geocentric(self.telescope_location[0],
                                                           self.telescope_location[1],
                                                           self.telescope_location[2],
                                                           unit='m')

        zenith_coord = SkyCoord(alt=Angle(90 * units.deg), az=Angle(0 * units.deg),
                                obstime=time, frame='altaz', location=telescope_location)

        obs_zenith_coord = zenith_coord.transform_to(phase_frame)
        zenith_ra = obs_zenith_coord.ra
        zenith_dec = obs_zenith_coord.dec

        self.phase(zenith_ra, zenith_dec, epoch='J2000', phase_frame=phase_frame,
                   use_ant_pos=use_ant_pos)

    def set_uvws_from_antenna_positions(self, allow_phasing=False,
                                        orig_phase_frame=None,
                                        output_phase_frame='icrs'):
        """
        Calculate UVWs based on antenna_positions

        Args:
            allow_phasing: Option for phased data. If data is phased and
                allow_phasing is set, data will be unphased, UVWs will be
                calculated, and then data will be rephased. Script will error
                if data is phased and allow_phasing is not set. Default is
                False.
            orig_phase_frame: The astropy frame to phase from. Either 'icrs' or
                'gcrs'. Defaults to using the 'phase_center_frame' attribute
                or 'icrs' if that attribute is None. Applied only if
                allow_phasing=True.
            output_phase_frame: The astropy frame to phase to. Either 'icrs' or
                'gcrs'. Default is 'icrs'. Applied only if allow_phasing=True.
        """
        phase_type = self.phase_type
        if phase_type == 'phased':
            if allow_phasing:
                warnings.warn('Warning: Data will be unphased and rephased '
                              'to calculate UVWs.'
                              )
                if orig_phase_frame not in [None, 'icrs', 'gcrs']:
                    raise ValueError('Invalid parameter orig_phase_frame. '
                                     'Options are "icrs", "gcrs", or None.')
                if output_phase_frame not in ['icrs', 'gcrs']:
                    raise ValueError('Invalid parameter output_phase_frame. '
                                     'Options are "icrs" or "gcrs".')
                phase_center_ra = self.phase_center_ra
                phase_center_dec = self.phase_center_dec
                phase_center_epoch = self.phase_center_epoch
                self.unphase_to_drift(phase_frame=orig_phase_frame)
            else:
                raise ValueError('UVW calculation requires unphased data. '
                                 'Use unphase_to_drift or set '
                                 'allow_phasing=True.'
                                 )
        antenna_locs_ENU = uvutils.ENU_from_ECEF(
            (self.antenna_positions + self.telescope_location),
            *self.telescope_location_lat_lon_alt)
        uvw_array = np.zeros((self.baseline_array.size, 3))
        for baseline in list(set(self.baseline_array)):
            baseline_inds = np.where(self.baseline_array == baseline)[0]
            ant1_index = np.where(self.antenna_numbers
                                  == self.ant_1_array[baseline_inds[0]])[0][0]
            ant2_index = np.where(self.antenna_numbers
                                  == self.ant_2_array[baseline_inds[0]])[0][0]
            uvw_array[baseline_inds, :] = (antenna_locs_ENU[ant2_index, :]
                                           - antenna_locs_ENU[ant1_index, :])
        self.uvw_array = uvw_array
        if phase_type == 'phased':
            self.phase(phase_center_ra, phase_center_dec, phase_center_epoch,
                       phase_frame=output_phase_frame)

    def __add__(self, other, run_check=True, check_extra=True,
                run_check_acceptability=True, inplace=False):
        """
        Combine two UVData objects. Objects can be added along frequency,
        polarization, and/or baseline-time axis.

        Args:
            other: Another UVData object which will be added to self.
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
        # Check that both objects are UVData and valid
        this.check(check_extra=check_extra, run_check_acceptability=run_check_acceptability)
        if not issubclass(other.__class__, this.__class__):
            if not issubclass(this.__class__, other.__class__):
                raise ValueError('Only UVData (or subclass) objects can be '
                                 'added to a UVData (or subclass) object')
        other.check(check_extra=check_extra, run_check_acceptability=run_check_acceptability)

        # Define parameters that must be the same to add objects
        # But phase_center should be the same, even if in drift (empty parameters)
        compatibility_params = ['_vis_units', '_channel_width', '_object_name',
                                '_telescope_name', '_instrument',
                                '_telescope_location', '_phase_type',
                                '_Nants_telescope', '_antenna_names',
                                '_antenna_numbers', '_antenna_positions',
                                '_phase_center_ra', '_phase_center_dec',
                                '_phase_center_epoch']

        # Build up history string
        history_update_string = ' Combined data along '
        n_axes = 0

        # Create blt arrays for convenience
        prec_t = - 2 * \
            np.floor(np.log10(this._time_array.tols[-1])).astype(int)
        prec_b = 8
        this_blts = np.array(["_".join(["{1:.{0}f}".format(prec_t, blt[0]),
                                        str(blt[1]).zfill(prec_b)]) for blt in
                              zip(this.time_array, this.baseline_array)])
        other_blts = np.array(["_".join(["{1:.{0}f}".format(prec_t, blt[0]),
                                         str(blt[1]).zfill(prec_b)]) for blt in
                               zip(other.time_array, other.baseline_array)])
        # Check we don't have overlapping data
        both_pol = np.intersect1d(
            this.polarization_array, other.polarization_array)
        both_freq = np.intersect1d(
            this.freq_array[0, :], other.freq_array[0, :])
        both_blts = np.intersect1d(this_blts, other_blts)
        if len(both_pol) > 0:
            if len(both_freq) > 0:
                if len(both_blts) > 0:
                    raise ValueError('These objects have overlapping data and'
                                     ' cannot be combined.')

        temp = np.nonzero(~np.in1d(other_blts, this_blts))[0]
        if len(temp) > 0:
            bnew_inds = temp
            new_blts = other_blts[temp]
            history_update_string += 'baseline-time'
            n_axes += 1
        else:
            bnew_inds, new_blts = ([], [])
            # add metadata to be checked to compatibility params
            extra_params = ['_integration_time', '_uvw_array', '_lst_array']
            compatibility_params.extend(extra_params)

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

        # Actually check compatibility parameters
        for a in compatibility_params:
            if getattr(this, a) != getattr(other, a):
                msg = 'UVParameter ' + \
                    a[1:] + ' does not match. Cannot combine objects.'
                raise ValueError(msg)

        # Pad out self to accommodate new data
        if len(bnew_inds) > 0:
            this_blts = np.concatenate((this_blts, new_blts))
            blt_order = np.argsort(this_blts)
            zero_pad = np.zeros(
                (len(bnew_inds), this.Nspws, this.Nfreqs, this.Npols))
            this.data_array = np.concatenate([this.data_array, zero_pad], axis=0)
            this.nsample_array = np.concatenate([this.nsample_array, zero_pad], axis=0)
            this.flag_array = np.concatenate([this.flag_array,
                                              1 - zero_pad], axis=0).astype(np.bool)
            this.uvw_array = np.concatenate([this.uvw_array,
                                             other.uvw_array[bnew_inds, :]], axis=0)[blt_order, :]
            this.time_array = np.concatenate([this.time_array,
                                              other.time_array[bnew_inds]])[blt_order]
            this.integration_time = np.concatenate([this.integration_time,
                                                    other.integration_time[bnew_inds]])[blt_order]
            this.lst_array = np.concatenate(
                [this.lst_array, other.lst_array[bnew_inds]])[blt_order]
            this.ant_1_array = np.concatenate([this.ant_1_array,
                                               other.ant_1_array[bnew_inds]])[blt_order]
            this.ant_2_array = np.concatenate([this.ant_2_array,
                                               other.ant_2_array[bnew_inds]])[blt_order]
            this.baseline_array = np.concatenate([this.baseline_array,
                                                  other.baseline_array[bnew_inds]])[blt_order]

        if len(fnew_inds) > 0:
            zero_pad = np.zeros((this.data_array.shape[0], this.Nspws, len(fnew_inds),
                                 this.Npols))
            this.freq_array = np.concatenate([this.freq_array,
                                              other.freq_array[:, fnew_inds]], axis=1)
            f_order = np.argsort(this.freq_array[0, :])
            this.data_array = np.concatenate([this.data_array, zero_pad], axis=2)
            this.nsample_array = np.concatenate([this.nsample_array, zero_pad], axis=2)
            this.flag_array = np.concatenate([this.flag_array, 1 - zero_pad],
                                             axis=2).astype(np.bool)
        if len(pnew_inds) > 0:
            zero_pad = np.zeros((this.data_array.shape[0], this.Nspws,
                                 this.data_array.shape[2], len(pnew_inds)))
            this.polarization_array = np.concatenate([this.polarization_array,
                                                      other.polarization_array[pnew_inds]])
            p_order = np.argsort(np.abs(this.polarization_array))
            this.data_array = np.concatenate([this.data_array, zero_pad], axis=3)
            this.nsample_array = np.concatenate([this.nsample_array, zero_pad], axis=3)
            this.flag_array = np.concatenate([this.flag_array, 1 - zero_pad],
                                             axis=3).astype(np.bool)

        # Now populate the data
        pol_t2o = np.nonzero(
            np.in1d(this.polarization_array, other.polarization_array))[0]
        freq_t2o = np.nonzero(
            np.in1d(this.freq_array[0, :], other.freq_array[0, :]))[0]
        blt_t2o = np.nonzero(np.in1d(this_blts, other_blts))[0]
        this.data_array[np.ix_(blt_t2o, [0], freq_t2o,
                               pol_t2o)] = other.data_array
        this.nsample_array[np.ix_(
            blt_t2o, [0], freq_t2o, pol_t2o)] = other.nsample_array
        this.flag_array[np.ix_(blt_t2o, [0], freq_t2o,
                               pol_t2o)] = other.flag_array
        if len(bnew_inds) > 0:
            this.data_array = this.data_array[blt_order, :, :, :]
            this.nsample_array = this.nsample_array[blt_order, :, :, :]
            this.flag_array = this.flag_array[blt_order, :, :, :]
        if len(fnew_inds) > 0:
            this.freq_array = this.freq_array[:, f_order]
            this.data_array = this.data_array[:, :, f_order, :]
            this.nsample_array = this.nsample_array[:, :, f_order, :]
            this.flag_array = this.flag_array[:, :, f_order, :]
        if len(pnew_inds) > 0:
            this.polarization_array = this.polarization_array[p_order]
            this.data_array = this.data_array[:, :, :, p_order]
            this.nsample_array = this.nsample_array[:, :, :, p_order]
            this.flag_array = this.flag_array[:, :, :, p_order]

        # Update N parameters (e.g. Npols)
        this.Ntimes = len(np.unique(this.time_array))
        this.Nbls = len(np.unique(this.baseline_array))
        this.Nblts = this.uvw_array.shape[0]
        this.Nfreqs = this.freq_array.shape[1]
        this.Npols = this.polarization_array.shape[0]
        this.Nants_data = len(
            np.unique(this.ant_1_array.tolist() + this.ant_2_array.tolist()))

        # Check specific requirements
        if this.Nfreqs > 1:
            freq_separation = np.diff(this.freq_array[0, :])
            if not np.isclose(np.min(freq_separation), np.max(freq_separation),
                              rtol=this._freq_array.tols[0], atol=this._freq_array.tols[1]):
                warnings.warn('Combined frequencies are not evenly spaced. This will '
                              'make it impossible to write this data out to some file types.')
            elif np.max(freq_separation) > this.channel_width:
                warnings.warn('Combined frequencies are not contiguous. This will make '
                              'it impossible to write this data out to some file types.')

        if this.Npols > 2:
            pol_separation = np.diff(this.polarization_array)
            if np.min(pol_separation) < np.max(pol_separation):
                warnings.warn('Combined polarizations are not evenly spaced. This will '
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
            other: Another UVData object which will be added to self.
        """
        self.__add__(other, inplace=True)
        return self

    def _select_preprocess(self, antenna_nums, antenna_names, ant_str, bls,
                           frequencies, freq_chans, times, polarizations, blt_inds):
        """
        Internal function to build up blt_inds, freq_inds, pol_inds
        and history_update_string for select.
        """
        # build up history string as we go
        history_update_string = '  Downselected to specific '
        n_selects = 0

        if ant_str is not None:
            if not (antenna_nums is None and antenna_names is None
                    and bls is None and polarizations is None):
                raise ValueError(
                    'Cannot provide ant_str with antenna_nums, antenna_names, '
                    'bls, or polarizations.')
            else:
                bls, polarizations = self.parse_ants(ant_str)

        # Antennas, times and blt_inds all need to be combined into a set of
        # blts indices to keep.

        # test for blt_inds presence before adding inds from antennas & times
        if blt_inds is not None:
            blt_inds = uvutils._get_iterable(blt_inds)
            if np.array(blt_inds).ndim > 1:
                blt_inds = np.array(blt_inds).flatten()
            history_update_string += 'baseline-times'
            n_selects += 1

        if antenna_names is not None:
            if antenna_nums is not None:
                raise ValueError(
                    'Only one of antenna_nums and antenna_names can be provided.')

            if not isinstance(antenna_names, (list, tuple, np.ndarray)):
                antenna_names = (antenna_names,)
            if np.array(antenna_names).ndim > 1:
                antenna_names = np.array(antenna_names).flatten()
            antenna_nums = []
            for s in antenna_names:
                if s not in self.antenna_names:
                    raise ValueError(
                        'Antenna name {a} is not present in the antenna_names array'.format(a=s))
                antenna_nums.append(self.antenna_numbers[np.where(
                    np.array(self.antenna_names) == s)][0])

        if antenna_nums is not None:
            antenna_nums = uvutils._get_iterable(antenna_nums)
            if np.array(antenna_nums).ndim > 1:
                antenna_nums = np.array(antenna_nums).flatten()
            if n_selects > 0:
                history_update_string += ', antennas'
            else:
                history_update_string += 'antennas'
            n_selects += 1
            inds1 = np.zeros(0, dtype=np.int)
            inds2 = np.zeros(0, dtype=np.int)
            for ant in antenna_nums:
                if ant in self.ant_1_array or ant in self.ant_2_array:
                    wh1 = np.where(self.ant_1_array == ant)[0]
                    wh2 = np.where(self.ant_2_array == ant)[0]
                    if len(wh1) > 0:
                        inds1 = np.append(inds1, list(wh1))
                    if len(wh2) > 0:
                        inds2 = np.append(inds2, list(wh2))
                else:
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=ant))

            ant_blt_inds = np.array(
                list(set(inds1).intersection(inds2)), dtype=np.int)
        else:
            ant_blt_inds = None

        if bls is not None:
            if isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                bls = [bls]
            if not all(isinstance(item, tuple) for item in bls):
                raise ValueError(
                    'bls must be a list of tuples of antenna numbers (optionally with polarization).')
            if not all([isinstance(item[0], six.integer_types + (np.integer,)) for item in bls]
                       + [isinstance(item[1], six.integer_types + (np.integer,)) for item in bls]):
                raise ValueError(
                    'bls must be a list of tuples of antenna numbers (optionally with polarization).')
            if all([len(item) == 3 for item in bls]):
                if polarizations is not None:
                    raise ValueError('Cannot provide length-3 tuples and also specify polarizations.')
                if not all([isinstance(item[2], str) for item in bls]):
                    raise ValueError('The third element in each bl must be a polarization string')

            if ant_str is None:
                if n_selects > 0:
                    history_update_string += ', baselines'
                else:
                    history_update_string += 'baselines'
            else:
                history_update_string += 'antenna pairs'
            n_selects += 1
            bls_blt_inds = np.zeros(0, dtype=np.int)
            bl_pols = set()
            for bl in bls:
                if not (bl[0] in self.ant_1_array or bl[0] in self.ant_2_array):
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=bl[0]))
                if not (bl[1] in self.ant_1_array or bl[1] in self.ant_2_array):
                    raise ValueError('Antenna number {a} is not present in the '
                                     'ant_1_array or ant_2_array'.format(a=bl[1]))
                wh1 = np.where(np.logical_and(
                    self.ant_1_array == bl[0], self.ant_2_array == bl[1]))[0]
                wh2 = np.where(np.logical_and(
                    self.ant_1_array == bl[1], self.ant_2_array == bl[0]))[0]
                if len(wh1) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh1))
                    if len(bl) == 3:
                        bl_pols.add(bl[2])
                elif len(wh2) > 0:
                    bls_blt_inds = np.append(bls_blt_inds, list(wh2))
                    if len(bl) == 3:
                        bl_pols.add(bl[2][::-1])  # reverse polarization string
                else:
                    raise ValueError('Antenna pair {p} does not have any data '
                                     'associated with it.'.format(p=bl))
            if len(bl_pols) > 0:
                polarizations = list(bl_pols)

            if ant_blt_inds is not None:
                # Use union (or) to join antenna_names/nums & ant_pairs_nums
                ant_blt_inds = np.array(list(set(ant_blt_inds).union(bls_blt_inds)))
            else:
                ant_blt_inds = bls_blt_inds

        if ant_blt_inds is not None:
            if blt_inds is not None:
                # Use intesection (and) to join antenna_names/nums/ant_pairs_nums with blt_inds
                # handled differently because of the time aspect (which is anded with antennas below)
                blt_inds = np.array(
                    list(set(blt_inds).intersection(ant_blt_inds)), dtype=np.int)
            else:
                blt_inds = ant_blt_inds

        if times is not None:
            times = uvutils._get_iterable(times)
            if np.array(times).ndim > 1:
                times = np.array(times).flatten()
            if n_selects > 0:
                history_update_string += ', times'
            else:
                history_update_string += 'times'
            n_selects += 1

            time_blt_inds = np.zeros(0, dtype=np.int)
            for jd in times:
                if jd in self.time_array:
                    time_blt_inds = np.append(
                        time_blt_inds, np.where(self.time_array == jd)[0])
                else:
                    raise ValueError(
                        'Time {t} is not present in the time_array'.format(t=jd))

            if blt_inds is not None:
                # Use intesection (and) to join antenna_names/nums/ant_pairs_nums/blt_inds with times
                blt_inds = np.array(
                    list(set(blt_inds).intersection(time_blt_inds)), dtype=np.int)
            else:
                blt_inds = time_blt_inds

        if blt_inds is not None:

            if len(blt_inds) == 0:
                raise ValueError(
                    'No baseline-times were found that match criteria')
            if max(blt_inds) >= self.Nblts:
                raise ValueError(
                    'blt_inds contains indices that are too large')
            if min(blt_inds) < 0:
                print(blt_inds)
                raise ValueError('blt_inds contains indices that are negative')

            blt_inds = list(sorted(set(list(blt_inds))))

        if freq_chans is not None:
            freq_chans = uvutils._get_iterable(freq_chans)
            if np.array(freq_chans).ndim > 1:
                freq_chans = np.array(freq_chans).flatten()
            if frequencies is None:
                frequencies = self.freq_array[0, freq_chans]
            else:
                frequencies = uvutils._get_iterable(frequencies)
                frequencies = np.sort(list(set(frequencies)
                                           | set(self.freq_array[0, freq_chans])))

        if frequencies is not None:
            frequencies = uvutils._get_iterable(frequencies)
            if np.array(frequencies).ndim > 1:
                frequencies = np.array(frequencies).flatten()
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
                    freq_inds = np.append(
                        freq_inds, np.where(freq_arr_use == f)[0])
                else:
                    raise ValueError(
                        'Frequency {f} is not present in the freq_array'.format(f=f))

            if len(frequencies) > 1:
                freq_ind_separation = freq_inds[1:] - freq_inds[:-1]
                if np.min(freq_ind_separation) < np.max(freq_ind_separation):
                    warnings.warn('Selected frequencies are not evenly spaced. This '
                                  'will make it impossible to write this data out to '
                                  'some file types')
                elif np.max(freq_ind_separation) > 1:
                    warnings.warn('Selected frequencies are not contiguous. This '
                                  'will make it impossible to write this data out to '
                                  'some file types.')

            freq_inds = list(sorted(set(list(freq_inds))))
        else:
            freq_inds = None

        if polarizations is not None:
            polarizations = uvutils._get_iterable(polarizations)
            if np.array(polarizations).ndim > 1:
                polarizations = np.array(polarizations).flatten()
            if n_selects > 0:
                history_update_string += ', polarizations'
            else:
                history_update_string += 'polarizations'
            n_selects += 1

            pol_inds = np.zeros(0, dtype=np.int)
            for p in polarizations:
                if isinstance(p, str):
                    p_num = uvutils.polstr2num(p)
                else:
                    p_num = p
                if p_num in self.polarization_array:
                    pol_inds = np.append(pol_inds, np.where(
                        self.polarization_array == p_num)[0])
                else:
                    raise ValueError(
                        'Polarization {p} is not present in the polarization_array'.format(p=p))

            if len(pol_inds) > 2:
                pol_ind_separation = pol_inds[1:] - pol_inds[:-1]
                if np.min(pol_ind_separation) < np.max(pol_ind_separation):
                    warnings.warn('Selected polarization values are not evenly spaced. This '
                                  'will make it impossible to write this data out to '
                                  'some file types')

            pol_inds = list(sorted(set(list(pol_inds))))
        else:
            pol_inds = None

        history_update_string += ' using pyuvdata.'

        return blt_inds, freq_inds, pol_inds, history_update_string

    def _select_metadata(self, blt_inds, freq_inds, pol_inds, history_update_string):
        """
        Internal function to perform select on everything except the data_array,
        flag_array and nsample_array to allow for re-use in uvfits reading
        """
        if blt_inds is not None:
            self.Nblts = len(blt_inds)
            self.baseline_array = self.baseline_array[blt_inds]
            self.Nbls = len(np.unique(self.baseline_array))
            self.time_array = self.time_array[blt_inds]
            self.integration_time = self.integration_time[blt_inds]
            self.lst_array = self.lst_array[blt_inds]
            self.uvw_array = self.uvw_array[blt_inds, :]

            self.ant_1_array = self.ant_1_array[blt_inds]
            self.ant_2_array = self.ant_2_array[blt_inds]
            self.Nants_data = int(
                len(set(self.ant_1_array.tolist() + self.ant_2_array.tolist())))

            self.Ntimes = len(np.unique(self.time_array))

        if freq_inds is not None:
            self.Nfreqs = len(freq_inds)
            self.freq_array = self.freq_array[:, freq_inds]

        if pol_inds is not None:
            self.Npols = len(pol_inds)
            self.polarization_array = self.polarization_array[pol_inds]

        self.history = self.history + history_update_string

    def select(self, antenna_nums=None, antenna_names=None, ant_str=None,
               bls=None, frequencies=None, freq_chans=None,
               times=None, polarizations=None, blt_inds=None, run_check=True,
               check_extra=True, run_check_acceptability=True, inplace=True):
        """
        Select specific antennas, antenna pairs, frequencies, times and
        polarizations to keep in the object while discarding others.

        Also supports selecting specific baseline-time indices to keep while
        discarding others, but this is not commonly used. The history attribute
        on the object will be updated to identify the operations performed.

        Args:
            antenna_nums: The antennas numbers to keep in the object (antenna
                positions and names for the removed antennas will be retained).
                This cannot be provided if antenna_names is also provided.
            antenna_names: The antennas names to keep in the object (antenna
                positions and names for the removed antennas will be retained).
                This cannot be provided if antenna_nums is also provided.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None.
            ant_str: A string containing information about what antenna numbers
                and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
                or combinations of antenna numbers and polarizations (e.g. '1',
                '1_2', '1x_2y').  See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be kept for both baselines (1,2) and (2,3) to return a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.  If this occurs, select raises a ValueError.
            frequencies: The frequencies to keep in the object.
            freq_chans: The frequency channel numbers to keep in the object.
            times: The times to keep in the object.
            polarizations: The polarizations to keep in the object.
            blt_inds: The baseline-time indices to keep in the object. This is
                not commonly used.
            run_check: Option to check for the existence and proper shapes of
                parameters after downselecting data on this object. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after downselecting data on this object. Default is True.
            inplace: Option to perform the select directly on self (True, default) or return
                a new UVData object, which is a subselection of self (False)
        """
        if inplace:
            uv_object = self
        else:
            uv_object = copy.deepcopy(self)

        blt_inds, freq_inds, pol_inds, history_update_string = \
            uv_object._select_preprocess(antenna_nums, antenna_names, ant_str, bls,
                                         frequencies, freq_chans, times, polarizations, blt_inds)

        # do select operations on everything except data_array, flag_array and nsample_array
        uv_object._select_metadata(blt_inds, freq_inds, pol_inds, history_update_string)

        if blt_inds is not None:
            uv_object.data_array = uv_object.data_array[blt_inds, :, :, :]
            uv_object.flag_array = uv_object.flag_array[blt_inds, :, :, :]
            uv_object.nsample_array = uv_object.nsample_array[blt_inds, :, :, :]

        if freq_inds is not None:
            uv_object.data_array = uv_object.data_array[:, :, freq_inds, :]
            uv_object.flag_array = uv_object.flag_array[:, :, freq_inds, :]
            uv_object.nsample_array = uv_object.nsample_array[:, :, freq_inds, :]

        if pol_inds is not None:
            uv_object.data_array = uv_object.data_array[:, :, :, pol_inds]
            uv_object.flag_array = uv_object.flag_array[:, :, :, pol_inds]
            uv_object.nsample_array = uv_object.nsample_array[:, :, :, pol_inds]

        # check if object is uv_object-consistent
        if run_check:
            uv_object.check(check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)

        if not inplace:
            return uv_object

    def _convert_from_filetype(self, other):
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        if filetype is 'uvfits':
            from . import uvfits
            other_obj = uvfits.UVFITS()
        elif filetype is 'fhd':
            from . import fhd
            other_obj = fhd.FHD()
        elif filetype is 'miriad':
            from . import miriad
            other_obj = miriad.Miriad()
        elif filetype is 'uvh5':
            from . import uvh5
            other_obj = uvh5.UVH5()
        else:
            raise ValueError('filetype must be uvfits, miriad, fhd, or uvh5')
        for p in self:
            param = getattr(self, p)
            setattr(other_obj, p, param)
        return other_obj

    def read_uvfits(self, filename, antenna_nums=None, antenna_names=None,
                    ant_str=None, bls=None, frequencies=None,
                    freq_chans=None, times=None, polarizations=None, blt_inds=None,
                    read_data=True, read_metadata=True, run_check=True,
                    check_extra=True, run_check_acceptability=True):
        """
        Read in header, metadata and data from uvfits file(s).

        Args:
            filename: The uvfits file or list of files to read from.
            antenna_nums: The antennas numbers to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided. Ignored if read_data is False.
            antenna_names: The antennas names to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided. Ignored if read_data is False.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None. Ignored if read_data is False.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when reading data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y').
                See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be kept for both baselines (1,2) and (2,3) to return a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.
                Ignored if read_data is False.
            frequencies: The frequencies to include when reading data into the
                object.  Ignored if read_data is False.
            freq_chans: The frequency channel numbers to include when reading
                data into the object. Ignored if read_data is False.
            times: The times to include when reading data into the object.
                Ignored if read_data is False.
            polarizations: The polarizations to include when reading data into
                the object.  Ignored if read_data is False.
            blt_inds: The baseline-time indices to include when reading data into
                the object. This is not commonly used. Ignored if read_data is False.
            read_data: Read in the visibility and flag data. If set to false,
                only the basic header info and metadata (if read_metadata is True)
                will be read in. Setting read_data to False results in an
                incompletely defined object (check will not pass). Default True.
            read_metadata: Read in metadata (times, baselines, uvws) as well as
                basic header info. Only used if read_data is False
                (metadata will be read if data is read). If both read_data and
                read_metadata are false, only basic header info is read in. Default True.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
                Ignored if read_data is False.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True. Ignored if read_data is False.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
                Ignored if read_data is False.
        """
        from . import uvfits
        # work out what function should be called depending on what's
        # already defined on the object
        if self.freq_array is not None:
            hdr_loaded = True
        else:
            hdr_loaded = False
        if self.data_array is not None:
            data_loaded = True
        else:
            data_loaded = False

        if not read_data and not read_metadata:
            # not reading data or metadata, use read_uvfits to get header
            func = 'read_uvfits'
        elif not read_data:
            # reading metadata but not data
            if hdr_loaded:
                # header already read, use read_uvfits_metadata
                # (which will error if the data have already been read)
                func = 'read_uvfits_metadata'
            else:
                # header not read, use read_uvfits
                func = 'read_uvfits'
        else:
            # reading data
            if hdr_loaded and not data_loaded:
                # header already read, data not read, use read_uvfits_data
                # (which will read metadata if it doesn't exist)
                func = 'read_uvfits_data'
            else:
                # header not read or object already fully defined,
                # use read_uvfits to get a new object
                func = 'read_uvfits'

        if isinstance(filename, (list, tuple)):
            if not read_data:
                raise ValueError('read_data cannot be False for a list of uvfits files')
            if func == 'read_uvfits_data':
                raise ValueError('A list of files cannot be used when just reading data')

            self.read_uvfits(filename[0], antenna_nums=antenna_nums,
                             antenna_names=antenna_names, ant_str=ant_str,
                             bls=bls, frequencies=frequencies,
                             freq_chans=freq_chans, times=times,
                             polarizations=polarizations, blt_inds=blt_inds,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
            if len(filename) > 1:
                for f in filename[1:]:
                    uv2 = UVData()
                    uv2.read_uvfits(f, antenna_nums=antenna_nums,
                                    antenna_names=antenna_names, ant_str=ant_str,
                                    bls=bls, frequencies=frequencies,
                                    freq_chans=freq_chans, times=times,
                                    polarizations=polarizations, blt_inds=blt_inds,
                                    run_check=run_check, check_extra=check_extra,
                                    run_check_acceptability=run_check_acceptability)
                    self += uv2
                del(uv2)
        else:
            if func == 'read_uvfits':
                uvfits_obj = uvfits.UVFITS()
                uvfits_obj.read_uvfits(filename, antenna_nums=antenna_nums,
                                       antenna_names=antenna_names, ant_str=ant_str,
                                       bls=bls, frequencies=frequencies,
                                       freq_chans=freq_chans, times=times,
                                       polarizations=polarizations, blt_inds=blt_inds,
                                       read_data=read_data, read_metadata=read_metadata,
                                       run_check=run_check, check_extra=check_extra,
                                       run_check_acceptability=run_check_acceptability)
                self._convert_from_filetype(uvfits_obj)
                del(uvfits_obj)
            elif func == 'read_uvfits_metadata':
                # can only be one file, it would have errored earlier because read_data=False
                uvfits_obj = self._convert_to_filetype('uvfits')
                uvfits_obj.read_uvfits_metadata(filename)
                self._convert_from_filetype(uvfits_obj)
                del(uvfits_obj)
            elif func == 'read_uvfits_data':
                uvfits_obj = self._convert_to_filetype('uvfits')
                uvfits_obj.read_uvfits_data(filename, antenna_nums=antenna_nums,
                                            antenna_names=antenna_names, ant_str=ant_str,
                                            bls=bls, frequencies=frequencies,
                                            freq_chans=freq_chans, times=times,
                                            polarizations=polarizations, blt_inds=blt_inds,
                                            run_check=run_check, check_extra=check_extra,
                                            run_check_acceptability=run_check_acceptability)
                self._convert_from_filetype(uvfits_obj)
                del(uvfits_obj)

    def write_uvfits(self, filename, spoof_nonessential=False,
                     force_phase=False, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Write the data to a uvfits file.

        Args:
            filename: The uvfits file to write to.
            spoof_nonessential: Option to spoof the values of optional
                UVParameters that are not set but are required for uvfits files.
                Default is False.
            force_phase: Option to automatically phase drift scan data to
                zenith of the first timestamp. Default is False.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
        """
        uvfits_obj = self._convert_to_filetype('uvfits')
        uvfits_obj.write_uvfits(filename, spoof_nonessential=spoof_nonessential,
                                force_phase=force_phase, run_check=run_check,
                                check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability)
        del(uvfits_obj)

    def read_ms(self, filepath, run_check=True, check_extra=True,
                run_check_acceptability=True, data_column='DATA', pol_order='AIPS'):
        """
        Read in data from a measurement set

        Args:
            filepath: The measurement set file directory or list of directories to read from.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check the values of parameters
                after reading in the file. Default is True.
            data_column: name of CASA data column to read into data_array.
                'DATA', 'MODEL', or 'CORRECTED_DATA'
            pol_order: specify whether you want polarizations ordered by
                'CASA' or 'AIPS' conventions.
        """
        from . import ms

        if isinstance(filepath, (list, tuple)):
            self.read_ms(filepath[0], run_check=run_check, check_extra=check_extra,
                         run_check_acceptability=run_check_acceptability,
                         data_column=data_column, pol_order=pol_order)
            if len(filepath) > 1:
                for f in filepath[1:]:
                    uv2 = UVData()
                    uv2.read_ms(f, run_check=run_check, check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability,
                                data_column=data_column, pol_order=pol_order)
                    self += uv2
                del(uv2)
        else:
            ms_obj = ms.MS()
            ms_obj.read_ms(filepath, run_check=run_check, check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability,
                           data_column=data_column, pol_order=pol_order)
            self._convert_from_filetype(ms_obj)
            del(ms_obj)

    def read_fhd(self, filelist, use_model=False, run_check=True, check_extra=True,
                 run_check_acceptability=True):
        """
        Read in data from a list of FHD files.

        Args:
            filelist: The list of FHD save files to read from. Must include at
                least one polarization file, a params file and a flag file.
                Can also be a list of lists to read multiple data sets.
            use_model: Option to read in the model visibilities rather than the
                dirty visibilities. Default is False.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        from . import fhd
        if isinstance(filelist[0], (list, tuple)):
            self.read_fhd(filelist[0], use_model=use_model, run_check=run_check,
                          check_extra=check_extra,
                          run_check_acceptability=run_check_acceptability)
            if len(filelist) > 1:
                for f in filelist[1:]:
                    uv2 = UVData()
                    uv2.read_fhd(f, use_model=use_model, run_check=run_check,
                                 check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)
                    self += uv2
                del(uv2)
        else:
            fhd_obj = fhd.FHD()
            fhd_obj.read_fhd(filelist, use_model=use_model, run_check=run_check,
                             check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(fhd_obj)
            del(fhd_obj)

    def read_miriad(self, filepath, antenna_nums=None, ant_str=None, bls=None,
                    polarizations=None, time_range=None, read_data=True,
                    phase_type=None, correct_lat_lon=True, run_check=True,
                    check_extra=True, run_check_acceptability=True):
        """
        Read in data from a miriad file.

        Args:
            filepath: The miriad file directory or list of directories to read from.
            antenna_nums: The antennas numbers to read into the object.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are
                provided, the polarizations argument below must be None.
            ant_str: A string containing information about what kinds of visibility data
                to read-in.  Can be 'auto', 'cross', 'all'. Cannot provide ant_str if
                antenna_nums and/or bls is not None.
            polarizations: List of polarization integers or strings to read-in.
                Ex: ['xx', 'yy', ...]
            time_range: len-2 list containing min and max range of times (Julian Date) to read-in.
                Ex: [2458115.20, 2458115.40]
            read_data: Read in the visibility and flag data. If set to false,
                only the metadata will be read in. Setting read_data to False
                results in an incompletely defined object (check will not pass).
                Default True.
            phase_type: Either 'drift' meaning zenith drift, 'phased' meaning
                the data are phased to a single RA/Dec or None and it will be
                guessed at based on the file. Default None.
            correct_lat_lon: flag -- that only matters if altitude is missing --
                to update the latitude and longitude from the known_telescopes list.
                Default True.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        from . import miriad
        if isinstance(filepath, (list, tuple)):
            if not read_data:
                raise ValueError('read_data cannot be False for a list of uvfits files')

            self.read_miriad(filepath[0], correct_lat_lon=correct_lat_lon,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability,
                             phase_type=phase_type, antenna_nums=antenna_nums,
                             ant_str=ant_str, bls=bls,
                             polarizations=polarizations, time_range=time_range)
            if len(filepath) > 1:
                for f in filepath[1:]:
                    uv2 = UVData()
                    uv2.read_miriad(f, correct_lat_lon=correct_lat_lon,
                                    run_check=run_check, check_extra=check_extra,
                                    run_check_acceptability=run_check_acceptability,
                                    phase_type=phase_type, antenna_nums=antenna_nums,
                                    ant_str=ant_str, bls=bls,
                                    polarizations=polarizations, time_range=time_range)
                    self += uv2
                del(uv2)
        else:
            # work out what function should be called
            if read_data:
                # reading data, use read_miriad
                miriad_obj = miriad.Miriad()
                miriad_obj.read_miriad(filepath, correct_lat_lon=correct_lat_lon,
                                       run_check=run_check, check_extra=check_extra,
                                       run_check_acceptability=run_check_acceptability,
                                       phase_type=phase_type, antenna_nums=antenna_nums,
                                       ant_str=ant_str, bls=bls,
                                       polarizations=polarizations, time_range=time_range)
                self._convert_from_filetype(miriad_obj)
                del(miriad_obj)
            else:
                # not reading data. Will error if data_array is already defined.
                miriad_obj = self._convert_to_filetype('miriad')
                miriad_obj.read_miriad_metadata(filepath, correct_lat_lon=correct_lat_lon)
                self._convert_from_filetype(miriad_obj)
                del(miriad_obj)

    def write_miriad(self, filepath, run_check=True, check_extra=True,
                     run_check_acceptability=True, clobber=False, no_antnums=False):
        """
        Write the data to a miriad file.

        Args:
            filename: The miriad file directory to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the filename if the file already exists.
                Default is False.
            no_antnums: Option to not write the antnums variable to the file.
                Should only be used for testing purposes.
        """
        miriad_obj = self._convert_to_filetype('miriad')
        miriad_obj.write_miriad(filepath, run_check=run_check, check_extra=check_extra,
                                run_check_acceptability=run_check_acceptability,
                                clobber=clobber, no_antnums=no_antnums)
        del(miriad_obj)

    def read_uvh5(self, filename, antenna_nums=None, antenna_names=None,
                  ant_str=None, bls=None, frequencies=None, freq_chans=None,
                  times=None, polarizations=None, blt_inds=None, read_data=True,
                  run_check=True, check_extra=True, run_check_acceptability=True):
        """
        Read a UVH5 file.

        Args:
            filename: The UVH5 file or list of files to read from.
            antenna_nums: The antennas numbers to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided. Ignored if read_data is False.
            antenna_names: The antennas names to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided. Ignored if read_data is False.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None. Ignored if read_data is False.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when reading data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y').
                See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be kept for both baselines (1,2) and (2,3) to return a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.
                Ignored if read_data is False.
            frequencies: The frequencies to include when reading data into the
                object.  Ignored if read_data is False.
            freq_chans: The frequency channel numbers to include when reading
                data into the object. Ignored if read_data is False.
            times: The times to include when reading data into the object.
                Ignored if read_data is False.
            polarizations: The polarizations to include when reading data into
                the object.  Ignored if read_data is False.
            blt_inds: The baseline-time indices to include when reading data into
                the object. This is not commonly used. Ignored if read_data is False.
            read_data: Read in the visibility and flag data. If set to false,
                only the basic metadata will be read in. Setting read_data to
                False results in an incompletely defined object (check will not
                pass). Default True.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.

        Returns:
            None
        """
        from . import uvh5
        if isinstance(filename, (list, tuple)):
            if not read_data and len(filename) > 1:
                raise ValueError('read_data cannot be False for a list of uvh5 files')

            self.read_uvh5(filename[0], antenna_nums=antenna_nums,
                           antenna_names=antenna_names, ant_str=ant_str, bls=bls,
                           frequencies=frequencies, freq_chans=freq_chans, times=times,
                           polarizations=polarizations, blt_inds=blt_inds,
                           read_data=read_data, run_check=run_check,
                           check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability)
            if len(filename) > 1:
                for f in filename[1:]:
                    uv2 = UVData()
                    uv2.read_uvh5(f, antenna_nums=antenna_nums,
                                  antenna_names=antenna_names, ant_str=ant_str, bls=bls,
                                  frequencies=frequencies, freq_chans=freq_chans,
                                  times=times, polarizations=polarizations,
                                  blt_inds=blt_inds, read_data=read_data,
                                  run_check=run_check, check_extra=check_extra,
                                  run_check_acceptability=run_check_acceptability)
                    self += uv2
                del(uv2)
        else:
            uvh5_obj = uvh5.UVH5()
            uvh5_obj.read_uvh5(filename, antenna_nums=antenna_nums,
                               antenna_names=antenna_names, ant_str=ant_str, bls=bls,
                               frequencies=frequencies, freq_chans=freq_chans, times=times,
                               polarizations=polarizations, blt_inds=blt_inds,
                               read_data=read_data, run_check=run_check, check_extra=check_extra,
                               run_check_acceptability=run_check_acceptability)
            self._convert_from_filetype(uvh5_obj)
            del(uvh5_obj)

    def write_uvh5(self, filename, run_check=True, check_extra=True,
                   run_check_acceptability=True, clobber=False,
                   data_compression=None, flags_compression="lzf",
                   nsample_compression="lzf"):
        """
        Write a completely in-memory UVData object to a UVH5 file.

        Args:
            filename: The UVH5 file to write to.
            run_check: Option to check for the existence and proper shapes of
                parameters before writing the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters before writing the file. Default is True.
            clobber: Option to overwrite the file if it already exists. Default is False.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                None (no filter/compression).
            flags_compression: HDF5 filter to apply when writing the flags_array. Default is
                the LZF filter.
            nsample_compression: HDF5 filter to apply when writing the nsample_array. Deafult is
                the LZF filter.

        Returns:
            None
        """
        uvh5_obj = self._convert_to_filetype('uvh5')
        uvh5_obj.write_uvh5(filename, run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                            clobber=clobber, data_compression=data_compression,
                            flags_compression=flags_compression,
                            nsample_compression=nsample_compression)
        del(uvh5_obj)

    def initialize_uvh5_file(self, filename, clobber=False, data_compression=None,
                             flags_compression="lzf", nsample_compression="lzf"):
        """
        Initialize a UVH5 file on disk with the header metadata and empty data arrays.

        Args:
            filename: The UVH5 file to write to.
            clobber: Option to overwrite the file if it already exists. Default is False.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                None (no filter/compression).
            flags_compression: HDF5 filter to apply when writing the flags_array. Default is
                the LZF filter.
            nsample_compression: HDF5 filter to apply when writing the nsample_array. Default is
                the LZF filter.

        Returns:
            None

        Notes:
            When partially writing out data, this function should be called first to initialize the
            file on disk. The data is then actually written by calling the write_uvh5_part method,
            with the same filename as the one specified in this function. See the tutorial for a
            worked example.
        """
        uvh5_obj = self._convert_to_filetype('uvh5')
        uvh5_obj.initialize_uvh5_file(filename, clobber=clobber,
                                      data_compression=data_compression,
                                      flags_compression=flags_compression,
                                      nsample_compression=nsample_compression)
        del(uvh5_obj)

    def write_uvh5_part(self, filename, data_array, flags_array, nsample_array, check_header=True,
                        antenna_nums=None, antenna_names=None, ant_str=None, bls=None,
                        frequencies=None, freq_chans=None, times=None, polarizations=None,
                        blt_inds=None):
        """
        Write data to a UVH5 file that has already been initialized.

        Args:
            filename: the file on disk to write data to. It must already exist,
                and is assumed to have been initialized with initialize_uvh5_file.
            data_array: the data to write to disk. A check is done to ensure that
                the dimensions of the data passed in conform to the ones specified by
                the "selection" arguments.
            flags_array: the flags array to write to disk. A check is done to ensure
                that the dimensions of the data passed in conform to the ones specified
                by the "selection" arguments.
            nsample_array: the nsample array to write to disk. A check is done to ensure
                that the dimensions fo the data passed in conform to the ones specified
                by the "selection" arguments.
            check_header: option to check that the metadata present in the header
                on disk matches that in the object. Default is True.
            antenna_nums: The antennas numbers to include when writing data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided.
            antenna_names: The antennas names to include when writing data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying baselines
                to write to the file. For length-2 tuples, the ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when writing data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y').
                See tutorial for more examples of valid strings and
                the behavior of different forms for ant_str.
                If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
                be written for both baselines (1,2) and (2,3) to reflect a valid
                pyuvdata object.
                An ant_str cannot be passed in addition to any of the above antenna
                args or the polarizations arg.
            frequencies: The frequencies to include when writing data to the file.
            freq_chans: The frequency channel numbers to include when writing data to the file.
            times: The times to include when writing data to the file.
            polarizations: The polarizations to include when writing data to the file.
            blt_inds: The baseline-time indices to include when writing data to the file.
                This is not commonly used.

        Returns:
            None
        """
        uvh5_obj = self._convert_to_filetype('uvh5')
        uvh5_obj.write_uvh5_part(filename, data_array, flags_array, nsample_array,
                                 check_header=check_header, antenna_nums=antenna_nums,
                                 antenna_names=antenna_names, bls=bls, ant_str=ant_str,
                                 frequencies=frequencies, freq_chans=freq_chans,
                                 times=times, polarizations=polarizations,
                                 blt_inds=blt_inds)
        del(uvh5_obj)

    def read(self, filename, file_type=None, antenna_nums=None, antenna_names=None,
             ant_str=None, bls=None, frequencies=None, freq_chans=None,
             times=None, polarizations=None, blt_inds=None, time_range=None,
             read_metadata=True, read_data=True, phase_type=None,
             correct_lat_lon=True, use_model=False, data_column='DATA',
             pol_order='AIPS', run_check=True, check_extra=True,
             run_check_acceptability=True):
        """
        Read a generic file into a UVData object.

        Args:
            filename: The file(s) or list(s) of files to read from.
            file_type: One of ['uvfits', 'miriad', 'fhd', 'ms', 'uvh5'] or None.
                If None, the code attempts to guess what the file type is.
                For miriad and ms types, this is based on the standard directory
                structure. For FHD, uvfits and uvh5 files it's based on file
                extensions (FHD: .sav, .txt; uvfits: .uvfits; uvh5: .uvh5).
                Note that if a list of datasets is passed, the file type is
                determined from the first dataset.
            antenna_nums: The antennas numbers to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_names is
                also provided. Ignored if read_data is False.
            antenna_names: The antennas names to include when reading data into
                the object (antenna positions and names for the excluded antennas
                will be retained). This cannot be provided if antenna_nums is
                also provided. Ignored if read_data is False.
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list
                of baseline 3-tuples (e.g. [(0,1,'xx'), (2,3,'yy')]) specifying
                baselines to keep in the object. For length-2 tuples, the
                ordering of the numbers within the tuple does not matter. For
                length-3 tuples, the polarization string is in the order of the
                two antennas. If length-3 tuples are provided, the polarizations
                argument below must be None. Ignored if read_data is False.
            ant_str: A string containing information about what antenna numbers
                and polarizations to include when reading data into the object.
                Can be 'auto', 'cross', 'all', or combinations of antenna numbers
                and polarizations (e.g. '1', '1_2', '1x_2y'). See tutorial for
                more examples of valid strings and the behavior of different
                forms for ant_str. If '1x_2y,2y_3y' is passed, both polarizations
                'xy' and 'yy' will be kept for both baselines (1,2) and (2,3)
                to return a valid pyuvdata object. An ant_str cannot be passed
                in addition to any of the above antenna keywords or the
                polarizations keyword. Ignored if read_data is False.
            frequencies: The frequencies to include when reading data into the
                object.  Ignored if read_data is False.
            freq_chans: The frequency channel numbers to include when reading
                data into the object. Ignored if read_data is False.
            times: The times to include when reading data into the object.
                Ignored if read_data is False.
            polarizations: The polarizations to include when reading data into
                the object.  Ignored if read_data is False.
            time_range: len-2 list containing min and max range of times (Julian Date) to read-in.
                Ex: [2458115.20, 2458115.40]. Cannot be set with times.
            blt_inds: The baseline-time indices to include when reading data into
                the object. This is not commonly used. Ignored if read_data is False.
            read_metadata: Read in metadata (times, baselines, uvws) as well as
                basic header info. Only used if file_type is 'uvfits' and read_data is False
                (metadata will be read if data is read). If file_type is 'uvfits'
                and both read_data and read_metadata are false, only basic header
                info is read in. Default True.
            read_data: Read in the data. Only used if file_type is 'uvfits',
                'miriad' or 'uvh5'. If set to False, only the metadata will be
                read in (for uvfits, this can be further restricted to just the
                header if read_metadata is False). Setting read_data to False
                results in an incompletely defined object (check will not pass).
                Default True.
            phase_type: Either 'drift' meaning zenith drift, 'phased' meaning
                the data are phased to a single RA/Dec or None and it will be
                guessed at based on the file. Only used if file_type is 'miriad'.
                Default None.
            correct_lat_lon: flag -- that only matters if altitude is missing --
                to update the latitude and longitude from the known_telescopes list.
                Only used if file_type is 'miriad'. Default True.
            use_model: Option to read in the model visibilities rather than the
                dirty visibilities. Only used if file_type is 'fhd'. Default is False.
            data_column: name of CASA data column to read into data_array.
                'DATA', 'MODEL', or 'CORRECTED_DATA'. Only used if file_type is 'ms'.
            pol_order: specify whether you want polarizations ordered by
                'CASA' or 'AIPS' conventions. Only used if file_type is 'ms'.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.

        Returns:
            None
        """
        if isinstance(filename, (list, tuple)):
            # this is either a list of separate files to read or a list of FHD files
            if isinstance(filename[0], (list, tuple)):
                # this must be a list of lists for FHD
                file_type = 'fhd'
                multi = True
            else:
                basename, extension = os.path.splitext(filename[0])
                if extension == '.sav' or extension == '.txt':
                    file_type = 'fhd'
                    multi = False
                else:
                    multi = True
        else:
            multi = False

        if file_type is None:
            if multi:
                file_test = filename[0]
            else:
                file_test = filename

            if os.path.isdir(file_test):
                # it's a directory, so it's either miriad or ms file type
                if os.path.exists(os.path.join(file_test, 'vartable')):
                    # It's miriad.
                    file_type = 'miriad'
                elif os.path.exists(os.path.join(file_test, 'OBSERVATION')):
                    # It's a measurement set.
                    file_type = 'ms'
            else:
                basename, extension = os.path.splitext(file_test)
                if extension == '.uvfits':
                    file_type = 'uvfits'
                elif extension == '.uvh5':
                    file_type = 'uvh5'

        if file_type is None:
            raise ValueError('File type could not be determined.')

        if (time_range is not None):
            if times is not None:
                raise ValueError(
                    'Only one of times and time_range can be provided.')

        if antenna_names is not None and antenna_nums is not None:
            raise ValueError('Only one of antenna_nums and antenna_names can be provided.')

        if file_type == 'uvfits':
            if (time_range is not None):
                select = True
                warnings.warn('Warning: "time_range" keyword is set which is not '
                              'supported by read_uvfits. This select will be '
                              'done after reading the file.')
            else:
                select = False

            self.read_uvfits(filename, antenna_nums=antenna_nums,
                             antenna_names=antenna_names, ant_str=ant_str,
                             bls=bls, frequencies=frequencies,
                             freq_chans=freq_chans, times=times,
                             polarizations=polarizations, blt_inds=blt_inds,
                             read_data=read_data, read_metadata=read_metadata,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)

            if select:
                unique_times = np.unique(self.time_array)
                times_to_keep = unique_times[np.where((unique_times >= np.min(time_range))
                                                      & (unique_times <= np.max(time_range)))]
                self.select(times=times_to_keep, run_check=run_check, check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)

        elif file_type == 'miriad':
            if (antenna_names is not None or frequencies is not None or freq_chans is not None
                    or times is not None or blt_inds is not None):

                if blt_inds is not None:
                    if (antenna_nums is not None or ant_str is not None
                            or bls is not None or time_range is not None):
                        warnings.warn('Warning: blt_inds is set along with select '
                                      'on read keywords that are supported by '
                                      'read_miriad and may downselect blts. '
                                      'This may result in incorrect results '
                                      'because the select on read will happen '
                                      'before the blt_inds selection so the '
                                      'indices may not match the expected locations.')
                else:
                    warnings.warn('Warning: a select on read keyword is set that is not '
                                  'supported by read_miriad. This select will be '
                                  'done after reading the file.')
                select = True
            else:
                select = False

            self.read_miriad(filename, antenna_nums=antenna_nums, ant_str=ant_str,
                             bls=bls, polarizations=polarizations,
                             time_range=time_range, read_data=read_data,
                             phase_type=phase_type, correct_lat_lon=correct_lat_lon,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)

            if select:
                self.select(antenna_names=antenna_names, frequencies=frequencies,
                            freq_chans=freq_chans, times=times,
                            blt_inds=blt_inds, run_check=run_check, check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)

        elif file_type == 'fhd':
            if (antenna_nums is not None or antenna_names is not None
                    or ant_str is not None or bls is not None
                    or frequencies is not None or freq_chans is not None
                    or times is not None or polarizations is not None
                    or blt_inds is not None):
                select = True
                warnings.warn('Warning: select on read keyword set, but '
                              'file_type is "fhd" which does not support select '
                              'on read. Entire file will be read and then select '
                              'will be performed')
            else:
                select = False

            self.read_fhd(filename, use_model=use_model, run_check=run_check,
                          check_extra=check_extra,
                          run_check_acceptability=run_check_acceptability)

            if select:
                self.select(antenna_nums=antenna_nums, antenna_names=antenna_names,
                            ant_str=ant_str, bls=bls, frequencies=frequencies,
                            freq_chans=freq_chans, times=times,
                            polarizations=polarizations, blt_inds=blt_inds,
                            run_check=run_check, check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)
        elif file_type == 'ms':
            if (antenna_nums is not None or antenna_names is not None
                    or ant_str is not None or bls is not None
                    or frequencies is not None or freq_chans is not None
                    or times is not None or polarizations is not None
                    or blt_inds is not None):
                select = True
                warnings.warn('Warning: select on read keyword set, but '
                              'file_type is "fhd" which does not support select '
                              'on read. Entire file will be read and then select '
                              'will be performed')
            else:
                select = False

            self.read_ms(filename, run_check=run_check, check_extra=check_extra,
                         run_check_acceptability=run_check_acceptability,
                         data_column=data_column, pol_order=pol_order)

            if select:
                self.select(antenna_nums=antenna_nums, antenna_names=antenna_names,
                            ant_str=ant_str, bls=bls, frequencies=frequencies,
                            freq_chans=freq_chans, times=times,
                            polarizations=polarizations, blt_inds=blt_inds,
                            run_check=run_check, check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)
        elif file_type == 'uvh5':
            if (time_range is not None):
                select = True
                warnings.warn('Warning: "time_range" keyword is set which is not '
                              'supported by read_uvh5. This select will be '
                              'done after reading the file.')
            else:
                select = False

            self.read_uvh5(filename, antenna_nums=antenna_nums,
                           antenna_names=antenna_names, ant_str=ant_str, bls=bls,
                           frequencies=frequencies, freq_chans=freq_chans, times=times,
                           polarizations=polarizations, blt_inds=blt_inds,
                           read_data=read_data, run_check=run_check, check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability)

            if select:
                unique_times = np.unique(self.time_array)
                times_to_keep = unique_times[np.where((unique_times >= np.min(time_range))
                                                      & (unique_times <= np.max(time_range)))]
                self.select(times=times_to_keep, run_check=run_check, check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability)

    def reorder_pols(self, order=None, run_check=True, check_extra=True,
                     run_check_acceptability=True):
        """
        Rearrange polarizations in the event they are not uvfits compatible.

        Args:
            order: Provide the order which to shuffle the data. Default will
                sort by absolute value of pol values.
            run_check: Option to check for the existence and proper shapes of
                parameters after reordering. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reordering. Default is True.
        """
        if order is None:
            order = np.argsort(np.abs(self.polarization_array))
        self.polarization_array = self.polarization_array[order]
        self.data_array = self.data_array[:, :, :, order]
        self.nsample_array = self.nsample_array[:, :, :, order]
        self.flag_array = self.flag_array[:, :, :, order]

        # check if object is self-consistent
        if run_check:
            self.check(check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)

    def get_ants(self):
        """
        Returns numpy array of unique antennas in the data.
        """
        return np.unique(np.append(self.ant_1_array, self.ant_2_array))

    def get_ENU_antpos(self, center=True, pick_data_ants=False):
        """
        Returns antenna positions in ENU (topocentric) coordinates in units of meters.

        Parameters:
        -----------
        center : bool, if True: subtract median of array position from antpos
        pick_data_ants : bool, if True: return only antennas found in data

        Returns: (antpos, ants)
        --------
        antpos : ndarray, antenna positions in TOPO frame and units of meters, shape=(Nants, 3)
        ants : ndarray, antenna numbers matching ordering of antpos, shape=(Nants,)
        """
        antpos = uvutils.ENU_from_ECEF((self.antenna_positions + self.telescope_location),
                                       *self.telescope_location_lat_lon_alt)
        ants = self.antenna_numbers

        if pick_data_ants:
            data_ants = np.unique(np.concatenate([self.ant_1_array, self.ant_2_array]))
            telescope_ants = self.antenna_numbers
            select = [x in data_ants for x in telescope_ants]
            antpos = antpos[select, :]
            ants = telescope_ants[select]

        if center is True:
            antpos -= np.median(antpos, axis=0)

        return antpos, ants

    def get_baseline_nums(self):
        """
        Returns numpy array of unique baseline numbers in data.
        """
        return np.unique(self.baseline_array)

    def get_antpairs(self):
        """
        Returns list of unique antpair tuples (ant1, ant2) in data.
        """
        return [self.baseline_to_antnums(bl) for bl in self.get_baseline_nums()]

    def get_pols(self):
        """
        Returns list of pols in string format.
        """
        return uvutils.polnum2str(self.polarization_array)

    def get_antpairpols(self):
        """
        Returns list of unique antpair + pol tuples (ant1, ant2, pol) in data.
        """
        bli = 0
        pols = self.get_pols()
        bls = self.get_antpairs()
        return [(bl) + (pol,) for bl in bls for pol in pols]

    def get_feedpols(self):
        """
        Returns list of unique antenna feed polarizations (e.g. ['X', 'Y']) in data.
        NOTE: Will return ValueError if any pseudo-Stokes visibilities are present.
        """
        if np.any(self.polarization_array > 0):
            raise ValueError('Pseudo-Stokes visibilities cannot be interpreted as feed polarizations')
        else:
            return list(set(''.join(self.get_pols())))

    def antpair2ind(self, ant1, ant2=None, ordered=True):
        """
        Given an antenna pair key, return indices along the baseline-time axis.
        This will search for either the key as specified, or the key and its
        conjugate.

        Args:
            ant1, ant2:
                Either an antenna-pair key, or key expanded as arguments.
                Example: antpair2ind( (10, 20) ) or antpair2ind(10, 20)
            ordered : Boolean, if True, search for antpair as provided, else
                search for it and it conjugate.

        Returns:
            inds: int-64 ndarray containing indices of the antpair along the
                baseline-time axis.
        """
        # check for expanded antpair or key
        if ant2 is None:
            if not isinstance(ant1, tuple):
                raise ValueError("antpair2ind must be fed an antpair tuple "
                                 "or expand it as arguments")
            ant2 = ant1[1]
            ant1 = ant1[0]
        else:
            if not isinstance(ant1, (int, np.integer)):
                raise ValueError("antpair2ind must be fed an antpair tuple or "
                                 "expand it as arguments")
        if not isinstance(ordered, (bool, np.bool)):
            raise ValueError("ordered must be a boolean")

        # if getting auto-corr, ordered must be True
        if ant1 == ant2:
            ordered = True

        # get indices
        inds = np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]
        if ordered:
            return inds
        else:
            ind2 = np.where((self.ant_1_array == ant2) & (self.ant_2_array == ant1))[0]
            inds = np.asarray(np.append(inds, ind2), dtype=np.int64)
            return inds

    def _key2inds(self, key):
        """
        Interpret user specified key as a combination of antenna pair and/or polarization.

        Args:
            key: Identifier of data. Key can be 1, 2, or 3 numbers:
                if len(key) == 1:
                    if (key < 5) or (type(key) is str):  interpreted as a
                                 polarization number/name, return all blts for that pol.
                    else: interpreted as a baseline number. Return all times and
                          polarizations for that baseline.
                if len(key) == 2: interpreted as an antenna pair. Return all
                    times and pols for that baseline.
                if len(key) == 3: interpreted as antenna pair and pol (ant1, ant2, pol).
                    Return all times for that baseline, pol. pol may be a string.

        Returns:
            blt_ind1: numpy array with blt indices for antenna pair.
            blt_ind2: numpy array with blt indices for conjugate antenna pair.
                      Note if a cross-pol baseline is requested, the polarization will
                      also be reversed so the appropriate correlations are returned.
                      e.g. asking for (1, 2, 'xy') may return conj(2, 1, 'yx'), which
                      is equivalent to the requesting baseline. See utils.conj_pol() for
                      complete conjugation mapping.
            pol_ind: tuple of numpy arrays with polarization indices for blt_ind1 and blt_ind2
        """
        key = uvutils._get_iterable(key)
        if type(key) is str:
            # Single string given, assume it is polarization
            pol_ind1 = np.where(self.polarization_array == uvutils.polstr2num(key))[0]
            if len(pol_ind1) > 0:
                blt_ind1 = np.arange(self.Nblts, dtype=np.int64)
                blt_ind2 = np.array([], dtype=np.int64)
                pol_ind2 = np.array([], dtype=np.int64)
                pol_ind = (pol_ind1, pol_ind2)
            else:
                raise KeyError('Polarization {pol} not found in data.'.format(pol=key))
        elif len(key) == 1:
            key = key[0]  # For simplicity
            if isinstance(key, collections.Iterable):
                # Nested tuple. Call function again.
                blt_ind1, blt_ind2, pol_ind = self._key2inds(key)
            elif key < 5:
                # Small number, assume it is a polarization number a la AIPS memo
                pol_ind1 = np.where(self.polarization_array == key)[0]
                if len(pol_ind1) > 0:
                    blt_ind1 = np.arange(self.Nblts)
                    blt_ind2 = np.array([], dtype=np.int64)
                    pol_ind2 = np.array([], dtype=np.int64)
                    pol_ind = (pol_ind1, pol_ind2)
                else:
                    raise KeyError('Polarization {pol} not found in data.'.format(pol=key))
            else:
                # Larger number, assume it is a baseline number
                inv_bl = self.antnums_to_baseline(self.baseline_to_antnums(key)[1],
                                                  self.baseline_to_antnums(key)[0])
                blt_ind1 = np.where(self.baseline_array == key)[0]
                blt_ind2 = np.where(self.baseline_array == inv_bl)[0]
                if len(blt_ind1) + len(blt_ind2) == 0:
                    raise KeyError('Baseline {bl} not found in data.'.format(bl=key))
                pol_ind = (np.arange(self.Npols), np.arange(self.Npols))
        elif len(key) == 2:
            # Key is an antenna pair
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError('Antenna pair {pair} not found in data'.format(pair=key))
            pol_ind = (np.arange(self.Npols), np.arange(self.Npols))
        elif len(key) == 3:
            # Key is an antenna pair + pol
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError('Antenna pair {pair} not found in '
                               'data'.format(pair=(key[0], key[1])))
            if type(key[2]) is str:
                # pol is str
                if len(blt_ind1) > 0:
                    pol_ind1 = np.where(self.polarization_array == uvutils.polstr2num(key[2]))[0]
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    pol_ind2 = np.where(self.polarization_array
                                        == uvutils.polstr2num(uvutils.conj_pol(key[2])))[0]
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
            else:
                # polarization number a la AIPS memo
                if len(blt_ind1) > 0:
                    pol_ind1 = np.where(self.polarization_array == key[2])[0]
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    pol_ind2 = np.where(self.polarization_array == uvutils.conj_pol(key[2]))[0]
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
            pol_ind = (pol_ind1, pol_ind2)
            if len(blt_ind1) * len(pol_ind[0]) + len(blt_ind2) * len(pol_ind[1]) == 0:
                raise KeyError('Polarization {pol} not found in data.'.format(pol=key[2]))
        # Catch autos
        if np.array_equal(blt_ind1, blt_ind2):
            blt_ind2 = np.array([], dtype=np.int64)
        return (blt_ind1, blt_ind2, pol_ind)

    def _smart_slicing(self, data, ind1, ind2, indp, **kwargs):
        """
        Method for quickly picking out the relevant section of data for get_data or get_flags

        Args:
            data: 4-dimensional array in the format of self.data_array
            ind1: list with blt indices for antenna pair (e.g. from self._key2inds)
            ind2: list with blt indices for conjugate antenna pair. (e.g. from self._key2inds)
            indp: tuple of lists with polarization indices for ind1 and ind2 (e.g. from self._key2inds)
        kwargs:
            squeeze: 'default': squeeze pol and spw dimensions if possible (default)
                     'none': no squeezing of resulting numpy array
                     'full': squeeze all length 1 dimensions
            force_copy: Option to explicitly make a copy of the data. Default is False.

        Returns:
            out: numpy array copy (or if possible, a read-only view) of relevant section of data
        """
        force_copy = kwargs.pop('force_copy', False)
        squeeze = kwargs.pop('squeeze', 'default')

        p_reg_spaced = [False, False]
        p_start = [0, 0]
        p_stop = [0, 0]
        dp = [1, 1]
        for i, pi in enumerate(indp):
            if len(pi) == 0:
                continue
            if len(set(np.ediff1d(pi))) <= 1:
                p_reg_spaced[i] = True
                p_start[i] = pi[0]
                p_stop[i] = pi[-1] + 1
                if len(pi) != 1:
                    dp[i] = pi[1] - pi[0]

        if len(ind2) == 0:
            # only unconjugated baselines
            if len(set(np.ediff1d(ind1))) <= 1:
                blt_start = ind1[0]
                blt_stop = ind1[-1] + 1
                if len(ind1) == 1:
                    dblt = 1
                else:
                    dblt = ind1[1] - ind1[0]
                if p_reg_spaced[0]:
                    out = data[blt_start:blt_stop:dblt, :, :, p_start[0]:p_stop[0]:dp[0]]
                else:
                    out = data[blt_start:blt_stop:dblt, :, :, indp[0]]
            else:
                out = data[ind1, :, :, :]
                if p_reg_spaced[0]:
                    out = out[:, :, :, p_start[0]:p_stop[0]:dp[0]]
                else:
                    out = out[:, :, :, indp[0]]
        elif len(ind1) == 0:
            # only conjugated baselines
            if len(set(np.ediff1d(ind2))) <= 1:
                blt_start = ind2[0]
                blt_stop = ind2[-1] + 1
                if len(ind2) == 1:
                    dblt = 1
                else:
                    dblt = ind2[1] - ind2[0]
                if p_reg_spaced[1]:
                    out = np.conj(data[blt_start:blt_stop:dblt, :, :, p_start[1]:p_stop[1]:dp[1]])
                else:
                    out = np.conj(data[blt_start:blt_stop:dblt, :, :, indp[1]])
            else:
                out = data[ind2, :, :, :]
                if p_reg_spaced[1]:
                    out = np.conj(out[:, :, :, p_start[1]:p_stop[1]:dp[1]])
                else:
                    out = np.conj(out[:, :, :, indp[1]])
        else:
            # both conjugated and unconjugated baselines
            out = (data[ind1, :, :, :], np.conj(data[ind2, :, :, :]))
            if p_reg_spaced[0] and p_reg_spaced[1]:
                out = np.append(out[0][:, :, :, p_start[0]:p_stop[0]:dp[0]],
                                out[1][:, :, :, p_start[1]:p_stop[1]:dp[1]], axis=0)
            else:
                out = np.append(out[0][:, :, :, indp[0]],
                                out[1][:, :, :, indp[1]], axis=0)

        if squeeze == 'full':
            out = np.squeeze(out)
        elif squeeze == 'default':
            if out.shape[3] is 1:
                # one polarization dimension
                out = np.squeeze(out, axis=3)
            if out.shape[1] is 1:
                # one spw dimension
                out = np.squeeze(out, axis=1)
        elif squeeze != 'none':
            raise ValueError('"' + str(squeeze) + '" is not a valid option for squeeze.'
                             'Only "default", "none", or "full" are allowed.')

        if force_copy:
            out = np.array(out)
        elif out.base is not None:
            # if out is a view rather than a copy, make it read-only
            out.flags.writeable = False

        return out

    def get_data(self, *args, **kwargs):
        """
        Function for quick access to numpy array with data corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.

        Returns:
            Numpy array of data corresponding to key.
            If data exists conjugate to requested antenna pair, it will be conjugated
            before returning.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.data_array, ind1, ind2, indp, **kwargs)
        return out

    def get_flags(self, *args, **kwargs):
        """
        Function for quick access to numpy array with flags corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.

        Returns:
            Numpy array of flags corresponding to key.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.flag_array, ind1, ind2, indp, **kwargs).astype(np.bool)
        return out

    def get_nsamples(self, *args, **kwargs):
        """
        Function for quick access to numpy array with nsamples corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.

        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.

        Returns:
            Numpy array of nsamples corresponding to key.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.nsample_array, ind1, ind2, indp, **kwargs)
        return out

    def get_times(self, *args):
        """
        Find the time_array entries for a given antpair or baseline number.
        Meant to be used in conjunction with get_data function.

        Args:
            args: antenna-pair(-pol) key expanded as arguments.

        Returns:
            Numpy array of times corresonding to key.
        """
        inds1, inds2, indp = self._key2inds(args)
        return self.time_array[np.append(inds1, inds2)]

    def antpairpol_iter(self, squeeze='default'):
        """
        Generates numpy arrays of data for each antpair, pol combination.

        Args:
            squeeze: 'default': squeeze pol and spw dimensions if possible
                     'none': no squeezing of resulting numpy array
                     'full': squeeze all length 1 dimensions

        Returns (for each iteration):
            key: tuple with antenna1, antenna2, and polarization string
            data: Numpy array with data which is the result of self[key]
        """
        antpairpols = self.get_antpairpols()
        for key in antpairpols:
            yield (key, self.get_data(key, squeeze=squeeze))

    def parse_ants(self, ant_str, print_toggle=False):
        """
        Generates two lists of antenna pair tuples and polarization indices based
        on parsing of the string ant_str.  If no valid polarizations (pseudo-Stokes
        params, or combinations of [lr] or [xy]) or antenna numbers are found in
        ant_str, ant_pairs_nums and polarizations are returned as None.

        Args:
            ant_str: String containing antenna information to pass to select
                function. Can be 'all', 'auto', cross, or combinations of antenna
                numbers and polarization indicators l and r or x and y.  Minus
                signs can also be used in front of an antenna number or baseline
                to exclude it from being output in ant_pairs_nums. If the antenna
                number attached with a minus sign is present in the outputted list
                ant_pairs_nums, it will be removed from the list.  If ant_str
                passed with a minus sign as the first character, 'all,' will be
                appended to the beginning of the string.  See the
                tutorial for examples of valid strings and their behavior.
            print_toggle: Boolean for printing parsed baselines for a visual user
                check.

        Output:
            ant_pairs_nums: List of tuples containing the parsed pairs of
                antenna numbers. If 'all' or pseudo-Stokes parameters are passed as
                ant_str, returned as None.
            polarizations: List of desired polarizations. If no polarizations
                found in ant_str then returned as None.
        """

        ant_re = r'(\(((-?\d+[lrxy]?,?)+)\)|-?\d+[lrxy]?)'
        bl_re = '(^(%s_%s|%s),?)' % (ant_re, ant_re, ant_re)
        str_pos = 0
        ant_pairs_nums = []
        polarizations = []
        ants_data = self.get_ants()
        ant_pairs_data = self.get_antpairs()
        pols_data = self.get_pols()
        warned_ants = []
        warned_pols = []
        warnings.simplefilter('always')

        if ant_str.startswith('-'):
            ant_str = 'all,' + ant_str

        while str_pos < len(ant_str):
            m = re.search(bl_re, ant_str[str_pos:])
            if m is None:
                if ant_str[str_pos:].upper().startswith('ALL'):
                    if len(ant_str[str_pos:].split(',')) > 1:
                        ant_pairs_nums = self.get_antpairs()
                elif ant_str[str_pos:].upper().startswith('AUTO'):
                    for pair in ant_pairs_data:
                        if (pair[0] == pair[1]
                                and pair not in ant_pairs_nums):
                            ant_pairs_nums.append(pair)
                elif ant_str[str_pos:].upper().startswith('CROSS'):
                    for pair in ant_pairs_data:
                        if not (pair[0] == pair[1]
                                or pair in ant_pairs_nums):
                            ant_pairs_nums.append(pair)
                elif ant_str[str_pos:].upper().startswith('PI'):
                    polarizations.append(uvutils.polstr2num('pI'))
                elif ant_str[str_pos:].upper().startswith('PQ'):
                    polarizations.append(uvutils.polstr2num('pQ'))
                elif ant_str[str_pos:].upper().startswith('PU'):
                    polarizations.append(uvutils.polstr2num('pU'))
                elif ant_str[str_pos:].upper().startswith('PV'):
                    polarizations.append(uvutils.polstr2num('pV'))
                else:
                    raise ValueError('Unparsible argument {s}'.format(s=ant_str))

                comma_cnt = ant_str[str_pos:].find(',')
                if comma_cnt >= 0:
                    str_pos += comma_cnt + 1
                else:
                    str_pos = len(ant_str)
            else:
                m = m.groups()
                str_pos += len(m[0])
                if m[2] is None:
                    ant_i_list = [m[8]]
                    ant_j_list = list(self.get_ants())
                else:
                    if m[3] is None:
                        ant_i_list = [m[2]]
                    else:
                        ant_i_list = m[3].split(',')

                    if m[6] is None:
                        ant_j_list = [m[5]]
                    else:
                        ant_j_list = m[6].split(',')

                for ant_i in ant_i_list:
                    include_i = True
                    if type(ant_i) == str and ant_i.startswith('-'):
                        ant_i = ant_i[1:]  # nibble the - off the string
                        include_i = False

                    for ant_j in ant_j_list:
                        include_j = True
                        if type(ant_j) == str and ant_j.startswith('-'):
                            ant_j = ant_j[1:]
                            include_j = False

                        pols = None
                        ant_i, ant_j = str(ant_i), str(ant_j)
                        if not ant_i.isdigit():
                            ai = re.search(r'(\d+)([x,y,l,r])', ant_i).groups()

                        if not ant_j.isdigit():
                            aj = re.search(r'(\d+)([x,y,l,r])', ant_j).groups()

                        if ant_i.isdigit() and ant_j.isdigit():
                            ai = [ant_i, '']
                            aj = [ant_j, '']
                        elif ant_i.isdigit() and not ant_j.isdigit():
                            if ('x' in ant_j or 'y' in ant_j):
                                pols = ['x' + aj[1], 'y' + aj[1]]
                            else:
                                pols = ['l' + aj[1], 'r' + aj[1]]
                            ai = [ant_i, '']
                        elif not ant_i.isdigit() and ant_j.isdigit():
                            if ('x' in ant_i or 'y' in ant_i):
                                pols = [ai[1] + 'x', ai[1] + 'y']
                            else:
                                pols = [ai[1] + 'l', ai[1] + 'r']
                            aj = [ant_j, '']
                        elif not ant_i.isdigit() and not ant_j.isdigit():
                            pols = [ai[1] + aj[1]]

                        ant_tuple = tuple((abs(int(ai[0])), abs(int(aj[0]))))

                        # Order tuple according to order in object
                        if ant_tuple in ant_pairs_data:
                            pass
                        elif ant_tuple[::-1] in ant_pairs_data:
                            ant_tuple = ant_tuple[::-1]
                        else:
                            if not (ant_tuple[0] in ants_data
                                    or ant_tuple[0] in warned_ants):
                                warned_ants.append(ant_tuple[0])
                            if not (ant_tuple[1] in ants_data
                                    or ant_tuple[1] in warned_ants):
                                warned_ants.append(ant_tuple[1])
                            if pols is not None:
                                for pol in pols:
                                    if not (pol.lower() in pols_data
                                            or pol in warned_pols):
                                        warned_pols.append(pol)
                            continue

                        if include_i and include_j:
                            if ant_tuple not in ant_pairs_nums:
                                ant_pairs_nums.append(ant_tuple)
                            if pols is not None:
                                for pol in pols:
                                    if (pol.lower() in pols_data
                                            and uvutils.polstr2num(pol) not in polarizations):
                                        polarizations.append(uvutils.polstr2num(pol))
                                    elif not (pol.lower() in pols_data
                                              or pol in warned_pols):
                                        warned_pols.append(pol)
                        else:
                            if pols is not None:
                                for pol in pols:
                                    if pol.lower() in pols_data:
                                        if (self.Npols == 1
                                                and [pol.lower()] == pols_data):
                                            ant_pairs_nums.remove(ant_tuple)
                                        if uvutils.polstr2num(pol) in polarizations:
                                            polarizations.remove(uvutils.polstr2num(pol))
                                    elif not (pol.lower() in pols_data
                                              or pol in warned_pols):
                                        warned_pols.append(pol)
                            elif ant_tuple in ant_pairs_nums:
                                ant_pairs_nums.remove(ant_tuple)

        if ant_str.upper() == 'ALL':
            ant_pairs_nums = None
        elif len(ant_pairs_nums) == 0:
            if (not ant_str.upper() in ['AUTO', 'CROSS']):
                ant_pairs_nums = None

        if len(polarizations) == 0:
            polarizations = None
        else:
            polarizations.sort(reverse=True)

        if print_toggle:
            print('\nParsed antenna pairs:')
            if ant_pairs_nums is not None:
                for pair in ant_pairs_nums:
                    print(pair)

            print('\nParsed polarizations:')
            if polarizations is not None:
                for pol in polarizations:
                    print(uvutils.polnum2str(pol))

        if len(warned_ants) > 0:
            warnings.warn('Warning: Antenna number {a} passed, but not present '
                          'in the ant_1_array or ant_2_array'
                          .format(a=(',').join(map(str, warned_ants))))

        if len(warned_pols) > 0:
            warnings.warn('Warning: Polarization {p} is not present in '
                          'the polarization_array'
                          .format(p=(',').join(warned_pols).upper()))

        return ant_pairs_nums, polarizations

    def _calc_single_integration_time(self):
        """Calculate a single integration time for a UVData object when not otherwise specified.

        Args:
            None

        Returns:
            int_time: integration time to be assigned to all samples in the data.

        Notes:
            This funciton computes the shortest time difference present in a UVData object's time_array,
            and returns that as the integration time to be used for all samples. Also, the time_array
            is in units of days, and integration_time has units of seconds, so we need to convert.
        """
        return np.diff(np.sort(list(set(self.time_array))))[0] * 86400
