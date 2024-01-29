# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing Miriad files."""
import itertools
import os
import shutil
import warnings

import numpy as np
import scipy
from astropy import constants as const
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import telescopes as uvtel
from .. import utils as uvutils
from ..docstrings import copy_replace_short_description
from .uvdata import UVData, _future_array_shapes_warning, reporting_request

__all__ = ["Miriad"]


class Miriad(UVData):
    """
    Defines a Miriad-specific subclass of UVData for reading and writing Miriad files.

    This class should not be interacted with directly, instead use the read
    and write_miriad methods on the UVData class.

    """

    def _pol_to_ind(self, pol):
        if self.polarization_array is None:
            raise ValueError(
                "Can't index polarization {p} because "
                "polarization_array is not set".format(p=pol)
            )
        pol_ind = np.argwhere(self.polarization_array == pol)
        if len(pol_ind) != 1:
            raise ValueError(
                "multiple matches for pol={pol} in "
                "polarization_array".format(pol=pol)
            )
        return pol_ind

    def _load_miriad_variables(self, uv):
        """
        Load miriad variables from an aipy.miriad UV descriptor onto self.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to load variables from.

        Returns
        -------
        default_miriad_variables : list of str
            list of default miriad variables
        other_miriad_variables: list of str
            list of other miriad variables
        extra_miriad_variables: list of str
            list of extra, non-standard variables

        """
        # list of miriad variables always read
        # NB: this includes variables in try/except (i.e. not all variables are
        # necessarily present in the miriad file)
        default_miriad_variables = [
            "nchan",
            "npol",
            "inttime",
            "sdf",
            "source",
            "telescop",
            "latitud",
            "longitu",
            "altitude",
            "history",
            "visunits",
            "instrume",
            "dut1",
            "gst0",
            "rdate",
            "timesys",
            "xorient",
            "cnt",
            "ra",
            "dec",
            "lst",
            "pol",
            "nants",
            "antnames",
            "nblts",
            "ntimes",
            "nbls",
            "sfreq",
            "epoch",
            "antpos",
            "antnums",
            "degpdy",
            "antdiam",
            "phsframe",
            "xorient",
            "bltorder",
        ]
        # list of miriad variables not read, but also not interesting
        # NB: nspect (I think) is number of spectral windows, will want one day
        # NB: xyphase & xyamp are "On-line X Y phase/amplitude measurements"
        #     which we may want in a calibration object some day
        # NB: systemp, xtsys & ytsys are "System temperatures of the antenna/X/Y feed"
        #     which we may want in a calibration object some day
        # NB: freqs, leakage and bandpass may be part of a calibration object some day
        other_miriad_variables = [
            "nspect",
            "obsdec",
            "vsource",
            "ischan",
            "restfreq",
            "nschan",
            "corr",
            "freq",
            "freqs",
            "leakage",
            "bandpass",
            "tscale",
            "coord",
            "veldop",
            "time",
            "obsra",
            "operator",
            "version",
            "axismax",
            "axisrms",
            "xyphase",
            "xyamp",
            "systemp",
            "xtsys",
            "ytsys",
            "baseline",
            "obspa",
            "antaz",
            "antel",
            "axisoff",
            "cable",
            "dazim",
            "delev",
            "jyperk",
            "jyperka",
            "phaselo1",
            "phaselo2",
            "phasem1",
            "themt",
            "wfreq",
            "wwidth",
            "wsystemp",
            "wcorr",
        ]

        extra_miriad_variables = []
        for variable in uv.variables():
            if (
                variable not in default_miriad_variables
                and variable not in other_miriad_variables
            ):
                extra_miriad_variables.append(variable)

        miriad_header_data = {
            "Nfreqs": "nchan",
            "Nspws": "nspect",
            "Npols": "npol",
            "channel_width": "sdf",  # in Ghz!
            "telescope_name": "telescop",
        }
        for item in miriad_header_data:
            if isinstance(uv[miriad_header_data[item]], str):
                header_value = uv[miriad_header_data[item]].replace("\x00", "")
            else:
                header_value = uv[miriad_header_data[item]]
            setattr(self, item, header_value)

        # Do the units and potential sign conversion for channel_width
        self.channel_width = np.abs(self.channel_width * 1e9)  # change from GHz to Hz

        # Future proof: always set the flex_spw_id_array.
        self.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)

        # Deal with the spectral axis now
        if self.Nspws > 1:
            self._set_flex_spw()
            # Channel widths are described per spw, just need to expand it out to be
            # for each frequency channel.
            self.channel_width = (
                np.concatenate(
                    tuple(
                        np.abs(chan_width) + np.zeros(nchan)
                        for (chan_width, nchan) in zip(uv["sdf"] * 1e9, uv["nschan"])
                    )
                )
                .flatten()
                .astype(np.float64)
            )
            # Now setup frequency array
            # TODO: Spw axis to be collapsed in future release
            self.freq_array = np.reshape(
                np.concatenate(
                    tuple(
                        chan_width * np.arange(nchan) + sfreq
                        for (chan_width, nchan, sfreq) in zip(
                            uv["sdf"] * 1e9, uv["nschan"], uv["sfreq"] * 1e9
                        )
                    )
                )
                .flatten()
                .astype(np.float64),
                (1, -1),
            )
            # TODO: Fix this to capture unsorted spectra
            self.flex_spw_id_array = np.concatenate(
                tuple(
                    idx + np.zeros(nchan, dtype=int)
                    for (idx, nchan) in zip(range(self.Nspws), uv["nschan"])
                )
            )
        else:
            self.freq_array = np.reshape(
                np.arange(self.Nfreqs) * self.channel_width + uv["sfreq"] * 1e9, (1, -1)
            )
            self.channel_width = np.float64(self.channel_width)

        self.spw_array = np.arange(self.Nspws)

        self.history = uv["history"]
        if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
            self.history += self.pyuvdata_version_str

        # check for pyuvdata variables that are not recognized miriad variables
        if "visunits" in uv.vartable.keys():
            self.vis_units = uv["visunits"].replace("\x00", "")
        else:
            self.vis_units = "uncalib"  # assume no calibration
        if "instrume" in uv.vartable.keys():
            self.instrument = uv["instrume"].replace("\x00", "")
        else:
            self.instrument = self.telescope_name  # set instrument = telescope

        if "dut1" in uv.vartable.keys():
            self.dut1 = uv["dut1"]
        if "degpdy" in uv.vartable.keys():
            self.earth_omega = uv["degpdy"]
        if "gst0" in uv.vartable.keys():
            self.gst0 = uv["gst0"]
        if "rdate" in uv.vartable.keys():
            self.rdate = uv["rdate"].replace("\x00", "")
        if "timesys" in uv.vartable.keys():
            self.timesys = uv["timesys"].replace("\x00", "")
        if "xorient" in uv.vartable.keys():
            self.x_orientation = uv["xorient"].replace("\x00", "")
        if "bltorder" in uv.vartable.keys():
            blt_order_str = uv["bltorder"].replace("\x00", "")
            self.blt_order = tuple(blt_order_str.split(", "))
            if self.blt_order == ("bda",):
                self._blt_order.form = (1,)

        return default_miriad_variables, other_miriad_variables, extra_miriad_variables

    def _load_telescope_coords(self, uv, correct_lat_lon=True):
        """
        Load telescope lat, lon, alt coordinates from aipy.miriad UV descriptor.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to load lat, lon, alt coordinates from.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        """
        # check if telescope name is present
        if self.telescope_name is None:
            self._load_miriad_variables(uv)

        latitude = uv["latitud"]  # in units of radians
        longitude = uv["longitu"]

        # Catch a weird case where where sometimes long is wrapped like RA (0 -> 2pi
        # instead of -pi -> pi)
        if longitude > np.pi:
            longitude -= 2 * np.pi
        try:
            altitude = uv["altitude"]
            self.telescope_location_lat_lon_alt = (latitude, longitude, altitude)
        except KeyError:
            # get info from known telescopes.
            # Check to make sure the lat/lon values match reasonably well
            telescope_obj = uvtel.get_telescope(self.telescope_name)
            if telescope_obj is not False:
                tol = 2 * np.pi * 1e-3 / (60.0 * 60.0 * 24.0)  # 1mas in radians
                lat_close = np.isclose(
                    telescope_obj.telescope_location_lat_lon_alt[0],
                    latitude,
                    rtol=0,
                    atol=tol,
                )
                lon_close = np.isclose(
                    telescope_obj.telescope_location_lat_lon_alt[1],
                    longitude,
                    rtol=0,
                    atol=tol,
                )
                if correct_lat_lon:
                    self.telescope_location_lat_lon_alt = (
                        telescope_obj.telescope_location_lat_lon_alt
                    )
                else:
                    self.telescope_location_lat_lon_alt = (
                        latitude,
                        longitude,
                        telescope_obj.telescope_location_lat_lon_alt[2],
                    )
                if lat_close and lon_close:
                    if correct_lat_lon:
                        warnings.warn(
                            "Altitude is not present in Miriad file, "
                            "using known location values for "
                            "{telescope_name}.".format(
                                telescope_name=telescope_obj.telescope_name
                            )
                        )
                    else:
                        warnings.warn(
                            "Altitude is not present in Miriad file, "
                            "using known location altitude value "
                            "for {telescope_name} and lat/lon from "
                            "file.".format(telescope_name=telescope_obj.telescope_name)
                        )
                else:
                    warn_string = "Altitude is not present in file "
                    if not lat_close and not lon_close:
                        warn_string = (
                            warn_string
                            + "and latitude and longitude values do not match values "
                        )
                    else:
                        if not lat_close:
                            warn_string = (
                                warn_string + "and latitude value does not match value "
                            )
                        else:
                            warn_string = (
                                warn_string
                                + "and longitude value does not match value "
                            )
                    if correct_lat_lon:
                        warn_string = (
                            warn_string + "for {telescope_name} in known "
                            "telescopes. Using values from known "
                            "telescopes.".format(
                                telescope_name=telescope_obj.telescope_name
                            )
                        )
                        warnings.warn(warn_string)
                    else:
                        warn_string = (
                            warn_string + "for {telescope_name} in known "
                            "telescopes. Using altitude value from known "
                            "telescopes and lat/lon from "
                            "file.".format(telescope_name=telescope_obj.telescope_name)
                        )
                        warnings.warn(warn_string)
            else:
                warnings.warn(
                    "Altitude is not present in Miriad file, and "
                    "telescope {telescope_name} is not in "
                    "known_telescopes. Telescope location will be "
                    "set using antenna positions.".format(
                        telescope_name=self.telescope_name
                    )
                )

    def _load_antpos(self, uv, sorted_unique_ants=None, correct_lat_lon=True):
        """
        Load antennas and their positions from a Miriad UV descriptor.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to antennas and positions from.
        sorted_unique_ants : list of ints, optional
            List of unique antennas.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        """
        # check if telescope coords exist
        if self.telescope_location_lat_lon_alt is None:
            self._load_telescope_coords(uv, correct_lat_lon=correct_lat_lon)

        latitude = uv["latitud"]  # in units of radians
        longitude = uv["longitu"]

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_postions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        try:
            # The antnums variable will only exist if the file was written with
            # pyuvdata.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so we convert to floats on write and back here
            self.antenna_numbers = uv["antnums"].astype(int)
            self.Nants_telescope = len(self.antenna_numbers)
        except KeyError:
            self.antenna_numbers = None
            self.Nants_telescope = None

        nants = uv["nants"]
        try:
            # Miriad stores antpos values in units of ns, pyuvdata uses meters.
            antpos = uv["antpos"].reshape(3, nants).T * const.c.to("m/ns").value

            # first figure out what are good antenna positions so we can only
            # use the non-zero ones to evaluate position information
            antpos_length = np.sqrt(np.sum(np.abs(antpos) ** 2, axis=1))
            good_antpos = np.where(antpos_length > 0)[0]
            mean_antpos_length = np.mean(antpos_length[good_antpos])
            if mean_antpos_length > 6.35e6 and mean_antpos_length < 6.39e6:
                absolute_positions = True
            else:
                absolute_positions = False

            # Miriad stores antpos values in a rotated ECEF coordinate system
            # where the x-axis goes through the local meridan. Need to convert
            # these positions back to standard ECEF and if they are absolute
            # positions, subtract off the telescope position to make them
            # relative to the array center.
            ecef_antpos = uvutils.ECEF_from_rotECEF(antpos, longitude)

            if self.telescope_location is not None:
                if absolute_positions:
                    rel_ecef_antpos = ecef_antpos - self.telescope_location
                    # maintain zeros because they mark missing data
                    rel_ecef_antpos[np.where(antpos_length == 0)[0]] = ecef_antpos[
                        np.where(antpos_length == 0)[0]
                    ]
                else:
                    rel_ecef_antpos = ecef_antpos
            else:
                self.telescope_location = np.mean(ecef_antpos[good_antpos, :], axis=0)
                valid_location = uvutils.check_surface_based_positions(
                    telescope_loc=self.telescope_location,
                    telescope_frame=self._telescope_location.frame,
                    raise_error=False,
                    raise_warning=False,
                )

                # check to see if this could be a valid telescope_location
                if valid_location:
                    mean_lat, mean_lon, mean_alt = self.telescope_location_lat_lon_alt
                    tol = 2 * np.pi / (60.0 * 60.0 * 24.0)  # 1 arcsecond in radians
                    mean_lat_close = np.isclose(mean_lat, latitude, rtol=0, atol=tol)
                    mean_lon_close = np.isclose(mean_lon, longitude, rtol=0, atol=tol)

                    if mean_lat_close and mean_lon_close:
                        # this looks like a valid telescope_location, and the
                        # mean antenna lat & lon values are close. Set the
                        # telescope_location using the file lat/lons and the mean alt.
                        # Then subtract it off of the antenna positions
                        warnings.warn(
                            "Telescope location is not set, but antenna "
                            "positions are present. Mean antenna latitude and "
                            "longitude values match file values, so "
                            "telescope_position will be set using the "
                            "mean of the antenna altitudes"
                        )
                        self.telescope_location_lat_lon_alt = (
                            latitude,
                            longitude,
                            mean_alt,
                        )
                        rel_ecef_antpos = ecef_antpos - self.telescope_location

                    else:
                        # this looks like a valid telescope_location, but the
                        # mean antenna lat & lon values are not close. Set the
                        # telescope_location using the file lat/lons at sea level.
                        # Then subtract it off of the antenna positions
                        self.telescope_location_lat_lon_alt = (latitude, longitude, 0)
                        warn_string = (
                            "Telescope location is set at sealevel at "
                            "the file lat/lon coordinates. Antenna "
                            "positions are present, but the mean "
                            "antenna "
                        )
                        rel_ecef_antpos = ecef_antpos - self.telescope_location

                        if not mean_lat_close and not mean_lon_close:
                            warn_string += (
                                "latitude and longitude values do not "
                                "match file values so they are not used "
                                "for altitude."
                            )
                        elif not mean_lat_close:
                            warn_string += (
                                "latitude value does not "
                                "match file values so they are not used "
                                "for altitude."
                            )
                        else:
                            warn_string += (
                                "longitude value does not "
                                "match file values so they are not used "
                                "for altitude."
                            )
                        warnings.warn(warn_string)

                else:
                    # This does not give a valid telescope_location. Instead
                    # calculate it from the file lat/lon and sea level for altitude
                    self.telescope_location_lat_lon_alt = (latitude, longitude, 0)
                    warn_string = (
                        "Telescope location is set at sealevel at "
                        "the file lat/lon coordinates. Antenna "
                        "positions are present, but the mean "
                        "antenna "
                    )

                    warn_string += (
                        "position does not give a "
                        "telescope_location on the surface of the "
                        "earth."
                    )
                    if absolute_positions:
                        rel_ecef_antpos = ecef_antpos - self.telescope_location
                    else:
                        warn_string += (
                            " Antenna positions do not appear to be "
                            "on the surface of the earth and will be treated "
                            "as relative."
                        )
                        rel_ecef_antpos = ecef_antpos

                    warnings.warn(warn_string)

            if self.Nants_telescope is not None:
                # in this case there is an antnums variable
                # (meaning that the file was written with pyuvdata), so we'll use it
                if nants == self.Nants_telescope:
                    # no inflation, so just use the positions
                    self.antenna_positions = rel_ecef_antpos
                else:
                    # there is some inflation, just use the ones that appear in antnums
                    self.antenna_positions = np.zeros(
                        (self.Nants_telescope, 3), dtype=antpos.dtype
                    )
                    for ai, num in enumerate(self.antenna_numbers):
                        self.antenna_positions[ai, :] = rel_ecef_antpos[num, :]
            else:
                # there is no antnums variable (meaning that this file was not
                # written by pyuvdata), so we test for antennas with non-zero
                # positions and/or that appear in the visibility data
                # (meaning that they have entries in ant_1_array or ant_2_array)
                antpos_length = np.sqrt(np.sum(np.abs(antpos) ** 2, axis=1))
                good_antpos = np.where(antpos_length > 0)[0]
                # take the union of the antennas with good positions (good_antpos)
                # and the antennas that have visisbilities (sorted_unique_ants)
                # if there are antennas with visibilities but zeroed positions
                # we issue a warning below
                if sorted_unique_ants is not None:
                    ants_use = set(good_antpos).union(sorted_unique_ants)
                else:
                    ants_use = set(good_antpos)
                # ants_use are the antennas we'll keep track of in the UVData
                # object, so they dictate Nants_telescope
                self.Nants_telescope = len(ants_use)
                self.antenna_numbers = np.array(list(ants_use))
                self.antenna_positions = np.zeros(
                    (self.Nants_telescope, 3), dtype=rel_ecef_antpos.dtype
                )
                for ai, num in enumerate(self.antenna_numbers):
                    if antpos_length[num] == 0:
                        warnings.warn(
                            "antenna number {n} has visibilities "
                            "associated with it, but it has a position"
                            " of (0,0,0)".format(n=num)
                        )
                    else:
                        # leave bad locations as zeros to make them obvious
                        self.antenna_positions[ai, :] = rel_ecef_antpos[num, :]

        except KeyError:
            # there is no antpos variable
            warnings.warn("Antenna positions are not present in the file.")
            self.antenna_positions = None

        if self.antenna_numbers is None:
            # there are no antenna_numbers or antenna_positions, so just use
            # the antennas present in the visibilities
            # (Nants_data will therefore match Nants_telescope)
            if sorted_unique_ants is not None:
                self.antenna_numbers = np.array(sorted_unique_ants)
                self.Nants_telescope = len(self.antenna_numbers)

        # antenna names is a foreign concept in miriad but required in other formats.
        try:
            # Here we deal with the way pyuvdata tacks it on to keep the
            # name information if we have it:
            # make it into one long comma-separated string
            ant_name_var = uv["antnames"]
            ant_name_str = ant_name_var.replace("\x00", "")
            ant_name_list = ant_name_str[1:-1].split(", ")
            self.antenna_names = ant_name_list
        except KeyError:
            self.antenna_names = self.antenna_numbers.astype(str).tolist()

        # check for antenna diameters
        try:
            self.antenna_diameters = uv["antdiam"]
        except KeyError:
            # backwards compatibility for when keyword was 'diameter'
            try:
                self.antenna_diameters = uv["diameter"]
                # if we find it, we need to remove it from extra_keywords to
                # keep from writing it out
                self.extra_keywords.pop("diameter")
            except KeyError:
                pass
        if self.antenna_diameters is not None:
            self.antenna_diameters = self.antenna_diameters * np.ones(
                self.Nants_telescope, dtype=np.float64
            )

    def _read_miriad_metadata(self, uv, correct_lat_lon=True):
        """
        Read in metadata (parameter info) but not data from a miriad file.

        Parameters
        ----------
        uv : aipy.miriad.UV
            aipy object to load metadata from.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing.

        Returns
        -------
        default_miriad_variables : list of str
            list of default miriad variables
        other_miriad_variables: list of str
            list of other miriad variables
        extra_miriad_variables: list of str
            list of extra, non-standard variables
        check_variables: dict
            dict of extra miriad variables to add to extra_keywords parameter.

        """
        # load miriad variables
        (default_miriad_variables, other_miriad_variables, extra_miriad_variables) = (
            self._load_miriad_variables(uv)
        )

        # dict of extra variables
        check_variables = {}
        for extra_variable in extra_miriad_variables:
            check_variables[extra_variable] = uv[extra_variable]

        # keep all single valued extra_variables as extra_keywords
        for key in check_variables.keys():
            if isinstance(check_variables[key], str):
                value = check_variables[key].replace("\x00", "")
                # check for booleans encoded as strings
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                self.extra_keywords[key] = value
            else:
                self.extra_keywords[key] = check_variables[key]

        # Check for items in itemtable to put into extra_keywords
        # These will end up as variables in written files, but is internally consistent.
        for key in uv.items():
            # A few items that are not needed, we read elsewhere, or is not supported
            # when downselecting, so we don't read here.
            if (
                key not in ["vartable", "history", "obstype"]
                and key not in other_miriad_variables
            ):
                if isinstance(uv[key], str):
                    value = uv[key].replace("\x00", "")
                    value = uv[key].replace("\x01", "")
                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                    self.extra_keywords[key] = value
                else:
                    self.extra_keywords[key] = uv[key]

        # load telescope coords
        self._load_telescope_coords(uv, correct_lat_lon=correct_lat_lon)

        # load antenna positions
        self._load_antpos(uv)

        return (
            default_miriad_variables,
            other_miriad_variables,
            extra_miriad_variables,
            check_variables,
        )

    @copy_replace_short_description(UVData.read_miriad, style=DocstringStyle.NUMPYDOC)
    def read_miriad(
        self,
        filepath,
        antenna_nums=None,
        ant_str=None,
        bls=None,
        polarizations=None,
        time_range=None,
        read_data=True,
        phase_type=None,
        projected=None,
        correct_lat_lon=True,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        calc_lst=True,
        fix_old_proj=False,
        fix_use_ant_pos=True,
        check_autos=True,
        fix_autos=True,
        use_future_array_shapes=False,
        astrometry_library=None,
    ):
        """Read in data from a miriad file."""
        from . import aipy_extracts

        if not os.path.exists(filepath):
            raise IOError(filepath + " not found")
        uv = aipy_extracts.UV(filepath)

        if phase_type is not None:
            warnings.warn(
                "The phase_type parameter is deprecated, use the projected parameter "
                "instead. This will become an error in version 3.0.",
                DeprecationWarning,
            )
            if projected is None:
                if phase_type not in ["phased", "drift"]:
                    raise ValueError(
                        "The phase_type was not one of the recognized options: "
                        "'drift', 'phased'. Instead, use the `projected` parameter."
                    )
                if phase_type == "drift":
                    projected = False
                else:
                    projected = True

        # load metadata
        (
            default_miriad_variables,
            other_miriad_variables,
            extra_miriad_variables,
            check_variables,
        ) = self._read_miriad_metadata(uv, correct_lat_lon=correct_lat_lon)

        # update filename attribute
        basename = filepath.rstrip("/")
        self.filename = [os.path.basename(basename)]
        self._filename.form = (1,)

        if not read_data:
            # don't read in the data. This means the object is incomplete,
            # but that may not matter for many purposes.
            return

        history_update_string = "  Downselected to specific "
        n_selects = 0

        # select on ant_str if provided
        if ant_str is not None:
            # type check
            if not isinstance(ant_str, (str, np.str_)):
                raise ValueError("ant_str must be a string")
            if antenna_nums is not None or bls is not None:
                raise ValueError("Cannot provide ant_str with antenna_nums or bls")

            aipy_extracts.uv_selector(uv, ant_str)
            if ant_str != "all":
                history_update_string += "antenna pairs"
                n_selects += 1

        # select on antenna_nums and/or bls using aipy_extracts.uv_selector
        if antenna_nums is not None or bls is not None:
            antpair_str = ""
            if antenna_nums is not None:
                # type check
                err_msg = "antenna_nums must be a list of antenna number integers"
                if not isinstance(antenna_nums, (np.ndarray, list)):
                    raise ValueError(err_msg)
                if not isinstance(antenna_nums[0], (int, np.integer)):
                    raise ValueError(err_msg)
                # get all possible combinations
                antpairs = list(
                    itertools.combinations_with_replacement(antenna_nums, 2)
                )
                # convert antenna numbers to string form required by
                # aipy_extracts.uv_selector
                antpair_str_list = ["_".join([str(a) for a in ap]) for ap in antpairs]
                history_update_string += "antennas"
                n_selects += 1

            if bls is not None:
                if isinstance(bls, tuple) and (len(bls) == 2 or len(bls) == 3):
                    bls = [bls]
                if not all(isinstance(item, tuple) for item in bls):
                    raise ValueError(
                        "bls must be a list of tuples of antenna numbers "
                        "(optionally with polarization)."
                    )
                if all(len(item) == 2 for item in bls):
                    if not all(
                        [isinstance(item[0], (int, np.integer)) for item in bls]
                        + [isinstance(item[1], (int, np.integer)) for item in bls]
                    ):
                        raise ValueError(
                            "bls must be a list of tuples of antenna numbers "
                            "(optionally with polarization)."
                        )
                elif all(len(item) == 3 for item in bls):
                    if polarizations is not None:
                        raise ValueError(
                            "Cannot provide length-3 tuples and also specify "
                            "polarizations."
                        )
                    if not all(isinstance(item[2], str) for item in bls):
                        raise ValueError(
                            "The third element in each bl must be a polarization string"
                        )
                else:
                    raise ValueError("bls tuples must be all length-2 or all length-3")

                bl_str_list = []
                bl_pols = set()
                for bl in bls:
                    if bl[0] <= bl[1]:
                        bl_str_list.append(str(bl[0]) + "_" + str(bl[1]))
                        if len(bl) == 3:
                            bl_pols.add(bl[2])
                    else:
                        bl_str_list.append(str(bl[1]) + "_" + str(bl[0]))
                        if len(bl) == 3:
                            bl_pols.add(uvutils.conj_pol(bl[2]))

                if n_selects > 0:
                    # combine antpair_str_list and bl_str_list with an intersection
                    antpair_str_list = list(
                        set(antpair_str_list).intersection(bl_str_list)
                    )
                else:
                    antpair_str_list = bl_str_list

                if len(bl_pols) > 0:
                    polarizations = list(bl_pols)

                if n_selects > 0:
                    history_update_string += ", baselines"
                else:
                    history_update_string += "baselines"
                n_selects += 1

            # convert antenna pair list to string form required by
            # aipy_extracts.uv_selector
            antpair_str += ",".join(antpair_str_list)
            aipy_extracts.uv_selector(uv, antpair_str)

        # select on time range
        if time_range is not None:
            # type check
            err_msg = (
                "time_range must be a len-2 list of Julian Date floats, "
                "Ex: [2458115.2, 2458115.6]"
            )
            if not isinstance(time_range, (list, np.ndarray)):
                raise ValueError(err_msg)
            if len(time_range) != 2:
                raise ValueError(err_msg)
            if not np.array(
                [isinstance(t, (float, np.floating, np.float64)) for t in time_range]
            ).all():
                raise ValueError(err_msg)

            # UVData.time_array marks center of integration, while Miriad
            # 'time' marks beginning
            # assume time_range refers to the center of the integrations,
            # so subtract 1/2 an integration before using with miriad select
            time_range_use = np.array(time_range) - uv["inttime"] / (24 * 3600.0) / 2

            uv.select("time", time_range_use[0], time_range_use[1], include=True)
            if n_selects > 0:
                history_update_string += ", times"
            else:
                history_update_string += "times"
            n_selects += 1

        # select on polarizations
        if polarizations is not None:
            # type check
            err_msg = (
                "pols must be a list of polarization strings or ints, "
                "Ex: ['xx', ...] or [-5, ...]"
            )
            if not isinstance(polarizations, (list, np.ndarray)):
                raise ValueError(err_msg)
            if not np.array(
                [isinstance(p, (str, np.str_, int, np.integer)) for p in polarizations]
            ).all():
                raise ValueError(err_msg)
            # convert to pol integer if string
            polarizations = [
                (
                    p
                    if isinstance(p, (int, np.integer))
                    else uvutils.polstr2num(p, x_orientation=self.x_orientation)
                )
                for p in polarizations
            ]
            # iterate through all possible pols and reject if not in pols
            pol_list = []
            for p in np.arange(-8, 5):
                if p not in polarizations:
                    uv.select("polarization", p, p, include=False)
                else:
                    pol_list.append(p)
            # check not empty
            if len(pol_list) == 0:
                raise ValueError("No polarizations in data matched input")
            if n_selects > 0:
                history_update_string += ", polarizations"
            else:
                history_update_string += "polarizations"
            n_selects += 1

        history_update_string += " using pyuvdata."
        if n_selects > 0:
            self.history += history_update_string

        data_accumulator = {}
        pol_list = []
        app_ra = None
        app_dec = None
        frame_pa = None
        epoch = None
        phase_frame = None
        sou_dict = {}
        Nphase = 0
        record_epoch = "epoch" in uv.vartable.keys()
        record_phase_frame = "phsframe" in uv.vartable.keys()
        record_app = ("obsra" in uv.vartable.keys()) and (
            "obsdec" in uv.vartable.keys()
        )
        record_pa = "obspa" in uv.vartable.keys()

        # dra_list - Stubbing out for later
        # ddec_list - Stubbing out for later
        for (uvw, t, (i, j)), d, f in uv.all_data(raw=True):
            # control for the case of only a single spw not showing up in
            # the dimension
            # Note that the (i, j) tuple is calculated from a baseline number in
            # _miriad (see miriad_wrap.h). The i, j values are also adjusted by _miriad
            # to start at 0 rather than 1.
            # N.B. (Karto): So the below is right, but for the wrong reasons. The data
            # array return is of length nchan, which encompasses the total number of
            # channels in all spectral windows, where 'ischan' and 'nschan' demark the
            # starting positions and length of the individual windows. I don't think
            # this array should ever _not_ be 1D, but the below does cast the shape of
            # the data array correctly so that the vestigial spw-axis is preserved. At
            # some point, the below will need to be fixed -- I'm keeping this here so
            # that I can skip reading through the MIRIAD programmers guide yet again.
            if len(d.shape) == 1:
                d.shape = (1,) + d.shape

            if np.size(d) != self.Nfreqs:
                raise ValueError("Number of channels in spectrum has changed!")
            try:
                cnt = uv["cnt"]
            except KeyError:
                cnt = np.ones(d.shape, dtype=np.float64)
            ra = uv["ra"]
            dec = uv["dec"]
            if record_epoch:
                epoch = uv["epoch"]
            if record_phase_frame:
                phase_frame = uv["phsframe"]
            else:
                phase_frame = None
            if record_app:
                app_ra = uv["obsra"]
                app_dec = uv["obsdec"]
            if record_pa:
                frame_pa = uv["obspa"]
            # LST is pulled from the file here, atlhough as some PAPER/early HERA
            # data calculated LST from pyephem (which is inconsistent w/ astropy
            # to the order of ~5 seconds), these values can be recalculated by
            # setting `calc_lst=True` when calling read_miriad.
            lst = uv["lst"]
            inttime = uv["inttime"]
            try:
                sou_id = sou_dict[uv["source"]]
            except KeyError:
                sou_dict[uv["source"]] = Nphase
                sou_id = Nphase
                Nphase += 1

            try:
                # TODO: Gotta clean this up at some point - there are going to be
                # addition things to stick into the metadata accumulator, and accessing
                # those values by index number will likely make it more difficult to
                # maintain/expand the code in the future.
                data_accumulator[uv["pol"]].append(
                    [
                        uvw,  # Entry 0
                        t,  # Entry 1
                        i,  # Entry 2
                        j,  # Entry 3
                        d,  # Entry 4
                        f,  # Entry 5
                        cnt,  # Entry 6
                        ra,  # Entry 7
                        dec,  # Entry 8
                        inttime,  # Entry 9
                        sou_id,  # Entry 10
                        epoch,  # Entry 11
                        app_ra,  # Entry 12
                        app_dec,  # Entry 13
                        frame_pa,  # Entry 14
                        lst,  # Entry 15
                        phase_frame,  # Entry 16
                    ]
                )
            except KeyError:
                data_accumulator[uv["pol"]] = [
                    [
                        uvw,  # Entry 0
                        t,  # Entry 1
                        i,  # Entry 2
                        j,  # Entry 3
                        d,  # Entry 4
                        f,  # Entry 5
                        cnt,  # Entry 6
                        ra,  # Entry 7
                        dec,  # Entry 8
                        inttime,  # Entry 9
                        sou_id,  # Entry 10
                        epoch,  # Entry 11
                        app_ra,  # Entry 12
                        app_dec,  # Entry 13
                        frame_pa,  # Entry 14
                        lst,  # Entry 15
                        phase_frame,  # Entry 16
                    ]
                ]
                pol_list.append(uv["pol"])
                # NB: flag types in miriad are usually ints

        if len(list(data_accumulator.keys())) == 0:
            raise ValueError(
                "No data is present, probably as a result of "
                "select on read that excludes all the data"
            )

        for pol, data in data_accumulator.items():
            # make array "object" dtype because underlying array may be ragged
            data_accumulator[pol] = np.array(data, dtype="object")

        self.polarization_array = np.array(pol_list)
        if polarizations is None:
            # A select on read would make the header npols not match the pols
            # in the data
            if len(self.polarization_array) != self.Npols:
                warnings.warn(
                    "npols={npols} but found {n} pols in data file".format(
                        npols=self.Npols, n=len(self.polarization_array)
                    )
                )
        self.Npols = len(pol_list)

        # makes a data_array (and flag_array) of zeroes to be filled in by
        #   data values
        # any missing data will have zeros

        # use set to get the unique list of all times ever listed in the file
        # iterate over polarizations and all spectra (bls and times) in two
        # nested loops, then flatten into a single vector, then set
        # then list again.

        times = list(
            set(np.concatenate([[k[1] for k in d] for d in data_accumulator.values()]))
        )
        times = np.sort(times)

        ant_i_unique = list(
            set(np.concatenate([[k[2] for k in d] for d in data_accumulator.values()]))
        )
        ant_j_unique = list(
            set(np.concatenate([[k[3] for k in d] for d in data_accumulator.values()]))
        )

        sorted_unique_ants = sorted(set(ant_i_unique + ant_j_unique))
        ant_i_unique = np.array(ant_i_unique)
        ant_j_unique = np.array(ant_j_unique)

        # Determine maximum digits needed to distinguish different values
        if sorted_unique_ants[-1] > 0:
            ndig_ant = np.ceil(np.log10(sorted_unique_ants[-1])).astype(int) + 1
        else:
            ndig_ant = 1
        # Be excessive in precision because we use the floating point values as
        # dictionary keys later
        prec_t = -2 * np.floor(np.log10(self._time_array.tols[-1])).astype(int)
        ndig_t = np.ceil(np.log10(times[-1])).astype(int) + prec_t + 2
        blts = []
        for d in data_accumulator.values():
            for k in d:
                blt = [
                    "{1:.{0}f}".format(prec_t, k[1]).zfill(ndig_t),
                    str(k[2]).zfill(ndig_ant),
                    str(k[3]).zfill(ndig_ant),
                    str(k[9]).zfill(ndig_t),
                ]
                blt = "_".join(blt)
                blts.append(blt)
        unique_blts = np.unique(np.array(blts))

        reverse_inds = dict(zip(unique_blts, range(len(unique_blts))))
        self.Nants_data = len(sorted_unique_ants)

        # load antennas and antenna positions using sorted unique ants list
        self._load_antpos(uv, sorted_unique_ants=sorted_unique_ants)

        # form up a grid which indexes time and baselines along the 'long'
        # axis of the visdata array
        tij_grid = np.array([list(map(float, x.split("_"))) for x in unique_blts])
        t_grid, ant_i_grid, ant_j_grid, int_grid = tij_grid.T
        # set the data sizes
        if (
            antenna_nums is None
            and bls is None
            and ant_str is None
            and time_range is None
        ):
            try:
                self.Nblts = uv["nblts"]
                if self.Nblts != len(t_grid):
                    warnings.warn(
                        "Nblts does not match the number of unique blts in the data"
                    )
                    self.Nblts = len(t_grid)
            except KeyError:
                self.Nblts = len(t_grid)
        else:
            # The select on read will make the header nblts not match the
            # number of unique blts
            self.Nblts = len(t_grid)
        if time_range is None:
            try:
                self.Ntimes = uv["ntimes"]
                if self.Ntimes != len(times):
                    warnings.warn(
                        "Ntimes does not match the number of unique times in the data"
                    )
                    self.Ntimes = len(times)
            except KeyError:
                self.Ntimes = len(times)
        else:
            # The select on read will make the header ntimes not match the
            # number of unique times
            self.Ntimes = len(times)

        # UVData.time_array marks center of integration, while Miriad 'time'
        # marks beginning
        # also, int_grid is in units of seconds, so we need to convert to days
        self.time_array = t_grid + int_grid / (24 * 3600.0) / 2

        self.integration_time = np.asarray(int_grid, dtype=np.float64)

        self.ant_1_array = ant_i_grid.astype(int)
        self.ant_2_array = ant_j_grid.astype(int)

        self.baseline_array = self.antnums_to_baseline(
            ant_i_grid.astype(int), ant_j_grid.astype(int)
        )
        if antenna_nums is None and bls is None and ant_str is None:
            try:
                self.Nbls = uv["nbls"]
                if self.Nbls != len(np.unique(self.baseline_array)):
                    warnings.warn(
                        "Nbls does not match the number of unique baselines in the data"
                    )
                    self.Nbls = len(np.unique(self.baseline_array))
            except KeyError:
                self.Nbls = len(np.unique(self.baseline_array))
        else:
            # The select on read will make the header nbls not match the
            # number of unique bls
            self.Nbls = len(np.unique(self.baseline_array))

        # slot the data into a grid
        # TODO: Spw axis to be collapsed in future release
        self.data_array = np.zeros(
            (self.Nblts, 1, self.Nfreqs, self.Npols), dtype=np.complex64
        )
        self.flag_array = np.ones(self.data_array.shape, dtype=np.bool_)
        self.uvw_array = np.zeros((self.Nblts, 3))
        # NOTE: Using our lst calculator, which uses astropy,
        # instead of _miriad values which come from pyephem.
        # The differences are of order 5 seconds.
        proc = None
        if (self.telescope_location is not None) and calc_lst:
            proc = self.set_lsts_from_time_array(
                background=background_lsts, astrometry_library=astrometry_library
            )
        self.nsample_array = np.ones(self.data_array.shape, dtype=np.float64)

        # Temporary arrays to hold polarization axis, which will be collapsed
        ra_pol_list = np.zeros((self.Nblts, self.Npols))
        dec_pol_list = np.zeros((self.Nblts, self.Npols))
        uvw_pol_list = np.zeros((self.Nblts, 3, self.Npols))
        sou_id_pol_list = np.zeros((self.Nblts, self.Npols), dtype=int)
        epoch_pol_list = np.zeros((self.Nblts, self.Npols))
        phase_frame_pol_list = np.zeros((self.Nblts, self.Npols), dtype=object)
        app_ra_pol_list = np.zeros((self.Nblts, self.Npols))
        app_dec_pol_list = np.zeros((self.Nblts, self.Npols))
        frame_pa_pol_list = np.zeros((self.Nblts, self.Npols))
        lst_pol_list = np.zeros((self.Nblts, self.Npols))

        c_ns = const.c.to("m/ns").value
        for pol, data in data_accumulator.items():
            pol_ind = self._pol_to_ind(pol)
            for d in data:
                blt = [
                    "{1:.{0}f}".format(prec_t, d[1]).zfill(ndig_t),
                    str(d[2]).zfill(ndig_ant),
                    str(d[3]).zfill(ndig_ant),
                    str(d[9]).zfill(ndig_t),
                ]
                blt = "_".join(blt)
                blt_index = reverse_inds[blt]

                self.data_array[blt_index, :, :, pol_ind] = d[4]
                self.flag_array[blt_index, :, :, pol_ind] = d[5]
                self.nsample_array[blt_index, :, :, pol_ind] = d[6]
                # because there are uvws/ra/dec for each pol, and one pol may not
                # have that visibility, we collapse along the polarization
                # axis but avoid any missing visbilities
                uvw = d[0] * c_ns
                uvw.shape = (1, 3)
                uvw_pol_list[blt_index, :, pol_ind] = uvw
                ra_pol_list[blt_index, pol_ind] = d[7]
                dec_pol_list[blt_index, pol_ind] = d[8]
                sou_id_pol_list[blt_index, pol_ind] = d[10]
                epoch_pol_list[blt_index, pol_ind] = d[11]
                app_ra_pol_list[blt_index, pol_ind] = d[12]
                app_dec_pol_list[blt_index, pol_ind] = d[13]
                frame_pa_pol_list[blt_index, pol_ind] = d[14]
                lst_pol_list[blt_index, pol_ind] = d[15]
                phase_frame_pol_list[blt_index, pol_ind] = d[16]

        # Collapse pol axis for ra_list, dec_list, and uvw_list
        ra_list = np.zeros(self.Nblts)
        dec_list = np.zeros(self.Nblts)
        sou_id_list = np.zeros(self.Nblts, dtype=int)
        epoch_list = np.zeros(self.Nblts)
        phase_frame_list = np.zeros(self.Nblts, dtype=object)
        app_ra_list = np.zeros(self.Nblts)
        app_dec_list = np.zeros(self.Nblts)
        frame_pa_list = np.zeros(self.Nblts)
        lst_list = np.zeros(self.Nblts)

        for blt_index in range(self.Nblts):
            test = ~np.all(self.flag_array[blt_index, :, :, :], axis=(0, 1))
            good_pol = np.where(test)[0]
            if len(good_pol) == 0:
                # No good pols for this blt. Fill with first one.
                good_pol = np.array([0])
            if len(good_pol) == 1:
                # Only one good pol, use it
                good_pol = good_pol[0]
                self.uvw_array[blt_index, :] = uvw_pol_list[blt_index, :, good_pol]
                ra_list[blt_index] = ra_pol_list[blt_index, good_pol]
                dec_list[blt_index] = dec_pol_list[blt_index, good_pol]
                sou_id_list[blt_index] = sou_id_pol_list[blt_index, good_pol]
                epoch_list[blt_index] = epoch_pol_list[blt_index, good_pol]
                phase_frame_list[blt_index] = phase_frame_pol_list[blt_index, good_pol]
                app_ra_list[blt_index] = app_ra_pol_list[blt_index, good_pol]
                app_dec_list[blt_index] = app_dec_pol_list[blt_index, good_pol]
                frame_pa_list[blt_index] = frame_pa_pol_list[blt_index, good_pol]
                lst_list[blt_index] = lst_pol_list[blt_index, good_pol]
            else:
                # Multiple good pols, check for consistency. pyuvdata does not
                # support pol-dependent uvw, ra, or dec.
                if np.any(np.diff(uvw_pol_list[blt_index, :, good_pol], axis=0)):
                    raise ValueError("uvw values are different by polarization.")
                else:
                    self.uvw_array[blt_index, :] = uvw_pol_list[
                        blt_index, :, good_pol[0]
                    ]

                check_list = [ra_pol_list, dec_pol_list, sou_id_pol_list, lst_pol_list]
                assign_list = [ra_list, dec_list, sou_id_list, lst_list]
                check_names = ["ra", "dec", "source id", "lst"]
                if record_epoch:
                    check_list.append(epoch_pol_list)
                    assign_list.append(epoch_list)
                    check_names.append("epoch")
                if record_phase_frame:
                    check_list.append(phase_frame_pol_list)
                    assign_list.append(phase_frame_list)
                    check_names.append("phsframe")
                if record_app:
                    check_list.extend([app_ra_pol_list, app_dec_pol_list])
                    assign_list.extend([app_ra_list, app_dec_list])
                    check_names.extend(["obsra", "obsdec"])
                if record_pa:
                    check_list.append(frame_pa_pol_list)
                    assign_list.append(frame_pa_list)
                    check_names.append("obspa")

                for item, name in zip(check_list, check_names):
                    if name == "phsframe":
                        arr = item[blt_index, good_pol]
                        if np.any(arr != arr[0]):
                            raise ValueError(
                                "%s values are different by polarization." % name
                                + reporting_request
                            )
                    else:
                        if np.any(np.diff(item[blt_index, good_pol])):
                            raise ValueError(
                                "%s values are different by polarization." % name
                                + reporting_request
                            )
                for item, target in zip(check_list, assign_list):
                    target[blt_index] = item[blt_index, good_pol[0]]

        # get unflagged blts
        # If we have a 1-baseline, single integration data set, set single_ra and
        # single_time to be true, otherwise evaluate the arrays
        blt_good = np.where(~np.all(self.flag_array, axis=(1, 2, 3)))
        single_ra = (
            True
            if (len(blt_good[0]) == 1)
            else (np.isclose(np.mean(np.diff(ra_list[blt_good])), 0.0))
        )
        single_time = (
            True
            if (len(blt_good[0]) == 1)
            else (np.isclose(np.mean(np.diff(self.time_array[blt_good])), 0.0))
        )

        if proc is not None:
            proc.join()

        if (self.telescope_location is None) or not calc_lst:
            # The float below is the number of sidereal days per solar day, and the
            # formula below adjusts for the fact that in MIRIAD, the lst is for the
            # start of the integration, as opposed to pyuvdata, which evaluates these
            # values at the midpoint of the integration.
            self.lst_array = lst_list + (
                np.pi * 1.002737909350795 * self.integration_time / (24.0 * 3600.0)
            )

        if projected is None:
            if record_phase_frame:
                projected = True
            elif not single_time:
                if single_ra or (Nphase > 1):
                    projected = True
                else:
                    projected = False
            else:
                # Finally check for the presence of an epoch variable, which isn't
                # really a good option, but at least it prevents crashes.
                warn_msg = (
                    "It is not clear from the file if the data are projected or not. "
                    "Since the 'epoch' variable is "
                )
                if "epoch" in uv.vartable.keys():
                    projected = True
                    warn_msg += "present it will be labeled as projected. "
                else:
                    projected = False
                    warn_msg += "not present it will be labeled as unprojected. "
                warn_msg += (
                    "If that is incorrect you can use the 'projected' parameter on "
                    "this method to set it properly."
                )
                warnings.warn(warn_msg)

        if record_app:
            self.phase_center_app_ra = app_ra_list
            self.phase_center_app_dec = app_dec_list
            if record_pa:
                self.phase_center_frame_pa = frame_pa_list

        if projected:
            # This presupposes that the data are already phased

            # Note that MIRIAD, AIPS, and (I think) CASA tasks assume the
            # coordinates are given in FK5 format (well, unless epoch <= 1984, in
            # which case MIRIAD assumes in semi-Orwellian fashion that you
            # _really_ wanted FK4/Bessel-Newcomb).
            self.phase_center_id_array = sou_id_list.astype(int)
            # Here is where we should package up sources
            for name in sou_dict.keys():
                select_mask = sou_id_list == sou_dict[name]

                # test for varying epochs. Unprojected phase centers have nans here
                # which do not test as matching, so also test for all nans
                if not np.all(
                    np.isnan(epoch_list[select_mask])
                ) and not uvutils._test_array_constant(
                    epoch_list[select_mask], tols=(1e-05, 1e-08)
                ):
                    # This is unusual but allowed within Miriad.
                    warnings.warn(
                        "Epoch values are varying within a single source. "
                        "Setting the epoch to the median." + reporting_request
                    )
                epoch_val = np.median(epoch_list[select_mask])
                cat_type = None
                if record_phase_frame:
                    unique_phase_frames = np.unique(phase_frame_list[select_mask])
                    # "phsframe" is not a standard Miriad keyword, it is only present
                    # in files written by pyuvdata, so this should not happen
                    assert_err_msg = (
                        "This is a bug, please make an issue in our issue log at "
                        "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues"
                    )
                    assert unique_phase_frames.size == 1, assert_err_msg
                    cat_frame = unique_phase_frames[0]

                    if cat_frame == "unprojected":
                        cat_type = "unprojected"
                        cat_frame = None
                        epoch_val = None
                else:
                    if epoch_val < 1984.0:
                        cat_frame = "fk4"
                    else:
                        cat_frame = "fk5"

                radian_tols = self._phase_center_app_ra.tols
                this_single_ra = uvutils._test_array_constant(
                    ra_list[select_mask], tols=radian_tols
                )
                this_single_dec = uvutils._test_array_constant(
                    dec_list[select_mask], tols=radian_tols
                )
                if not cat_type == "unprojected" and (
                    not this_single_ra or not this_single_dec
                ):
                    cat_type = "ephem"

                    lon_use = ra_list[select_mask]
                    lat_use = dec_list[select_mask]
                    times_use = self.time_array[select_mask]

                    unique_times, unique_inds, inverse, counts = np.unique(
                        times_use,
                        return_index=True,
                        return_inverse=True,
                        return_counts=True,
                    )
                    if np.max(counts) > 1:
                        for t_ind in np.arange(unique_times.size):
                            if not uvutils._test_array_constant(
                                lon_use[inverse == t_ind], tols=radian_tols
                            ):
                                raise ValueError(
                                    f"Source {name} has different RA values for "
                                    "different baselines at the same time."
                                    + reporting_request
                                )
                            if not uvutils._test_array_constant(
                                lat_use[inverse == t_ind], tols=radian_tols
                            ):
                                raise ValueError(
                                    f"Source {name} has different RA values for "
                                    "different baselines at the same time."
                                    + reporting_request
                                )
                        times_use = unique_times
                        lon_use = lon_use[unique_inds]
                        lat_use = lat_use[unique_inds]

                elif cat_type == "unprojected":
                    lon_use = None
                    lat_use = None
                    times_use = None
                else:
                    cat_type = "sidereal"
                    lon_use = np.median(ra_list[select_mask])
                    lat_use = np.median(dec_list[select_mask])
                    times_use = None

                self._add_phase_center(
                    name,
                    cat_type,
                    cat_lon=lon_use,
                    cat_lat=lat_use,
                    cat_frame=cat_frame,
                    cat_epoch=epoch_val,
                    cat_id=sou_dict[name],
                    cat_times=times_use,
                    info_source="file",
                )

        else:
            # check that the RA values are not constant (if more than one time
            # present)
            if Nphase > 1:
                raise ValueError("projected is False but there are multiple sources.")
            if single_ra and not single_time:
                raise ValueError("projected is False but the RA values are constant.")

            # use skycoord to simplify calculating sky separations.
            # Note, this should be done in the TEE frame, which isn't supported
            # by astropy. Frame doesn't really matter, though, because this is just
            # geometrical, so use icrs.
            pointing_coords = SkyCoord(
                ra=ra_list, dec=dec_list, unit="radian", frame="icrs"
            )
            zenith_coord = SkyCoord(
                ra=self.lst_array,
                dec=self.telescope_location_lat_lon_alt[0],
                unit="radian",
                frame="icrs",
            )

            separation_angles = pointing_coords.separation(zenith_coord)
            acceptable_offset = Angle("1d")
            if np.max(separation_angles.rad) > acceptable_offset.rad:
                warnings.warn(
                    "projected is False, but RA, Dec is off from lst, latitude by more "
                    f"than {acceptable_offset}, so it appears that it is not an "
                    "unphased file. Setting cat_type to unprojected, but that might be "
                    "inaccurate."
                )

            # there can only be one source name or it would have errored earlier.
            cat_name = list(sou_dict.keys())[0]

            cat_id = self._add_phase_center(cat_name=cat_name, cat_type="unprojected")
            self.phase_center_id_array = np.zeros((self.Nblts), dtype=int) + cat_id

        # close out now that we're done
        uv.close()

        if not (record_app and record_pa):
            # At this point, if we are missing information about the sky position
            # of the source, we want to fill it in now. If we have the apparent
            # coords, but are only missing the frame position angles (common given
            # that obspa is not a standard keyword), then we can _just_ fill those
            # in.
            self._set_app_coords_helper(pa_only=record_app)
        try:
            self.set_telescope_params()
        except ValueError as ve:
            warnings.warn(str(ve))

        # if blt_order is defined, reorder data to match that order
        # this is required because the data are ordered by (time, baseline) on the read
        if self.blt_order is not None:
            if len(list(self.blt_order)) == 2:
                order, minor_order = self.blt_order
            else:
                order = self.blt_order[0]
                minor_order = None
            self.reorder_blts(order=order, minor_order=minor_order)

        # If the data set was recorded using the old phasing method, fix that now.
        if fix_old_proj and projected:
            # this will error if it could not have been phased with the old method
            self.fix_phase(use_ant_pos=fix_use_ant_pos)

        if use_future_array_shapes:
            self.use_future_array_shapes()
        else:
            warnings.warn(_future_array_shapes_warning, DeprecationWarning)

        # check if object has all required uv_properties set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

    def write_miriad(
        self,
        filepath,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        no_antnums=False,
        calc_lst=False,
        check_autos=True,
        fix_autos=False,
    ):
        """
        Write the data to a miriad file.

        Parameters
        ----------
        filename : str
            The miriad root directory to write to.
        clobber : bool
            Option to overwrite the filename if the file already exists.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after before writing the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        no_antnums : bool
            Option to not write the antnums variable to the file.
            Should only be used for testing purposes.
        calc_lst : bool
            Recalculate the LST values upon writing the file. This is done to perform
            higher-precision accounting for the difference in MIRAD timestamps vs
            pyuvdata (the former marks the beginning of an integration, the latter
            marks the midpoint). Default is False, which instead uses a simple formula
            for correcting the LSTs, expected to be accurate to approximately 0.1 µsec
            precision.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is False.

        Raises
        ------
        ValueError
            If the frequencies are not evenly spaced or are separated by more
            than their channel width.
        TypeError
            If any entry in extra_keywords is not a single string or number.

        """
        from . import aipy_extracts

        if self._telescope_location.frame != "itrs":
            raise ValueError(
                "Only ITRS telescope locations are supported in Miriad files."
            )

        # change time_array and lst_array to mark beginning of integration,
        # per Miriad format
        miriad_time_array = self.time_array - self.integration_time / (24 * 3600.0) / 2
        if (self.telescope_location is not None) and calc_lst:
            latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
            miriad_lsts = uvutils.get_lst_for_time(
                miriad_time_array, latitude, longitude, altitude
            )
        else:
            # The long float below is the number of sidereal days per day. The below
            # equation should be accurate to _much_ better than 1 µsec.
            miriad_lsts = self.lst_array - (
                np.pi * 1.002737909350795 * self.integration_time / (24.0 * 3600.0)
            )

        # Miriad requires j>i which we call ant1<ant2
        self.conjugate_bls(convention="ant1<ant2")

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_freq_spacing=True,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

        if os.path.exists(filepath):
            if clobber:
                print("File exists: clobbering")
                shutil.rmtree(filepath)
            else:
                raise IOError("File exists: skipping")

        uv = aipy_extracts.UV(filepath, status="new")

        # initialize header variables
        uv._wrhd("obstype", "mixed-auto-cross")
        # avoid inserting extra \n.
        if not self.history[-1] == "\n":
            self.history += "\n"
        uv._wrhd("history", self.history)

        # recognized miriad variables
        uv.add_var("nchan", "i")
        uv["nchan"] = self.Nfreqs
        uv.add_var("npol", "i")
        uv["npol"] = self.Npols

        #####################################################
        # Frequency information here
        uv.add_var("nspect", "i")
        uv["nspect"] = self.Nspws

        if self.future_array_shapes:
            freq_array_use = self.freq_array
        else:
            freq_array_use = self.freq_array[0, :]
        if self.flex_spw:
            win_start_pos = np.insert(
                np.where(self.flex_spw_id_array[1:] != self.flex_spw_id_array[:-1])[0]
                + 1,
                0,
                0,
            ).astype(int)

            uv.add_var("ischan", "i")  # Starting chan of window
            uv["ischan"] = win_start_pos + 1  # Miriad is 1-based indexed

            uv.add_var("nschan", "i")  # Number of chan per window
            uv["nschan"] = np.diff(np.append(win_start_pos, self.Nfreqs))

            uv.add_var("sfreq", "d")  # Freq of first channel of the window, Hz -> GHz
            uv["sfreq"] = (freq_array_use[win_start_pos] / 1e9).astype(np.double)

            # Need the array direction here since channel_width is always supposed
            # to be > 0, but channels can be in descending freq order
            freq_dir = np.sign(
                freq_array_use[np.append(win_start_pos[1:] - 1, self.Nfreqs - 1)]
                - freq_array_use[win_start_pos]
            )

            uv.add_var("sdf", "d")  # Channel width, Hz -> GHz
            uv["sdf"] = self.channel_width[win_start_pos] * freq_dir / 1e9
        else:
            uv.add_var("ischan", "i")  # Starting chan of window
            uv["ischan"] = 1  # Miriad is 1-based indexed

            uv.add_var("nschan", "i")  # Number of chan per window
            uv["nschan"] = self.Nfreqs

            # Need the array direction here since channel_width is always supposed
            # to be > 0, but channels can be in decending freq order
            freq_dir = np.sign(np.diff(freq_array_use[([0, -1])]))

            uv.add_var("sfreq", "d")  # Freq of first channel of the window, in GHz
            uv["sfreq"] = (freq_array_use[0] / 1e9).astype(np.double)  # Hz -> GHz

            uv.add_var("sdf", "d")  # Channel width, in GHz
            # we've already run the check_freq_spacing, so channel widths are the
            # same to our tolerances
            uv["sdf"] = np.median(self.channel_width) * freq_dir / 1e9  # Hz -> GHz

        uv.add_var("telescop", "a")
        uv["telescop"] = self.telescope_name
        uv.add_var("latitud", "d")
        uv["latitud"] = self.telescope_location_lat_lon_alt[0].astype(np.double)
        uv.add_var("longitu", "d")
        uv["longitu"] = self.telescope_location_lat_lon_alt[1].astype(np.double)
        uv.add_var("nants", "i")

        # DCP 2024.01.12 - Adding defaults required for basic imaging
        #############################################################
        miriad_defaults = {
            "restfreq": ("d", np.float64(0.0)),
            "jyperk": ("r", np.float32(1.0)),
            "systemp": ("r", np.float32(1.0)),
            "veldop": ("r", np.float32(0.0)),
            "vsource": ("r", np.float32(0.0)),
        }

        for key, (miriad_dtype, val) in miriad_defaults.items():
            uv.add_var(key, miriad_dtype)
            uv[key] = val

        warnings.warn(
            "writing default values for restfreq, vsource, "
            "veldop, jyperk, and systemp"
        )

        if self.antenna_diameters is not None:
            if not np.allclose(self.antenna_diameters, self.antenna_diameters[0]):
                warnings.warn(
                    "Antenna diameters are not uniform, but miriad only "
                    "supports a single diameter. Skipping."
                )
            else:
                uv.add_var("antdiam", "d")
                uv["antdiam"] = float(self.antenna_diameters[0])

        # Miriad has no way to keep track of antenna numbers, so the antenna
        # numbers are simply the index for each antenna in any array that
        # describes antenna attributes (e.g. antpos for the antenna_postions).
        # Therefore on write, nants (which gives the size of the antpos array)
        # needs to be increased to be the max value of antenna_numbers+1 and the
        # antpos array needs to be inflated with zeros at locations where we
        # don't have antenna information. These inflations need to be undone at
        # read. If the file was written by pyuvdata, then the variable antnums
        # will be present and we can use it, otherwise we need to test for zeros
        # in the antpos array and/or antennas with no visibilities.
        nants = np.max(self.antenna_numbers) + 1
        uv["nants"] = nants
        if self.antenna_positions is not None:
            # Miriad wants antenna_positions to be in absolute coordinates
            # (not relative to array center) in a rotated ECEF frame where the
            # x-axis goes through the local meridian.
            rel_ecef_antpos = np.zeros((nants, 3), dtype=self.antenna_positions.dtype)
            for ai, num in enumerate(self.antenna_numbers):
                rel_ecef_antpos[num, :] = self.antenna_positions[ai, :]

            # find zeros so antpos can be zeroed there too
            antpos_length = np.sqrt(np.sum(np.abs(rel_ecef_antpos) ** 2, axis=1))

            ecef_antpos = rel_ecef_antpos + self.telescope_location
            longitude = self.telescope_location_lat_lon_alt[1]
            antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)

            # zero out bad locations (these are checked on read)
            antpos[np.where(antpos_length == 0), :] = [0, 0, 0]

            uv.add_var("antpos", "d")
            # Miriad stores antpos values in units of ns, pyuvdata uses meters.
            uv["antpos"] = (antpos.T.flatten() / const.c.to("m/ns").value).astype(
                np.double
            )

        # required pyuvdata variables that are not recognized miriad variables
        uv.add_var("ntimes", "i")
        uv["ntimes"] = self.Ntimes
        uv.add_var("nbls", "i")
        uv["nbls"] = self.Nbls
        uv.add_var("nblts", "i")
        uv["nblts"] = self.Nblts
        uv.add_var("visunits", "a")
        uv["visunits"] = self.vis_units
        uv.add_var("instrume", "a")
        uv["instrume"] = self.instrument
        uv.add_var("altitude", "d")
        uv["altitude"] = self.telescope_location_lat_lon_alt[2].astype(np.double)

        # optional pyuvdata variables that are not recognized miriad variables
        if self.dut1 is not None:
            uv.add_var("dut1", "d")
            uv["dut1"] = self.dut1
        if self.earth_omega is not None:
            uv.add_var("degpdy", "d")
            uv["degpdy"] = self.earth_omega
        if self.gst0 is not None:
            uv.add_var("gst0", "d")
            uv["gst0"] = self.gst0
        if self.rdate is not None:
            uv.add_var("rdate", "a")
            uv["rdate"] = self.rdate
        if self.timesys is not None:
            uv.add_var("timesys", "a")
            uv["timesys"] = self.timesys
        if self.x_orientation is not None:
            uv.add_var("xorient", "a")
            uv["xorient"] = self.x_orientation
        if self.blt_order is not None:
            blt_order_str = ", ".join(self.blt_order)
            uv.add_var("bltorder", "a")
            uv["bltorder"] = blt_order_str

        # other extra keywords
        # set up dictionaries to map common python types to miriad types
        # NB: arrays/lists/dicts could potentially be written as strings or 1D
        # vectors.  This is not supported at present!
        # NB: complex numbers *should* be supportable, but are not currently
        # supported due to unexplained errors in _miriad and/or its underlying libraries
        types = {
            str: "a",
            int: "i",
            float: "d",
            bool: "a",  # booleans are stored as strings and changed back on read
        }
        for key, value in self.extra_keywords.items():
            raise_type_error = False
            if isinstance(value, np.number):
                if issubclass(value.dtype.type, np.integer):
                    value = int(value)
                elif issubclass(value.dtype.type, np.floating):
                    value = float(value)
                elif issubclass(value.dtype.type, np.complexfloating):
                    raise_type_error = True
            elif isinstance(value, bool):
                value = str(value)
            elif type(value) not in types.keys():
                raise_type_error = True

            if raise_type_error:
                raise TypeError(
                    "Extra keyword {keyword} is of {keytype}. "
                    "Only strings and real numbers are "
                    "supported in miriad.".format(keyword=key, keytype=type(value))
                )

            if len(str(key)) > 8:
                warnings.warn(
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 as required "
                    "by the miriad file format.".format(key=key)
                )

            uvkeyname = str(key)[:8]  # name must be string, max 8 letters
            typestring = types[type(value)]
            uv.add_var(uvkeyname, typestring)
            uv[uvkeyname] = value

        if not no_antnums:
            # Add in the antenna_numbers so we have them if we read this file back in.
            # For some reason Miriad doesn't handle an array of integers properly,
            # so convert to floats here and integers on read.
            uv.add_var("antnums", "d")
            uv["antnums"] = self.antenna_numbers.astype(np.float64)

        # antenna names is a foreign concept in miriad but required in other formats.
        # Miriad can't handle arrays of strings, so we make it into one long
        # comma-separated string and convert back on read.
        ant_name_str = "[" + ", ".join(self.antenna_names) + "]"
        uv.add_var("antnames", "a")
        uv["antnames"] = ant_name_str

        # variables that can get updated with every visibility
        uv.add_var("pol", "i")
        uv.add_var("lst", "d")
        uv.add_var("cnt", "d")
        uv.add_var("source", "a")
        uv.add_var("ra", "d")
        uv.add_var("dec", "d")
        uv.add_var("inttime", "r")

        uv.add_var("epoch", "r")
        uv.add_var("phsframe", "a")  # Non-standard MIRIAD keyword

        uv.add_var("obsra", "d")
        uv.add_var("obsdec", "d")
        uv.add_var("obspa", "d")  # Non-standard MIRIAD keyword

        # write data
        c_ns = const.c.to("m/ns").value
        any_ephem = np.any(self._check_for_cat_type("ephem"))
        any_driftscan = np.any(self._check_for_cat_type("driftscan"))
        if any_driftscan:
            warnings.warn(
                "This object has a driftscan phase center. Miriad does not really "
                "support driftscans, writing this out in the same way 'ephem' phase "
                "centers are written by converting alt/az to ra/dec at each time. The "
                "data will not be changed, but if this file  is read back in it will "
                "be represented as an ephem phase center rather than a driftscan phase "
                "center."
            )
            driftscan_ids = []
            driftscan_coords = {}
            for cat_id, phase_dict in self.phase_center_catalog.items():
                if phase_dict["cat_type"] == "driftscan":
                    driftscan_ids.append(cat_id)
                    times = np.unique(
                        self.time_array[self.phase_center_id_array == cat_id]
                    )
                    this_altaz = SkyCoord(
                        alt=np.zeros_like(times) + phase_dict["cat_lat"],
                        az=np.zeros_like(times) + phase_dict["cat_lon"],
                        frame="altaz",
                        unit="rad",
                        location=EarthLocation.from_geocentric(
                            *self.telescope_location, unit="m"
                        ),
                        obstime=Time(times, format="jd"),
                    )
                    driftscan_coords[cat_id] = {
                        "times": times,
                        "coord": this_altaz.transform_to("fk5"),
                    }
        if any_ephem:
            ephem_interp = False
            ra_interp_func = {}
            dec_interp_func = {}
            ra_use = {}
            dec_use = {}
            for cat_id in self.phase_center_catalog.keys():
                if self.phase_center_catalog[cat_id]["cat_type"] != "ephem":
                    continue
                npts = self.phase_center_catalog[cat_id]["cat_times"].size
                if npts == 1:
                    continue
                if npts <= 4:
                    interp_kind = "slinear"
                else:
                    interp_kind = "cubic"
                unique_times, unique_inds = np.unique(
                    self.phase_center_catalog[cat_id]["cat_times"], return_index=True
                )
                # generate interp functions in case they're needed
                ra_interp_func[cat_id] = scipy.interpolate.interp1d(
                    unique_times,
                    self.phase_center_catalog[cat_id]["cat_lon"][unique_inds],
                    kind=interp_kind,
                )
                dec_interp_func[cat_id] = scipy.interpolate.interp1d(
                    unique_times,
                    self.phase_center_catalog[cat_id]["cat_lat"][unique_inds],
                    kind=interp_kind,
                )
        for viscnt in range(self.data_array.shape[0]):
            uvw = (self.uvw_array[viscnt] / c_ns).astype(np.double)
            this_t = miriad_time_array[viscnt]
            this_i = self.ant_1_array[viscnt]
            this_j = self.ant_2_array[viscnt]

            uv["lst"] = miriad_lsts[viscnt].astype(np.double)
            uv["inttime"] = self.integration_time[viscnt].astype(np.float32)

            cat_id = self.phase_center_id_array[viscnt]
            uv["source"] = self.phase_center_catalog[cat_id]["cat_name"]
            cat_type = self.phase_center_catalog[cat_id]["cat_type"]
            if cat_type == "unprojected":
                uv["ra"] = self.phase_center_app_ra[viscnt]
                uv["dec"] = self.phase_center_app_dec[viscnt]
                uv["phsframe"] = "unprojected"
            elif cat_type == "sidereal":
                uv["ra"] = self.phase_center_catalog[cat_id]["cat_lon"]
                uv["dec"] = self.phase_center_catalog[cat_id]["cat_lat"]
                uv["epoch"] = self.phase_center_catalog[cat_id]["cat_epoch"]
                uv["phsframe"] = self.phase_center_catalog[cat_id]["cat_frame"]
            elif cat_type == "ephem":
                if self.phase_center_catalog[cat_id]["cat_times"].size == 1:
                    ephem_interp = True
                    # if there's only one time, just use the values
                    uv["ra"] = self.phase_center_catalog[cat_id]["cat_lon"]
                    uv["dec"] = self.phase_center_catalog[cat_id]["cat_lat"]
                else:
                    # there are multiple times. find closest time. Use the
                    # integration center time NOT the miriad time
                    t_use = self.time_array[viscnt]

                    if cat_id in ra_use and t_use in ra_use[cat_id]:
                        # already calculated the ra/dec to use for this cat_id & time
                        uv["ra"] = ra_use[cat_id][t_use]
                        uv["dec"] = dec_use[cat_id][t_use]
                    else:
                        if cat_id not in ra_use:
                            ra_use[cat_id] = {}
                            dec_use[cat_id] = {}
                        t_diffs = np.abs(
                            self.phase_center_catalog[cat_id]["cat_times"] - t_use
                        )
                        t_min_loc = np.argmin(t_diffs)
                        tols = self._time_array.tols
                        if np.isclose(
                            0, t_diffs[t_min_loc], rtol=tols[0], atol=tols[1]
                        ):
                            ra_use[cat_id][t_use] = self.phase_center_catalog[cat_id][
                                "cat_lon"
                            ][t_min_loc]
                            dec_use[cat_id][t_use] = self.phase_center_catalog[cat_id][
                                "cat_lat"
                            ][t_min_loc]

                            uv["ra"] = ra_use[cat_id][t_use]
                            uv["dec"] = dec_use[cat_id][t_use]
                        else:
                            ephem_interp = True
                            try:
                                ra_use[cat_id][t_use] = np.asarray(
                                    [ra_interp_func[cat_id](t_use)]
                                )
                                dec_use[cat_id][t_use] = np.asarray(
                                    [dec_interp_func[cat_id](t_use)]
                                )
                            except ValueError:
                                # If t_use would require extrapolation, use the closest
                                # time
                                ra_use[cat_id][t_use] = self.phase_center_catalog[
                                    cat_id
                                ]["cat_lon"][t_min_loc]
                                dec_use[cat_id][t_use] = self.phase_center_catalog[
                                    cat_id
                                ]["cat_lat"][t_min_loc]
                            uv["ra"] = ra_use[cat_id][t_use]
                            uv["dec"] = dec_use[cat_id][t_use]
                uv["epoch"] = self.phase_center_catalog[cat_id]["cat_epoch"]
                uv["phsframe"] = self.phase_center_catalog[cat_id]["cat_frame"]
            else:
                # This is a driftscan, use driftscan_coords to set ra/dec
                t_ind = np.nonzero(
                    driftscan_coords[cat_id]["times"] == self.time_array[viscnt]
                )[0]
                uv["ra"] = driftscan_coords[cat_id]["coord"][t_ind].ra.rad
                uv["dec"] = driftscan_coords[cat_id]["coord"][t_ind].dec.rad
                uv["epoch"] = driftscan_coords[cat_id]["coord"].equinox.jyear
                uv["phsframe"] = driftscan_coords[cat_id]["coord"].frame.name

            uv["obspa"] = self.phase_center_frame_pa[viscnt]
            uv["obsra"] = self.phase_center_app_ra[viscnt]
            uv["obsdec"] = self.phase_center_app_dec[viscnt]

            for polcnt, pol in enumerate(self.polarization_array):
                uv["pol"] = pol.astype(np.int64)
                if self.future_array_shapes:
                    uv["cnt"] = self.nsample_array[viscnt, :, polcnt].astype(np.double)
                else:
                    uv["cnt"] = self.nsample_array[viscnt, 0, :, polcnt].astype(
                        np.double
                    )

                if self.future_array_shapes:
                    data = self.data_array[viscnt, :, polcnt]
                    flags = self.flag_array[viscnt, :, polcnt]
                else:
                    data = self.data_array[viscnt, 0, :, polcnt]
                    flags = self.flag_array[viscnt, 0, :, polcnt]
                # Using an assert here because it should be guaranteed by an earlier
                # method call.
                assert this_j >= this_i, (
                    "Miriad requires ant1<ant2 which should be "
                    "guaranteed by prior conjugate_bls call"
                )
                preamble = (uvw, this_t, (this_i, this_j))

                uv.write(preamble, data, flags)

        if any_ephem and ephem_interp:
            warnings.warn(
                "Some visibility times did not match ephem times so the ra and dec "
                "values for those visibilities were interpolated or set to the "
                "closest time if they would have required extrapolation."
            )

        # close out now that we're done
        uv.close()

        return
