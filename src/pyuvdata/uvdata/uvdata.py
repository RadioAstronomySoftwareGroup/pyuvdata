# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Primary container for radio interferometer datasets."""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import threading
import warnings
from collections.abc import Iterable
from typing import Literal

import numpy as np
from astropy import constants as const, coordinates as coord, units
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time
from docstring_parser import DocstringStyle
from scipy import ndimage as nd

from .. import Telescope, parameter as uvp, utils
from ..docstrings import combine_docstrings, copy_replace_short_description
from ..telescopes import known_telescopes
from ..utils import phasing as phs_utils
from ..utils.io import hdf5 as hdf5_utils
from ..utils.phasing import _get_focus_xyz, _get_nearfield_delay
from ..uvbase import UVBase
from .initializers import new_uvdata

__all__ = ["UVData"]

logger = logging.getLogger(__name__)

reporting_request = (
    " Please report this in our issue log, we have not been able to find a file with "
    "this feature, we would like to investigate this more."
)


class UVData(UVBase):
    """
    A class for defining a radio interferometer dataset.

    Attributes
    ----------
    UVParameter objects :
        For full list see the documentation on ReadTheDocs:
        http://pyuvdata.readthedocs.io/en/latest/.
        Some are always required, and others are always optional.

    """

    def __init__(self):
        """Create a new UVData object."""
        # add the UVParameters to the class

        self._Ntimes = uvp.UVParameter(
            "Ntimes", description="Number of times.", expected_type=int
        )
        self._Nbls = uvp.UVParameter(
            "Nbls", description="Number of baselines.", expected_type=int
        )
        desc = (
            "Number of baseline-times (i.e. number of spectra). Not necessarily "
            "equal to Nbls * Ntimes."
        )
        self._Nblts = uvp.UVParameter("Nblts", description=desc, expected_type=int)
        self._Nfreqs = uvp.UVParameter(
            "Nfreqs", description="Number of frequency channels.", expected_type=int
        )
        self._Npols = uvp.UVParameter(
            "Npols", description="Number of polarizations.", expected_type=int
        )

        desc = (
            "Array of the visibility data, shape: (Nblts, Nfreqs, Npols), "
            "type = complex float, in units of self.vis_units."
        )
        self._data_array = uvp.UVParameter(
            "data_array",
            description=desc,
            form=("Nblts", "Nfreqs", "Npols"),
            expected_type=complex,
        )

        desc = 'Visibility units, options are: "uncalib", "Jy" or "K str".'
        self._vis_units = uvp.UVParameter(
            "vis_units",
            description=desc,
            form="str",
            expected_type=str,
            acceptable_vals=["uncalib", "Jy", "K str"],
        )

        desc = (
            "Number of data points averaged into each data element, "
            "NOT required to be an integer, type = float, same shape as data_array."
            "The product of the integration_time and the nsample_array "
            "value for a visibility reflects the total amount of time "
            "that went into the visibility. Best practice is for the "
            "nsample_array to be used to track flagging within an integration_time "
            "(leading to a decrease of the nsample array value below 1) and "
            "LST averaging (leading to an increase in the nsample array "
            "value). So datasets that have not been LST averaged should "
            "have nsample array values less than or equal to 1."
            "Note that many files do not follow this convention, but it is "
            "safe to assume that the product of the integration_time and "
            "the nsample_array is the total amount of time included in a visibility."
        )
        self._nsample_array = uvp.UVParameter(
            "nsample_array",
            description=desc,
            form=("Nblts", "Nfreqs", "Npols"),
            expected_type=float,
        )

        desc = "Boolean flag, True is flagged, same shape as data_array."
        self._flag_array = uvp.UVParameter(
            "flag_array",
            description=desc,
            form=("Nblts", "Nfreqs", "Npols"),
            expected_type=bool,
        )

        self._Nspws = uvp.UVParameter(
            "Nspws",
            description=(
                "Number of spectral windows (ie non-contiguous spectral chunks). "
            ),
            expected_type=int,
        )

        self._spw_array = uvp.UVParameter(
            "spw_array",
            description="Array of spectral window numbers, shape (Nspws).",
            form=("Nspws",),
            expected_type=int,
        )

        desc = (
            "Projected baseline vectors relative to phase center, "
            "shape (Nblts, 3), units meters. "
            "Convention is: uvw = xyz(ant2) - xyz(ant1)."
            "Note that this is the Miriad convention but it is different "
            "from the AIPS/FITS convention (where uvw = xyz(ant1) - xyz(ant2))."
        )
        self._uvw_array = uvp.UVParameter(
            "uvw_array",
            description=desc,
            form=("Nblts", 3),
            expected_type=np.float64,
            strict_type_check=True,
            acceptable_range=(0, 1e8),
            tols=1e-3,
        )

        desc = (
            "Array of times, center of integration, shape (Nblts), units Julian Date."
        )
        self._time_array = uvp.UVParameter(
            "time_array",
            description=desc,
            form=("Nblts",),
            expected_type=np.float64,
            strict_type_check=True,
            tols=1e-3 / (60.0 * 60.0 * 24.0),
        )  # 1 ms in days

        desc = (
            "Array of local apparent sidereal times (LAST) at the center of "
            "integration, shape (Nblts), units radians."
        )
        self._lst_array = uvp.UVParameter(
            "lst_array",
            description=desc,
            form=("Nblts",),
            expected_type=np.float64,
            strict_type_check=True,
            tols=utils.RADIAN_TOL,
        )

        desc = (
            "Array of numbers for the first antenna, which is matched to that in "
            "the antenna_numbers attribute. Shape (Nblts), type = int."
        )
        self._ant_1_array = uvp.UVParameter(
            "ant_1_array",
            description=desc,
            expected_type=int,
            form=("Nblts",),
            acceptable_range=(0, 2147483647),
            setter=self._clear_antpair2ind_cache,
        )

        desc = (
            "Array of numbers for the second antenna, which is matched to that in "
            "the antenna_numbers attribute. Shape (Nblts), type = int."
        )
        self._ant_2_array = uvp.UVParameter(
            "ant_2_array",
            description=desc,
            expected_type=int,
            form=("Nblts",),
            acceptable_range=(0, 2147483647),
            setter=self._clear_antpair2ind_cache,
        )

        desc = (
            "Array of baseline numbers, shape (Nblts), type = int; "
            "by default baseline = 2048 * ant1 + ant2 + 2^16, "
            "though other conventions are available."
        )
        self._baseline_array = uvp.UVParameter(
            "baseline_array",
            description=desc,
            expected_type=int,
            form=("Nblts",),
            acceptable_range=(0, 4611686018498691072),
        )

        # this dimensionality of freq_array does not allow for different spws
        # to have different dimensions
        desc = "Array of frequencies, center of the channel, shape (Nfreqs,), units Hz."
        self._freq_array = uvp.UVParameter(
            "freq_array",
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )  # mHz

        desc = (
            "Array of polarization integers, shape (Npols). "
            "AIPS Memo 117 says: pseudo-stokes 1:4 (pI, pQ, pU, pV);  "
            "circular -1:-4 (RR, LL, RL, LR); linear -5:-8 (XX, YY, XY, YX). "
            "NOTE: AIPS Memo 117 actually calls the pseudo-Stokes polarizations "
            '"Stokes", but this is inaccurate as visibilities cannot be in '
            "true Stokes polarizations for physical antennas. We adopt the "
            "term pseudo-Stokes to refer to linear combinations of instrumental "
            "visibility polarizations (e.g. pI = xx + yy). A value of 0 indicates that "
            "the polarization is different for different spectral windows and is only "
            "allowed if flex_spw_polarization_array is defined (not None)."
        )
        self._polarization_array = uvp.UVParameter(
            "polarization_array",
            description=desc,
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 5)),
            form=("Npols",),
            setter=self._clear_key2ind_cache,
        )

        desc = (
            "Length of the integration in seconds, shape (Nblts). "
            "The product of the integration_time and the nsample_array "
            "value for a visibility reflects the total amount of time "
            "that went into the visibility. Best practice is for the "
            "integration_time to reflect the length of time a visibility "
            "was integrated over (so it should vary in the case of "
            "baseline-dependent averaging and be a way to do selections "
            "for differently integrated baselines)."
            "Note that many files do not follow this convention, but it is "
            "safe to assume that the product of the integration_time and "
            "the nsample_array is the total amount of time included in a visibility."
        )
        self._integration_time = uvp.UVParameter(
            "integration_time",
            description=desc,
            form=("Nblts",),
            expected_type=float,
            tols=1e-3,
        )  # 1 ms

        desc = (
            "Width of frequency channels (Hz). Array of shape (Nfreqs), type = float."
        )
        self._channel_width = uvp.UVParameter(
            "channel_width",
            description=desc,
            form=("Nfreqs",),
            expected_type=float,
            tols=1e-3,
        )  # 1 mHz

        self._history = uvp.UVParameter(
            "history",
            description="String of history, units English.",
            form="str",
            expected_type=str,
        )

        # --- flexible spectral window information ---

        desc = (
            "Maps individual channels along the frequency axis to individual spectral "
            "windows, as listed in the spw_array. Shape (Nfreqs), type = int."
        )
        self._flex_spw_id_array = uvp.UVParameter(
            "flex_spw_id_array", description=desc, form=("Nfreqs",), expected_type=int
        )

        desc = (
            "Optional parameter, allows for labeling individual spectral windows with "
            "different polarizations. If set, Npols must be set to 1 (i.e., only one "
            "polarization per spectral window allowed). Shape (Nspws), type = int."
        )
        self._flex_spw_polarization_array = uvp.UVParameter(
            "flex_spw_polarization_array",
            description=desc,
            form=("Nspws",),
            expected_type=int,
            acceptable_vals=list(np.arange(-8, 0)) + list(np.arange(1, 5)),
            required=False,
        )

        # --- phasing information ---

        desc = "Specifies the number of phase centers contained within the data set."
        self._Nphase = uvp.UVParameter("Nphase", description=desc, expected_type=int)

        desc = (
            "Dictionary that acts as a catalog, containing information on individual "
            "phase centers. Keys are the catalog IDs of the different phase centers in "
            "the object (matched to the parameter ``phase_center_id_array``). At a "
            "minimum, each dictionary must contain the keys: "
            "'cat_name' giving the phase center name (this does not have to be unique, "
            "non-unique values can be used to indicate sets of phase centers that make "
            "up a mosaic observation), "
            "'cat_type', which can be 'sidereal' (fixed position in RA/Dec), 'ephem' "
            "(position in RA/Dec which moves with time), 'driftscan' (fixed postion in "
            "Az/El, NOT the same as the old ``phase_type`` = 'drift') or 'unprojected' "
            "(baseline coordinates in ENU, but data are not phased, similar to "
            "the old ``phase_type`` = 'drift') "
            "'cat_lon' (longitude coord, e.g. RA, either a single value or a one "
            "dimensional array of length Npts --the number of ephemeris data points-- "
            "for ephem type phase centers), "
            "'cat_lat' (latitude coord, e.g. Dec., either a single value or a one "
            "dimensional array of length Npts --the number of ephemeris data points-- "
            "for ephem type phase centers), "
            "'cat_frame' (coordinate frame, e.g. icrs, must be a frame supported by "
            "astropy). "
            "Other optional keys include "
            "'cat_epoch' (epoch and equinox of the coordinate frame, not needed for "
            "frames without an epoch (e.g. ICRS) unless the there is proper motion), "
            "'cat_times' (times for the coordinates, only used for 'ephem' types), "
            "'cat_pm_ra' (proper motion in RA), "
            "'cat_pm_dec' (proper motion in Dec), "
            "'cat_dist' (physical distance to the source in parsec, useful if parallax "
            "is important, either a single value or a one dimensional array of length "
            "Npts --the number of ephemeris data points-- for ephem type phase "
            "centers.), "
            "'cat_vrad' (rest frame velocity in km/s, either a single value or a one "
            "dimensional array of length Npts --the number of ephemeris data points-- "
            "for ephem type phase centers.), and "
            "'info_source' (describes where catalog info came from). "
            "See the documentation of the `phase` method for more details."
        )
        self._phase_center_catalog = uvp.UVParameter(
            "phase_center_catalog", description=desc, expected_type=dict
        )

        desc = (
            "Apparent right ascension of phase center in the topocentric frame of the "
            "observatory, units radians. Shape (Nblts,), type = float."
        )
        self._phase_center_app_ra = uvp.AngleParameter(
            "phase_center_app_ra",
            form=("Nblts",),
            expected_type=float,
            description=desc,
            tols=utils.RADIAN_TOL,
        )

        desc = (
            "Apparent Declination of phase center in the topocentric frame of the "
            "observatory, units radians. Shape (Nblts,), type = float."
        )
        self._phase_center_app_dec = uvp.AngleParameter(
            "phase_center_app_dec",
            form=("Nblts",),
            expected_type=float,
            description=desc,
            tols=utils.RADIAN_TOL,
        )

        desc = (
            "Position angle between the hour circle (which is a great circle that goes "
            "through the target postion and both poles) in the apparent/topocentric "
            "frame, and the frame given in the phase_center_frame attribute."
            "Shape (Nblts,), type = float."
        )
        # The tolerance here is set by the fact that is is calculated using an arctan,
        # the limiting precision of which happens around values of 1.
        self._phase_center_frame_pa = uvp.AngleParameter(
            "phase_center_frame_pa",
            form=("Nblts",),
            expected_type=float,
            description=desc,
            tols=2e-8,
        )

        desc = (
            "Maps individual indices along the Nblt axis to a key in "
            "`phase_center_catalog`, which maps to a dict "
            "containing the other metadata for each phase center."
            "Shape (Nblts), type = int."
        )
        self._phase_center_id_array = uvp.UVParameter(
            "phase_center_id_array",
            description=desc,
            form=("Nblts",),
            expected_type=int,
        )

        desc = (
            "Optional when reading a MS. Retains the scan number when reading a MS."
            " Shape (Nblts), type = int."
        )
        self._scan_number_array = uvp.UVParameter(
            "scan_number_array",
            description=desc,
            form=("Nblts",),
            expected_type=int,
            required=False,
        )

        # --- antenna information ----
        desc = (
            "Number of antennas with data present (i.e. number of unique "
            "entries in ant_1_array and ant_2_array). May be smaller "
            "than the number of antennas in the array."
        )
        self._Nants_data = uvp.UVParameter(
            "Nants_data", description=desc, expected_type=int
        )

        self._telescope = uvp.UVParameter(
            "telescope",
            description=(
                ":class:`pyuvdata.Telescope` object containing the telescope metadata."
            ),
            expected_type=Telescope,
        )

        # -------- extra, non-required parameters ----------

        blt_order_options = ["time", "baseline", "ant1", "ant2", "bda"]
        desc = (
            "Ordering of the data array along the blt axis. A tuple with "
            'the major and minor order (minor order is omitted if order is "bda"). '
            "The allowed values are: "
            + " ,".join([str(val) for val in blt_order_options])
            + "."
        )
        self._blt_order = uvp.UVParameter(
            "blt_order",
            description=desc,
            form=(2,),
            required=False,
            expected_type=str,
            acceptable_vals=blt_order_options,
            ignore_eq_none=True,
        )

        desc = (
            "Whether the baseline-time axis is rectangular. If not provided, the "
            "rectangularity will be determined from the data. This is a non-negligible"
            "operation, so if you know it, it can be provided."
        )
        self._blts_are_rectangular = uvp.UVParameter(
            "blts_are_rectangular",
            description=desc,
            required=False,
            expected_type=bool,
            ignore_eq_none=True,
        )

        desc = (
            "If blts are rectangular, this variable specifies whether the time axis is"
            "the fastest-moving virtual axis. Various reading/indexing functions "
            "benefit from knowing this, so if it is known, it should be provided."
        )
        self._time_axis_faster_than_bls = uvp.UVParameter(
            "time_axis_faster_than_bls",
            description=desc,
            required=False,
            expected_type=bool,
            ignore_eq_none=True,
        )

        desc = (
            "Any user supplied extra keywords, type=dict. Keys should be "
            "8 character or less strings if writing to uvfits or miriad files. "
            'Use the special key "comment" for long multi-line string comments.'
        )
        self._extra_keywords = uvp.UVParameter(
            "extra_keywords",
            required=False,
            description=desc,
            value={},
            spoof_val={},
            expected_type=dict,
        )

        # --- other stuff ---
        # the below are copied from AIPS memo 117, but could be revised to
        # merge with other sources of data.
        self._gst0 = uvp.UVParameter(
            "gst0",
            required=False,
            description="Greenwich sidereal time at midnight on reference date.",
            spoof_val=0.0,
            expected_type=float,
        )
        self._rdate = uvp.UVParameter(
            "rdate",
            required=False,
            description="Date for which the GST0 applies.",
            spoof_val="",
            form="str",
        )
        self._earth_omega = uvp.UVParameter(
            "earth_omega",
            required=False,
            description="Earth's rotation rate in degrees per day.",
            spoof_val=360.985,
            expected_type=float,
        )
        self._dut1 = uvp.UVParameter(
            "dut1",
            required=False,
            description="DUT1 (google it) AIPS 117 calls it UT1UTC.",
            spoof_val=0.0,
            expected_type=float,
        )
        self._timesys = uvp.UVParameter(
            "timesys",
            required=False,
            description="We only support UTC.",
            spoof_val="UTC",
            form="str",
        )

        desc = (
            "FHD thing we do not understand, something about the time "
            "at which the phase center is normal to the chosen UV plane "
            "for phasing."
        )
        self._uvplane_reference_time = uvp.UVParameter(
            "uvplane_reference_time", required=False, description=desc, spoof_val=0
        )

        desc = "Per-antenna and per-frequency equalization coefficients."
        self._eq_coeffs = uvp.UVParameter(
            "eq_coeffs",
            required=False,
            description=desc,
            form=("Nants_telescope", "Nfreqs"),
            expected_type=float,
            spoof_val=1.0,
        )

        desc = "Convention for how to remove eq_coeffs from data."
        self._eq_coeffs_convention = uvp.UVParameter(
            "eq_coeffs_convention",
            required=False,
            description=desc,
            form="str",
            spoof_val="divide",
        )

        desc = (
            "List of strings containing the unique basenames (not the full path) of "
            "input files."
        )
        self._filename = uvp.UVParameter(
            "filename", required=False, description=desc, expected_type=str
        )

        desc = (
            "The convention for how instrumental polarizations (e.g. XX and YY) "
            "are converted to Stokes parameters. Options are 'sum' and 'avg', "
            "corresponding to I=XX+YY and I=(XX+YY)/2 (for linear instrumental "
            "polarizations) respectively. This parameter is not required, and "
            "only makes sense for calibrated data. If pol_convention is set, "
            "vis_units should be set to real units (as opposed to 'uncalib')."
        )
        self._pol_convention = uvp.UVParameter(
            "pol_convention",
            required=False,
            description=desc,
            form="str",
            spoof_val="avg",
            acceptable_vals=["sum", "avg"],
        )

        self.__antpair2ind_cache = {}
        self.__key2ind_cache = {}

        super().__init__()

        # Assign attributes to UVParameters after initialization, since UVBase.__init__
        # will link the properties to the underlying UVParameter.value attributes
        # initialize the telescope object
        self.telescope = Telescope()

        # set the appropriate telescope attributes as required
        self._set_telescope_requirements()

    def _set_telescope_requirements(self):
        """Set the UVParameter required fields appropriately for UVData."""
        self.telescope._instrument.required = True
        self.telescope._feed_array.required = False
        self.telescope._feed_angle.required = False
        self.telescope._mount_type.required = False

    # This is required for eq_coeffs, which has Nants_telescope as one of its
    # shapes. That's to allow us to line up the antenna_numbers/names with
    # eq_coeffs so that we know which antenna each eq_coeff goes with.
    # TODO: do we want a setter on UVData for this?
    @property
    def Nants_telescope(self):  # noqa N802
        """
        The number of antennas in the telescope.

        This property is stored on the Telescope object internally.
        """
        return self._telescope.value.Nants

    @staticmethod
    def _clear_antpair2ind_cache(obj):
        """Clear the antpair2ind cache."""
        obj.__antpair2ind_cache = {}
        obj.__key2ind_cache = {}

    @staticmethod
    def _clear_key2ind_cache(obj):
        """Clear the antpair2ind cache."""
        obj.__key2ind_cache = {}

    @staticmethod
    @combine_docstrings(new_uvdata, style=DocstringStyle.NUMPYDOC)
    def new(**kwargs):  # noqa: D102
        return new_uvdata(**kwargs)

    def _set_scan_numbers(self, override=False):
        """
        Set scan numbers by grouping consecutive integrations on the same phase center.

        This approach mimics the definition of scan number in measurement sets and is
        especially helpful for distinguishing between repeated visits to multiple
        phase centers.

        Parameters
        ----------
        override : bool
            When True, will redefine existing scan numbers. Default is False.
        """
        if self.scan_number_array is None or override:
            slice_list = []
            # This loops over phase centers, finds contiguous integrations with
            # ndimage.label, and then finds the slices to return those contiguous
            # integrations with nd.find_objects.
            for cat_id in self.phase_center_catalog:
                slice_list.extend(
                    nd.find_objects(nd.label(self.phase_center_id_array == cat_id)[0])
                )

            # Sort by start integration number, which we can extract from
            # the start of each slice in the list.
            slice_list_ord = sorted(slice_list, key=lambda x: x[0].start)

            # Incrementally increase the scan number with each group in
            # slice_list_ord
            scan_array = np.zeros_like(self.phase_center_id_array)
            for ii, slice_scan in enumerate(slice_list_ord):
                scan_array[slice_scan] = ii + 1

            self.scan_number_array = scan_array

    def _add_phase_center(
        self,
        cat_name,
        *,
        cat_type=None,
        cat_lon=None,
        cat_lat=None,
        cat_frame=None,
        cat_epoch=None,
        cat_times=None,
        cat_pm_ra=None,
        cat_pm_dec=None,
        cat_dist=None,
        cat_vrad=None,
        info_source="user",
        force_update=False,
        cat_id=None,
    ):
        """
        Add an entry to the internal object/source catalog or find a matching one.

        This is a helper function for identifying a adding a phase center to the
        internal catalog, contained within the attribute `phase_center_catalog`, unless
        a phase center already exists that matches the passed parameters. If a matching
        phase center is found, the catalog ID associated with that phase center is
        returned.

        Parameters
        ----------
        cat_name : str
            Name of the phase center to be added.
        cat_type : str
            Type of phase center to be added. Must be one of:
                "sidereal" (fixed RA/Dec),
                "ephem" (RA/Dec that moves with time),
                "driftscan" (fixed az/el position),
                "unprojected" (no w-projection, equivalent to the old
                `phase_type` == "drift").
        cat_lon : float or ndarray
            Value of the longitudinal coordinate (e.g., RA, Az, l) in radians of the
            phase center. No default unless `cat_type="unprojected"`, in which case the
            default is zero. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_lat : float or ndarray
            Value of the latitudinal coordinate (e.g., Dec, El, b) in radians of the
            phase center. No default unless `cat_type="unprojected"`, in which case the
            default is pi/2. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_frame : str
            Coordinate frame that cat_lon and cat_lat are given in. Only used
            for sidereal and ephem targets. Can be any of the several supported frames
            in astropy (a limited list: fk4, fk5, icrs, gcrs, cirs, galactic).
        cat_epoch : str or float
            Epoch of the coordinates, only used when cat_frame = fk4 or fk5. Given
            in units of fractional years, either as a float or as a string with the
            epoch abbreviation (e.g, Julian epoch 2000.0 would be J2000.0).
        cat_times : ndarray of floats
            Only used when `cat_type="ephem"`. Describes the time for which the values
            of `cat_lon` and `cat_lat` are calculated, in units of JD. Shape is (Npts,).
        cat_pm_ra : float
            Proper motion in RA, in units of mas/year. Only used for sidereal phase
            centers.
        cat_pm_dec : float
            Proper motion in Dec, in units of mas/year. Only used for sidereal phase
            centers.
        cat_dist : float or ndarray of float
            Distance of the source, in units of pc. Only used for sidereal and ephem
            phase centers. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_vrad : float or ndarray of float
            Radial velocity of the source, in units of km/s. Only used for sidereal and
            ephem phase centers. Expected to be a float for sidereal and driftscan phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        info_source : str
            Optional string describing the source of the information provided. Used
            primarily in UVData to denote when an ephemeris has been supplied by the
            JPL-Horizons system, user-supplied, or read in by one of the various file
            interpreters. Default is 'user'.
        force_update : bool
            Normally, `_add_phase_center` will throw an error if there already exists a
            phase_center with the given cat_id. However, if one sets
            `force_update=True`, the method will overwrite the existing entry in
            `phase_center_catalog` with the parameters supplied. Note that doing this
            will _not_ update other attributes of the `UVData` object. Default is False.
        cat_id : int
            An integer signifying the ID number for the phase center, used in the
            `phase_center_id_array` attribute. If a matching phase center entry exists
            already, that phase center ID will be returned, which may be different than
            the value specified to this parameter. The default is for the method to
            assign this value automatically.

        Returns
        -------
        cat_id : int
            The unique ID number for the phase center that either matches the specified
            parameters or was added to the internal catalog. If a matching entry was
            found, this may not be the value passed to the `cat_id` parameter. This
            value is used in the `phase_center_id_array` attribute to denote which
            source a given baseline-time corresponds to.

        Raises
        ------
        ValueError
            If attempting to add a non-unique source name or if adding a sidereal
            source without coordinates.

        """
        cat_entry = utils.phase_center_catalog.generate_phase_center_cat_entry(
            cat_name=cat_name,
            cat_type=cat_type,
            cat_lon=cat_lon,
            cat_lat=cat_lat,
            cat_frame=cat_frame,
            cat_epoch=cat_epoch,
            cat_times=cat_times,
            cat_pm_ra=cat_pm_ra,
            cat_pm_dec=cat_pm_dec,
            cat_dist=cat_dist,
            cat_vrad=cat_vrad,
            info_source=info_source,
        )

        # We want to create a unique ID for each source, for use in indexing arrays.
        # The logic below ensures that we pick the lowest positive integer that is
        # not currently being used by another source
        if cat_id is None or not force_update:
            cat_id = utils.phase_center_catalog.generate_new_phase_center_id(
                phase_center_catalog=self.phase_center_catalog, cat_id=cat_id
            )

        if self.phase_center_catalog is None:
            # Initialize an empty dict to plug entries into
            self.phase_center_catalog = {}
        else:
            # Let's warn if this entry has the same name as an existing one
            temp_id, cat_diffs = utils.phase_center_catalog.look_in_catalog(
                self.phase_center_catalog, phase_dict=cat_entry
            )

            # If the source does have the same name, check to see if all the
            # attributes match. If so, no problem, go about your business
            if temp_id is not None:
                if cat_diffs == 0:
                    # Everything matches, return the catalog ID of the matching entry
                    return temp_id
                warnings.warn(
                    f"The provided name {cat_name} is already used but has different "
                    "parameters. Adding another entry with the same name but a "
                    "different ID and parameters."
                )

        # If source is unique, begin creating a dictionary for it
        self.phase_center_catalog[cat_id] = cat_entry
        self.Nphase = len(self.phase_center_catalog)

        return cat_id

    def _remove_phase_center(self, defunct_id):
        """
        Remove an entry from the internal object/source catalog.

        Removes an entry from the attribute `phase_center_catalog`.

        Parameters
        ----------
        defunct_id : int
            Catalog ID of the source to be removed

        Raises
        ------
        IndexError
            If the name provided is not found as a key in `phase_center_catalog`
        """
        if defunct_id not in self.phase_center_catalog:
            raise IndexError("No source by that ID contained in the catalog.")

        del self.phase_center_catalog[defunct_id]
        self.Nphase = len(self.phase_center_catalog)

    def _clear_unused_phase_centers(self):
        """
        Remove objects dictionaries and names that are no longer in use.

        Goes through the `phase_center_catalog` attribute in of a UVData object and
        clears out entries that are no longer being used, and appropriately updates
        `phase_center_id_array` accordingly. This function is not typically called
        by users, but instead is used by other methods.

        """
        unique_cat_ids = np.unique(self.phase_center_id_array)
        defunct_list = []
        Nphase = 0
        for cat_id in self.phase_center_catalog:
            if cat_id in unique_cat_ids:
                Nphase += 1
            else:
                defunct_list.append(cat_id)

        # Check the number of "good" sources we have -- if we haven't dropped any,
        # then we are free to bail, otherwise update the Nphase attribute
        if Nphase == self.Nphase:
            return

        # Time to kill the entries that are no longer in the source stack
        for defunct_id in defunct_list:
            self._remove_phase_center(defunct_id)

    def _check_for_cat_type(self, cat_type):
        """
        Check which Nblts have a cat_type in `cat_type`.

        This convenience method returns back a boolean mask to identify which data
        along the Blt axis contains data in cat_type.

        Parameters
        ----------
        cat_type : str or list of str
            Phase types to check for.

        Returns
        -------
        blt_mask : ndarray of bool
            A boolean mask for identifying which elements contain unprojected objects.
            True where not projected, False where projected.
        """
        # Check and see if we have any data with cat_type.
        if not isinstance(cat_type, list | tuple | np.ndarray):
            cat_type = [cat_type]
        cat_type_list = [
            cat_id
            for cat_id, cat_dict in self.phase_center_catalog.items()
            if cat_dict["cat_type"] in cat_type
        ]

        # Construct a bool mask
        blt_mask = np.isin(self.phase_center_id_array, cat_type_list)

        return blt_mask

    def rename_phase_center(self, catalog_identifier, new_name):
        """
        Rename a phase center/catalog entry within a multi phase center data set.

        Parameters
        ----------
        catalog_identifier : str or int or list of int
            Unique identifier of a phase center to be renamed. If supplied as a str,
            will be matched against the phase center name. Otherwise if an int or list
            of int, assumed to be the catalog ID number(s).
        new_name : str
            New name for the phase center.


        Raises
        ------
        ValueError
            If attempting to run the method on a non multi phase center data set, or if
            `catalog_identifier` is not found in `phase_center_catalog`.
        TypeError
            If `new_name` is not actually a string or if `catalog_identifier` is not a
            string or an integer.
        """
        if (
            isinstance(catalog_identifier, str | int)
            or isinstance(catalog_identifier, list)
            and all(isinstance(cat, int) for cat in catalog_identifier)
        ):
            pass
        else:
            raise TypeError(
                "catalog_identifier must be a string, an integer or a list of integers."
            )

        if isinstance(catalog_identifier, str):
            cat_id = []
            for key, ps_dict in self.phase_center_catalog.items():
                if ps_dict["cat_name"] == catalog_identifier:
                    cat_id.append(key)
            if len(cat_id) == 0:
                raise ValueError(
                    f"No entry by the name {catalog_identifier} in the catalog."
                )
        else:
            # Force cat_id to be a list to make downstream code simpler. If cat_id is
            # an int, it will throw a TypeError on casting to list, which we can catch.
            try:
                cat_id = list(catalog_identifier)
            except TypeError:
                cat_id = [catalog_identifier]

            for key in cat_id:
                if key not in self.phase_center_catalog:
                    raise ValueError(f"No entry with the ID {key} in the catalog.")

        if not isinstance(new_name, str):
            raise TypeError("Value provided to new_name must be a string.")

        if (new_name == catalog_identifier) or (len(cat_id) == 0):
            # This is basically just a no-op, so return to user
            return

        for key in cat_id:
            self.phase_center_catalog[key]["cat_name"] = new_name

    def split_phase_center(
        self,
        catalog_identifier,
        *,
        new_name=None,
        select_mask=None,
        new_id=None,
        downselect=False,
    ):
        """
        Rename the phase center (but preserve other properties) of a subset of data.

        Allows you to rename a subset of the data phased to a particular phase center,
        marked by a different name than the original. Useful when you want to phase to
        one position, but want to differentiate different groups of data (e.g., marking
        every other integration to make jackknifing easier).

        Parameters
        ----------
        catalog_identifier : str or int
            Unique identifier of a phase center to be renamed. If supplied as a str,
            will be matched against the phase center name. Otherwise if an int, assumed
            to be the catalog ID number.
        new_name : str
            Name for the "split" portion of the phase center. Optional argument, default
            is to use the same name as the existing phase center.
        select_mask : array_like
            Selection mask for which data should be identified as belonging to the phase
            center labeled by `new_name`. Any array-like able to be used as an index
            is suitable -- the most typical is an array of bool with length `Nblts`,
            or an array of ints within the range (-Nblts, Nblts).
        new_id : int
            Catalog ID to assign to the new phase center. Optional argument, a unique
            value will be automatically assigned if not provided.
        downselect : bool
            If selecting data that is not marked as belonging to `cat_name`,
            normally an error is thrown. By setting this to True, `select_mask` will
            be modified to exclude data not marked as belonging to `cat_name`.

        Raises
        ------
        ValueError
            If  catalog_identifier is not an int or string or if it is not found as a
            cat_name in `phase_center_catalog` or if it is a string and is found
            multiple times. If new_id is not an int or already exists in the catalog.
            If new_name is not a string. If `select_mask` contains data that doesn't
            belong to catalog_identifier, unless `downselect` is True.
        IndexError
            If select_mask is not a valid indexing array.
        UserWarning
            If all data for `cat_name` was selected (in which case `rename_phase_center`
            is called instead), or if no valid data was selected.
        """
        # Check to make sure that everything lines up with
        if not isinstance(catalog_identifier, str | int):
            raise TypeError("catalog_identifier must be a string or an integer.")

        if isinstance(catalog_identifier, str):
            cat_id = None
            for pc_id, pc_dict in self.phase_center_catalog.items():
                if pc_dict["cat_name"] == catalog_identifier:
                    if cat_id is None:
                        cat_id = pc_id
                    else:
                        raise ValueError(
                            f"The cat_name {catalog_identifier} has multiple entries "
                            "in the catalog. Please specify a cat_id in order to "
                            "eliminate ambiguity (which you can see using the "
                            "`print_phase_center_info` method)."
                        )
            if cat_id is None:
                raise ValueError(
                    f"No catalog entries matching the name {catalog_identifier}."
                )
        else:
            cat_id = catalog_identifier
            if cat_id not in self.phase_center_catalog:
                raise ValueError(f"No entry with the ID {cat_id} found in the catalog.")

        if new_id is None:
            new_id = (
                set(range(self.Nphase + 1)).difference(self.phase_center_catalog).pop()
            )
        elif not isinstance(new_id, int):
            raise TypeError("Value provided to new_id must be an int.")

        if new_id in self.phase_center_catalog:
            raise ValueError(
                f"The ID {new_id} is already in the catalog, choose another value for "
                "new_id."
            )

        if not (isinstance(new_name, str) or new_name is None):
            raise TypeError("Value provided to new_name must be a string.")

        try:
            inv_mask = np.ones(self.Nblts, dtype=bool)
            inv_mask[select_mask] = False
        except IndexError as err:
            raise IndexError(
                "select_mask must be an array-like, either of ints with shape (Nblts), "
                "or  of ints within the range (-Nblts, Nblts)."
            ) from err

        # If we have selected any entries that don't correspond to the cat_id
        # in question, either downselect or raise an error.
        if np.any(
            np.isin(self.phase_center_id_array[select_mask], cat_id, invert=True)
        ):
            if downselect:
                inv_mask |= np.isin(self.phase_center_id_array, cat_id, invert=True)
                select_mask = ~inv_mask
            else:
                raise ValueError(
                    "Data selected with select_mask includes data which is not part of "
                    "the selected phase_center ({cat_id}).  You can fix this by either "
                    "revising select_mask or setting downselect=True."
                )

        # Now check for no(-ish) ops
        if np.all(inv_mask):
            # You didn't actually select anything we could change
            warnings.warn("No relevant data selected - check select_mask.")
        elif not np.any(np.isin(self.phase_center_id_array[inv_mask], cat_id)):
            # No matching catalog IDs found outside the range, so this is really a
            # replace more than a split.
            warnings.warn(
                "All data for the source selected - updating the cat_id instead."
            )
            self._update_phase_center_id(cat_id, new_id=new_id)
            if new_name is not None:
                self.rename_phase_center(new_id, new_name)
        else:
            temp_dict = self.phase_center_catalog[cat_id]
            cat_id = self._add_phase_center(
                temp_dict["cat_name"] if new_name is None else new_name,
                cat_type=temp_dict["cat_type"],
                cat_lon=temp_dict.get("cat_lon"),
                cat_lat=temp_dict.get("cat_lat"),
                cat_frame=temp_dict.get("cat_frame"),
                cat_epoch=temp_dict.get("cat_epoch"),
                cat_times=temp_dict.get("cat_times"),
                cat_pm_ra=temp_dict.get("cat_pm_ra"),
                cat_pm_dec=temp_dict.get("cat_pm_dec"),
                cat_dist=temp_dict.get("cat_dist"),
                cat_vrad=temp_dict.get("cat_vrad"),
                cat_id=new_id,
                force_update=True,
            )
            self.phase_center_id_array[select_mask] = cat_id

    def merge_phase_centers(
        self, catalog_identifier, *, force_merge=False, ignore_name=False
    ):
        """
        Merge two differently named objects into one within a multi-phase-ctr data set.

        Recombines two different objects into a single catalog entry -- useful if
        having previously used `split_phase_center` or when multiple objects with
        different names share the same source parameters.

        Parameters
        ----------
        catalog_identifier : str or int or list of str or int
            Unique identifier(s) of a phase center to be merged. If supplied as strings,
            will be matched against the phase center name. Otherwise if supplied as
            integers, assumed to be the catalog ID number(s).
        force_merge : bool
            Normally, the method will throw an error if the phase center properties
            differ for the catalogs listed in catalog_identifier. This can be overriden
            by setting this to True. Default is False.
        ignore_name : bool
            When comparing phase centers, all attributes are normally checked. However,
            if set to True, the catalog name ("cat_name") will be ignored when
            performing the comparison. Default is False.

        Raises
        ------
        ValueError
            If catalog_identifiers are not found in the UVData object, of if their
            properties differ (and `force_merge` is not set to True).

        Warns
        -----
        UserWarning
            If forcing the merge of two objects with different properties.

        """
        if (
            isinstance(catalog_identifier, str | int)
            or isinstance(catalog_identifier, list)
            and all(isinstance(cat, str | int) for cat in catalog_identifier)
        ):
            pass
        else:
            raise TypeError(
                "catalog_identifier must be a string, an integer or a list of strings "
                "or integers."
            )

        if not isinstance(catalog_identifier, list):
            catalog_identifier = [catalog_identifier]

        cat_id_list = []
        for cat in catalog_identifier:
            if isinstance(cat, str):
                this_list = []
                for key, ps_dict in self.phase_center_catalog.items():
                    if ps_dict["cat_name"] == cat:
                        this_list.append(key)
                if len(this_list) == 0:
                    raise ValueError(f"No entry by the name {cat} in the catalog.")
                cat_id_list.extend(this_list)
            else:
                # Force cat_id to be a list to make downstream code simpler. If cat_id
                # is an int, it will throw a TypeError on casting to list, which we can
                # catch.
                if cat not in self.phase_center_catalog:
                    raise ValueError(f"No entry with the ID {cat} in the catalog.")
                cat_id_list.append(cat)

        # Check for the no-op
        if len(cat_id_list) < 2:
            warnings.warn(
                "Selection matches less than two phase centers, no need to merge."
            )
            return

        # First, let's check and see if the dict entries are identical
        for cat_id in cat_id_list[1:]:
            pc_id, pc_diffs = utils.phase_center_catalog.look_in_catalog(
                self.phase_center_catalog,
                phase_dict=self.phase_center_catalog[cat_id],
                ignore_name=ignore_name,
                target_cat_id=cat_id_list[0],
            )
            if (pc_diffs != 0) or (pc_id is None):
                if force_merge:
                    warnings.warn(
                        "Forcing fields together, even though their attributes differ."
                    )
                else:
                    raise ValueError(
                        "Attributes of phase centers differ in phase_center_catalog. "
                        "You can ignore this error and force merge_phase_centers to "
                        "complete by setting force_merge=True, but this should be done "
                        "with substantial caution."
                    )

        # Set everything to the first cat ID in the list
        self.phase_center_id_array[np.isin(self.phase_center_id_array, cat_id_list)] = (
            cat_id_list[0]
        )

        # Finally, remove the defunct cat IDs
        for cat_id in cat_id_list[1:]:
            self._remove_phase_center(cat_id)

    def print_phase_center_info(
        self,
        catalog_identifier=None,
        *,
        hms_format=None,
        return_str=False,
        print_table=True,
    ):
        """
        Print out the details of the phase centers.

        Prints out an ASCII table that contains the details of the
        `phase_center_catalog` attribute, which acts as the internal source catalog
        for UVData objects.

        Parameters
        ----------
        catalog_identifier : str or int or list of str or int
            Optional parameter which, if provided, will cause the method to only return
            information on the phase center(s) with the matching name(s) or catalog ID
            number(s). Default is to print out information on all catalog entries.
        hms_format : bool
            Optional parameter, which if selected, can be used to force coordinates to
            be printed out in Hours-Min-Sec (if set to True) or Deg-Min-Sec (if set to
            False) format. Default is to print out in HMS if all the objects have
            coordinate frames of icrs, gcrs, fk5, fk4, and top; otherwise, DMS format
            is used.
        return_str: bool
            If set to True, the method returns an ASCII string which contains all the
            table infrmation. Default is False.
        print_table : bool
            If set to True, prints the table to the terminal window. Default is True.

        Returns
        -------
        table_str : bool
            If return_str=True, an ASCII string containing the entire table text

        Raises
        ------
        ValueError
            If `cat_name` matches no keys in `phase_center_catalog`.
        """
        return utils.phase_center_catalog.print_phase_center_info(
            self.phase_center_catalog,
            catalog_identifier=catalog_identifier,
            hms_format=hms_format,
            return_str=return_str,
            print_table=print_table,
        )

    def _update_phase_center_id(self, cat_id, *, new_id=None, reserved_ids=None):
        """
        Update a phase center with a new catalog ID number.

        Parameters
        ----------
        cat_id : int
            Current catalog ID of the phase center, which corresponds to a key in the
            attribute `phase_center_catalog`.
        new_id : int
            Optional argument. If supplied, then the method will attempt to use the
            provided value as the new catalog ID, provided that an existing catalog
            entry is not already using the same value. If not supplied, then the
            method will automatically assign a value.
        reserved_ids : array-like in int
            Optional argument. An array-like of ints that denotes which ID numbers
            are already reserved. Useful for when combining two separate catalogs.

        Raises
        ------
        ValueError
            If not using the method on a multi-phase-ctr data set, if there's no entry
            that matches `cat_name`, or of the value `new_id` is already taken.
        """
        new_id = utils.phase_center_catalog.generate_new_phase_center_id(
            phase_center_catalog=self.phase_center_catalog,
            cat_id=new_id,
            old_id=cat_id,
            reserved_ids=reserved_ids,
        )

        # If new_id is None, it means that the existing ID was fine and needs no update
        if new_id is not None:
            self.phase_center_id_array[self.phase_center_id_array == cat_id] = new_id
            self.phase_center_catalog[new_id] = self.phase_center_catalog.pop(cat_id)

    def _consolidate_phase_center_catalogs(
        self, *, reference_catalog=None, other=None, ignore_name=False
    ):
        """
        Consolidate phase center catalogs with a reference or another object.

        This is a helper method which updates the phase_center_catalog and related
        parameters to make this object consistent with a reference catalog or with
        another object so the second object can be added or concatenated to this object.
        If both `reference_catalog` and `other` are provided, both this object and the
        one passed to `other` will have their catalogs updated.

        Parameters
        ----------
        reference_catalog : dict
            A reference catalog to make this object consistent with.
        other : UVData object
            A UVData object which self needs to be consistent with because it will be
            added to self. The phase_center_catalog from other is used as the reference
            catalog if the reference_catalog is None. If `reference_catalog` is also
            set, the phase_center_catalog on other will also be modified to be
            consistent with the `reference_catalog`.
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when identifying matching phase centers. If set to
            True, phase centers that are the same up to their name will be combined with
            the name set to the reference catalog name or the name found in the first
            UVData object. If set to False, phase centers that are the same up to the
            name will be kept as separate phase centers. Default is False.

        """
        if reference_catalog is None and other is None:
            raise ValueError(
                "Either the reference_catalog or the other parameter must be set."
            )

        if other is not None:
            # If exists, first update other to be consistent with the reference
            if reference_catalog is not None:
                other._consolidate_phase_center_catalogs(
                    reference_catalog=reference_catalog
                )
            # then use the updated other as the reference
            reference_catalog = other.phase_center_catalog

        reserved_ids = list(reference_catalog)
        # First loop, we want to update all the catalog IDs so that we know there
        # are no conflicts with the reference
        for cat_id in list(self.phase_center_catalog):
            self._update_phase_center_id(cat_id, reserved_ids=reserved_ids)

        # Next loop, we want to update the IDs of sources that are in the reference.
        for cat_id in list(reference_catalog):
            # Normally one would wrap this in an items() call above, except that for
            # testing it's sometimes convenient to use self.phase_center_catalog as
            # the ref catalog, which causes a RunTime error due to updates to the dict.
            cat_entry = reference_catalog[cat_id]
            match_id, match_diffs = utils.phase_center_catalog.look_in_catalog(
                self.phase_center_catalog, phase_dict=cat_entry, ignore_name=ignore_name
            )
            if match_id is None or match_diffs != 0:
                # If no match, just add the entry
                self._add_phase_center(cat_id=cat_id, **cat_entry)
                continue

            # If match_diffs is 0 then all the keys in the phase center catalog
            # match, so this is functionally the same source
            self._update_phase_center_id(match_id, new_id=cat_id)
            # look_in_catalog ignores the "info_source" field, so update it to match
            self.phase_center_catalog[cat_id]["info_source"] = cat_entry.get(
                "info_source"
            )
            if ignore_name:
                # Make the names match if names were ignored in matching
                self.phase_center_catalog[cat_id]["cat_name"] = cat_entry["cat_name"]

    @property
    def _data_params(self):
        """List of strings giving the data-like parameters."""
        return ["data_array", "nsample_array", "flag_array"]

    @property
    def data_like_parameters(self):
        """Iterate defined parameters which are data-like (not metadata-like)."""
        for key in self._data_params:
            if hasattr(self, key):
                yield getattr(self, key)

    @property
    def metadata_only(self):
        """
        Property that determines whether this is a metadata only object.

        An object is metadata only if data_array, nsample_array and flag_array
        are all None.
        """
        metadata_only = all(d is None for d in self.data_like_parameters)

        for param_name in self._data_params:
            getattr(self, "_" + param_name).required = not metadata_only

        return metadata_only

    def _old_phase_attributes_compatible(self):
        """
        Check if this object is compatible with the old phase attributes.

        Returns
        -------
        compatible : bool
            True if this object is compatible with the old phase attributes, False
            otherwise
        reason : str
            Reason it is not compatible. None if compatible is True.

        """
        if self.phase_center_catalog is None:
            return True, None

        if self.Nphase > 1:
            return False, "multiple phase centers"
        phase_dict = list(self.phase_center_catalog.values())[0]
        if phase_dict["cat_type"] not in ["sidereal", "unprojected"]:
            return False, f"{phase_dict['cat_type']} phase centers"
        if phase_dict["cat_type"] == "sidereal" and phase_dict["cat_frame"] not in [
            "icrs",
            "gcrs",
        ]:
            return False, f"{phase_dict['cat_frame']} phase frames"
        else:
            return True, None

    def known_telescopes(self):
        """
        Get a list of telescopes known to pyuvdata.

        This is just a shortcut to uvdata.telescopes.known_telescopes()

        Returns
        -------
        list of str
            List of names of known telescopes
        """
        return known_telescopes()

    def set_telescope_params(
        self,
        *,
        x_orientation=None,
        mount_type=None,
        overwrite=False,
        warn=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Set telescope related parameters.

        If the telescope_name is in astropy sites or known_telescopes, set any
        missing telescope parameters (e.g. telescope location, antenna information)
        to the value(s) from astropy sites or known telescopes.

        Parameters
        ----------
        x_orientation : str or None
            String describing how the x-orientation is oriented. Must be either "north"/
            "n"/"ns" (x-polarization of antenna has a position angle of 0 degrees with
            respect to zenith/north) or "east"/"e"/"ew" (x-polarization of antenna has a
            position angle of 90 degrees with respect to zenith/north). Ignored if
            "x_orientation" is relevant entry for the known telescope, or if set to
            None.
        mount_type : str or None
            String describing the mount amount type, which describes the optics.
            Supported options include: "alt-az" (primary rotates in azimuth and
            elevation), "equatorial" (primary rotates in hour angle and declination),
            "orbiting" (antenna is in motion, and its orientation depends on orbital
            parameters), "x-y" (primary rotates first in the plane connecting east,
            west, and zenith, and then perpendicular to that plane),
            "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
            mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
            tertiary mirror), "phased" (antenna is "electronically steered" by
            summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
            beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
            (also referred to in some formats as "bizarre"). See the "Conventions"
            page of the documentation for further details.
        overwrite : bool
            Option to overwrite existing telescope-associated parameters with
            the values from the known telescope. Default is False.
        warn : bool
            Option to issue a warning listing all modified parameters. Default is True
            if `overwrite=True`, otherwise False.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after updating. Default is True.
        check_extra : bool
            Option to check optional parameters as well as required ones. Default is
            True.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            updating. Default is True

        Raises
        ------
        ValueError
            If the telescope_name is not in astropy sites or known telescopes.
        """
        self.telescope.update_params_from_known_telescopes(
            overwrite=overwrite,
            warn=overwrite if warn is None else warn,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            x_orientation=x_orientation,
            mount_type=mount_type,
            polarization_array=self.polarization_array,
            flex_polarization_array=self.flex_spw_polarization_array,
            override_known_params=False,
        )

    def _calc_single_integration_time(self):
        """
        Calculate a single integration time in seconds when not otherwise specified.

        This function computes the shortest time difference present in the
        time_array, and returns it to be used as the integration time for all
        samples.

        Returns
        -------
        int_time : int
            integration time in seconds to be assigned to all samples in the data.

        """
        # The time_array is in units of days, and integration_time has units of
        # seconds, so we need to convert.
        return np.diff(np.sort(list(set(self.time_array))))[0] * 86400

    def _set_lsts_helper(self, *, astrometry_library=None):
        # the utility function is efficient -- it only calculates unique times
        self.lst_array = utils.get_lst_for_time(
            jd_array=self.time_array,
            telescope_loc=self.telescope.location,
            frame=self.telescope._location.frame,
            ellipsoid=self.telescope._location.ellipsoid,
            astrometry_library=astrometry_library,
        )
        return

    def _set_app_coords_helper(self, *, pa_only=False):
        """
        Set values for the apparent coordinate arrays.

        This is an internal helper function, which is not designed to be called by
        users, but rather individual read/write functions for the UVData object.
        Users should use the phase() method for updating/adjusting coordinate values.

        Parameters
        ----------
        pa_only : bool, False
            Skip the calculation of the apparent RA/Dec, and only calculate the
            position angle between `cat_frame` and the apparent coordinate
            system. Useful for reading in data formats that do not calculate a PA.
        """
        if pa_only:
            app_ra = self.phase_center_app_ra
            app_dec = self.phase_center_app_dec
        else:
            app_ra = np.zeros(self.Nblts, dtype=float)
            app_dec = np.zeros(self.Nblts, dtype=float)
            cat_ids = np.unique(self.phase_center_id_array)
            for cat_id in cat_ids:
                temp_dict = self.phase_center_catalog[cat_id]
                select_mask = self.phase_center_id_array == cat_id
                cat_type = temp_dict["cat_type"]
                lon_val = temp_dict.get("cat_lon")
                lat_val = temp_dict.get("cat_lat")
                time_val = temp_dict.get("cat_times")
                epoch = temp_dict.get("cat_epoch")
                frame = temp_dict.get("cat_frame")
                pm_ra = temp_dict.get("cat_pm_ra")
                pm_dec = temp_dict.get("cat_pm_dec")
                vrad = temp_dict.get("vrad")
                dist = temp_dict.get("cat_dist")

                if self.blts_are_rectangular and len(cat_ids) == 1:
                    slc = (
                        slice(None, self.Ntimes)
                        if self.time_axis_faster_than_bls
                        else slice(None, None, self.Nbls)
                    )
                    _app_ra, _app_dec = phs_utils.calc_app_coords(
                        lon_coord=lon_val,
                        lat_coord=lat_val,
                        coord_frame=frame,
                        coord_epoch=epoch,
                        coord_times=time_val,
                        pm_ra=pm_ra,
                        pm_dec=pm_dec,
                        vrad=vrad,
                        dist=dist,
                        time_array=self.time_array[slc],
                        lst_array=self.lst_array[slc],
                        telescope_loc=self.telescope.location,
                        coord_type=cat_type,
                        all_times_unique=True,
                    )
                    if self.time_axis_faster_than_bls:
                        app_ra = np.tile(_app_ra, self.Nbls)
                        app_dec = np.tile(_app_dec, self.Nbls)
                    else:
                        app_ra = np.repeat(_app_ra, self.Nbls)
                        app_dec = np.repeat(_app_dec, self.Nbls)
                else:
                    app_ra[select_mask], app_dec[select_mask] = (
                        phs_utils.calc_app_coords(
                            lon_coord=lon_val,
                            lat_coord=lat_val,
                            coord_frame=frame,
                            coord_epoch=epoch,
                            coord_times=time_val,
                            pm_ra=pm_ra,
                            pm_dec=pm_dec,
                            vrad=vrad,
                            dist=dist,
                            time_array=self.time_array[select_mask],
                            lst_array=self.lst_array[select_mask],
                            telescope_loc=self.telescope.location,
                            coord_type=cat_type,
                        )
                    )

        # Now that we have the apparent coordinates sorted out, we can figure
        # out what it is we want to do with the position angle
        frame_pa = np.zeros(self.Nblts, dtype=float)
        for cat_id in self.phase_center_catalog:
            temp_dict = self.phase_center_catalog[cat_id]
            select_mask = self.phase_center_id_array == cat_id
            if not np.any(select_mask):
                continue
            frame = temp_dict.get("cat_frame")
            epoch = temp_dict.get("cat_epoch")
            if frame != "altaz":
                frame_pa[select_mask] = phs_utils.calc_frame_pos_angle(
                    time_array=self.time_array[select_mask],
                    app_ra=app_ra[select_mask],
                    app_dec=app_dec[select_mask],
                    telescope_loc=self.telescope.location,
                    ref_frame=frame,
                    ref_epoch=epoch,
                )
        self.phase_center_app_ra = app_ra
        self.phase_center_app_dec = app_dec
        self.phase_center_frame_pa = frame_pa

    def set_lsts_from_time_array(self, *, background=False, astrometry_library=None):
        """Set the lst_array based from the time_array.

        Parameters
        ----------
        background : bool, False
            When set to True, start the calculation on a threading.Thread in the
            background and return the thread to the user.
        astrometry_library : str
            Library used for calculating the LSTs. Allowed options are
            'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas
            library), and 'astropy' (which uses the astropy utilities). Default is erfa
            unless the telescope_location frame is MCMF (on the moon), in which case the
            default is astropy.

        Returns
        -------
        proc : None or threading.Thread instance
            When background is set to True, a thread is returned which must be
            joined before the lst_array exists on the UVData object.

        """
        if not background:
            self._set_lsts_helper(astrometry_library=astrometry_library)
            return
        else:
            proc = threading.Thread(
                target=self._set_lsts_helper,
                kwargs={"astrometry_library": astrometry_library},
            )
            proc.start()
            return proc

    def _check_flex_spw_contiguous(self, *, raise_errors=True):
        """
        Check if the spectral windows are contiguous.

        This checks the flex_spw_id_array to make sure that all channels for each
        spectral window are together in one block, versus being interspersed (e.g.,
        channel #1 and #3 is in spw #1, channels #2 and #4 are in spw #2). In theory,
        UVH5 and UVData objects can handle this, but MIRIAD, MIR, UVFITS, and MS file
        formats cannot, so we just consider it forbidden.

        Parameters
        ----------
        raise_errors : bool or None
            Option to raise errors if the various checks do not pass. If True, and
            error is raised. If False, then a warning is raised. If None, no
            errors or warnings are raised.
        """
        utils.frequency._check_flex_spw_contiguous(
            spw_array=self.spw_array,
            flex_spw_id_array=self.flex_spw_id_array,
            strict=raise_errors,
        )

    def _check_freq_spacing(self, *, raise_errors=True):
        """
        Check if frequencies are evenly spaced and separated by their channel width.

        This is a requirement for writing uvfits & miriad files.

        Parameters
        ----------
        raise_errors : bool
            Option to raise errors if the various checks do not pass. If True, and
            error is raised. If False, then a warning is raised. If None, no
            errors or warnings are raised.

        Returns
        -------
        spacing_error : bool
            Flag that channel spacings or channel widths are not equal.
        chanwidth_error : bool
            Flag that channel spacing does not match channel width.

        """
        return utils.frequency._check_freq_spacing(
            freq_array=self._freq_array,
            channel_width=self._channel_width,
            spw_array=self.spw_array,
            flex_spw_id_array=self.flex_spw_id_array,
            strict=raise_errors,
        )

    def _check_pol_spacing(self, *, raise_errors=True):
        """
        Check if polarizations are evenly spaced.

        This is a requirement for writing uvfits files.

        Parameters
        ----------
        raise_errors : bool
            If set to True, then the function will raise an error if checks are failed.
            If set to False, then a warning is raised instead. If set to None, then
            no errors or warnings are raised.

        """
        # Resort allowed since UVFITS code will reorder pols if needed.
        return utils.pol._check_pol_spacing(
            polarization_array=self._polarization_array,
            strict=raise_errors,
            allow_resort=True,
        )

    def remove_flex_pol(self, *, combine_spws=True):
        """
        Convert a flex-pol UVData object into one with a standard polarization axis.

        This will convert a flexible-polarization dataset into one with standard
        polarization handling, which is required for some operations or writing in
        certain filetypes. Note that depending on how it is used, this can inflate
        the size of data-like parameters by up to a factor of Nspws (the true value
        depends on the number of unique entries in `flex_spw_polarization_array`).

        Parameters
        ----------
        combine_spws : bool
            If set to True, the method will attempt to recombine multiple windows
            carrying different polarization information into a single (multi-pol)
            spectral window. Functionally, this is the inverse of what is done in the
            `convert_to_flex_pol` method. If set to False, the method will effectively
            "inflate" the polarization-axis of UVData parameters such that all windows
            have the same polarization codes (though the added entries will be flagged
            and will carry no data). Default is True.
        """
        if self.flex_spw_polarization_array is None:
            # There isn't anything to do, so just move along
            return

        unique_pols = np.unique(self.flex_spw_polarization_array)
        n_pols = len(unique_pols)

        if self.Nspws == 1 or n_pols == 1:
            # Just remove the flex_spw_polarization_array and fix the polarization array
            self.polarization_array = unique_pols
            self.flex_spw_polarization_array = None
            return

        if combine_spws:
            # check to see if there are spectral windows that have matching freq_array
            # and channel_width (up to sorting). If so, they need to be combined.
            freq_array_use = self.freq_array

            # Now find matching sets of spws
            # order spws by order of appearance in flex_spw_id_array
            # this tends to get back to the original order in a convert/remove loop
            spws_remaining = self.flex_spw_id_array[
                np.sort(np.unique(self.flex_spw_id_array, return_index=True)[1])
            ].tolist()
            # key is first spw in a set, value is another dict with keys:
            #  - "freqs" sorted array of frequencies
            #  - "widths" channel widths sorted to match frequencies (using argsort)
            #  - "spws" list of spws in this set
            #  - "pols" list of pols for the spws in this set
            spw_dict = {}
            # key is spw, value is first spw that matches (key into spw_dict)
            first_spw_dict = {}
            while len(spws_remaining) > 0:
                this_spw = spws_remaining[0]
                this_pol = self.flex_spw_polarization_array[self.spw_array == this_spw][
                    0
                ]
                spw_inds = np.nonzero(self.flex_spw_id_array == this_spw)[0]
                spw_inds = spw_inds[np.argsort(freq_array_use[spw_inds])]
                this_freqs = freq_array_use[spw_inds]
                this_widths = self.channel_width[spw_inds]
                match = False
                for spw1, fset1 in spw_dict.items():
                    if np.array_equal(fset1["freqs"], this_freqs) and np.array_equal(
                        fset1["widths"], this_widths
                    ):
                        if this_pol in spw_dict[spw1]["pols"]:
                            raise ValueError(
                                "Some spectral windows have identical frequencies, "
                                "channel widths and polarizations, so spws cannot be "
                                "combined. Set combine_spws=False to avoid this error."
                            )
                        # this spw matches an existing set with no overlapping pols
                        spw_dict[spw1]["spws"].append(this_spw)
                        spw_dict[spw1]["pols"].append(this_pol)
                        first_spw_dict[this_spw] = spw1
                        spws_remaining.remove(this_spw)
                        match = True
                        continue
                if not match:
                    # this spw does not match an existing set
                    spw_dict[this_spw] = {
                        "freqs": this_freqs,
                        "widths": this_widths,
                        "spws": [this_spw],
                        "pols": [this_pol],
                    }
                    first_spw_dict[this_spw] = this_spw
                    spws_remaining.remove(this_spw)

            n_sets = len(spw_dict)
            n_spws_per_set = []
            n_freqs = 0
            reorder_channels = False
            for spw1, spw_set in spw_dict.items():
                n_spws_per_set.append(len(spw_set["spws"]))
                n_freqs += spw_set["freqs"].size
                spw1_mask = self.flex_spw_id_array == spw1
                for spw2 in spw_set["spws"][1:]:
                    spw2_mask = self.flex_spw_id_array == spw2
                    if not (
                        np.array_equal(
                            freq_array_use[spw1_mask], freq_array_use[spw2_mask]
                        )
                        and np.array_equal(
                            self.channel_width[spw1_mask], self.channel_width[spw2_mask]
                        )
                    ):
                        reorder_channels = True

            n_spws_per_set = np.array(n_spws_per_set)

            if not np.all(n_spws_per_set == n_pols):
                # If all the pols are not present in all sets, we cannot combine spws
                warnings.warn(
                    "combine_spws is True but there are not matched spws for all "
                    "polarizations, so spws will not be combined."
                )
                combine_spws = False

        if combine_spws:
            # figure out whether reordering is required to use an inplace reshape
            # Criteria:
            #  1) polarization is the slowest changing axis
            #  2) freqs and spw sets are in the same order for each pol
            reorder_spws = False
            spw_inds = np.zeros_like(self.flex_spw_id_array)
            first_spw_array = np.zeros_like(self.flex_spw_id_array)
            for spw_ind, spw in enumerate(self.spw_array):
                these_freq_inds = np.nonzero(self.flex_spw_id_array == spw)[0]
                spw_inds[these_freq_inds] = spw_ind
                first_spw_array[these_freq_inds] = first_spw_dict[spw]
            pol_array_check = self.flex_spw_polarization_array[spw_inds]
            if np.nonzero(np.diff(pol_array_check))[0].size != n_pols - 1:
                reorder_spws = True

            pol0_spw_order = first_spw_array[pol_array_check == unique_pols[0]]
            for pol in unique_pols[1:]:
                this_spw_order = first_spw_array[pol_array_check == pol]
                if not np.array_equal(this_spw_order, pol0_spw_order):
                    reorder_spws = True

            if reorder_channels or reorder_spws:
                if reorder_channels:
                    channel_order = "freq"
                else:
                    channel_order = None
                if reorder_spws:
                    # note: spw_order is an index array into spw_array
                    spw_order = np.zeros(self.Nspws, dtype=int)
                    if np.all(unique_pols < 0):
                        # use more standard ordering for polarizations
                        unique_pols = unique_pols[np.argsort(np.abs(unique_pols))]
                    for pol_ind, pol in enumerate(unique_pols):
                        for spw_ind, (_, spw_set) in enumerate(spw_dict.items()):
                            this_ind = pol_ind * n_sets + spw_ind
                            this_spw = np.array(spw_set["spws"])[spw_set["pols"] == pol]
                            spw_order[this_ind] = np.nonzero(
                                self.spw_array == this_spw
                            )[0][0]
                else:
                    spw_order = None

                self.reorder_freqs(channel_order=channel_order, spw_order=spw_order)

                # recalculate arrays used below
                freq_array_use = self.freq_array
                first_spw_array = np.zeros_like(self.flex_spw_id_array)
                for spw_ind, spw in enumerate(self.spw_array):
                    these_freq_inds = np.nonzero(self.flex_spw_id_array == spw)[0]
                    spw_inds[these_freq_inds] = spw_ind
                    first_spw_array[these_freq_inds] = first_spw_dict[spw]
                pol_array_check = self.flex_spw_polarization_array[spw_inds]

            self.Npols = n_pols
            self.Nspws = n_sets
            self.Nfreqs = n_freqs

            # now things are in the correct order to do a simple reshape
            self.freq_array = freq_array_use[: self.Nfreqs]
            self.channel_width = self.channel_width[: self.Nfreqs]

            self.polarization_array = pol_array_check[
                np.sort(np.unique(pol_array_check, return_index=True)[1])
            ]
            self.flex_spw_polarization_array = None

            self.spw_array = first_spw_array[
                np.sort(np.unique(first_spw_array, return_index=True)[1])
            ]
            self.flex_spw_id_array = first_spw_array[: self.Nfreqs]
            if not self.metadata_only:
                # we use the order="F" parameter here to undo the reshape done in
                # `convert_to_flexp_pol` (which uses it to ensure that polarization is
                # the slowest changing axis)
                self.data_array = self.data_array.reshape(
                    self.Nblts, self.Nfreqs, self.Npols, order="F"
                )
                self.flag_array = self.flag_array.reshape(
                    self.Nblts, self.Nfreqs, self.Npols, order="F"
                )
                self.nsample_array = self.nsample_array.reshape(
                    self.Nblts, self.Nfreqs, self.Npols, order="F"
                )
            return

        self.Npols = n_pols
        # If we have metadata only, or there was only one pol we were working with,
        # then we do not need to do anything further aside from removing the array
        # associated with flex_spw_polarization_array
        if self.metadata_only:
            self.polarization_array = unique_pols
            self.flex_spw_polarization_array = None
            return

        # Otherwise, move through all of the data params
        self.polarization_array = unique_pols
        for name, param in zip(
            self._data_params, self.data_like_parameters, strict=True
        ):
            # We need to construct arrays with the appropriate shape
            new_shape = [self.Nblts, self.Nfreqs, self.Npols]

            # Use full here, since we want zeros if we are working with nsample_array
            # or data_array, otherwise we want True if working w/ flag_array.
            new_param = np.full(new_shape, name == "flag_array", dtype=param.dtype)

            # Now we have to iterate through each spectral window
            for spw, pol in zip(
                self.spw_array, self.flex_spw_polarization_array, strict=True
            ):
                pol_idx = np.intersect1d(
                    pol, self.polarization_array, return_indices=True
                )[2][0]
                spw_screen = self.flex_spw_id_array == spw

                # Note that this works because pol_idx is an integer, ergo a simple
                # slice (wherease spw_screen is a complex slice, which we can only have
                # one of for an array).
                new_param[:, spw_screen, pol_idx] = param[:, spw_screen, 0]

            # With the new array defined and filled, set the attribute equal to it
            setattr(self, name, new_param)

        # Finally, remove the flex-pol attribute
        self.flex_spw_polarization_array = None

    def _make_flex_pol(self, *, raise_error=False, raise_warning=True):
        """
        Convert a regular UVData object into one with flex-polarization enabled.

        This is an internal helper function, which is not designed to be called by
        users, but rather individual read/write functions for the UVData object.
        This will convert a regular UVData object into one that uses flexible
        polarization, which allows for each spectral window to have its own unique
        polarization code, useful for storing data more compactly when different
        windows have different polarizations recorded. Note that at this time,
        only one polarization code per-spw is allowed -- if more than one polarization
        is found to have unflagged data in a given spectral window, then the object
        will not be converted.

        Parameters
        ----------
        raise_error : bool
            If an object cannot be converted to flex-pol, then raise a ValueError.
            Default is False.
        raise_warning : bool
            If an object cannot be converted to flex-pol, and `raise_error=False`, then
            raise a warning that the conversion failed. Default is True.
        """
        if self.metadata_only:
            msg = (
                "Cannot make a metadata_only UVData object flex-pol because flagging "
                "info is required. Consider using `convert_to_flex_pol` instead, but "
                "be aware that the behavior is somewhat different"
            )
            if raise_error:
                raise ValueError(msg)
            if raise_warning:
                warnings.warn(msg)
            return

        if self.Npols == 1:
            # This is basically a no-op, fix the to pol-array attributes and exit
            if self.flex_spw_polarization_array is None:
                self.flex_spw_polarization_array = (
                    np.zeros_like(self.spw_array) + self.polarization_array[0]
                )
                self.polarization_array = np.array([0])
            return

        flex_pol_idx = np.zeros_like(self.spw_array)
        for idx, spw in enumerate(self.spw_array):
            spw_screen = self.flex_spw_id_array == spw

            # For each window, we want to check that there is only one polarization with
            # any unflagged data, which we can do by seeing if not all of the flags
            # are set across the non-polarization axes (hence the ~np.all()).
            pol_check = ~np.all(self.flag_array[:, spw_screen], axis=(0, 1))

            if sum(pol_check) > 1:
                msg = (
                    "Cannot make a flex-pol UVData object, as some windows have "
                    "unflagged data in mutiple polarizations."
                )
                if raise_error:
                    raise ValueError(msg)
                if raise_warning:
                    warnings.warn(msg)
                return
            elif not np.any(pol_check):
                flex_pol_idx[idx] = -1
            else:
                flex_pol_idx[idx] = np.where(pol_check)[0][0]

        # If one window was all flagged out, but the others all belong to the same pol,
        # assume we just want that polarization.
        if len(np.unique(flex_pol_idx[flex_pol_idx >= 0])) == 1:
            flex_pol_idx[:] = np.unique(flex_pol_idx[flex_pol_idx >= 0])

        # Now that we have polarizations sorted out, update metadata attributes
        self.flex_spw_polarization_array = self.polarization_array[flex_pol_idx]
        self.polarization_array = np.array([0])
        self.Npols = 1

        # Finally, prep for working w/ data-like attibutes. Start by determining the
        # right shape for the new values
        new_shape = [self.Nblts, self.Nfreqs, 1]

        # Now go through one-by-one with data-like parameters and update
        for name, param in zip(
            self._data_params, self.data_like_parameters, strict=True
        ):
            # We can use empty here, since we know that we will be filling all
            # values of his array (and empty allows us to forgo the extra overhead
            # of setting all the elements to a particular value).
            new_param = np.empty(new_shape, dtype=param.dtype)

            # Now we have to iterate through each spectral window
            for spw, pol_idx in zip(self.spw_array, flex_pol_idx, strict=True):
                spw_screen = self.flex_spw_id_array == spw

                # Note that this works because pol_idx is an integer, ergo a simple
                # slice (wherease spw_screen is a complex slice, which we can only have
                # one of for an array).
                new_param[:, spw_screen, 0] = param[:, spw_screen, pol_idx]

            # With the new array defined and filled, set the attribute equal to it
            setattr(self, name, new_param)

    def convert_to_flex_pol(self):
        """
        Convert a regular UVData object into a flex-polarization object.

        This effectively combines the frequency and polarization axis with polarization
        changing slowest. Saving data to uvh5 files this way can speed up some kinds
        of data access.

        """
        if self.flex_spw_polarization_array is not None:
            raise ValueError("This is already a flex-pol object")

        new_spw_array = self.spw_array
        new_flex_pol_array = np.full(self.Nspws, self.polarization_array[0])
        new_spw_id_array = np.zeros((self.Nfreqs, self.Npols), dtype=int)
        for pol_ind, pol in enumerate(self.polarization_array):
            if pol_ind == 0:
                new_spw_id_array[:, 0] = self.flex_spw_id_array
            else:
                for spw in self.spw_array:
                    new_spw = (
                        set(range(self.Nspws * self.Npols + 1))
                        .difference(new_spw_array)
                        .pop()
                    )
                    new_spw_array = np.concatenate((new_spw_array, np.array([new_spw])))
                    new_flex_pol_array = np.concatenate(
                        (new_flex_pol_array, np.array([pol]))
                    )
                    spw_inds = np.nonzero(self.flex_spw_id_array == spw)[0]
                    new_spw_id_array[spw_inds, pol_ind] = new_spw

        spw_sort = np.argsort(new_spw_array)
        self.spw_array = new_spw_array[spw_sort]
        self.flex_spw_polarization_array = new_flex_pol_array[spw_sort]
        # we use the order="F" parameter here to ensure that polarization is the slowest
        # changing axis
        self.flex_spw_id_array = new_spw_id_array.reshape(
            self.Nfreqs * self.Npols, order="F"
        )
        self.Nspws = self.spw_array.size
        self.polarization_array = np.array([0])
        freq_array_use = self.freq_array
        self.freq_array = np.tile(freq_array_use, self.Npols)
        self.channel_width = np.tile(self.channel_width, self.Npols)
        if not self.metadata_only:
            # we use the order="F" parameter here to ensure that polarization is the
            # slowest changing axis
            self.data_array = self.data_array.reshape(
                self.Nblts, self.Nfreqs * self.Npols, 1, order="F"
            )
            self.flag_array = self.flag_array.reshape(
                self.Nblts, self.Nfreqs * self.Npols, 1, order="F"
            )
            self.nsample_array = self.nsample_array.reshape(
                self.Nblts, self.Nfreqs * self.Npols, 1, order="F"
            )

        self.Nfreqs = self.Nfreqs * self.Npols
        self.Npols = 1

        return

    def _calc_nants_data(self):
        """Calculate the number of antennas from ant_1_array and ant_2_array arrays."""
        return int(np.union1d(self.ant_1_array, self.ant_2_array).size)

    def _fix_autos(self):
        """Remove imaginary component of auto-correlations."""
        if self.polarization_array is None or (
            self.ant_1_array is None or self.ant_2_array is None
        ):
            warnings.warn(
                "Cannot use _fix_autos if ant_1_array, ant_2_array, or "
                "polarization_array are None. Leaving data_array untouched."
            )
            return

        # Select out the autos
        auto_screen = self.ant_1_array == self.ant_2_array

        # Only these pols have "true" auto-correlations, that we'd expect
        # to be real only. Select on only them
        auto_pol_list = ["xx", "yy", "rr", "ll", "pI", "pQ", "pU", "pV"]
        pol_screen = np.array(
            [
                utils.POL_NUM2STR_DICT[pol] in auto_pol_list
                for pol in self.polarization_array
            ]
        )

        # Make sure we actually have work to do here, otherwise skip all of this
        if (np.any(pol_screen) and np.any(auto_screen)) and not (
            pol_screen is None or auto_screen is None
        ):
            # Select out the relevant data. Need to do this because we have two
            # complex slices we need to do
            auto_data = self.data_array[auto_screen]

            # Set the autos to be real-only by taking the absolute value
            auto_data[:, :, pol_screen] = np.abs(auto_data[:, :, pol_screen])

            # Finally, plug the modified values back into data_array
            self.data_array[auto_screen] = auto_data

    def check(
        self,
        *,
        check_extra=True,
        run_check_acceptability=True,
        check_freq_spacing=False,
        check_pol_spacing=False,
        raise_spacing_errors=True,
        strict_uvw_antpos_check=False,
        allow_flip_conj=False,
        check_autos=False,
        fix_autos=False,
        lst_tol=utils.LST_RAD_TOL,
    ):
        """
        Add some extra checks on top of checks on UVBase class.

        Check that required parameters exist. Check that parameters have
        appropriate shapes and optionally that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check all parameters, otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.
        check_freq_spacing :  bool
            Option to check if frequencies are evenly spaced and the spacing is
            equal to their channel_width. This is not required for UVData
            objects in general but is required to write to uvfits and miriad files.
        check_pol_spacing :  bool
            Option to check if polarizations are evenly spaced. This is not required for
            UVData objects in general but is required to write to uvfits and miriad
            files.
        raise_spacing_errors : bool or None
            If set to True, then the function will raise an error if spacing checks are
            failed. If set to False, then a warning is raised instead. If set to None,
            then no errors or warnings are raised.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        allow_flip_conj : bool
            If set to True, and the UVW coordinates do not match antenna positions,
            check and see if flipping the conjugation of the baselines (i.e, multiplying
            the UVWs by -1) resolves  the apparent discrepancy -- and if it does, fix
            the apparent conjugation error in `uvw_array` and `data_array`. Default is
            False.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is False.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        lst_tol : float or None
            Tolerance level at which to test LSTs against their expected values. If
            provided as a float, must be in units of radians. If set to None, the
            default precision tolerance from the `lst_array` parameter is used (1 mas).
            Default value is 75 mas,  which is set by the predictive uncertainty in IERS
            calculations of DUT1 (RMS is of order 1 ms, with with a 5-sigma threshold
            for detection is used to prevent false issues from being reported), which
            for some observatories sets the precision with which these values are
            written.

        Returns
        -------
        bool
            True if check passes

        Raises
        ------
        ValueError
            if parameter shapes or types are wrong or do not have acceptable
            values (if run_check_acceptability is True)

        """
        self._set_telescope_requirements()

        # call metadata_only to make sure that parameter requirements are set properly
        self.metadata_only  # noqa B018

        # first run the basic check from UVBase

        logger.debug("Doing UVBase check...")
        super().check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )
        logger.debug("... Done UVBase Check")

        # Check consistency between pol_convention and units of data
        if self.vis_units == "uncalib" and self.pol_convention is not None:
            raise ValueError(
                "pol_convention is set but the data is uncalibrated. This "
                "is not allowed."
            )

        # then run telescope object check
        self.telescope.check(
            check_extra=check_extra, run_check_acceptability=run_check_acceptability
        )

        # Check that all values in flex_spw_id_array are entries in the spw_array
        if not np.all(np.isin(self.flex_spw_id_array, self.spw_array)):
            raise ValueError(
                "All values in the flex_spw_id_array must exist in the spw_array."
            )

        # Check blt axis rectangularity arguments
        if self.time_axis_faster_than_bls and not self.blts_are_rectangular:
            raise ValueError(
                "time_axis_faster_than_bls is True but blts_are_rectangular is False. "
                "This is not allowed."
            )
        if (
            self.time_axis_faster_than_bls
            and self.Ntimes > 1
            and self.time_array[1] == self.time_array[0]
        ):
            raise ValueError(
                "time_axis_faster_than_bls is True but time_array does not move first"
            )

        # Check internal consistency of numbers which don't explicitly correspond
        # to the shape of another array.
        if self.Nants_data != self._calc_nants_data():
            raise ValueError(
                "Nants_data must be equal to the number of unique "
                "values in ant_1_array and ant_2_array"
            )

        if self.Nbls != len(np.unique(self.baseline_array)):
            raise ValueError(
                "Nbls must be equal to the number of unique "
                f"baselines in the data_array. Got {self.Nbls}, not"
                f"{len(np.unique(self.baseline_array))}"
            )

        if self.Ntimes != len(np.unique(self.time_array)):
            raise ValueError(
                "Ntimes must be equal to the number of unique "
                f"times in the time_array. Got {self.Ntimes}, not "
                f"{len(np.unique(self.time_array))}."
            )

        for val in np.unique(self.phase_center_id_array):
            if val not in self.phase_center_catalog:
                raise ValueError(
                    f"Phase center id {val} is does not have an entry in "
                    "`phase_center_catalog`, which has keys "
                    f"{self.phase_center_catalog.keys()}. All values in "
                    "`phase_center_id_array` must be keys in `phase_center_catalog`. "
                )

        if self.flex_spw_polarization_array is not None:
            # Check that usage of flex_spw_polarization_array follows the rule that
            # each window only has a single polarization per spectral window.
            if self.Npols != 1:
                raise ValueError(
                    "Npols must be equal to 1 if flex_spw_polarization_array is set."
                    f"Got {self.Npols}"
                )
            if np.any(self.polarization_array != 0):
                raise ValueError(
                    "polarization_array must all be equal to 0 if "
                    "flex_spw_polarization_array is set."
                )
        elif np.any(self.polarization_array == 0):
            # If flex_spw_polarization_array is not set, then make sure that
            # no entries in polarization_array are equal to zero.
            raise ValueError(
                "polarization_array may not be equal to 0 if "
                "flex_spw_polarization_array is not set."
            )

        # require that all entries in ant_1_array and ant_2_array exist in
        # antenna_numbers
        logger.debug("Doing Antenna Uniqueness Check...")
        if not set(np.unique(self.ant_1_array)).issubset(
            self.telescope.antenna_numbers
        ):
            raise ValueError("All antennas in ant_1_array must be in antenna_numbers.")
        if not set(np.unique(self.ant_2_array)).issubset(
            self.telescope.antenna_numbers
        ):
            raise ValueError("All antennas in ant_2_array must be in antenna_numbers.")
        logger.debug("... Done Antenna Uniqueness Check")

        if run_check_acceptability:
            # Check antenna positions
            utils.coordinates.check_surface_based_positions(
                antenna_positions=self.telescope.antenna_positions,
                telescope_loc=self.telescope.location,
                raise_error=False,
            )

            # Check the LSTs against what we expect given up-to-date IERS data
            utils.times.check_lsts_against_times(
                jd_array=self.time_array,
                lst_array=self.lst_array,
                lst_tols=self._lst_array.tols if lst_tol is None else [0, lst_tol],
                telescope_loc=self.telescope.location,
            )

            # create a metadata copy to do operations on
            temp_obj = self.copy(metadata_only=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logger.debug("Setting UVWs from antenna positions...")
                temp_obj.set_uvws_from_antenna_positions()
                logger.debug("... Done Setting UVWs")

            # check that the uvws make sense given the antenna positions
            # make a metadata only copy of this object to properly calculate uvws
            if not np.allclose(temp_obj.uvw_array, self.uvw_array, atol=1):
                max_diff = np.max(np.abs(temp_obj.uvw_array - self.uvw_array))
                if allow_flip_conj and np.allclose(
                    -temp_obj.uvw_array, self.uvw_array, atol=1
                ):
                    warnings.warn(
                        "UVW orientation appears to be flipped, attempting to "
                        "fix by changing conjugation of baselines."
                    )
                    self.uvw_array *= -1
                    self.data_array = np.conj(self.data_array)
                    logger.info("Flipped Array")
                elif not strict_uvw_antpos_check:
                    warnings.warn(
                        "The uvw_array does not match the expected values given "
                        "the antenna positions. The largest discrepancy is "
                        f"{max_diff} meters. This is a fairly common situation "
                        "but might indicate an error in the antenna positions, "
                        "the uvws or the phasing."
                    )
                else:
                    raise ValueError(
                        "The uvw_array does not match the expected values given "
                        "the antenna positions. The largest discrepancy is "
                        f"{max_diff} meters."
                    )

            # check auto and cross-corrs have sensible uvws
            logger.debug("Checking autos...")
            autos = self.ant_1_array == self.ant_2_array
            if not np.all(
                np.isclose(
                    self.uvw_array[autos],
                    0.0,
                    rtol=self._uvw_array.tols[0],
                    atol=self._uvw_array.tols[1],
                )
            ):
                raise ValueError(
                    "Some auto-correlations have non-zero uvw_array coordinates."
                )
            logger.debug("... Done Checking Autos")
            if (self.data_array is not None and np.any(autos)) and check_autos:
                # Verify here that the autos do not have any imaginary components
                # Only these pols have "true" auto-correlations, that we'd expect
                # to be real only. Select on only them
                auto_pol_list = ["xx", "yy", "rr", "ll", "pI", "pQ", "pU", "pV"]
                if self.flex_spw_polarization_array is not None:
                    pol_screen = np.array(
                        [
                            utils.POL_NUM2STR_DICT[pol] in auto_pol_list
                            for pol in self.flex_spw_polarization_array
                        ]
                    )
                    # There should be a better way...
                    spw_inds = np.zeros_like(self.flex_spw_id_array)
                    for spw_ind, spw in enumerate(self.spw_array):
                        these_freq_inds = np.nonzero(self.flex_spw_id_array == spw)[0]
                        spw_inds[these_freq_inds] = spw_ind
                    freq_screen = pol_screen[spw_inds]
                else:
                    pol_screen = np.array(
                        [
                            utils.POL_NUM2STR_DICT[pol] in auto_pol_list
                            for pol in self.polarization_array
                        ]
                    )

                # Check autos if they have imag component -- doing iscomplex first and
                # then pol select was faster in every case checked in test files.
                if not np.any(pol_screen):
                    # There's no relevant pols to check, just skip the rest of this
                    auto_imag = False
                else:
                    auto_imag = np.iscomplex(self.data_array[autos])
                    if np.all(pol_screen):
                        auto_imag = np.any(auto_imag)
                    elif self.flex_spw_polarization_array is not None:
                        auto_imag = np.any(auto_imag[:, freq_screen])
                    else:
                        auto_imag = np.any(auto_imag[:, :, pol_screen])
                if auto_imag:
                    if np.all(pol_screen):
                        temp_data = self.data_array[autos]
                    else:
                        auto_data = self.data_array[autos]
                        if self.flex_spw_polarization_array is not None:
                            temp_data = auto_data[:, freq_screen]
                        else:
                            temp_data = auto_data[:, :, pol_screen]
                    temp_data = temp_data[temp_data.imag != 0]
                    max_imag = np.max(np.abs(temp_data.imag))
                    max_imag_ratio = np.max(np.abs(temp_data.imag / temp_data.real))
                    if fix_autos:
                        warnings.warn(
                            "Fixing auto-correlations to be be real-only, after some "
                            "imaginary values were detected in data_array. "
                            f"Largest imaginary component was {max_imag}, largest "
                            f"imaginary/real ratio was {max_imag_ratio}."
                        )
                        self._fix_autos()
                    else:
                        raise ValueError(
                            "Some auto-correlations have non-real values in data_array."
                            f" Largest imaginary component was {max_imag}, largest "
                            f"imaginary/real ratio was {max_imag_ratio}."
                            " You can attempt to fix this by setting fix_autos=True."
                        )

            if np.any(
                np.isclose(
                    # this line used to use np.linalg.norm but it turns out
                    # squaring and sqrt is slightly more efficient unless the array
                    # is "very large". Square the tols is equivalent to getting the
                    # sqrt of the uvw magnitude, but much faster.
                    np.sum(self.uvw_array[~autos] ** 2, axis=1),
                    0.0,
                    rtol=self._uvw_array.tols[0] ** 2,
                    atol=self._uvw_array.tols[1] ** 2,
                )
            ):
                raise ValueError(
                    "Some cross-correlations have near-zero uvw_array magnitudes."
                )

        if check_freq_spacing:
            self._check_freq_spacing(raise_errors=raise_spacing_errors)

        if check_pol_spacing:
            self._check_pol_spacing(raise_errors=raise_spacing_errors)

        return True

    def copy(self, *, metadata_only=False):
        """
        Make and return a copy of the UVData object.

        Parameters
        ----------
        metadata_only : bool
            If True, only copy the metadata of the object.

        Returns
        -------
        UVData
            Copy of self.
        """
        if not metadata_only:
            return super().copy()
        else:
            uv = UVData()
            # include all attributes, not just UVParameter ones.
            for attr in self.__iter__(uvparams_only=False):
                # skip properties
                if isinstance(getattr(type(self), attr, None), property):
                    continue

                # skip data like parameters
                # parameter names have a leading underscore we want to ignore
                if attr.lstrip("_") in self._data_params:
                    continue
                setattr(uv, attr, copy.deepcopy(getattr(self, attr)))

        return uv

    def baseline_to_antnums(self, baseline):
        """
        Get the antenna numbers corresponding to a given baseline number.

        Parameters
        ----------
        baseline : int or array_like of int
            baseline number

        Returns
        -------
        int or array_like of int
            first antenna number(s)
        int or array_like of int
            second antenna number(s)
        """
        return utils.baseline_to_antnums(baseline, Nants_telescope=self.telescope.Nants)

    def antnums_to_baseline(
        self, ant1, ant2, *, attempt256=False, use_miriad_convention=False
    ):
        """
        Get the baseline number corresponding to two given antenna numbers.

        Parameters
        ----------
        ant1 : int or array_like of int
            first antenna number
        ant2 : int or array_like of int
            second antenna number
        attempt256 : bool
            Option to try to use the older 256 standard used in many uvfits files
            (will use 2048 standard if there are more than 256 antennas).
        use_miriad_convention : bool
            Option to use the MIRIAD convention where BASELINE id is
            `bl = 256 * ant1 + ant2` if `ant2 < 256`, otherwise
            `bl = 2048 * ant1 + ant2 + 2**16`.
            Note MIRIAD uses 1-indexed antenna IDs, but this code accepts 0-based.

        Returns
        -------
        int or array of int
            baseline number corresponding to the two antenna numbers.
        """
        # set attempt256 to false if using miriad convention
        attempt256 = False if use_miriad_convention else attempt256
        return utils.antnums_to_baseline(
            ant1,
            ant2,
            Nants_telescope=self.telescope.Nants,
            attempt256=attempt256,
            use_miriad_convention=use_miriad_convention,
        )

    def antpair2ind(
        self,
        ant1: int | tuple[int, int],
        ant2: int | None = None,
        *,
        ordered: bool = True,
    ) -> np.ndarray | slice | None:
        """
        Get indices along the baseline-time axis for a given antenna pair.

        This will search for either the key as specified, or the key and its
        conjugate.

        Parameters
        ----------
        ant1, ant2 : int
            Either an antenna-pair key, or key expanded as arguments,
            e.g. antpair2ind( (10, 20) ) or antpair2ind(10, 20)
        ordered : bool
            If True, search for antpair as provided, else search for it and
            its conjugate.

        Returns
        -------
        inds : ndarray of int-64 or slice
            If possible, returns a slice object that can be used to index the blt
            axis and get back the data associated with antpair. If not, returns indices
            of the antpair along the baseline-time axis. If the antpair does not exist
            in the data, returns None.
        """
        # check for expanded antpair or key
        if ant2 is None:
            if not isinstance(ant1, tuple):
                raise ValueError(
                    "antpair2ind must be fed an antpair tuple or expand it as arguments"
                )
            ant2 = ant1[1]
            ant1 = ant1[0]
        else:
            if not isinstance(ant1, int | np.integer):
                raise ValueError(
                    "antpair2ind must be fed an antpair tuple or expand it as arguments"
                )
        if not isinstance(ordered, bool | np.bool_):
            raise ValueError("ordered must be a boolean")

        # if getting auto-corr, ordered must be True
        if ant1 == ant2:
            ordered = True

        # get indices
        if (ant1, ant2, ordered) in self.__antpair2ind_cache:
            return self.__antpair2ind_cache[(ant1, ant2, ordered)]

        if self.blts_are_rectangular:
            antpairs = self.get_antpairs()
            try:
                idx = antpairs.index((ant1, ant2))
                if self.time_axis_faster_than_bls:
                    inds = slice(self.Ntimes * idx, self.Ntimes * (idx + 1))
                else:
                    inds = slice(idx, None, self.Nbls)

            except ValueError:
                # antpair is not in data
                inds = None

            if not ordered:
                try:
                    idx = antpairs.index((ant2, ant1))
                    if self.time_axis_faster_than_bls:
                        ind2 = slice(self.Ntimes * idx, self.Ntimes * (idx + 1))
                    else:
                        ind2 = slice(idx, None, self.Nbls)

                    if inds is None:
                        inds = ind2
                    else:
                        # concatenate them.
                        indxs = np.arange(self.Nblts)
                        inds = np.asarray(
                            np.append(indxs[inds], indxs[ind2]), dtype=np.int64
                        )
                except ValueError:
                    # inverse antpair is not in data
                    pass
        else:
            # get indices
            inds = np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]

            if not ordered:
                ind2 = np.where(
                    (self.ant_1_array == ant2) & (self.ant_2_array == ant1)
                )[0]
                inds = np.asarray(np.append(inds, ind2), dtype=np.int64)

            if inds.size == 0:
                inds = None

        inds = utils.tools.slicify(inds)
        self.__antpair2ind_cache[(ant1, ant2, ordered)] = inds
        return inds

    def _key2inds(self, key: str | tuple[int] | tuple[int, int] | tuple[int, int, str]):
        """
        Interpret user specified key as antenna pair and/or polarization.

        Parameters
        ----------
        key : str tuple of int
            Identifier of data. Key can be length 1, 2, or 3:

            if len(key) == 1:
                if (key < 5) or (type(key) is str):  interpreted as a
                             polarization number/name, return all blts for that pol.
                else: interpreted as a baseline number. Return all times and
                      polarizations for that baseline.

            if len(key) == 2: interpreted as an antenna pair. Return all
                times and pols for that baseline.

            if len(key) == 3: interpreted as antenna pair and pol (ant1, ant2, pol).
                Return all times for that baseline, pol. pol may be a string.

        Returns
        -------
        blt_ind1 : ndarray of int or slice or None
            blt indices for antenna pair. None if antpair is not in data.
        blt_ind2 : ndarray of int or slice or None
            blt indices for conjugate antenna pair.
            Note if a cross-pol baseline is requested, the polarization will
            also be reversed so the appropriate correlations are returned.
            e.g. asking for (1, 2, 'xy') may return conj(2, 1, 'yx'), which
            is equivalent to the requested baseline. See utils.conj_pol() for
            complete conjugation mapping.
        pol_ind : tuple of ndarray of int or slice or None
            polarization indices for blt_ind1 and blt_ind2

        Raises
        ------
        KeyError
            If the requested key is not in the data at all (either in given form
            or its conjugate).
        """
        orig_key = key

        key = utils.tools._get_iterable(key)
        if not isinstance(key, str):
            key = tuple(key)

        if key in self.__key2ind_cache:
            return self.__key2ind_cache[key]

        if isinstance(key, str):
            # Single string given, assume it is polarization
            pol_ind1 = np.where(
                self.polarization_array
                == utils.polstr2num(
                    key, x_orientation=self.telescope.get_x_orientation_from_feeds()
                )
            )[0]
            if len(pol_ind1) > 0:
                blt_ind1 = slice(None)
                blt_ind2 = None
                pol_ind2 = None
                pol_ind = (pol_ind1, pol_ind2)
            else:
                raise KeyError(f"Polarization {key} not found in data.")
        elif len(key) == 1:
            key = key[0]  # For simplicity
            if isinstance(key, Iterable):
                # Nested tuple. Call function again.
                return self._key2inds(key)
            elif key < 5:
                # Small number, assume it is a polarization number a la AIPS memo
                pol_ind1 = np.where(self.polarization_array == key)[0]
                if len(pol_ind1) > 0:
                    blt_ind1 = slice(None)
                    blt_ind2 = None
                    pol_ind2 = None
                    pol_ind = (pol_ind1, pol_ind2)
                else:
                    raise KeyError(f"Polarization {int(key)} not found in data.")
            else:
                # Larger number, assume it is a baseline number
                key = self.baseline_to_antnums(key)  # turns it into a len-2 key.

        if isinstance(key, tuple) and len(key) >= 2:
            # Key is an antenna pair
            blt_ind1 = self.antpair2ind(key[0], key[1])
            if key[0] == key[1]:  # catch autos
                blt_ind2 = None
            else:
                blt_ind2 = self.antpair2ind(key[1], key[0])

            if blt_ind1 is None and blt_ind2 is None:
                if isinstance(orig_key, int):
                    raise KeyError(f"Baseline {int(orig_key)} not found in data")
                else:
                    key_print = (int(key[0]), int(key[1]))
                    raise KeyError(f"Antenna pair {key_print} not found in data")

            if len(key) == 3:
                orig_pol = key[2]
                if isinstance(key[2], str):
                    pol = utils.polstr2num(
                        key[2],
                        x_orientation=self.telescope.get_x_orientation_from_feeds(),
                    )
                else:
                    pol = key[2]

            if blt_ind1 is None:
                pol_ind1 = None
            else:
                if len(key) == 2:
                    pol_ind1 = slice(None)
                else:
                    pol_ind1 = np.where(self.polarization_array == pol)[0]
                    if pol_ind1.size == 0:
                        pol_ind1 = None

            if blt_ind2 is None:
                pol_ind2 = None
            else:
                if len(key) == 2:
                    try:
                        pol_ind2 = utils.pol.reorder_conj_pols(self.polarization_array)
                    except ValueError as err:
                        if blt_ind1 is None:
                            if isinstance(orig_key, int):
                                key_print = int(orig_key)
                            else:
                                key_print = (int(orig_key[0]), int(orig_key[1]))

                            raise KeyError(
                                f"Baseline {key_print} not found for polarization "
                                "array in data."
                            ) from err
                        else:
                            pol_ind2 = None
                            blt_ind2 = None
                else:
                    pol_ind2 = np.where(self.polarization_array == utils.conj_pol(pol))[
                        0
                    ]
                    if pol_ind2.size == 0:
                        pol_ind2 = None

            pol_ind = (pol_ind1, pol_ind2)
            if (blt_ind1 is None or pol_ind1 is None) and (
                blt_ind2 is None or pol_ind2 is None
            ):
                if isinstance(orig_pol, str):
                    key_print = orig_pol
                else:
                    key_print = int(orig_pol)
                raise KeyError(f"Polarization {key_print} not found in data.")

        # Convert to slices if possible
        pol_ind = (utils.tools.slicify(pol_ind[0]), utils.tools.slicify(pol_ind[1]))

        self.__key2ind_cache[key] = (blt_ind1, blt_ind2, pol_ind)
        return (blt_ind1, blt_ind2, pol_ind)

    def _smart_slicing(
        self, data, ind1, ind2, indp, *, squeeze="default", force_copy=False
    ):
        """
        Quickly get the relevant section of a data-like array.

        Used in get_data, get_flags and get_nsamples.

        Parameters
        ----------
        data : ndarray
            4-dimensional array shaped like self.data_array
        ind1 : array_like of int
            blt indices for antenna pair (e.g. from self._key2inds)
        ind2 : array_like of int
            blt indices for conjugate antenna pair. (e.g. from self._key2inds)
        indp : tuple array_like of int
            polarization indices for ind1 and ind2 (e.g. from self._key2inds)
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of data
        """
        if squeeze not in ["full", "default", "none"]:
            raise ValueError(
                f'"{squeeze}" is not a valid option for squeeze.'
                'Only "default", "none", or "full" are allowed.'
            )

        if ind1 is None or ind2 is None:
            ind = ind1 if ind2 is None else ind2
            indp = indp[0] if ind2 is None else indp[1]

            if isinstance(ind, slice):
                out = data[ind, ..., indp]
            else:
                out = data[ind][..., indp]

            if ind1 is None:
                out = np.conj(out)
        else:
            # both conjugated and unconjugated baselines
            out = np.append(
                data[ind1][..., indp[0]], np.conj(data[ind2][..., indp[1]]), axis=0
            )

        if squeeze == "full":
            out = np.squeeze(out)
        elif squeeze == "default" and out.shape[2] == 1:
            # one polarization dimension
            out = np.squeeze(out, axis=2)

        if force_copy:
            out = np.array(out)
        elif out.base is not None:
            # if out is a view rather than a copy, make it read-only
            out.flags.writeable = False

        return out

    def get_ants(self):
        """
        Get the unique antennas that have data associated with them.

        Returns
        -------
        ndarray of int
            Array of unique antennas with data associated with them.
        """
        if self.blts_are_rectangular:
            if self.time_axis_faster_than_bls:
                ant1 = self.ant_1_array[:: self.Ntimes]
                ant2 = self.ant_2_array[:: self.Ntimes]
            else:
                ant1 = self.ant_1_array[: self.Nbls]
                ant2 = self.ant_2_array[: self.Nbls]
            return np.unique(np.append(ant1, ant2))
        else:
            return np.unique(np.append(self.ant_1_array, self.ant_2_array))

    def get_baseline_nums(self):
        """
        Get the unique baselines that have data associated with them.

        Returns
        -------
        ndarray of int
            Array of unique baselines with data associated with them.
        """
        if self.blts_are_rectangular:
            if self.time_axis_faster_than_bls:
                return self.baseline_array[:: self.Ntimes]
            else:
                return self.baseline_array[: self.Nbls]
        else:
            return np.unique(self.baseline_array)

    def get_antpairs(self):
        """
        Get the unique antpair tuples that have data associated with them.

        Returns
        -------
        list of tuples of int
            list of unique antpair tuples (ant1, ant2) with data associated with them.
        """
        ant1_arr, ant2_arr = self.baseline_to_antnums(self.get_baseline_nums())
        return list(zip((ant1_arr).tolist(), (ant2_arr).tolist(), strict=True))

    def get_pols(self):
        """
        Get the polarizations in the data.

        Returns
        -------
        list of str
            list of polarizations (as strings) in the data.
        """
        return utils.polnum2str(
            self.polarization_array,
            x_orientation=self.telescope.get_x_orientation_from_feeds(),
        )

    def get_antpairpols(self):
        """
        Get the unique antpair + pol tuples that have data associated with them.

        Returns
        -------
        list of tuples of int
            list of unique antpair + pol tuples (ant1, ant2, pol) with data
            associated with them.
        """
        pols = self.get_pols()
        bls = self.get_antpairs()
        return [(bl) + (pol,) for bl in bls for pol in pols]

    def get_feedpols(self):
        """
        Get the unique antenna feed polarizations in the data.

        Returns
        -------
        list of str
            list of antenna feed polarizations (e.g. ['X', 'Y']) in the data.

        Raises
        ------
        ValueError
            If any pseudo-Stokes visibilities are present
        """
        if np.any(self.polarization_array > 0):
            raise ValueError(
                "Pseudo-Stokes visibilities cannot be interpreted as feed polarizations"
            )
        else:
            return list(set("".join(self.get_pols())))

    def get_data(
        self, key1, key2=None, key3=None, *, squeeze="default", force_copy=False
    ):
        """
        Get the data corresonding to a baseline and/or polarization.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all data for
                    that pol.
                else:
                    interpreted as a baseline number, get all data for that baseline.

            if key is length 2: interpreted as an antenna pair, get all data
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all data for that baseline, pol. pol may be a string or int.
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of data.
            If data exists conjugate to requested antenna pair, it will be conjugated
            before returning.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(utils.tools._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        out = self._smart_slicing(
            self.data_array, ind1, ind2, indp, squeeze=squeeze, force_copy=force_copy
        )
        return out

    def get_flags(
        self, key1, key2=None, key3=None, *, squeeze="default", force_copy=False
    ):
        """
        Get the flags corresonding to a baseline and/or polarization.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all flags for
                    that pol.
                else:
                    interpreted as a baseline number, get all flags for that baseline.

            if key is length 2: interpreted as an antenna pair, get all flags
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all flags for that baseline, pol. pol may be a string or int.
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of flags.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(utils.tools._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        # When we select conjugated baselines, there is a call to np.conj()
        # inside of _smart_slicing to correct the data array. This has the
        # unintended consequence of promoting the dtype of an array of np.bool_
        # to np.int8. Rather than having a bunch of special handling for this
        # ~corner case, we instead explicitly cast back to np.bool_ before we
        # hand back to the user.
        out = self._smart_slicing(
            self.flag_array, ind1, ind2, indp, squeeze=squeeze, force_copy=force_copy
        ).astype(np.bool_)
        return out

    def get_nsamples(
        self, key1, key2=None, key3=None, *, squeeze="default", force_copy=False
    ):
        """
        Get the nsamples corresonding to a baseline and/or polarization.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all nsamples for
                    that pol.
                else:
                    interpreted as a baseline number, get all nsamples for that
                    baseline.

            if key is length 2: interpreted as an antenna pair, get all nsamples
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all nsamples for that baseline, pol. pol may be a string or int.
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.
        force_copy : bool
            Option to explicitly make a copy of the data.

        Returns
        -------
        ndarray
            copy (or if possible, a read-only view) of relevant section of
            nsample_array.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(utils.tools._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        out = self._smart_slicing(
            self.nsample_array, ind1, ind2, indp, squeeze=squeeze, force_copy=force_copy
        )
        return out

    def get_times(self, key1, key2=None, key3=None):
        """
        Get the times for a given antpair or baseline number.

        Meant to be used in conjunction with get_data function.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all times.
                else:
                    interpreted as a baseline number, get all times for that baseline.

            if key is length 2: interpreted as an antenna pair, get all times
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all times for that baseline.

        Returns
        -------
        ndarray
            times from the time_array for the given antpair or baseline.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(utils.tools._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        inds1, inds2, indp = self._key2inds(key)
        if inds1 is None:
            inds1 = slice(0, 0)
        if inds2 is None:
            inds2 = slice(0, 0)
        return np.append(self.time_array[inds1], self.time_array[inds2])

    def get_lsts(self, key1, key2=None, key3=None):
        """
        Get the LSTs for a given antpair or baseline number.

        Meant to be used in conjunction with get_data function.

        Parameters
        ----------
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to get, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all times.
                else:
                    interpreted as a baseline number, get all times for that baseline.

            if key is length 2: interpreted as an antenna pair, get all times
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all times for that baseline.

        Returns
        -------
        ndarray
            LSTs from the lst_array for the given antpair or baseline.
        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(utils.tools._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        inds1, inds2, indp = self._key2inds(key)
        if inds1 is None:
            inds1 = slice(0, 0)
        if inds2 is None:
            inds2 = slice(0, 0)
        return np.append(self.lst_array[inds1], self.lst_array[inds2])

    def get_enu_data_ants(self):
        """
        Get positions for antennas with data in East, North, Up coordinates.

        The difference between this method and `self.telescope.get_enu_antpos()`
        is that this method only returns positions information for antennas
        that have visibilities associated with them (the set returned by
        `self.get_ants()`). It also returns the array of antenna numbers
        corresponding to the first axis of the returned positions array.

        Returns
        -------
        antpos : ndarray
            Antenna positions in East, North, Up coordinates in units of
            meters, shape=(Nants, 3)
        ants : ndarray
            Antenna numbers matching ordering of antpos, shape=(Nants,)

        """
        antpos = self.telescope.get_enu_antpos()
        data_ants = self.get_ants()
        telescope_ants = self.telescope.antenna_numbers
        select = np.isin(telescope_ants, data_ants)
        antpos = antpos[select, :]
        ants = telescope_ants[select]

        return antpos, ants

    def _set_method_helper(self, dshape, key1, key2=None, key3=None):
        """
        Extract the indices for setting data, flags, or nsample arrays.

        This is a helper method designed to work with set_data, set_flags, and
        set_nsamples. Given the shape of the data-like array and the keys
        corresponding to where the data should end up, it finds the indices
        that are needed for the `_index_dset` method.

        Parameters
        ----------
        dshape : tuple of int
            The shape of the data-like array. This is used to ensure the array
            is compatible with the indices selected.
        key1, key2, key3 : int or tuple of ints
            Identifier of which flags to set, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, set all flags for
                    that pol.
                else:
                    interpreted as a baseline number, set all flags for that baseline.

            if key is length 2: interpreted as an antenna pair, set all flags
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                set all flags for that baseline, pol. pol may be a string or int.

        Returns
        -------
        inds : tuple of int
            The indices in the data-like array to slice into.

        Raises
        ------
        ValueError:
            If more than 3 keys are passed, if the requested indices are
            conjugated in the data, if the data array shape is not compatible
            with the indices.

        """
        key = []
        for val in [key1, key2, key3]:
            if isinstance(val, str):
                key.append(val)
            elif val is not None:
                key += list(utils.tools._get_iterable(val))
        if len(key) > 3:
            raise ValueError("no more than 3 key values can be passed")
        ind1, ind2, indp = self._key2inds(key)
        if ind2 is not None:
            raise ValueError(
                "the requested key is present on the object, but conjugated. Please "
                "conjugate data and keys appropriately and try again"
            )

        nbltinds = len(np.arange(self.Nblts)[ind1])
        npolinds = len(np.arange(self.Npols)[indp[0]])
        expected_shape = (nbltinds, self.Nfreqs, npolinds)
        if dshape != expected_shape:
            raise ValueError(
                "the input array is not compatible with the shape of the destination. "
                f"Input array shape is {dshape}, expected shape is {expected_shape}."
            )

        blt_slices, blt_sliceable = utils.tools._convert_to_slices(
            ind1, max_nslice_frac=0.1
        )
        pol_slices, pol_sliceable = utils.tools._convert_to_slices(
            indp[0], max_nslice_frac=0.5
        )

        inds = [ind1, np.s_[:], indp[0]]
        if blt_sliceable:
            inds[0] = blt_slices
        if pol_sliceable:
            inds[-1] = pol_slices

        return tuple(inds)

    def set_data(self, data, key1, key2=None, key3=None):
        """
        Set the data array to some values provided by the user.

        Parameters
        ----------
        data : ndarray of complex
            The data to overwrite into the data_array. Must be the same shape as
            the target indices.
        key1, key2, key3 : int or tuple of ints
            Identifier of which data to set, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, get all data for
                    that pol.
                else:
                    interpreted as a baseline number, get all data for that baseline.

            if key is length 2: interpreted as an antenna pair, get all data
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                get all data for that baseline, pol. pol may be a string or int.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If more than 3 keys are passed, if the requested indices are
            conjugated in the data, if the data array shape is not compatible
            with the indices.

        """
        dshape = data.shape
        inds = self._set_method_helper(dshape, key1, key2, key3)
        hdf5_utils._index_dset(self.data_array, inds, input_array=data)

        return

    def set_flags(self, flags, key1, key2=None, key3=None):
        """
        Set the flag array to some values provided by the user.

        Parameters
        ----------
        flag : ndarray of boolean
            The flags to overwrite into the fkag_array. Must be the same shape
            as the target indices.
        key1, key2, key3 : int or tuple of ints
            Identifier of which flags to set, can be passed as 1, 2, or 3 arguments
            or as a single tuple of length 1, 2, or 3. These are collectively
            called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, set all flags for
                    that pol.
                else:
                    interpreted as a baseline number, set all flags for that baseline.

            if key is length 2: interpreted as an antenna pair, set all flags
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2, pol),
                set all flags for that baseline, pol. pol may be a string or int.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If more than 3 keys are passed, if the requested indices are
            conjugated in the data, if the data array shape is not compatible
            with the indices.

        """
        dshape = flags.shape
        inds = self._set_method_helper(dshape, key1, key2, key3)
        hdf5_utils._index_dset(self.flag_array, inds, input_array=flags)

        return

    def set_nsamples(self, nsamples, key1, key2=None, key3=None):
        """
        Set the nsamples array to some values provided by the user.

        Parameters
        ----------
        nsamples : ndarray of float
            The nsamples to overwrite into the nsample_array. Must be the same
            shape as the target indices.
        key1, key2, key3 : int or tuple of ints
            Identifier of which nsamples to set, can be passed as 1, 2, or 3
            arguments or as a single tuple of length 1, 2, or 3. These are
            collectively called the key.

            If key is length 1:
                if (key < 5) or (type(key) is str):
                    interpreted as a polarization number/name, set all data for
                    that pol.
                else:
                    interpreted as a baseline number, set all nsamples for that
                    baseline.

            if key is length 2: interpreted as an antenna pair, set all nsamples
                for that baseline.

            if key is length 3: interpreted as antenna pair and pol (ant1, ant2,
                pol), set all nsamples for that baseline, pol. pol may be a
                string or int.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If more than 3 keys are passed, if the requested indices are
            conjugated in the data, if the data array shape is not compatible
            with the indices.

        """
        dshape = nsamples.shape
        inds = self._set_method_helper(dshape, key1, key2, key3)
        hdf5_utils._index_dset(self.nsample_array, inds, input_array=nsamples)

        return

    def antpairpol_iter(self, *, squeeze="default"):
        """
        Iterate the data for each antpair, polarization combination.

        Parameters
        ----------
        squeeze : str
            string specifying how to squeeze the returned array. Options are:
            'default': squeeze pol and spw dimensions if possible;
            'none': no squeezing of resulting numpy array;
            'full': squeeze all length 1 dimensions.

        Yields
        ------
        key : tuple
            antenna1, antenna2, and polarization string
        data : ndarray of complex
            data for the ant pair and polarization specified in key
        """
        antpairpols = self.get_antpairpols()
        for key in antpairpols:
            yield (key, self.get_data(key, squeeze=squeeze))

    def conjugate_bls(self, convention="ant1<ant2", *, use_enu=True, uvw_tol=0.0):
        """
        Conjugate baselines according to one of the supported conventions.

        This will fail if only one of the cross pols is present (because
        conjugation requires changing the polarization number for cross pols).

        Parameters
        ----------
        convention : str or array_like of int
            A convention for the directions of the baselines, options are:
            'ant1<ant2', 'ant2<ant1', 'u<0', 'u>0', 'v<0', 'v>0' or an
            index array of blt indices to conjugate.
        use_enu : bool
            Use true antenna positions to determine uv location (as opposed to
            uvw array). Only applies if `convention` is 'u<0', 'u>0', 'v<0', 'v>0'.
            Set to False to use uvw array values.
        uvw_tol : float
            Defines a tolerance on uvw coordinates for setting the
            u>0, u<0, v>0, or v<0 conventions. Defaults to 0m.

        Raises
        ------
        ValueError
            If convention is not an allowed value or if not all conjugate pols exist.

        """
        if isinstance(convention, np.ndarray | list | tuple):
            convention = np.array(convention)
            if (
                np.max(convention) >= self.Nblts
                or np.min(convention) < 0
                or convention.dtype not in [int, np.int_, np.int32, np.int64]
            ):
                raise ValueError(
                    "If convention is an index array, it must "
                    "contain integers and have values greater "
                    "than zero and less than NBlts"
                )
        else:
            if convention not in ["ant1<ant2", "ant2<ant1", "u<0", "u>0", "v<0", "v>0"]:
                raise ValueError(
                    "convention must be one of 'ant1<ant2', "
                    "'ant2<ant1', 'u<0', 'u>0', 'v<0', 'v>0' or "
                    "an index array with values less than NBlts"
                )

        if isinstance(convention, str):
            if convention in ["u<0", "u>0", "v<0", "v>0"]:
                if use_enu is True:
                    enu = self.telescope.get_enu_antpos()
                    anum = self.telescope.antenna_numbers.tolist()
                    uvw_array_use = np.zeros_like(self.uvw_array)
                    for i in range(self.baseline_array.size):
                        a1, a2 = self.ant_1_array[i], self.ant_2_array[i]
                        i1, i2 = anum.index(a1), anum.index(a2)
                        uvw_array_use[i, :] = enu[i2] - enu[i1]
                else:
                    uvw_array_use = copy.copy(self.uvw_array)

            if convention == "ant1<ant2":
                index_array = np.asarray(self.ant_1_array > self.ant_2_array).nonzero()
            elif convention == "ant2<ant1":
                index_array = np.asarray(self.ant_2_array > self.ant_1_array).nonzero()
            elif convention == "u<0":
                index_array = np.asarray(
                    (uvw_array_use[:, 0] > uvw_tol)
                    | (uvw_array_use[:, 1] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    | (uvw_array_use[:, 2] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                ).nonzero()
            elif convention == "u>0":
                index_array = np.asarray(
                    (uvw_array_use[:, 0] < -uvw_tol)
                    | (
                        (uvw_array_use[:, 1] < -uvw_tol)
                        & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    )
                    | (
                        (uvw_array_use[:, 2] < -uvw_tol)
                        & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                        & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                    )
                ).nonzero()
            elif convention == "v<0":
                index_array = np.asarray(
                    (uvw_array_use[:, 1] > uvw_tol)
                    | (uvw_array_use[:, 0] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                    | (uvw_array_use[:, 2] > uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                ).nonzero()
            elif convention == "v>0":
                index_array = np.asarray(
                    (uvw_array_use[:, 1] < -uvw_tol)
                    | (uvw_array_use[:, 0] < -uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                    | (uvw_array_use[:, 2] < -uvw_tol)
                    & np.isclose(uvw_array_use[:, 0], 0, atol=uvw_tol)
                    & np.isclose(uvw_array_use[:, 1], 0, atol=uvw_tol)
                ).nonzero()
        else:
            index_array = convention

        if index_array[0].size > 0:
            new_pol_inds = utils.pol.reorder_conj_pols(self.polarization_array)

            self.uvw_array[index_array] *= -1

            if not self.metadata_only:
                orig_data_array = copy.copy(self.data_array)
                for pol_ind in np.arange(self.Npols):
                    self.data_array[index_array, :, new_pol_inds[pol_ind]] = np.conj(
                        orig_data_array[index_array, :, pol_ind]
                    )

            ant_1_vals = self.ant_1_array[index_array]
            ant_2_vals = self.ant_2_array[index_array]
            self.ant_1_array[index_array] = ant_2_vals
            self.ant_2_array[index_array] = ant_1_vals
            self.baseline_array[index_array] = self.antnums_to_baseline(
                self.ant_1_array[index_array], self.ant_2_array[index_array]
            )
            self.Nbls = np.unique(self.baseline_array).size
            self._clear_antpair2ind_cache(self)

    def reorder_pols(
        self,
        order="AIPS",
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Arrange polarization axis according to desired order.

        Parameters
        ----------
        order : str
            Either a string specifying a canonical ordering ("AIPS" or "CASA")
            or an index array of length Npols that specifies how to shuffle the
            data (this is not the desired final pol order).
            CASA ordering has cross-pols in between (e.g. XX,XY,YX,YY)
            AIPS ordering has auto-pols followed by cross-pols (e.g. XX,YY,XY,YX)
            Default ("AIPS") will sort by absolute value of pol values.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reordering.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reordering.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If the order is not one of the allowed values.

        """
        if isinstance(order, np.ndarray | list | tuple):
            order = np.array(order)
            if (
                order.size != self.Npols
                or order.dtype not in [int, np.int_, np.int32, np.int64]
                or np.min(order) < 0
                or np.max(order) >= self.Npols
            ):
                raise ValueError(
                    "If order is an index array, it must "
                    "contain integers and be length Npols."
                )
            index_array = order
        elif (order == "AIPS") or (order == "CASA"):
            index_array = utils.pol.determine_pol_order(
                self.polarization_array, order=order
            )
        else:
            raise ValueError(
                "order must be one of: 'AIPS', 'CASA', or an "
                "index array of length Npols"
            )

        self._select_along_param_axis({"Npols": index_array})

        # check if object is self-consistent
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def set_rectangularity(self, *, force: bool = False) -> None:
        """
        Set the rectangularity attributes of the object.

        Parameters
        ----------
        force : bool, optional
            Whether to force setting the rectangularity attributes, even if they are
            unset. Default is to leave them unset if they are not already set, but
            otherwise just ensure correctness.

        Returns
        -------
        None
        """
        if self.blts_are_rectangular is not None and not force:
            return

        rect, time = utils.bltaxis.determine_rectangularity(
            time_array=self.time_array,
            baseline_array=self.baseline_array,
            nbls=self.Nbls,
            ntimes=self.Ntimes,
            blt_order=self.blt_order,
        )
        self.blts_are_rectangular = rect
        self.time_axis_faster_than_bls = time

    def determine_blt_order(self) -> tuple[str] | tuple[str, str] | None:
        """Determine, set and return the baseline-time ordering."""
        if self.blt_order is not None:
            return self.blt_order

        order = utils.bltaxis.determine_blt_order(
            time_array=self.time_array,
            baseline_array=self.baseline_array,
            ant_1_array=self.ant_1_array,
            ant_2_array=self.ant_2_array,
            Nbls=self.Nbls,
            Ntimes=self.Ntimes,
        )
        self.blt_order = order
        return order

    def reorder_blts(
        self,
        order="time",
        *,
        minor_order=None,
        autos_first=False,
        conj_convention=None,
        uvw_tol=0.0,
        conj_convention_use_enu=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Arrange baseline-times axis according to desired order.

        If the specified order is an index type, it will be the slowest changing index
        along the baseline-time axis (the minor order will be the next slowest).
        If the `conj_convention` is set, this method can also conjugate some baselines
        using the `conjugate_bls` method.

        Parameters
        ----------
        order : str or array_like of int
            A string describing the desired order along the blt axis or an
            index array of length Nblts that specifies the new order.
            If a string, the options are: `time`, `baseline`, `ant1`, `ant2`, `bda`.
            If this is `time`, `baseline`, `ant1`, `ant2` then this will be the
            slowest changing index.
        minor_order : str
            Optionally specify a secondary ordering.
            Default depends on how order is set:
            if order is 'time', this defaults to `baseline`,
            if order is `ant1`, or `ant2` this defaults to the other antenna,
            if order is `baseline` the only allowed value is `time`.
            Ignored if order is `bda` or an integer array.
            If this is the same as order, it is reset to the default.
            This will be the next slowest changing index.
        autos_first : bool
            If True, sort the autos before all the crosses. The autos and crosses will
            each be sorted according to the order and minor order keywords. Ignored if
            order is an integer array.
        conj_convention : str or array_like of int
            Optionally conjugate baselines to make the baselines have the
            desired orientation. See conjugate_bls for allowed values and details.
        uvw_tol : float
            If conjugating baselines, sets a tolerance for determining the signs
            of u,v, and w, and whether or not they are zero.
            See conjugate_bls for details.
        conj_convention_use_enu: bool
            If `conj_convention` is set, this is passed to conjugate_bls, see that
            method for details.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reordering.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reordering.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Raises
        ------
        ValueError
            If parameter values are inappropriate

        """
        if isinstance(order, np.ndarray | list | tuple):
            order = np.array(order)
            if order.size != self.Nblts or order.dtype not in [
                int,
                np.int_,
                np.int32,
                np.int64,
            ]:
                raise ValueError(
                    "If order is an index array, it must "
                    "contain integers and be length Nblts."
                )
            if minor_order is not None:
                raise ValueError(
                    "Minor order cannot be set if order is an index array."
                )
        else:
            if order not in ["time", "baseline", "ant1", "ant2", "bda"]:
                raise ValueError(
                    "order must be one of 'time', 'baseline', "
                    "'ant1', 'ant2', 'bda' or an index array of "
                    "length Nblts"
                )

            if minor_order == order:
                minor_order = None

            if minor_order is not None:
                if minor_order not in ["time", "baseline", "ant1", "ant2"]:
                    raise ValueError(
                        "minor_order can only be one of 'time', "
                        "'baseline', 'ant1', 'ant2'"
                    )
                if isinstance(order, np.ndarray) or order == "bda":
                    raise ValueError(
                        "minor_order cannot be specified if order is "
                        "'bda' or an index array."
                    )
                if order == "baseline" and minor_order in ["ant1", "ant2"]:
                    raise ValueError("minor_order conflicts with order")
            else:
                if order == "time":
                    minor_order = "baseline"
                elif order == "ant1":
                    minor_order = "ant2"
                elif order == "ant2":
                    minor_order = "ant1"
                elif order == "baseline":
                    minor_order = "time"

        if conj_convention is not None:
            self.conjugate_bls(
                conj_convention, use_enu=conj_convention_use_enu, uvw_tol=uvw_tol
            )

        if isinstance(order, str):
            if minor_order is None:
                self.blt_order = (order,)
                self._blt_order.form = (1,)
            else:
                self.blt_order = (order, minor_order)
                # set it back to the right shape in case it was set differently before
                self._blt_order.form = (2,)
        else:
            self.blt_order = None

        if autos_first:
            # find the auto indices
            auto_inds = np.nonzero(self.ant_1_array == self.ant_2_array)[0]
            cross_inds = np.nonzero(self.ant_1_array != self.ant_2_array)[0]
            inds_use_list = [auto_inds, cross_inds]
        else:
            inds_use_list = [np.arange(self.Nblts)]

        if not isinstance(order, np.ndarray):
            index_array = []
            for inds_use in inds_use_list:
                if inds_use.size == 0:
                    continue
                # Use lexsort to sort along different arrays in defined order.
                if order == "time":
                    arr1 = self.time_array[inds_use]
                    if minor_order == "ant1":
                        arr2 = self.ant_1_array[inds_use]
                        arr3 = self.ant_2_array[inds_use]
                    elif minor_order == "ant2":
                        arr2 = self.ant_2_array[inds_use]
                        arr3 = self.ant_1_array[inds_use]
                    else:
                        # minor_order is baseline
                        arr2 = self.baseline_array[inds_use]
                        arr3 = self.baseline_array[inds_use]
                elif order == "ant1":
                    arr1 = self.ant_1_array[inds_use]
                    if minor_order == "time":
                        arr2 = self.time_array[inds_use]
                        arr3 = self.ant_2_array[inds_use]
                    elif minor_order == "ant2":
                        arr2 = self.ant_2_array[inds_use]
                        arr3 = self.time_array[inds_use]
                    else:  # minor_order is baseline
                        arr2 = self.baseline_array[inds_use]
                        arr3 = self.time_array[inds_use]
                elif order == "ant2":
                    arr1 = self.ant_2_array[inds_use]
                    if minor_order == "time":
                        arr2 = self.time_array[inds_use]
                        arr3 = self.ant_1_array[inds_use]
                    elif minor_order == "ant1":
                        arr2 = self.ant_1_array[inds_use]
                        arr3 = self.time_array[inds_use]
                    else:
                        # minor_order is baseline
                        arr2 = self.baseline_array[inds_use]
                        arr3 = self.time_array[inds_use]
                elif order == "baseline":
                    arr1 = self.baseline_array[inds_use]
                    # only allowed minor order is time
                    arr2 = self.time_array[inds_use]
                    arr3 = self.time_array[inds_use]
                elif order == "bda":
                    arr1 = self.integration_time[inds_use]
                    # only allowed minor order is time
                    arr2 = self.baseline_array[inds_use]
                    arr3 = self.time_array[inds_use]

                # lexsort uses the listed arrays from last to first
                # (so the primary sort is on the last one)
                index_array.extend(inds_use[np.lexsort((arr3, arr2, arr1))].tolist())
        else:
            index_array = order

        self._select_along_param_axis({"Nblts": index_array})

        self.set_rectangularity(force=True)

        # check if object is self-consistent
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def reorder_freqs(
        self,
        *,
        spw_order=None,
        channel_order=None,
        select_spw=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
    ):
        """
        Arrange frequency axis according to desired order.

        Can be applied across the entire frequency axis, or just a subset.

        Parameters
        ----------
        spw_order : str or array_like of int
            A string describing the desired order of spectral windows along the
            frequecy axis. Allowed strings include `number` (sort on spectral window
            number) and `freq` (sort on median frequency). A '-' can be prepended
            to signify descending order instead of the default ascending order,
            e.g., if you have SPW #1 and 2, and wanted them ordered as [2, 1],
            you would specify `-number`. Alternatively, one can supply an index array
            of length Nspws that specifies how to shuffle the spws (this is not the
            desired final spw order). Default is to apply no sorting of spectral
            windows.
        channel_order : str or array_like of int
            A string describing the desired order of frequency channels within a
            spectral window. Allowed strings are "freq" and "-freq", which will sort
            channels within a spectral window by ascending or descending frequency
            respectively. Alternatively, one can supply an index array of length Nfreqs
            that specifies the new order. Default is to apply no sorting of channels
            within a single spectral window. Note that proving an array_like of ints
            will cause the values given to `spw_order` and `select_spw` to be ignored.
        select_spw : int or array_like of int
            An int or array_like of ints which specifies which spectral windows to
            apply sorting. Note that setting this argument will cause the value
            given to `spw_order` to be ignored.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after reordering.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reordering.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.

        Returns
        -------
        None

        Raises
        ------
        UserWarning
            Raised if providing arguments to select_spw and channel_order (the latter
            overrides the former).
        ValueError
            Raised if select_spw contains values not in spw_array, or if channel_order
            is not the same length as freq_array.

        """
        index_array = utils.frequency._sort_freq_helper(
            Nfreqs=self.Nfreqs,
            freq_array=self.freq_array,
            Nspws=self.Nspws,
            spw_array=self.spw_array,
            flex_spw_id_array=self.flex_spw_id_array,
            spw_order=spw_order,
            channel_order=channel_order,
            select_spw=select_spw,
        )

        if index_array is None:
            # This only happens if no sorting is needed
            return

        self._select_along_param_axis({"Nfreqs": index_array})

        if (self.flex_spw_id_array is not None) and (self.Nspws > 1):
            # Reorder the spw-axis items based on their first appearance in the data
            # Note that the dict will preserve first order.
            new_spw = dict.fromkeys(self.flex_spw_id_array)
            spw_map = {spw: idx for idx, spw in enumerate(self.spw_array)}
            self._select_along_param_axis({"Nspws": [spw_map[key] for key in new_spw]})

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

    def remove_eq_coeffs(self):
        """
        Remove equalization coefficients from the data.

        Some telescopes, e.g. HERA, apply per-antenna, per-frequency gain
        coefficients as part of the signal chain. These are stored in the
        `eq_coeffs` attribute of the object. This method will remove them, so
        that the data are in "unnormalized" raw units.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if eq_coeffs or eq_coeffs_convention are not defined on the
            object, or if eq_coeffs_convention is not one of "multiply" or "divide".
        """
        if self.eq_coeffs is None:
            raise ValueError(
                "The eq_coeffs attribute must be defined on the object to apply them."
            )
        if self.eq_coeffs_convention is None:
            raise ValueError(
                "The eq_coeffs_convention attribute must be defined on the object "
                "to apply them."
            )
        if self.eq_coeffs_convention not in ("multiply", "divide"):
            raise ValueError(
                f"Got unknown convention {self.eq_coeffs_convention}. Must be one of: "
                '"multiply", "divide"'
            )

        # apply coefficients for each baseline
        for key in self.get_antpairs():
            # get indices for this key
            blt_inds = self.antpair2ind(key)

            ant1_index = np.asarray(self.telescope.antenna_numbers == key[0]).nonzero()[
                0
            ][0]
            ant2_index = np.asarray(self.telescope.antenna_numbers == key[1]).nonzero()[
                0
            ][0]

            eq_coeff1 = self.eq_coeffs[ant1_index, :]
            eq_coeff2 = self.eq_coeffs[ant2_index, :]

            # make sure coefficients are the right size to broadcast
            eq_coeff1 = np.repeat(eq_coeff1[:, np.newaxis], self.Npols, axis=1)
            eq_coeff2 = np.repeat(eq_coeff2[:, np.newaxis], self.Npols, axis=1)

            if self.eq_coeffs_convention == "multiply":
                self.data_array[blt_inds] *= eq_coeff1 * eq_coeff2
            else:
                self.data_array[blt_inds] /= eq_coeff1 * eq_coeff2

        return

    def _apply_w_proj(self, *, new_w_vals, old_w_vals, select_mask=None):
        """
        Apply corrections based on changes to w-coord.

        Adjusts the data to account for a change along the w-axis of a baseline.

        Parameters
        ----------
        new_w_vals: float or ndarray of float
            New w-coordinates for the baselines, in units of meters. Can either be a
            solitary float (helpful for unphasing data, where new_w_vals can be set to
            0.0) or an array of shape (Nblts,).
        old_w_vals: float or ndarray of float
            Old w-coordinates for the baselines, in units of meters. Can either be a
            solitary float (helpful for phasing unprojected data, where old_w_vals can
            be set to 0.0) or an array of shape (Nblts,).
        select_mask: ndarray of bool
            Array of shape (Nblts,), which identifies which records to change.

        Raises
        ------
        IndexError
            If the length of new_w_vals or old_w_vals isn't compatible with
            select_mask, or if select mask isn't the right length.
        """
        # If we only have metadata, then we have no work to do. W00t!
        if self.metadata_only or (self.data_array is None):
            return

        if select_mask is None:
            select_mask = ...
        else:
            try:
                inv_mask = np.ones(self.Nblts, dtype=bool)
                inv_mask[select_mask] = False

                if all(inv_mask):
                    # If nothing is selected, then bail
                    return

                # If everything is selected by the mask, then no entries pop up in the
                # inverse, in which case it's faster to use the Ellipsis
                select_mask = ~inv_mask if any(inv_mask) else ...
            except IndexError as err:
                raise IndexError(
                    "select_mask must be an array-like, either of bool with shape "
                    "(Nblts), or of ints within the range (-Nblts, Nblts)."
                ) from err

        # Promote everything to float64 ndarrays if they aren't already
        old_w_vals = np.array(old_w_vals, dtype=np.float64)
        new_w_vals = np.array(new_w_vals, dtype=np.float64)

        if old_w_vals.shape not in [(), (1,), (self.Nblts,)]:
            raise IndexError(
                f"The length of old_w_vals is wrong (expected 1 or {self.Nblts}, "
                f"got {old_w_vals.size})!"
            )
        if new_w_vals.shape not in [(), (1,), (self.Nblts,)]:
            raise IndexError(
                f"The length of new_w_vals is wrong (expected 1 or {self.Nblts}, "
                f"got {new_w_vals.size})!"
            )

        # Calculate the difference in w terms.
        delta_w = (new_w_vals - old_w_vals).reshape(-1, 1)

        # Convert w into wavelengths as a function of freq. Note that the
        # 1/c is there to speed of processing (faster to multiply than divide).
        # Check for singleton w arrays, in which case no select mask gets applied.
        delta_w_lambda = (
            delta_w[... if delta_w.shape[0] == 1 else select_mask]
            * (1.0 / const.c.to_value("m/s"))
            * self.freq_array.reshape(1, self.Nfreqs)
        )

        self.data_array[select_mask] *= np.exp(
            (-1j * 2 * np.pi) * delta_w_lambda[:, :, None]
        )

    def unproject_phase(
        self, *, use_ant_pos=True, select_mask=None, cat_name="unprojected"
    ):
        """
        Undo phasing to get back to an `unprojected` state.

        See the phasing memo under docs/references for more documentation.

        Parameters
        ----------
        use_ant_pos : bool
            If True, calculate the uvws directly from the antenna positions
            rather than from the existing uvws. Default is True.
        select_mask : ndarray of bool
            Optional mask for selecting which data to operate on along the blt-axis.
            Shape is (Nblts,).
        cat_name : str
            Name for the newly unprojected entry in the phase_center_catalog.

        Raises
        ------
        ValueError
            If the object is alread unprojected.
        """
        # select_mask_use is length Nblts, True means should be unprojected
        # only select blts that are actually phased.
        if select_mask is not None:
            if len(select_mask) != self.Nblts:
                raise IndexError("Selection mask must be of length Nblts.")
            if not isinstance(select_mask[0], bool | np.bool_):
                raise ValueError("Selection mask must be a boolean array")
            select_mask_use = ~self._check_for_cat_type("unprojected") & select_mask
        else:
            select_mask_use = ~self._check_for_cat_type("unprojected")

        if np.all(~select_mask_use):
            warnings.warn("No selected baselines are projected, doing nothing")

        new_uvw = phs_utils.calc_uvw(
            lst_array=self.lst_array,
            use_ant_pos=use_ant_pos,
            uvw_array=self.uvw_array,
            antenna_positions=self.telescope.antenna_positions,
            antenna_numbers=self.telescope.antenna_numbers,
            ant_1_array=self.ant_1_array,
            ant_2_array=self.ant_2_array,
            old_app_ra=self.phase_center_app_ra,
            old_app_dec=self.phase_center_app_dec,
            old_frame_pa=self.phase_center_frame_pa,
            telescope_lat=self.telescope.location.lat.rad,
            telescope_lon=self.telescope.location.lon.rad,
            to_enu=True,
        )

        self._apply_w_proj(
            new_w_vals=0.0, old_w_vals=self.uvw_array[:, 2], select_mask=select_mask_use
        )
        self.uvw_array = new_uvw

        # remove/update phase center
        match_id, match_diffs = utils.phase_center_catalog.look_in_catalog(
            self.phase_center_catalog, cat_name=cat_name, cat_type="unprojected"
        )
        if match_diffs == 0:
            self.phase_center_id_array[select_mask_use] = match_id
        else:
            self.phase_center_id_array[select_mask_use] = self._add_phase_center(
                cat_name, cat_type="unprojected"
            )
        self._clear_unused_phase_centers()
        self.phase_center_app_ra[select_mask_use] = self.lst_array[
            select_mask_use
        ].copy()
        self.phase_center_app_dec[select_mask_use] = self.telescope.location.lat.rad
        self.phase_center_frame_pa[select_mask_use] = 0

        return

    def _phase_dict_helper(
        self,
        *,
        lon,
        lat,
        epoch,
        phase_frame,
        ephem_times,
        cat_type,
        pm_ra,
        pm_dec,
        dist,
        vrad,
        cat_name,
        lookup_name,
        time_array,
    ):
        """
        Supplies a dictionary with parameters for the phase method to use.

        This method should not be called directly by users; it is instead a function
        called by the `phase` method, which packages up phase center information
        into a single dictionary to allow for consistent behavior between different
        instantiations of `UVData` objects.
        """
        cat_id = None
        info_source = "user"

        name_dict = {
            pc_dict["cat_name"]: pc_id
            for pc_id, pc_dict in self.phase_center_catalog.items()
        }

        if lookup_name and (
            len(
                utils.phase_center_catalog.look_for_name(
                    self.phase_center_catalog, cat_name
                )
            )
            > 1
        ):
            raise ValueError(
                "Name of object has multiple matches in phase center catalog. "
                "Set lookup_name=False in order to continue."
            )

        if lookup_name and (cat_name not in name_dict):
            if (cat_type is None) or (cat_type == "ephem"):
                [cat_times, cat_lon, cat_lat, cat_dist, cat_vrad] = (
                    phs_utils.lookup_jplhorizons(
                        cat_name, time_array, telescope_loc=self.telescope.location
                    )
                )
                cat_type = "ephem"
                cat_pm_ra = cat_pm_dec = None
                cat_epoch = 2000.0
                cat_frame = "icrs"
                info_source = "jplh"
            else:
                raise ValueError(
                    f"Unable to find {cat_name} in among the existing sources "
                    "recorded in the catalog. Please supply source "
                    "information (e.g., RA and Dec coordinates) and "
                    "set lookup_name=False."
                )
        elif cat_name in name_dict:
            # If the name of the source matches, then verify that all of its
            # properties are the same as what is stored in phase_center_catalog.
            if lookup_name:
                cat_id = name_dict[cat_name]
                cat_diffs = 0
            else:
                cat_id, cat_diffs = utils.phase_center_catalog.look_in_catalog(
                    self.phase_center_catalog,
                    cat_name=cat_name,
                    cat_type=cat_type,
                    cat_lon=lon,
                    cat_lat=lat,
                    cat_frame=phase_frame,
                    cat_epoch=epoch,
                    cat_times=ephem_times,
                    cat_pm_ra=pm_ra,
                    cat_pm_dec=pm_dec,
                    cat_dist=dist,
                    cat_vrad=vrad,
                )
            # If cat_diffs > 0, it means that the catalog entries dont match
            if cat_diffs != 0:
                warnings.warn(
                    f"The entry name {cat_name} is not unique inside the phase "
                    "center catalog, adding anyways."
                )
                cat_type = "sidereal" if cat_type is None else cat_type
                cat_lon = lon
                cat_lat = lat
                cat_frame = phase_frame
                cat_epoch = epoch
                cat_times = ephem_times
                cat_pm_ra = pm_ra
                cat_pm_dec = pm_dec
                cat_dist = dist
                cat_vrad = vrad
            else:
                temp_dict = self.phase_center_catalog[cat_id]
                cat_type = temp_dict["cat_type"]
                info_source = temp_dict["info_source"]
                # Get here will return None if no key found, which we want
                cat_lon = temp_dict.get("cat_lon")
                cat_lat = temp_dict.get("cat_lat")
                cat_frame = temp_dict.get("cat_frame")
                cat_epoch = temp_dict.get("cat_epoch")
                cat_times = temp_dict.get("cat_times")
                cat_pm_ra = temp_dict.get("cat_pm_ra")
                cat_pm_dec = temp_dict.get("cat_pm_dec")
                cat_dist = temp_dict.get("cat_dist")
                cat_vrad = temp_dict.get("cat_vrad")
        else:
            # The name of the source is unique!
            cat_type = "sidereal" if cat_type is None else cat_type
            cat_lon = lon
            cat_lat = lat
            cat_frame = phase_frame
            cat_epoch = epoch
            cat_times = ephem_times
            cat_pm_ra = pm_ra
            cat_pm_dec = pm_dec
            cat_dist = dist
            cat_vrad = vrad

        if (cat_epoch is None) and (cat_type != "unprojected"):
            cat_epoch = 1950.0 if (cat_frame in ["fk4", "fk4noeterms"]) else 2000.0
        if isinstance(cat_epoch, str | Time):
            cat_epoch = Time(cat_epoch).to_value(
                "byear" if cat_frame in ["fk4", "fk4noeterms"] else "jyear"
            )

        # One last check - if we have an ephem phase center, lets make sure that the
        # time range of the ephemeris encapsulates the entire range of time_array
        check_ephem = False
        if cat_type == "ephem":
            # Take advantage of this to make sure that lat, lon, and times are all
            # ndarray types
            cat_lon = np.array(cat_lon, dtype=float)
            cat_lat = np.array(cat_lat, dtype=float)
            cat_times = np.array(cat_times, dtype=float)
            cat_lon.shape += (1,) if (cat_lon.ndim == 0) else ()
            cat_lat.shape += (1,) if (cat_lat.ndim == 0) else ()
            cat_times.shape += (1,) if (cat_times.ndim == 0) else ()
            check_ephem = np.min(time_array) < np.min(cat_times)
            check_ephem = check_ephem or (np.max(time_array) > np.max(cat_times))
            # If the ephem was supplied by JPL-Horizons, then we can easily expand
            # it to cover the requested range.
            if check_ephem and (info_source == "jplh"):
                # Concat the two time ranges to make sure that we cover both the
                # requested time range _and_ the original time range.
                [cat_times, cat_lon, cat_lat, cat_dist, cat_vrad] = (
                    phs_utils.lookup_jplhorizons(
                        cat_name,
                        np.concatenate((np.reshape(time_array, -1), cat_times)),
                        telescope_loc=self.telescope.location,
                    )
                )
            elif check_ephem:
                # The ephem was user-supplied during the call to the phase method,
                # raise an error to ask for more ephem data.
                raise ValueError(
                    "Ephemeris data does not cover the entirety of the time range "
                    "attempted to be phased. Please supply additional ephem data "
                    "(and if used, set lookup_name=False)."
                )
        # Time to repackage everything into a dict
        phase_dict = {
            "cat_name": cat_name,
            "cat_type": cat_type,
            "cat_lon": cat_lon,
            "cat_lat": cat_lat,
            "cat_frame": cat_frame,
            "cat_epoch": cat_epoch,
            "cat_times": cat_times,
            "cat_pm_ra": cat_pm_ra,
            "cat_pm_dec": cat_pm_dec,
            "cat_dist": cat_dist,
            "cat_vrad": cat_vrad,
            "info_source": info_source,
            "cat_id": cat_id,
        }

        # Finally, make sure everything is a float or an ndarray of floats
        for key in phase_dict:
            if isinstance(phase_dict[key], np.ndarray):
                phase_dict[key] = phase_dict[key].astype(float)
            elif (key == "cat_id") and (phase_dict[key] is not None):
                # If this is the cat_id, make it an int
                phase_dict[key] = int(phase_dict[key])
            elif not ((phase_dict[key] is None) or isinstance(phase_dict[key], str)):
                phase_dict[key] = float(phase_dict[key])
        return phase_dict

    def _apply_near_field_corrections(self, focus, ra, dec):
        """
        Apply near-field corrections by focusing the array to the specified focal point.

        Parameters
        ----------
        focus : astropy.units.Quantity object
            Focal point of the array
        ra : ndarray
            Right ascension of the focal point ie phase center (rad; shape (Ntimes,))
        dec : ndarray
            Declination of the focal point ie phase center (rad; shape (Ntimes,))

        Returns
        -------
        None (performs operations inplace)
        """
        # Obtain focal distance in km
        focus = focus.to(units.km).value

        # Convert ra, dec from radians to degrees
        ra, dec = np.degrees(ra), np.degrees(dec)

        # Calculate the x, y, z coordinates of the focal point
        # in ENU frame for each vis along Nblts axis
        focus_x, focus_y, focus_z = _get_focus_xyz(self, focus, ra, dec)

        # Calculate near-field correction at the specified timestep
        # for each vis along Nblts axis
        new_w = _get_nearfield_delay(self, focus_x, focus_y, focus_z)

        # Update phase and w
        self._apply_w_proj(new_w_vals=new_w, old_w_vals=self.uvw_array[:, -1])
        self.uvw_array[:, -1] = new_w

    def phase(
        self,
        *,
        cat_name,
        lon=None,
        lat=None,
        epoch="J2000",
        phase_frame="icrs",
        ra=None,
        dec=None,
        cat_type=None,
        ephem_times=None,
        pm_ra=None,
        pm_dec=None,
        dist=None,
        vrad=None,
        lookup_name=False,
        use_ant_pos=True,
        select_mask=None,
        cleanup_old_sources=True,
    ):
        """
        Phase data to a new direction, supports sidereal, ephemeris and driftscan types.

        Can be used to phase all or a subset of the baseline-times. Types of phase
        centers (`cat_type`) that are supported include:

            - sidereal (fixed RA/Dec)
            - ephem (RA/Dec that moves with time)
            - driftscan (fixed az/el position)

        See the phasing memo under docs/references for more documentation.

        Tested against MWA_Tools/CONV2UVFITS/convutils.

        Parameters
        ----------
        lon : float
            The longitude coordinate (e.g. RA or Azimuth) to phase to in radians.
        lat : float
            The latitude coordinate (e.g. Dec or Altitude) to phase to in radians.
        epoch : astropy.time.Time object or str
            The epoch to use for phasing. Either an astropy Time object or the
            string "J2000" (which is the default).
            Note that the epoch is only used to evaluate the ra & dec values,
            if the epoch is not J2000, the ra & dec values are interpreted
            as FK5 ra/dec values and transformed to J2000, the data are then
            phased to the J2000 ra/dec values.
        phase_frame : str
            The astropy frame to phase to, any astropy-supported frame is allowed unless
            use_old_proj is True, in which case it can only be 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
        ra : float
            An alias for `lon`.
        dec : float
            An alias for `lat`.
        cat_type : str
            Type of phase center to be added. Must be one of:
            "sidereal" (fixed RA/Dec), "ephem" (RA/Dec that moves with time),
            "driftscan" (fixed az/el position), "near_field" (first applies far-field
            phasing assuming sidereal phase center, then applies near-field
            corrections to the specified dist). Default is "sidereal".
        ephem_times : ndarray of float
            Only used when `cat_type="ephem"`. Describes the time for which the values
            of `cat_lon` and `cat_lat` are caclulated, in units of JD. Shape is (Npts,).
        pm_ra : float
            Proper motion in RA, in units of mas/year. Only used for sidereal phase
            centers.
        pm_dec : float
            Proper motion in Dec, in units of mas/year. Only used for sidereal phase
            centers.
        dist : float or ndarray of float or astropy.units.Quantity object.
            Distance to the source. Used for sidereal and ephem phase centers,
            and for applying near-field corrections. If passed either as a float
            (for sidereal phase centers) or as an ndarray of floats of shape (Npts,)
            (for ephem phase centers), will be interpreted in units of parsec for all
            cat_types except near_field; in the latter case it will be interpreted
            in meters. Alternatively, an astropy.units.Quantity object may be passed
            instead, in which case the units will be infered automatically.
        vrad : float or ndarray of float
            Radial velocity of the source, in units of km/s. Only used for sidereal and
            ephem phase centers. Expected to be a float for sidereal phase
            centers, and an ndarray of floats of shape (Npts,) for ephem phase centers.
        cat_name : str
            Name of the phase center being phased to, required.
        lookup_name : bool
            Allows the user to lookup phase center infomation in `phase_center_catalog`
            (for the entry matching `cat_name`). Setting this to `True` will ignore the
            values supplied to the `ra`, `dec`, `epoch`, `phase_frame`, `pm_ra`,
            `pm_dec`, `dist`, `vrad`.
        use_ant_pos : bool
            If True, calculate the uvws directly from the antenna positions
            rather than from the existing uvws.
        select_mask : ndarray of bool
            Optional mask for selecting which data to operate on along the blt-axis.
            Shape is (Nblts,). Ignored if `use_old_proj` is True.

        Raises
        ------
        ValueError
            If the `cat_name` is None.

        """
        if cat_type != "unprojected":
            if lon is None:
                if ra is None:
                    raise ValueError(
                        "lon parameter must be set if cat_type is not 'unprojected'"
                    )
                else:
                    lon = ra
            if lat is None:
                if dec is None:
                    raise ValueError(
                        "lat parameter must be set if cat_type is not 'unprojected'"
                    )
                else:
                    lat = dec

        # Before moving forward with the heavy calculations, we need to do some
        # basic housekeeping to make sure that we've got the coordinate data that
        # we need in order to proceed.
        if dist is not None:
            if isinstance(dist, units.Quantity):
                dist_qt = copy.deepcopy(dist)
            else:
                if cat_type == "near_field":
                    dist_qt = dist * units.m
                else:
                    dist_qt = dist * units.parsec

            dist = dist_qt.to(
                units.parsec
            ).value  # phase_dict internally stores in parsecs
        elif dist is None and cat_type == "near_field":
            raise ValueError(
                "dist parameter must be specified for cat_type 'near_field'"
            )

        phase_dict = self._phase_dict_helper(
            lon=lon,
            lat=lat,
            epoch=epoch,
            phase_frame=phase_frame,
            ephem_times=ephem_times,
            cat_type=cat_type,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            dist=dist,
            vrad=vrad,
            cat_name=cat_name,
            lookup_name=lookup_name,
            time_array=self.time_array,
        )

        if phase_dict["cat_type"] not in ["ephem", "unprojected"]:
            if np.array(lon).size > 1:
                raise ValueError(
                    "lon parameter must be a single value for cat_type "
                    f"{phase_dict['cat_type']}"
                )

            if np.array(lat).size > 1:
                raise ValueError(
                    "lat parameter must be a single value for cat_type "
                    f"{phase_dict['cat_type']}"
                )

        # Grab all the meta-data we need for the rotations
        time_array = self.time_array
        lst_array = self.lst_array
        uvw_array = self.uvw_array
        ant_1_array = self.ant_1_array
        ant_2_array = self.ant_2_array
        old_w_vals = self.uvw_array[:, 2].copy()
        old_w_vals[self._check_for_cat_type("unprojected")] = 0.0
        old_app_ra = self.phase_center_app_ra
        old_app_dec = self.phase_center_app_dec
        old_frame_pa = self.phase_center_frame_pa
        # Check and see if we have any unprojected objects, in which case
        # their w-values should be zeroed out.

        if select_mask is None:
            # If no selection mask is specified, then use the Ellipsis to access
            # the full array (and not have to check for None later on)
            select_mask = ...
        else:
            if len(select_mask) != self.Nblts:
                raise IndexError("Selection mask must be of length Nblts.")
            if not isinstance(select_mask[0], bool | np.bool_):
                raise ValueError("Selection mask must be a boolean array")

        time_array = time_array[select_mask]
        lst_array = lst_array[select_mask]
        uvw_array = uvw_array[select_mask, :]
        ant_1_array = ant_1_array[select_mask]
        ant_2_array = ant_2_array[select_mask]

        # We got the meta-data, now handle calculating the apparent coordinates.
        # First, check if we need to look up the phase center in question
        new_app_ra, new_app_dec = phs_utils.calc_app_coords(
            lon_coord=phase_dict["cat_lon"],
            lat_coord=phase_dict["cat_lat"],
            coord_frame=phase_dict["cat_frame"],
            coord_epoch=phase_dict["cat_epoch"],
            coord_times=phase_dict["cat_times"],
            coord_type=phase_dict["cat_type"],
            time_array=time_array,
            lst_array=lst_array,
            pm_ra=phase_dict["cat_pm_ra"],
            pm_dec=phase_dict["cat_pm_dec"],
            vrad=phase_dict["cat_vrad"],
            dist=phase_dict["cat_dist"],
            telescope_loc=self.telescope.location,
        )

        # Now calculate position angles.
        if phase_frame != "altaz":
            new_frame_pa = phs_utils.calc_frame_pos_angle(
                time_array=time_array,
                app_ra=new_app_ra,
                app_dec=new_app_dec,
                telescope_loc=self.telescope.location,
                ref_frame=phase_frame,
                ref_epoch=epoch,
            )
        else:
            new_frame_pa = np.zeros(time_array.shape, dtype=float)

        # Now its time to do some rotations and calculate the new coordinates
        new_uvw = phs_utils.calc_uvw(
            app_ra=new_app_ra,
            app_dec=new_app_dec,
            frame_pa=new_frame_pa,
            lst_array=lst_array,
            use_ant_pos=use_ant_pos,
            uvw_array=uvw_array,
            antenna_positions=self.telescope.antenna_positions,
            antenna_numbers=self.telescope.antenna_numbers,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            old_app_ra=old_app_ra,
            old_app_dec=old_app_dec,
            old_frame_pa=old_frame_pa,
            telescope_lat=self.telescope.location.lat.rad,
            telescope_lon=self.telescope.location.lon.rad,
        )

        # With all operations complete, we now start manipulating the UVData object
        cat_id = self._add_phase_center(
            cat_name=phase_dict["cat_name"],
            cat_type=phase_dict["cat_type"],
            cat_lon=phase_dict["cat_lon"],
            cat_lat=phase_dict["cat_lat"],
            cat_frame=phase_dict["cat_frame"],
            cat_epoch=phase_dict["cat_epoch"],
            cat_times=phase_dict["cat_times"],
            cat_pm_ra=phase_dict["cat_pm_ra"],
            cat_pm_dec=phase_dict["cat_pm_dec"],
            cat_dist=phase_dict["cat_dist"],
            cat_vrad=phase_dict["cat_vrad"],
            info_source=phase_dict["info_source"],
            cat_id=phase_dict["cat_id"],
            force_update=True,
        )

        # Extract out information for applying w-projection
        if cat_type == "unprojected":
            new_w_vals = 0.0
        else:
            # Create a blank array and fill in w-vals based on the selection mask, so
            # that the full array is the right shape when calling _apply_w-p
            new_w_vals = np.zeros(self.Nblts)
            new_w_vals[select_mask] = new_uvw[:, 2]

        # Now its time to update the raw data. This will return empty if
        # metadata_only is set to True.
        self._apply_w_proj(
            new_w_vals=new_w_vals, old_w_vals=old_w_vals, select_mask=select_mask
        )

        # Finally, we now take it upon ourselves to update some metadata.
        self.uvw_array[select_mask] = new_uvw
        self.phase_center_app_ra[select_mask] = new_app_ra
        self.phase_center_app_dec[select_mask] = new_app_dec
        self.phase_center_frame_pa[select_mask] = new_frame_pa
        self.phase_center_id_array[select_mask] = cat_id

        # If not multi phase center, make sure to update the ra/dec values, since
        # otherwise we'll have no record of source properties.
        if cleanup_old_sources:
            self._clear_unused_phase_centers()

        # Lastly, apply near-field corrections if specified
        if cat_type == "near_field":
            self._apply_near_field_corrections(
                focus=dist_qt, ra=phase_dict["cat_lon"], dec=phase_dict["cat_lat"]
            )

    def phase_to_time(
        self, time, *, phase_frame="icrs", use_ant_pos=True, select_mask=None
    ):
        """
        Phase to the ra/dec of zenith at a particular time.

        See the phasing memo under docs/references for more documentation.

        Parameters
        ----------
        time : astropy.time.Time object or float
            The time to phase to, an astropy Time object or a float Julian Date
        phase_frame : str
            The astropy frame to phase to, any astropy-supported frame is allowed unless
            use_old_proj is True, in which case it can only be 'icrs' or 'gcrs'.
            'gcrs' accounts for precession & nutation,
            'icrs' accounts for precession, nutation & abberation.
        use_ant_pos : bool
            If True, calculate the uvws directly from the antenna positions
            rather than from the existing uvws.
        select_mask : array_like
            Selection mask for which data should be rephased. Any array-like able to be
            used as an index is suitable -- the most typical is an array of bool with
            length `Nblts`, or an array of ints within the range (-Nblts, Nblts).

        Raises
        ------
        TypeError
            If time is not an astropy.time.Time object or Julian Date as a float

        """
        if isinstance(time, float | np.floating):
            time = Time(time, format="jd")

        if not isinstance(time, Time):
            raise TypeError("time must be an astropy.time.Time object or a float")

        # Generate ra/dec of zenith at time in the phase_frame coordinate
        # system to use for phasing
        on_moon = False
        if not isinstance(self.telescope.location, EarthLocation):
            with contextlib.suppress(ImportError):
                from lunarsky import MoonLocation, SkyCoord as LunarSkyCoord

                if isinstance(self.telescope.location, MoonLocation):
                    on_moon = True

        if on_moon:
            zenith_coord = LunarSkyCoord(
                alt=Angle(90 * units.deg),
                az=Angle(0 * units.deg),
                obstime=time,
                frame="lunartopo",
                location=self.telescope.location,
            )
        else:
            zenith_coord = SkyCoord(
                alt=Angle(90 * units.deg),
                az=Angle(0 * units.deg),
                obstime=time,
                frame="altaz",
                location=self.telescope.location,
            )

        obs_zenith_coord = zenith_coord.transform_to(phase_frame)
        zenith_ra = obs_zenith_coord.ra.rad
        zenith_dec = obs_zenith_coord.dec.rad

        self.phase(
            lon=zenith_ra,
            lat=zenith_dec,
            epoch="J2000",
            phase_frame=phase_frame,
            use_ant_pos=use_ant_pos,
            select_mask=select_mask,
            cat_name=(f"zenith_at_jd{time.jd:f}"),
        )

    def set_uvws_from_antenna_positions(self, *, update_vis=True):
        """
        Calculate UVWs based on antenna_positions.

        Parameters
        ----------
        update_vis : bool
            Option to update visibilities based on the new uvws (only has an effect if
            visibilities have been phased). This should only be set to False in limited
            circumstances (e.g., when certain metadata like exact times are not
            trusted), as misuse can significantly corrupt data.

        """
        unprojected_blts = self._check_for_cat_type("unprojected")

        if self.blts_are_rectangular and np.all(unprojected_blts):
            # The calculation is much more simple. We get a significant speed boost
            # by only sending one times-worth of data in.
            slc = (
                slice(None, None, self.Ntimes)
                if self.time_axis_faster_than_bls
                else slice(None, self.Nbls)
            )
            new_uvw = phs_utils.calc_uvw(
                lst_array=self.lst_array[slc],
                use_ant_pos=True,
                to_enu=True,
                antenna_positions=self.telescope.antenna_positions,
                antenna_numbers=self.telescope.antenna_numbers,
                ant_1_array=self.ant_1_array[slc],
                ant_2_array=self.ant_2_array[slc],
                telescope_lat=self.telescope.location.lat.rad,
                telescope_lon=self.telescope.location.lon.rad,
            )
            if self.time_axis_faster_than_bls:
                new_uvw = np.repeat(new_uvw, self.Ntimes, axis=0)
            else:
                new_uvw = np.tile(new_uvw, (self.Ntimes, 1))

        else:
            new_uvw = phs_utils.calc_uvw(
                app_ra=self.phase_center_app_ra,
                app_dec=self.phase_center_app_dec,
                frame_pa=self.phase_center_frame_pa,
                lst_array=self.lst_array,
                use_ant_pos=True,
                antenna_positions=self.telescope.antenna_positions,
                antenna_numbers=self.telescope.antenna_numbers,
                ant_1_array=self.ant_1_array,
                ant_2_array=self.ant_2_array,
                telescope_lat=self.telescope.location.lat.rad,
                telescope_lon=self.telescope.location.lon.rad,
                to_enu=np.all(unprojected_blts),
            )
        if np.any(~unprojected_blts):
            # At least some are phased
            if update_vis:
                old_w_vals = self.uvw_array[:, 2].copy()
                # Treat the unprojected values as having no w-proj previously
                old_w_vals[unprojected_blts] = 0.0
                self._apply_w_proj(new_w_vals=new_uvw[:, 2], old_w_vals=old_w_vals)
            else:
                warnings.warn(
                    "Recalculating uvw_array without adjusting visibility "
                    "phases -- this can introduce significant errors if used "
                    "incorrectly."
                )

        # If the data are phased, we've already adjusted the phases. Now we just
        # need to update the uvw's and we are home free.
        self.uvw_array = new_uvw
        return

    def update_antenna_positions(
        self, new_positions=None, delta_antpos=False, update_vis=True
    ):
        """
        Update antenna positions and associated (meta)data.

        This method will update the antenna positions for a UVData object, and
        correspondingly will adjust the uvw-coordinates for each of the baselines, as
        well as the phases of the visibilities (if requested).

        Parameters
        ----------
        new_positions : dict
            Dictionary containing new antenna positions, where the key is the antenna
            number (which should match to an entry in UVData.antenna_numbers), and the
            value is a 3-element array corresponding to the ECEF/MCMF position relative
            to the array center.
        delta_antpos : bool
            When set to True, uvws are updated by calculating the difference between
            the old and new antenna positions. This option should be used with care,
            and should only be used when warranted (e.g., antenna positions are stored
            with higher precision than the baselines). Default is False.
        update_vis : bool
            Option to update visibilities based on the new uvws (only has an effect if
            visibilities have been phased). This should only be set to False in limited
            circumstances (e.g., when certain metadata like exact times are not
            trusted), as misuse can significantly corrupt data.
        """
        new_antpos = self.telescope.antenna_positions.copy()
        for idx, ant in enumerate(self.telescope.antenna_numbers):
            # If no updated position is found, then just keep going
            with contextlib.suppress(KeyError):
                new_antpos[idx] = new_positions[ant]

        if np.array_equal(new_antpos, self.telescope.antenna_positions):
            warnings.warn("No antenna positions appear to have changed, returning.")
            return

        if delta_antpos:
            # We want to calculate the _relative_ changes in uvw for the new positions.
            # Take advantage of the distributive property of matrix multiplication,
            # where R(A - B) == R(A) - R(B), and R is the rotation matrix, A is the
            # upated antenna positions, and B is the old positions. I.e., this is the
            # same as independently calculating uvws from old and new and subtracting
            # one from the other.
            delta_uvw = phs_utils.calc_uvw(
                app_ra=self.phase_center_app_ra,
                app_dec=self.phase_center_app_dec,
                frame_pa=self.phase_center_frame_pa,
                lst_array=self.lst_array,
                use_ant_pos=True,
                antenna_positions=new_antpos - self.telescope.antenna_positions,
                antenna_numbers=self.telescope.antenna_numbers,
                ant_1_array=self.ant_1_array,
                ant_2_array=self.ant_2_array,
                telescope_lat=self.telescope.location.lat.rad,
                telescope_lon=self.telescope.location.lon.rad,
            )

            # Calculate the new uvw values, relate to the old ones, and add that to
            # what has been recorded as the uvw-array.
            if update_vis:
                # If requested, update the visibilities at this time. Note we screen
                # out the unprojected baselines since no phases are touched for them.
                # Also Note that the old values here are being treated as zeros since
                # we've calcualted relative deltas.
                self._apply_w_proj(
                    new_w_vals=0.0,
                    old_w_vals=delta_uvw[:, 2],  # <-- w-col of the uvws
                    select_mask=self._check_for_cat_type("unprojected"),
                )

            # Assign the new antenna position values.
            self.telescope.antenna_positions = new_antpos

            # Finally, add the deltas to the original uvw array.
            self.uvw_array += delta_uvw
        else:
            # Otherwise under "normal" circumstances, just plug in the new values and
            # update the uvws accordingly.
            self.telescope.antenna_positions = new_antpos
            self.set_uvws_from_antenna_positions(update_vis=update_vis)

    def fix_phase(self, *, use_ant_pos=True):
        """
        Fix the data to be consistent with the new phasing method.

        This is a simple utility function for updating UVW coordinates calculated using
        the 'old' phasing algorithm with those calculated by the 'new' algorithm. Note
        that this step is required for using the new methods with data phased using the
        `phase` methiod prior to pyuvdata v2.2.

        Parameters
        ----------
        use_ant_pos : bool
            Use the antenna positions for determining UVW coordinates. Default is True.
        """
        # datasets phased with new phasing should not use the this method
        compatible, reason = self._old_phase_attributes_compatible()
        if not compatible:
            raise ValueError(
                "Objects with "
                + reason
                + " were not phased with the old method, so no fixing is required."
            )

        # Record the old values
        # if we get here, we can only have one phase center
        phase_dict = list(self.phase_center_catalog.values())[0]

        # unprojected data were not phased at all, no need to fix them!
        if phase_dict["cat_type"] == "unprojected":
            raise ValueError("Data are unprojected, no phase fixing required.")

        # If we are just using the antenna positions, we don't actually need to do
        # anything, since the new baseline vectors will be unaffected by the prior
        # phasing method, and the delta_w values get correctly corrected for in
        # set_uvws_from_antenna_positions.
        if use_ant_pos:
            warnings.warn("Fixing phases using antenna positions.")

            self.set_uvws_from_antenna_positions()
        else:
            warnings.warn(
                "Attempting to fix residual phasing errors from the old `phase` method "
                "without using the antenna positions. This can result in closure "
                "errors if the data were not actually phased using the old method -- "
                "caution is advised."
            )
            # Bring the UVWs back to ENU/unprojected
            # This is the code that used to be in `unproject_phase` for the
            # old method without using antenna positions
            phase_frame = phase_dict["cat_frame"]

            icrs_coord = SkyCoord(
                ra=phase_dict["cat_lon"],
                dec=phase_dict["cat_lat"],
                unit="radian",
                frame="icrs",
            )
            if phase_frame == "icrs":
                frame_phase_center = icrs_coord
            else:
                # use center of observation for obstime for gcrs
                center_time = np.mean(
                    [np.max(self.time_array), np.min(self.time_array)]
                )
                icrs_coord.obstime = Time(center_time, format="jd")
                frame_phase_center = icrs_coord.transform_to("gcrs")

            # This promotion is REQUIRED to get the right answer when we
            # add in the telescope location for ICRS
            # In some cases, the uvws are already float64, but sometimes they're not
            self.uvw_array = np.float64(self.uvw_array)

            # apply -w phasor
            if not self.metadata_only:
                w_lambda = (
                    self.uvw_array[:, 2].reshape(self.Nblts, 1)
                    / const.c.to_value("m/s")
                    * self.freq_array.reshape(1, self.Nfreqs)
                )
                phs = np.exp(-1j * 2 * np.pi * (-1) * w_lambda[:, :, None])
                self.data_array *= phs

            unique_times, _ = np.unique(self.time_array, return_index=True)

            obs_times = Time(unique_times, format="jd")
            itrs_telescope_locations = self.telescope.location.get_itrs(
                obstime=obs_times
            )
            itrs_telescope_locations = SkyCoord(itrs_telescope_locations)
            # just calling transform_to(coord.GCRS) will delete the obstime information
            # need to re-add obstimes for a GCRS transformation

            if phase_frame == "gcrs":
                frame_telescope_locations = itrs_telescope_locations.transform_to(
                    getattr(coord, f"{phase_frame}".upper())(obstime=obs_times)
                )
            else:
                frame_telescope_locations = itrs_telescope_locations.transform_to(
                    getattr(coord, f"{phase_frame}".upper())
                )

            frame_telescope_locations.representation_type = "cartesian"

            for ind, jd in enumerate(unique_times):
                inds = np.where(self.time_array == jd)[0]

                obs_time = obs_times[ind]

                frame_telescope_location = frame_telescope_locations[ind]

                uvws_use = self.uvw_array[inds, :]

                uvw_rel_positions = phs_utils.undo_old_uvw_calc(
                    frame_phase_center.ra.rad, frame_phase_center.dec.rad, uvws_use
                )

                frame_uvw_coord = SkyCoord(
                    x=uvw_rel_positions[:, 0] * units.m + frame_telescope_location.x,
                    y=uvw_rel_positions[:, 1] * units.m + frame_telescope_location.y,
                    z=uvw_rel_positions[:, 2] * units.m + frame_telescope_location.z,
                    frame=phase_frame,
                    obstime=obs_time,
                    representation_type="cartesian",
                )

                itrs_uvw_coord = frame_uvw_coord.transform_to("itrs")
                # now convert them to ENU, which is the space uvws are in
                self.uvw_array[inds, :] = utils.ENU_from_ECEF(
                    itrs_uvw_coord.cartesian.get_xyz().value.T,
                    center_loc=self.telescope.location,
                )

            # remove/add phase center
            self.phase_center_id_array[:] = self._add_phase_center(
                "unprojected", cat_type="unprojected"
            )
            self._clear_unused_phase_centers()

            self.phase_center_app_ra = self.lst_array.copy()
            self.phase_center_app_dec[:] = (
                np.zeros(self.Nblts) + self.telescope.location.lat.rad
            )
            self.phase_center_frame_pa = np.zeros(self.Nblts)

            # Check for any autos, since their uvws get potentially corrupted
            # by the above operation
            auto_mask = self.ant_1_array == self.ant_2_array
            if any(auto_mask):
                self.uvw_array[auto_mask, :] = 0.0

            # And rephase the data using the new algorithm
            self.phase(
                lon=phase_dict["cat_lon"],
                lat=phase_dict["cat_lat"],
                phase_frame=phase_dict["cat_frame"],
                epoch=phase_dict["cat_epoch"],
                cat_name=phase_dict["cat_name"],
                use_ant_pos=False,
            )

    def __add__(
        self,
        other,
        *,
        inplace=False,
        verbose_history=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        ignore_name=False,
        warn_spacing=False,
    ):
        """
        Combine two UVData objects along frequency, polarization and/or baseline-time.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (if many objects are combined in succession this can
            lead to very long histories). If False and if the histories for the two
            objects are different, the combined object will have the history of the
            first object and only the parts of the second object history that are unique
            (this is done word by word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when combining two UVData objects. If set to True,
            phase centers that are the same up to their name will be combined with the
            name set to the name found in the first UVData object in the sum. If set to
            False, phase centers that are the same up to the name will be kept as
            separate phase centers. Default is False.
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            uvfits or mirad file formats. Default is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, self and other are not compatible
            or if data in self and other overlap.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are UVData and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        if not issubclass(other.__class__, this.__class__) and not issubclass(
            this.__class__, other.__class__
        ):
            raise ValueError(
                "Only UVData (or subclass) objects can be "
                "added to a UVData (or subclass) object"
            )
        other.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )

        # Define parameters that must be the same to add objects
        compatibility_params = ["_vis_units"]

        # Build up history string
        history_update_string = " Combined data along "
        n_axes = 0

        # Create blt arrays for convenience
        prec_t = -2 * np.floor(np.log10(this._time_array.tols[-1])).astype(int)
        prec_b = 8
        this_blts = np.array(
            [
                "_".join(
                    ["{1:.{0}f}".format(prec_t, blt[0]), str(blt[1]).zfill(prec_b)]
                )
                for blt in zip(this.time_array, this.baseline_array, strict=True)
            ]
        )
        other_blts = np.array(
            [
                "_".join(
                    ["{1:.{0}f}".format(prec_t, blt[0]), str(blt[1]).zfill(prec_b)]
                )
                for blt in zip(other.time_array, other.baseline_array, strict=True)
            ]
        )
        # Check we don't have overlapping data
        both_pol, this_pol_ind, other_pol_ind = np.intersect1d(
            this.polarization_array, other.polarization_array, return_indices=True
        )

        # If we have a flexible spectral window, the handling here becomes a bit funky,
        # because we are allowed to have channels with the same frequency *if* they
        # belong to different spectral windows (one real-life example: you might want
        # to preserve guard bands in the correlator, which can have overlaping RF
        # frequency channels)
        this_freq_ind = np.array([], dtype=np.int64)
        other_freq_ind = np.array([], dtype=np.int64)
        both_freq = np.array([], dtype=float)
        both_spw = np.intersect1d(this.spw_array, other.spw_array)
        for idx in both_spw:
            this_mask = np.where(this.flex_spw_id_array == idx)[0]
            other_mask = np.where(other.flex_spw_id_array == idx)[0]
            both_spw_freq, this_spw_ind, other_spw_ind = np.intersect1d(
                this.freq_array[this_mask],
                other.freq_array[other_mask],
                return_indices=True,
            )
            this_freq_ind = np.append(this_freq_ind, this_mask[this_spw_ind])
            other_freq_ind = np.append(other_freq_ind, other_mask[other_spw_ind])
            both_freq = np.append(both_freq, both_spw_freq)

        both_blts, this_blts_ind, other_blts_ind = np.intersect1d(
            this_blts, other_blts, return_indices=True
        )
        if not self.metadata_only and (
            len(both_pol) > 0 and len(both_freq) > 0 and len(both_blts) > 0
        ):
            # check that overlapping data is not valid
            this_inds = np.ravel_multi_index(
                (
                    this_blts_ind[:, np.newaxis, np.newaxis],
                    this_freq_ind[np.newaxis, :, np.newaxis],
                    this_pol_ind[np.newaxis, np.newaxis, :],
                ),
                this.data_array.shape,
            ).flatten()
            other_inds = np.ravel_multi_index(
                (
                    other_blts_ind[:, np.newaxis, np.newaxis],
                    other_freq_ind[np.newaxis, :, np.newaxis],
                    other_pol_ind[np.newaxis, np.newaxis, :],
                ),
                other.data_array.shape,
            ).flatten()
            this_all_zero = np.all(this.data_array.flatten()[this_inds] == 0)
            this_all_flag = np.all(this.flag_array.flatten()[this_inds])
            other_all_zero = np.all(other.data_array.flatten()[other_inds] == 0)
            other_all_flag = np.all(other.flag_array.flatten()[other_inds])

            if this_all_zero and this_all_flag:
                # we're fine to overwrite; update history accordingly
                history_update_string = " Overwrote invalid data using pyuvdata."
                this.history += history_update_string
            elif other_all_zero and other_all_flag:
                raise ValueError(
                    "To combine these data, please run the add operation again, "
                    "but with the object whose data is to be overwritten as the "
                    "first object in the add operation."
                )
            else:
                raise ValueError(
                    "These objects have overlapping data and cannot be combined."
                )

        # find the blt indices in "other" but not in "this"
        temp = np.nonzero(~np.isin(other_blts, this_blts))[0]
        if len(temp) > 0:
            bnew_inds = temp
            new_blts = other_blts[temp]
            history_update_string += "baseline-time"
            n_axes += 1
        else:
            bnew_inds, new_blts = ([], [])

        # if there's any overlap in blts, check extra params
        temp = np.nonzero(np.isin(other_blts, this_blts))[0]
        if len(temp) > 0:
            # add metadata to be checked to compatibility params
            extra_params = [
                "_integration_time",
                "_lst_array",
                "_phase_center_catalog",
                "_phase_center_id_array",
                "_phase_center_app_ra",
                "_phase_center_app_dec",
                "_phase_center_frame_pa",
                "_Nphase",
                "_uvw_array",
            ]
            compatibility_params.extend(extra_params)

        # find the freq indices in "other" but not in "this"
        if (this.flex_spw_polarization_array is None) != (
            other.flex_spw_polarization_array is None
        ):
            raise ValueError(
                "Cannot add a flex-pol and non-flex-pol UVData objects. Use "
                "the `remove_flex_pol` method to convert the objects to "
                "have a regular polarization axis."
            )
        elif this.flex_spw_polarization_array is not None:
            this_flexpol_dict = dict(
                zip(this.spw_array, this.flex_spw_polarization_array, strict=True)
            )
            other_flexpol_dict = dict(
                zip(other.spw_array, other.flex_spw_polarization_array, strict=True)
            )
            for key in other_flexpol_dict:
                try:
                    if this_flexpol_dict[key] != other_flexpol_dict[key]:
                        raise ValueError(
                            "Cannot add a flex-pol UVData objects where the same "
                            "spectral window contains different polarizations. Use "
                            "the `remove_flex_pol` method to convert the objects "
                            "to have a regular polarization axis."
                        )
                except KeyError:
                    this_flexpol_dict[key] = other_flexpol_dict[key]

        other_mask = np.ones_like(other.flex_spw_id_array, dtype=bool)
        for idx in np.intersect1d(this.spw_array, other.spw_array):
            other_mask[other.flex_spw_id_array == idx] = np.isin(
                other.freq_array[other.flex_spw_id_array == idx],
                this.freq_array[this.flex_spw_id_array == idx],
                invert=True,
            )
        temp = np.where(other_mask)[0]
        if len(temp) > 0:
            fnew_inds = temp
            if n_axes > 0:
                history_update_string += ", frequency"
            else:
                history_update_string += "frequency"
            n_axes += 1
        else:
            fnew_inds = []

        # if channel width is an array and there's any overlap in freqs,
        # check extra params
        temp = np.nonzero(np.isin(other.freq_array, this.freq_array))[0]
        if len(temp) > 0:
            # add metadata to be checked to compatibility params
            extra_params = ["_channel_width"]
            compatibility_params.extend(extra_params)

        # find the pol indices in "other" but not in "this"
        temp = np.nonzero(~np.isin(other.polarization_array, this.polarization_array))[
            0
        ]
        if len(temp) > 0:
            pnew_inds = temp
            if n_axes > 0:
                history_update_string += ", polarization"
            else:
                history_update_string += "polarization"
            n_axes += 1
        else:
            pnew_inds = []

        # Actually check compatibility parameters
        blt_inds_params = [
            "_integration_time",
            "_lst_array",
            "_phase_center_app_ra",
            "_phase_center_app_dec",
            "_phase_center_frame_pa",
            "_phase_center_id_array",
        ]
        for cp in compatibility_params:
            if cp in blt_inds_params:
                # only check that overlapping blt indices match
                this_param = getattr(this, cp)
                other_param = getattr(other, cp)
                params_match = np.allclose(
                    this_param.value[this_blts_ind],
                    other_param.value[other_blts_ind],
                    rtol=this_param.tols[0],
                    atol=this_param.tols[1],
                )
            elif cp == "_uvw_array":
                # only check that overlapping blt indices match
                params_match = np.allclose(
                    this.uvw_array[this_blts_ind, :],
                    other.uvw_array[other_blts_ind, :],
                    rtol=this._uvw_array.tols[0],
                    atol=this._uvw_array.tols[1],
                )
            elif cp == "_channel_width":
                # only check that overlapping freq indices match
                params_match = np.allclose(
                    this.channel_width[this_freq_ind],
                    other.channel_width[other_freq_ind],
                    rtol=this._channel_width.tols[0],
                    atol=this._channel_width.tols[1],
                )
            else:
                params_match = getattr(this, cp) == getattr(other, cp)
            if not params_match:
                msg = (
                    "UVParameter " + cp[1:] + " does not match. Cannot combine objects."
                )
                raise ValueError(msg)

        # Begin manipulating the objects.
        # Note that this will check to see if we can merge the telescopes (if they are
        # different, otherwise the underlying checking is the same).
        this.telescope += other.telescope

        # First, handle the internal source catalogs, since merging them is kind of a
        # weird, one-off process (i.e., nothing is cat'd across a particular axis)
        this._consolidate_phase_center_catalogs(other=other, ignore_name=ignore_name)

        # Next, we want to make sure that the ordering of the _overlapping_ data is
        # the same, so that things can get plugged together in a sensible way.
        if len(this_blts_ind) != 0:
            this_argsort = np.argsort(this_blts_ind)
            other_argsort = np.argsort(other_blts_ind)

            if np.any(this_argsort != other_argsort):
                temp_ind = np.arange(this.Nblts)
                temp_ind[this_blts_ind[this_argsort]] = temp_ind[
                    this_blts_ind[other_argsort]
                ]

                this.reorder_blts(order=temp_ind)

        if len(this_freq_ind) != 0:
            this_argsort = np.argsort(this_freq_ind)
            other_argsort = np.argsort(other_freq_ind)
            if np.any(this_argsort != other_argsort):
                temp_ind = np.arange(this.Nfreqs)
                temp_ind[this_freq_ind[this_argsort]] = temp_ind[
                    this_freq_ind[other_argsort]
                ]

                this.reorder_freqs(channel_order=temp_ind)

        if len(this_pol_ind) != 0:
            this_argsort = np.argsort(this_pol_ind)
            other_argsort = np.argsort(other_pol_ind)
            if np.any(this_argsort != other_argsort):
                temp_ind = np.arange(this.Npols)
                temp_ind[this_pol_ind[this_argsort]] = temp_ind[
                    this_pol_ind[other_argsort]
                ]

                this.reorder_pols(temp_ind)

        # Pad out self to accommodate new data
        blt_order = None
        if len(bnew_inds) > 0:
            this_blts = np.concatenate((this_blts, new_blts))
            blt_order = np.argsort(this_blts)
            if not self.metadata_only:
                zero_pad = np.zeros((len(bnew_inds), this.Nfreqs, this.Npols))
                this.data_array = np.concatenate([this.data_array, zero_pad], axis=0)
                this.nsample_array = np.concatenate(
                    [this.nsample_array, zero_pad], axis=0
                )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=0
                ).astype(np.bool_)
            this.uvw_array = np.concatenate(
                [this.uvw_array, other.uvw_array[bnew_inds, :]], axis=0
            )[blt_order, :]
            this.time_array = np.concatenate(
                [this.time_array, other.time_array[bnew_inds]]
            )[blt_order]
            this.integration_time = np.concatenate(
                [this.integration_time, other.integration_time[bnew_inds]]
            )[blt_order]
            this.lst_array = np.concatenate(
                [this.lst_array, other.lst_array[bnew_inds]]
            )[blt_order]
            this.ant_1_array = np.concatenate(
                [this.ant_1_array, other.ant_1_array[bnew_inds]]
            )[blt_order]
            this.ant_2_array = np.concatenate(
                [this.ant_2_array, other.ant_2_array[bnew_inds]]
            )[blt_order]
            this.baseline_array = np.concatenate(
                [this.baseline_array, other.baseline_array[bnew_inds]]
            )[blt_order]
            this.phase_center_app_ra = np.concatenate(
                [this.phase_center_app_ra, other.phase_center_app_ra[bnew_inds]]
            )[blt_order]
            this.phase_center_app_dec = np.concatenate(
                [this.phase_center_app_dec, other.phase_center_app_dec[bnew_inds]]
            )[blt_order]
            this.phase_center_frame_pa = np.concatenate(
                [this.phase_center_frame_pa, other.phase_center_frame_pa[bnew_inds]]
            )[blt_order]
            this.phase_center_id_array = np.concatenate(
                [this.phase_center_id_array, other.phase_center_id_array[bnew_inds]]
            )[blt_order]

        f_order = None
        if len(fnew_inds) > 0:
            this.freq_array = np.concatenate(
                [this.freq_array, other.freq_array[fnew_inds]]
            )
            this.channel_width = np.concatenate(
                [this.channel_width, other.channel_width[fnew_inds]]
            )

            this.flex_spw_id_array = np.concatenate(
                [this.flex_spw_id_array, other.flex_spw_id_array[fnew_inds]]
            )
            this.spw_array = np.concatenate([this.spw_array, other.spw_array])
            # We want to preserve per-spw information based on first appearance
            # in the concatenated array.
            unique_index = np.sort(
                np.unique(this.flex_spw_id_array, return_index=True)[1]
            )
            this.spw_array = this.flex_spw_id_array[unique_index]
            this.Nspws = len(this.spw_array)

            if this.flex_spw_polarization_array is not None:
                this.flex_spw_polarization_array = np.array(
                    [this_flexpol_dict[key] for key in this.spw_array]
                )
            # Need to sort out the order of the individual windows first.
            f_order = np.concatenate(
                [
                    np.where(this.flex_spw_id_array == idx)[0]
                    for idx in sorted(this.spw_array)
                ]
            )

            # With spectral windows sorted, check and see if channels within
            # windows need sorting. If they are ordered in ascending or descending
            # fashion, leave them be. If not, sort in ascending order
            for idx in this.spw_array:
                select_mask = this.flex_spw_id_array[f_order] == idx
                check_freqs = this.freq_array[f_order[select_mask]]
                if (not np.all(check_freqs[1:] > check_freqs[:-1])) and (
                    not np.all(check_freqs[1:] < check_freqs[:-1])
                ):
                    subsort_order = f_order[select_mask]
                    f_order[select_mask] = subsort_order[np.argsort(check_freqs)]

            if not self.metadata_only:
                zero_pad = np.zeros(
                    (this.data_array.shape[0], len(fnew_inds), this.Npols)
                )
                this.data_array = np.concatenate([this.data_array, zero_pad], axis=1)
                this.nsample_array = np.concatenate(
                    [this.nsample_array, zero_pad], axis=1
                )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=1
                ).astype(np.bool_)

        p_order = None
        if len(pnew_inds) > 0:
            this.polarization_array = np.concatenate(
                [this.polarization_array, other.polarization_array[pnew_inds]]
            )
            p_order = np.argsort(np.abs(this.polarization_array))
            if not self.metadata_only:
                zero_pad = np.zeros(
                    (this.data_array.shape[0], this.data_array.shape[1], len(pnew_inds))
                )
                this.data_array = np.concatenate([this.data_array, zero_pad], axis=2)
                this.nsample_array = np.concatenate(
                    [this.nsample_array, zero_pad], axis=2
                )
                this.flag_array = np.concatenate(
                    [this.flag_array, 1 - zero_pad], axis=2
                ).astype(np.bool_)

        # Now populate the data
        pol_t2o = np.nonzero(
            np.isin(this.polarization_array, other.polarization_array)
        )[0]
        this_freqs = this.freq_array
        other_freqs = other.freq_array

        freq_t2o = np.zeros(this_freqs.shape, dtype=bool)
        for spw_id in set(this.spw_array).intersection(other.spw_array):
            mask = this.flex_spw_id_array == spw_id
            freq_t2o[mask] |= np.isin(
                this_freqs[mask], other_freqs[other.flex_spw_id_array == spw_id]
            )
        freq_t2o = np.nonzero(freq_t2o)[0]
        blt_t2o = np.nonzero(np.isin(this_blts, other_blts))[0]

        if not self.metadata_only:
            this.data_array[np.ix_(blt_t2o, freq_t2o, pol_t2o)] = other.data_array
            this.nsample_array[np.ix_(blt_t2o, freq_t2o, pol_t2o)] = other.nsample_array
            this.flag_array[np.ix_(blt_t2o, freq_t2o, pol_t2o)] = other.flag_array

            # Fix ordering
            axis_dict = {
                0: {"inds": bnew_inds, "order": blt_order},
                1: {"inds": fnew_inds, "order": f_order},
                2: {"inds": pnew_inds, "order": p_order},
            }
            for axis, subdict in axis_dict.items():
                for name, param in zip(
                    this._data_params, this.data_like_parameters, strict=True
                ):
                    if len(subdict["inds"]) > 0:
                        unique_order_diffs = np.unique(np.diff(subdict["order"]))
                        if np.array_equal(unique_order_diffs, np.array([1])):
                            # everything is already in order
                            continue
                        setattr(this, name, np.take(param, subdict["order"], axis=axis))

        if len(fnew_inds) > 0:
            this.freq_array = this.freq_array[f_order]
            this.channel_width = this.channel_width[f_order]
            this.flex_spw_id_array = this.flex_spw_id_array[f_order]

        if len(pnew_inds) > 0:
            this.polarization_array = this.polarization_array[p_order]

        # Update N parameters (e.g. Npols)
        this.Ntimes = len(np.unique(this.time_array))
        this.Nbls = len(np.unique(this.baseline_array))
        this.Nblts = this.uvw_array.shape[0]
        this.Nfreqs = this.freq_array.size
        this.Npols = this.polarization_array.shape[0]
        this.Nants_data = this._calc_nants_data()

        # Update filename parameter
        this.filename = utils.tools._combine_filenames(this.filename, other.filename)
        if this.filename is not None:
            this._filename.form = (len(this.filename),)

        if n_axes > 0:
            history_update_string += " axis using pyuvdata."

            histories_match = utils.history._check_histories(
                this.history, other.history
            )

            this.history += history_update_string
            if not histories_match:
                if verbose_history:
                    this.history += " Next object history follows. " + other.history
                else:
                    extra_history = utils.history._combine_history_addition(
                        this.history, other.history
                    )
                    if extra_history is not None:
                        this.history += (
                            " Unique part of next object history follows. "
                            + extra_history
                        )

        # Reset blt_order if blt axis was added to and it is set
        if len(blt_t2o) > 0:
            this.blt_order = None

        this.set_rectangularity(force=True)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                check_freq_spacing=warn_spacing,
                check_pol_spacing=warn_spacing,
                raise_spacing_errors=False,
            )

        if not inplace:
            return this

    def __iadd__(
        self,
        other,
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        ignore_name=False,
        warn_spacing=False,
    ):
        """
        In place add.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (if many objects are combined in succession this can
            lead to very long histories). If False and if the histories for the two
            objects are different, the combined object will have the history of the
            first object and only the parts of the second object history that are unique
            (this is done word by word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when combining two UVData objects. If set to True,
            phase centers that are the same up to their name will be combined with the
            name set to the name found in the first UVData object in the sum. If set to
            False, phase centers that are the same up to the name will be kept as
            separate phase centers. Default is False.
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            uvfits or mirad file formats. Default is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, self and other are not compatible
            or if data in self and other overlap.
            If `phase_center_radec` is not None and is not length 2.

        """
        self.__add__(
            other,
            inplace=True,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
            ignore_name=ignore_name,
            warn_spacing=warn_spacing,
        )
        return self

    def fast_concat(
        self,
        other,
        axis,
        *,
        inplace=False,
        verbose_history=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        ignore_name=None,
        warn_spacing=False,
    ):
        """
        Concatenate two UVData objects along specified axis with almost no checking.

        Warning! This method assumes all the metadata along other axes is sorted
        the same way. The __add__ method is much safer, it checks all the metadata,
        but it is slower. Some quick checks are run, but this method doesn't
        make any guarantees that the resulting object is correct.

        Note that if objects have different phasing (different phase_center_catalogs)
        they can still be combined and the phasing information will be preserved.
        However, the phase center ID numbers may be changed on any of the objects passed
        into this method.

        Parameters
        ----------
        other : UVData object or list of UVData objects
            UVData object or list of UVData objects which will be added to self.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. Allowed values are: 'blt', 'freq', 'polarization'.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            objects are different, the combined object will keep all the history of
            all input objects (if many objects are combined this can lead to very long
            histories). If False and if the histories for the objects are different,
            the combined object will have the history of the first object and only the
            parts of the other object histories that are unique (this is done word by
            word and can result in hard to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        ignore_name : bool
            Option to ignore the name of the phase center (`cat_name` in
            `phase_center_catalog`) when combining two UVData objects. If set to True,
            phase centers that are the same up to their name will be combined with the
            name set to the name found in the first UVData object in the sum. If set to
            False, phase centers that are the same up to the name will be kept as
            separate phase centers. Default is False.
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            uvfits or miriad formats. Default is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, axis is not an allowed value or if
            self and other are not compatible.

        """
        allowed_axes = ["blt", "freq", "polarization"]
        if axis not in allowed_axes:
            raise ValueError("Axis must be one of: " + ", ".join(allowed_axes))

        if inplace:
            this = self
        else:
            this = self.copy()
        if not isinstance(other, list | tuple | np.ndarray):
            # if this is a UVData object already, stick it in a list
            other = [other]

        # Check that both objects are UVData and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        for obj in other:
            if not issubclass(obj.__class__, this.__class__) and not issubclass(
                this.__class__, obj.__class__
            ):
                raise ValueError(
                    "Only UVData (or subclass) objects can be "
                    "added to a UVData (or subclass) object"
                )
            obj.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        # update the phase_center_catalog to make them consistent across objects
        # Doing this as a binary tree merge
        # The left object in each loop will have its phase center IDs updated.
        uv_list = [this] + other
        while len(uv_list) > 1:
            # for an odd number of files, the second argument will be shorter
            # so the last element in the first list won't be combined, but it
            # will not be lost, so it's ok.
            for uv1, uv2 in zip(uv_list[0::2], uv_list[1::2], strict=False):
                uv1._consolidate_phase_center_catalogs(
                    other=uv2, ignore_name=ignore_name
                )
            uv_list = uv_list[0::2]
        # Because self was at the beginning of the list,
        # all the phase centers are merged into it at the end of this loop

        compatibility_params = ["_vis_units"]

        history_update_string = " Combined data along "

        if axis == "freq":
            history_update_string += "frequency"
            compatibility_params += [
                "_polarization_array",
                "_ant_1_array",
                "_ant_2_array",
                "_integration_time",
                "_uvw_array",
                "_lst_array",
                "_phase_center_id_array",
            ]
        elif axis == "polarization":
            history_update_string += "polarization"
            compatibility_params += [
                "_freq_array",
                "_channel_width",
                "_flex_spw_id_array",
                "_ant_1_array",
                "_ant_2_array",
                "_integration_time",
                "_uvw_array",
                "_lst_array",
                "_phase_center_id_array",
            ]
        elif axis == "blt":
            history_update_string += "baseline-time"
            compatibility_params += [
                "_freq_array",
                "_polarization_array",
                "_flex_spw_id_array",
            ]

        history_update_string += " axis using pyuvdata."

        histories_match = []
        for obj in other:
            histories_match.append(
                utils.history._check_histories(this.history, obj.history)
            )

        this.history += history_update_string
        for obj_num, obj in enumerate(other):
            if not histories_match[obj_num]:
                if verbose_history:
                    this.history += " Next object history follows. " + obj.history
                else:
                    extra_history = utils.history._combine_history_addition(
                        this.history, obj.history
                    )
                    if extra_history is not None:
                        this.history += (
                            " Unique part of next object history follows. "
                            + extra_history
                        )

        # Actually check compatibility parameters
        tel_obj = this.telescope.copy() if inplace else this.telescope
        for obj in other:
            tel_obj += obj.telescope
            for a in compatibility_params:
                params_match = getattr(this, a) == getattr(obj, a)
                if not params_match:
                    msg = (
                        "UVParameter "
                        + a[1:]
                        + " does not match. Cannot combine objects."
                    )
                    raise ValueError(msg)

        this.telescope = tel_obj

        if axis == "freq":
            this.Nfreqs = sum([this.Nfreqs] + [obj.Nfreqs for obj in other])
            this.freq_array = np.concatenate(
                [this.freq_array] + [obj.freq_array for obj in other]
            )
            this.channel_width = np.concatenate(
                [this.channel_width] + [obj.channel_width for obj in other]
            )
            this.flex_spw_id_array = np.concatenate(
                [this.flex_spw_id_array] + [obj.flex_spw_id_array for obj in other]
            )
            this.spw_array = np.concatenate(
                [this.spw_array] + [obj.spw_array for obj in other]
            )
            # We want to preserve per-spw information based on first appearance
            # in the concatenated array.
            unique_index = np.sort(
                np.unique(this.flex_spw_id_array, return_index=True)[1]
            )
            this.spw_array = this.flex_spw_id_array[unique_index]

            this.Nspws = len(this.spw_array)

            if not self.metadata_only:
                this.data_array = np.concatenate(
                    [this.data_array] + [obj.data_array for obj in other], axis=1
                )
                this.nsample_array = np.concatenate(
                    [this.nsample_array] + [obj.nsample_array for obj in other], axis=1
                )
                this.flag_array = np.concatenate(
                    [this.flag_array] + [obj.flag_array for obj in other], axis=1
                )
        elif axis == "polarization":
            this.polarization_array = np.concatenate(
                [this.polarization_array] + [obj.polarization_array for obj in other]
            )
            this.Npols = sum([this.Npols] + [obj.Npols for obj in other])

            if not self.metadata_only:
                this.data_array = np.concatenate(
                    [this.data_array] + [obj.data_array for obj in other], axis=2
                )
                this.nsample_array = np.concatenate(
                    [this.nsample_array] + [obj.nsample_array for obj in other], axis=2
                )
                this.flag_array = np.concatenate(
                    [this.flag_array] + [obj.flag_array for obj in other], axis=2
                )
        elif axis == "blt":
            this.Nblts = sum([this.Nblts] + [obj.Nblts for obj in other])
            this.ant_1_array = np.concatenate(
                [this.ant_1_array] + [obj.ant_1_array for obj in other]
            )
            this.ant_2_array = np.concatenate(
                [this.ant_2_array] + [obj.ant_2_array for obj in other]
            )
            this.Nants_data = this._calc_nants_data()
            this.uvw_array = np.concatenate(
                [this.uvw_array] + [obj.uvw_array for obj in other], axis=0
            )
            this.time_array = np.concatenate(
                [this.time_array] + [obj.time_array for obj in other]
            )
            this.Ntimes = len(np.unique(this.time_array))
            this.lst_array = np.concatenate(
                [this.lst_array] + [obj.lst_array for obj in other]
            )
            this.baseline_array = np.concatenate(
                [this.baseline_array] + [obj.baseline_array for obj in other]
            )
            this.Nbls = len(np.unique(this.baseline_array))
            this.integration_time = np.concatenate(
                [this.integration_time] + [obj.integration_time for obj in other]
            )
            this.phase_center_app_ra = np.concatenate(
                [this.phase_center_app_ra] + [obj.phase_center_app_ra for obj in other]
            )
            this.phase_center_app_dec = np.concatenate(
                [this.phase_center_app_dec]
                + [obj.phase_center_app_dec for obj in other]
            )
            this.phase_center_frame_pa = np.concatenate(
                [this.phase_center_frame_pa]
                + [obj.phase_center_frame_pa for obj in other]
            )
            this.phase_center_id_array = np.concatenate(
                [this.phase_center_id_array]
                + [obj.phase_center_id_array for obj in other]
            )
            if not self.metadata_only:
                this.data_array = np.concatenate(
                    [this.data_array] + [obj.data_array for obj in other], axis=0
                )
                this.nsample_array = np.concatenate(
                    [this.nsample_array] + [obj.nsample_array for obj in other], axis=0
                )
                this.flag_array = np.concatenate(
                    [this.flag_array] + [obj.flag_array for obj in other], axis=0
                )

        # update filename attribute
        for obj in other:
            this.filename = utils.tools._combine_filenames(this.filename, obj.filename)
        if this.filename is not None:
            this._filename.form = len(this.filename)

        this.set_rectangularity(force=True)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                check_freq_spacing=warn_spacing,
                check_pol_spacing=warn_spacing,
                raise_spacing_errors=False,
            )

        if not inplace:
            return this

    def sum_vis(
        self,
        other,
        *,
        inplace=False,
        difference=False,
        verbose_history=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        override_params=None,
    ):
        """
        Sum visibilities between two UVData objects.

        By default requires that all UVParameters are the same on the two objects
        except for `history`, `data_array`, `extra_keywords` and `filename`.
        If keys in `extra_keywords` have different values the values from the first
        object are taken.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        difference : bool
            If True, differences the visibilities of the two UVData objects
            rather than summing them.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        verbose_history : bool
            Option to allow more verbose history. If True and if the histories for the
            two objects are different, the combined object will keep all the history of
            both input objects (this can lead to long histories). If False and if the
            histories for the two objects are different, the combined object will have
            the history of the first object and only the parts of the second object
            history that are unique (this is done word by word and can result in hard
            to interpret histories).
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        override_params : array_like of strings
            List of object UVParameters to omit from compatibility check. Overridden
            parameters will not be compared between the objects, and the values
            for these parameters will be taken from the first object.

        Returns
        -------
        UVData Object
            If inplace parameter is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, or if self and other
            are not compatible.

        """
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that both objects are UVData and valid
        this.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )
        if not issubclass(other.__class__, this.__class__) and not issubclass(
            this.__class__, other.__class__
        ):
            raise ValueError(
                "Only UVData (or subclass) objects can be "
                "added to a UVData (or subclass) object"
            )
        other.check(
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
        )

        compatibility_params = list(this.__iter__())
        remove_params = ["_history", "_data_array", "_extra_keywords", "_filename"]

        # Add underscores to override_params to match list from __iter__()
        # Add to parameters to be removed
        if override_params and all(isinstance(param, str) for param in override_params):
            for param in override_params:
                if param[0] != "_":
                    param = "_" + param
                if param not in compatibility_params:
                    msg = (
                        "Provided parameter "
                        + param[1:]
                        + " is not a recognizable UVParameter."
                    )
                    raise ValueError(msg)
                remove_params.append(param)

        # compatibility_params should define the parameters that need to
        # be the same for objects to be summed or diffed
        compatibility_params = list(set(compatibility_params) - set(remove_params))

        # Check each UVParameter in compatibility_params
        for param in compatibility_params:
            params_match = getattr(this, param) == getattr(other, param)
            if not params_match:
                msg = (
                    "UVParameter "
                    + param[1:]
                    + " does not match. Cannot combine objects."
                )
                raise ValueError(msg)

        # Merge extra keywords
        for intersection in set(this.extra_keywords.keys()) & set(
            other.extra_keywords.keys()
        ):
            if this.extra_keywords[intersection] != other.extra_keywords[intersection]:
                warnings.warn(
                    "Keyword " + intersection + " in _extra_keywords is different "
                    "in the two objects. Taking the first object's entry."
                )

        # Merge extra_keywords lists, taking values from the first object
        this.extra_keywords = dict(
            list(other.extra_keywords.items()) + list(this.extra_keywords.items())
        )

        # Do the summing / differencing
        if difference:
            this.data_array = this.data_array - other.data_array
            history_update_string = " Visibilities differenced using pyuvdata."
        else:
            this.data_array = this.data_array + other.data_array
            history_update_string = " Visibilities summed using pyuvdata."

        histories_match = utils.history._check_histories(this.history, other.history)

        this.history += history_update_string
        if not histories_match:
            if verbose_history:
                this.history += " Second object history follows. " + other.history
            else:
                extra_history = utils.history._combine_history_addition(
                    this.history, other.history
                )
                if extra_history is not None:
                    this.history += (
                        " Unique part of second object history follows. "
                        + extra_history
                    )

        # merge file names
        this.filename = utils.tools._combine_filenames(this.filename, other.filename)
        this._filename.form = (len(this.filename),)

        # Check final object is self-consistent
        if run_check:
            this.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if not inplace:
            return this

    def diff_vis(
        self,
        other,
        *,
        inplace=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        override_params=None,
    ):
        """
        Difference visibilities between two UVData objects.

        By default requires that all UVParameters are the same on the two objects
        except for `history`, `data_array`, and `extra_keywords`.
        If keys in `extra_keywords` have different values the values from the first
        object are taken.

        Parameters
        ----------
        other : UVData object
            Another UVData object which will be added to self.
        inplace : bool
            If True, overwrite self as we go, otherwise create a third object
            as the sum of the two.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after combining objects.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            combining objects.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        override_params : array_like of strings
            List of object UVParameters to omit from compatibility check. Overridden
            parameters will not be compared between the objects, and the values
            for these parameters will be taken from the first object.

        Returns
        -------
        UVData Object
            If inplace parameter is False.

        Raises
        ------
        ValueError
            If other is not a UVData object, or if self and other
            are not compatible.

        """
        if inplace:
            self.sum_vis(
                other,
                difference=True,
                inplace=inplace,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                override_params=override_params,
            )
        else:
            return self.sum_vis(
                other,
                difference=True,
                inplace=inplace,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                override_params=override_params,
            )

    def parse_ants(self, ant_str, *, print_toggle=False):
        """
        Get antpair and polarization from parsing an aipy-style ant string.

        Used to support the select function. Generates two lists of antenna pair
        tuples and polarization indices based on parsing of the string ant_str.
        If no valid polarizations (pseudo-Stokes params, or combinations of [lr]
        or [xy]) or antenna numbers are found in ant_str, ant_pairs_nums and
        polarizations are returned as None.

        Parameters
        ----------
        ant_str : str
            String containing antenna information to parse. Can be 'all',
            'auto', 'cross', or combinations of antenna numbers and polarization
            indicators 'l' and 'r' or 'x' and 'y'.  Minus signs can also be used
            in front of an antenna number or baseline to exclude it from being
            output in ant_pairs_nums. If ant_str has a minus sign as the first
            character, 'all,' will be added to the beginning of the string.
            See the tutorial for examples of valid strings and their behavior.
        print_toggle : bool
            Boolean for printing parsed baselines for a visual user check.

        Returns
        -------
        ant_pairs_nums : list of tuples of int or None
            List of tuples containing the parsed pairs of antenna numbers, or
            None if ant_str is 'all' or a pseudo-Stokes polarizations.
        polarizations : list of int or None
            List of desired polarizations or None if ant_str does not contain a
            polarization specification.

        """
        return utils.bls.parse_ants(
            uv=self,
            ant_str=ant_str,
            print_toggle=print_toggle,
            x_orientation=self.telescope.get_x_orientation_from_feeds(),
        )

    def _select_preprocess(
        self,
        *,
        antenna_nums,
        antenna_names,
        ant_str,
        bls,
        frequencies,
        freq_chans,
        spws,
        times,
        time_range,
        lsts,
        lst_range,
        polarizations,
        blt_inds,
        phase_center_ids,
        catalog_names,
        invert=False,
        strict=False,
        warn_spacing=False,
    ):
        """
        Build up blt_inds, freq_inds, pol_inds and history_update_string for select.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
        bls : list of tuple or list of int, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]), a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]), or a list of
            baseline numbers (e.g. [67599, 71699, 73743]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the
            numbers within the tuple does not matter. For length-3 tuples, the
            polarization string is in the order of the two antennas. If
            length-3 tuples are provided, `polarizations` must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
            or combinations of antenna numbers and polarizations (e.g. '1',
            '1_2', '1x_2y').  See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        spws : array_like of int, optional
            The spectral window numbers to keep in the object.
        times : array_like of float, optional
            The times to keep in the object, each value passed here should exist
            in the time_array. Cannot be used with `time_range`, `lsts`, or
            `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be length
            2. Some of the times in the object should fall between the first and
            last elements. Cannot be used with `times`, `lsts`, or `lst_array`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int or str, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array. If passing strings, the
            canonical polarization strings (e.g. "xx", "rr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "nn", "ee") are also supported.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        phase_center_ids : array_like of int, optional
            Phase center IDs to keep on the object (effectively a selection on
            baseline-times). Cannot be used with `catalog_names`.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to keep in the object, which should
            match exactly in spelling and capitalization. Cannot be used with
            `phase_center_ids`.
        invert : bool
            Normally records matching given criteria are what are included in the
            subsequent object. However, if set to True, these records are excluded
            instead. Default is False.
        strict : bool or None
            Normally, select will warn when no records match a one element of a
            parameter, as long as *at least one* element matches with what is in the
            object. However, if set to True, an error is thrown if any element
            does not match. If set to None, then neither errors nor warnings are raised.
            Default is False.
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            uvfits or miriad file-format. Default is False.

        Returns
        -------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.

        """
        selections = []
        if ant_str is not None:
            if not (
                antenna_nums is None
                and antenna_names is None
                and bls is None
                and polarizations is None
            ):
                raise ValueError(
                    "Cannot provide ant_str with antenna_nums, antenna_names, "
                    "bls, or polarizations."
                )
            else:
                bls, polarizations = self.parse_ants(ant_str)
                if bls is not None and len(bls) == 0:
                    raise ValueError(
                        f"There is no data matching ant_str={ant_str} in this object."
                    )
                if invert and polarizations is not None:
                    raise ValueError(
                        "Cannot set invert=True if using ant_str with polarizations."
                    )

        if (phase_center_ids is not None) and (catalog_names is not None):
            raise ValueError("Cannot set both phase_center_ids and catalog_names.")

        if catalog_names is not None:
            phase_center_ids = utils.phase_center_catalog.look_for_name(
                self.phase_center_catalog, catalog_names
            )
            selections.append("catalog names")
        elif phase_center_ids is not None:
            selections.append("phase center IDs")

        if bls is not None:
            bls, polarizations = utils.bls._extract_bls_pol(
                bls=bls,
                polarizations=polarizations,
                baseline_array=self.baseline_array,
                ant_1_array=self.ant_1_array,
                ant_2_array=self.ant_2_array,
                nants_telescope=self.telescope.Nants,
                strict=strict,
                invert=invert,
            )
        blt_inds, blt_selections = utils.bltaxis._select_blt_preprocess(
            select_antenna_nums=antenna_nums,
            select_antenna_names=antenna_names,
            bls=bls,
            times=times,
            time_range=time_range,
            lsts=lsts,
            lst_range=lst_range,
            blt_inds=blt_inds,
            phase_center_ids=phase_center_ids,
            antenna_names=self.telescope.antenna_names,
            antenna_numbers=self.telescope.antenna_numbers,
            ant_1_array=self.ant_1_array,
            ant_2_array=self.ant_2_array,
            baseline_array=self.baseline_array,
            time_array=self.time_array,
            time_tols=self._time_array.tols,
            lst_array=self.lst_array,
            lst_tols=self._lst_array.tols,
            phase_center_id_array=self.phase_center_id_array,
            invert=invert,
            strict=strict,
        )
        selections.extend(blt_selections)

        freq_inds, spw_inds, freq_selections = utils.frequency._select_freq_helper(
            frequencies=frequencies,
            freq_chans=freq_chans,
            obj_freq_array=self.freq_array,
            freq_tols=self._freq_array.tols,
            obj_channel_width=self.channel_width,
            channel_width_tols=self._channel_width.tols,
            spws=spws,
            obj_spw_array=self.spw_array,
            obj_spw_id_array=self.flex_spw_id_array,
            obj_flex_spw_pol_array=self.flex_spw_polarization_array,
            polarizations=polarizations,
            obj_x_orientation=self.telescope.get_x_orientation_from_feeds(),
            invert=invert,
            strict=strict,
            warn_spacing=warn_spacing,
        )
        selections.extend(freq_selections)

        pol_inds, pol_selections = utils.pol._select_pol_helper(
            polarizations=polarizations,
            obj_pol_array=self.polarization_array,
            obj_x_orientation=self.telescope.get_x_orientation_from_feeds(),
            flex_pol=self.flex_spw_polarization_array is not None,
            invert=invert,
            strict=strict,
            warn_spacing=warn_spacing,
        )
        selections.extend(pol_selections)

        # build up history string from selections
        history_update_string = ""
        if len(selections) > 0:
            history_update_string = (
                "  Downselected to specific "
                + ", ".join(selections)
                + " using pyuvdata."
            )

        return blt_inds, freq_inds, spw_inds, pol_inds, history_update_string

    def _select_by_index(
        self,
        *,
        blt_inds,
        freq_inds,
        spw_inds,
        pol_inds,
        history_update_string,
        keep_all_metadata=True,
    ):
        """
        Perform select based on indexing arrays.

        Parameters
        ----------
        blt_inds : list of int
            list of baseline-time indices to keep. Can be None (to keep everything).
        freq_inds : list of int
            list of frequency indices to keep. Can be None (to keep everything).
        spw_inds : list of int
            list of spw indices to keep. Can be None (to keep everything).
        pol_inds : list of int
            list of polarization indices to keep. Can be None (to keep everything).
        history_update_string : str
            string to append to the end of the history.
        keep_all_metadata : bool
            Option to keep metadata for antennas that are no longer in the dataset.
        """
        # Create a dictionary to pass to _select_along_param_axis
        ind_dict = {
            "Nblts": blt_inds,
            "Nfreqs": freq_inds,
            "Nspws": spw_inds,
            "Npols": pol_inds,
        }

        self._select_along_param_axis(ind_dict)

        if blt_inds is not None:
            # Process post blt-specific selection actions, including counting
            # unique times antennas/baselines in the data.
            self.Nbls = len(np.unique(self.baseline_array))
            self.Ntimes = len(np.unique(self.time_array))
            self.Nants_data = self._calc_nants_data()

            if not keep_all_metadata:
                # If we are dropping metadata and selecting on blts, then add
                # evaluate the antenna axis of all parameters
                use_ants = list(set(self.ant_1_array).union(self.ant_2_array))
                ind_arr = np.nonzero(np.isin(self.telescope.antenna_numbers, use_ants))[
                    0
                ].tolist()
                self.telescope._select_along_param_axis({"Nants": ind_arr})

        # Update the history string
        self.history += history_update_string

    def select(
        self,
        *,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        blt_inds=None,
        phase_center_ids=None,
        catalog_names=None,
        invert=False,
        strict=False,
        inplace=True,
        keep_all_metadata=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        warn_spacing=False,
    ):
        """
        Downselect data to keep on the object along various axes.

        Axes that can be selected along include antenna names or numbers,
        antenna pairs, frequencies, times and polarizations. Specific
        baseline-time indices can also be selected, but this is not commonly
        used.
        The history attribute on the object will be updated to identify the
        operations performed.

        Parameters
        ----------
        antenna_nums : array_like of int, optional
            The antennas numbers to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to keep in the object (antenna positions and
            names for the removed antennas will be retained unless
            `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to keep in the object. For length-2 tuples, the ordering of the numbers
            within the tuple does not matter. For length-3 tuples, the polarization
            string is in the order of the two antennas. If length-3 tuples are
            provided, `polarizations` must be None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to keep in the object.  Can be 'auto', 'cross', 'all',
            or combinations of antenna numbers and polarizations (e.g. '1',
            '1_2', '1x_2y').  See tutorial for more examples of valid strings and
            the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to keep in the object, each value passed here should
            exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to keep in the object.
        spws : array_like of int, optional
            The spectral window numbers to keep in the object.
        times : array_like of float, optional
            The times to keep in the object, each value passed here should
            exist in the time_array. Cannot be used with `time_range`, `lsts`, or
            `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int or str, optional
            The polarizations numbers to keep in the object, each value passed
            here should exist in the polarization_array. If passing strings, the
            canonical polarization strings (e.g. "xx", "rr") are supported and if the
            `x_orientation` attribute is set, the physical dipole strings
            (e.g. "nn", "ee") are also supported.
        blt_inds : array_like of int, optional
            The baseline-time indices to keep in the object. This is
            not commonly used.
        phase_center_ids : array_like of int, optional
            Phase center IDs to keep on the object (effectively a selection on
            baseline-times). Cannot be used with `catalog_names`.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to keep in the object, which should
            match exactly in spelling and capitalization. Cannot be used with
            `phase_center_ids`.
        invert : bool
            Normally records matching given criteria are what are included in the
            subsequent object. However, if set to True, these records are excluded
            instead. Default is False.
        strict : bool or None
            Normally, select will warn when no records match a one element of a
            parameter, as long as *at least one* element matches with what is in the
            object. However, if set to True, an error is thrown if any element
            does not match. If set to None, then neither errors nor warnings are raised.
            Default is False.
        inplace : bool
            Option to perform the select directly on self or return a new UVData
            object with just the selected data (the default is True, meaning the
            select will be done on self).
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do do not have data associated with them after the select option.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after downselecting data on this object (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            downselecting data on this object (the default is True, meaning the
            acceptable range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        warn_spacing : bool
            Option to raise warnings about spacing that would prevent writing to
            uvfits and miriad file-format. Default is False.

        Returns
        -------
        UVData object or None
            None is returned if inplace is True, otherwise a new UVData object
            with just the selected data is returned

        Raises
        ------
        ValueError
            If any of the parameters are set to inappropriate values.

        """
        if inplace:
            uv_obj = self
        else:
            uv_obj = self.copy()

        # Figure out which index positions we want to hold on to.
        blt_inds, freq_inds, spw_inds, pol_inds, history_update_string = (
            uv_obj._select_preprocess(
                antenna_nums=antenna_nums,
                antenna_names=antenna_names,
                ant_str=ant_str,
                bls=bls,
                frequencies=frequencies,
                freq_chans=freq_chans,
                spws=spws,
                times=times,
                time_range=time_range,
                lsts=lsts,
                lst_range=lst_range,
                polarizations=polarizations,
                blt_inds=blt_inds,
                phase_center_ids=phase_center_ids,
                catalog_names=catalog_names,
                invert=invert,
                strict=strict,
                warn_spacing=warn_spacing,
            )
        )

        # Call the low-level selection method.
        uv_obj._select_by_index(
            blt_inds=blt_inds,
            freq_inds=freq_inds,
            spw_inds=spw_inds,
            pol_inds=pol_inds,
            history_update_string=history_update_string,
            keep_all_metadata=keep_all_metadata,
        )

        # Update the rectangularity attributes
        if blt_inds is not None:
            uv_obj.set_rectangularity(force=True)

        # If we have a flex-pol data set, but we only have one pol, then this doesn't
        # need to be flex-pol anymore, and we can drop it here
        if (
            uv_obj.flex_spw_polarization_array is not None
            and len(np.unique(uv_obj.flex_spw_polarization_array)) == 1
        ):
            uv_obj.remove_flex_pol()

        # check if object is uv_object-consistent
        if run_check:
            uv_obj.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
            )

        if not inplace:
            return uv_obj

    def _harmonize_resample_arrays(
        self,
        *,
        inds_to_keep,
        temp_baseline,
        temp_id_array,
        temp_time,
        temp_int_time,
        temp_data,
        temp_flag,
        temp_nsample,
        astrometry_library=None,
    ):
        """
        Make a self-consistent object after up/downsampling.

        This function is called by both upsample_in_time and downsample_in_time.
        See those functions for more information about arguments.
        """
        self.baseline_array = self.baseline_array[inds_to_keep]
        self.time_array = self.time_array[inds_to_keep]
        self.integration_time = self.integration_time[inds_to_keep]
        self.phase_center_id_array = self.phase_center_id_array[inds_to_keep]

        self.baseline_array = np.concatenate((self.baseline_array, temp_baseline))
        self.time_array = np.concatenate((self.time_array, temp_time))
        self.integration_time = np.concatenate((self.integration_time, temp_int_time))
        self.phase_center_id_array = np.concatenate(
            (self.phase_center_id_array, temp_id_array)
        )
        if not self.metadata_only:
            self.data_array = self.data_array[inds_to_keep]
            self.flag_array = self.flag_array[inds_to_keep]
            self.nsample_array = self.nsample_array[inds_to_keep]

            # concatenate temp array with existing arrays
            self.data_array = np.concatenate((self.data_array, temp_data), axis=0)
            self.flag_array = np.concatenate((self.flag_array, temp_flag), axis=0)
            self.nsample_array = np.concatenate(
                (self.nsample_array, temp_nsample), axis=0
            )

        # set antenna arrays from baseline_array
        self.ant_1_array, self.ant_2_array = self.baseline_to_antnums(
            self.baseline_array
        )

        # update metadata
        self.Nblts = self.baseline_array.shape[0]
        self.Ntimes = np.unique(self.time_array).size
        self.uvw_array = np.zeros((self.Nblts, 3))
        self.set_rectangularity(force=True)

        # set lst array
        self.set_lsts_from_time_array(astrometry_library=astrometry_library)

        # update app source coords to new times
        self._set_app_coords_helper()

        # temporarily store the metadata only to calculate UVWs correctly
        uv_temp = self.copy(metadata_only=True)

        # properly calculate the UVWs self-consistently
        uv_temp.set_uvws_from_antenna_positions()
        self.uvw_array = uv_temp.uvw_array

        return

    def upsample_in_time(
        self,
        max_int_time,
        *,
        blt_order="time",
        minor_order="baseline",
        summing_correlator_mode=False,
        allow_drift=False,
        astrometry_library=None,
    ):
        """
        Resample to a shorter integration time.

        This method will resample a UVData object such that all data samples have
        an integration time less than or equal to the `max_int_time`. The new
        samples are copied from the original samples (not interpolated).

        Parameters
        ----------
        max_int_time : float
            Maximum integration time to upsample to in seconds.
        blt_order : str
            Major baseline ordering for output object. Default is "time". See
            the documentation on the `reorder_blts` method for more info.
        minor_order : str
            Minor baseline ordering for output object. Default is "baseline".
        summing_correlator_mode : bool
            Option to split the flux from the original samples into the new
            samples rather than duplicating the original samples in all the new
            samples (undoing an integration rather than an average) to emulate
            undoing the behavior in some correlators (e.g. HERA).
        allow_drift : bool
            Option to allow resampling of unprojected or driftscan data. If this is
            False, unprojected or driftscan data will be phased to the ra/dec of zenith
            before resampling and then unprojected or rephased to a driftscan after
            resampling. Note that resampling unprojected or driftscan phased data may
            result in unexpected behavior.
        astrometry_library : str
            Library to use for calculating the LSTs after upsampling. Allowed options
            are 'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas
            library), and 'astropy' (which uses the astropy utilities). Default is erfa
            unless the telescope_location is a MoonLocation object, in which case the
            default is astropy.

        Returns
        -------
        None

        """
        # check that max_int_time is sensible given integration_time
        min_integration_time = np.amin(self.integration_time)
        sensible_min = 1e-2 * min_integration_time
        if max_int_time < sensible_min:
            raise ValueError(
                "Decreasing the integration time by more than a "
                "factor of 100 is not supported. Also note that "
                "max_int_time should be in seconds."
            )

        # figure out where integration_time is longer than max_int_time
        inds_to_upsample = np.nonzero(
            (self.integration_time > max_int_time)
            & (
                ~np.isclose(
                    self.integration_time,
                    max_int_time,
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                )
            )
        )
        if len(inds_to_upsample[0]) == 0:
            warnings.warn(
                "All values in the integration_time array are already "
                "longer than the value specified; doing nothing."
            )
            return

        unprojected_blts = self._check_for_cat_type("unprojected")
        driftscan_blts = self._check_for_cat_type("driftscan")
        initial_driftscan = np.any(driftscan_blts)
        initial_unprojected = np.any(unprojected_blts)
        initial_nphase_ids = np.unique(self.phase_center_id_array).size
        if initial_driftscan:
            initial_phase_catalog = self.phase_center_catalog.copy()
            initial_ids = self.phase_center_id_array
        phased = False
        if initial_unprojected or initial_driftscan:
            if allow_drift:
                print(
                    "Data are unprojected or phased as a driftscan and allow_drift is "
                    "True, so resampling will be done without phasing."
                )
            else:
                phased = True
                # phase to RA/dec of zenith
                print(
                    "Data are unprojected or phased as a driftscan, phasing before "
                    "resampling."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)

        # we want the ceil of this, but we don't want to get the wrong answer
        # when the number is very close to an integer but just barely above it.
        temp_new_samples = self.integration_time[inds_to_upsample] / max_int_time
        mask_close_floor = np.isclose(temp_new_samples, np.floor(temp_new_samples))
        temp_new_samples[mask_close_floor] = np.floor(
            temp_new_samples[mask_close_floor]
        )

        n_new_samples = np.asarray(list(map(int, np.ceil(temp_new_samples))))

        temp_Nblts = np.sum(n_new_samples)

        temp_baseline = np.zeros((temp_Nblts,), dtype=np.uint64)
        temp_id_array = np.zeros((temp_Nblts,), dtype=int)
        if initial_nphase_ids > 1 and initial_driftscan:
            temp_initial_ids = np.zeros((temp_Nblts,), dtype=int)
        else:
            temp_initial_ids = None
        if initial_nphase_ids > 1 and initial_unprojected:
            temp_unprojected_blts = np.zeros((temp_Nblts,), dtype=bool)
        else:
            temp_unprojected_blts = None
        temp_time = np.zeros((temp_Nblts,))
        temp_int_time = np.zeros((temp_Nblts,))
        if self.metadata_only:
            temp_data = None
            temp_flag = None
            temp_nsample = None
        else:
            new_data_shape = (temp_Nblts, self.Nfreqs, self.Npols)
            temp_data = np.zeros(new_data_shape, dtype=self.data_array.dtype)
            temp_flag = np.zeros(new_data_shape, dtype=self.flag_array.dtype)
            temp_nsample = np.zeros(new_data_shape, dtype=self.nsample_array.dtype)

        i0 = 0
        for i, ind in enumerate(inds_to_upsample[0]):
            i1 = i0 + n_new_samples[i]
            temp_baseline[i0:i1] = self.baseline_array[ind]
            temp_id_array[i0:i1] = self.phase_center_id_array[ind]
            if initial_nphase_ids > 1:
                if initial_driftscan:
                    temp_initial_ids[i0:i1] = initial_ids[ind]
                if initial_unprojected:
                    temp_unprojected_blts[i0:i1] = unprojected_blts[ind]

            if not self.metadata_only:
                if summing_correlator_mode:
                    temp_data[i0:i1] = self.data_array[ind] / n_new_samples[i]
                else:
                    temp_data[i0:i1] = self.data_array[ind]
                temp_flag[i0:i1] = self.flag_array[ind]
                temp_nsample[i0:i1] = self.nsample_array[ind]

            # compute the new times of the upsampled array
            t0 = self.time_array[ind]
            dt = self.integration_time[ind] / n_new_samples[i]

            # `offset` will be 0.5 or 1, depending on whether n_new_samples for
            # this baseline is even or odd.
            offset = 0.5 + 0.5 * (n_new_samples[i] % 2)
            n2 = n_new_samples[i] // 2

            # Figure out the new center for sample ii taking offset into
            # account. Because `t0` is the central time for the original time
            # sample, `nt` will range from negative to positive so that
            # `temp_time` will result in the central time for the new samples.
            # `idx2` tells us how to far to shift and in what direction for each
            # new sample.
            for ii, idx in enumerate(range(i0, i1)):
                idx2 = ii + offset + n2 - n_new_samples[i]
                nt = ((t0 * units.day) + (dt * idx2 * units.s)).to_value(units.day)
                temp_time[idx] = nt

            temp_int_time[i0:i1] = dt

            i0 = i1

        # harmonize temporary arrays with existing ones
        inds_to_keep = np.nonzero(self.integration_time <= max_int_time)
        self._harmonize_resample_arrays(
            inds_to_keep=inds_to_keep,
            temp_baseline=temp_baseline,
            temp_id_array=temp_id_array,
            temp_time=temp_time,
            temp_int_time=temp_int_time,
            temp_data=temp_data,
            temp_flag=temp_flag,
            temp_nsample=temp_nsample,
            astrometry_library=astrometry_library,
        )

        if phased:
            print("Undoing phasing.")
            if initial_unprojected:
                if initial_nphase_ids > 1:
                    select_mask = unprojected_blts[inds_to_keep]
                    select_mask = np.concatenate((select_mask, temp_unprojected_blts))
                else:
                    select_mask = None
                self.unproject_phase(select_mask=select_mask)
            if initial_driftscan:
                if initial_nphase_ids > 1:
                    initial_ids = initial_ids[inds_to_keep]
                    initial_ids = np.concatenate((initial_ids, temp_initial_ids))
                for cat_id, cat_dict in initial_phase_catalog.items():
                    if cat_dict["cat_type"] != "driftscan":
                        continue
                    if initial_nphase_ids > 1:
                        select_mask = initial_ids == cat_id
                        if not np.any(select_mask):
                            select_mask = None
                    else:
                        select_mask = None
                    self.phase(
                        lon=cat_dict["cat_lon"],
                        lat=cat_dict["cat_lat"],
                        cat_name=cat_dict["cat_name"],
                        cat_type=cat_dict["cat_type"],
                        phase_frame=cat_dict["cat_frame"],
                        epoch=cat_dict["cat_epoch"],
                        select_mask=select_mask,
                    )

        # reorganize along blt axis
        self.reorder_blts(order=blt_order, minor_order=minor_order)

        # check the resulting object
        self.check()

        # add to the history
        history_update_string = (
            f" Upsampled data to {max_int_time:f} second integration time using "
            "pyuvdata."
        )
        self.history = self.history + history_update_string

        return

    def downsample_in_time(
        self,
        *,
        min_int_time=None,
        n_times_to_avg=None,
        blt_order="time",
        minor_order="baseline",
        keep_ragged=True,
        summing_correlator_mode=False,
        allow_drift=False,
        astrometry_library=None,
    ):
        """
        Average to a longer integration time.

        This method will average a UVData object either by an integer factor
        (by setting `n_times_to_avg`) or by a factor that can differ by
        baseline-time sample such that after averaging, the samples have an
        integration time greater than or equal to the `min_int_time` (up to the
        tolerance on the integration_time).

        Note that if the integrations for a baseline do not divide evenly by the
        `n_times_to_avg` or into the specified `min_int_time`, the final
        integrations for that baseline may have integration times less than
        `min_int_time` or be composed of fewer input integrations than `n_times_to_avg`.
        This behavior can be controlled with the `keep_ragged` argument.
        The new samples are averages of the original samples (not interpolations).

        Parameters
        ----------
        min_int_time : float
            Minimum integration time to average the UVData integration_time to
            in seconds.
        n_times_to_avg : int
            Number of time integrations to average together.
        blt_order : str
            Major baseline ordering for output object. Default is "time". See the
            documentation on the `reorder_blts` method for more details.
        minor_order : str
            Minor baseline ordering for output object. Default is "baseline".
        keep_ragged : bool
            When averaging baselines that do not evenly divide into min_int_time,
            or that have a number of integrations that do not evenly divide by
            n_times_to_avg, keep_ragged controls whether to keep the (averaged)
            integrations corresponding to the remaining samples (keep_ragged=True),
            or discard them (keep_ragged=False).
        summing_correlator_mode : bool
            Option to integrate the flux from the original samples rather than
            average the flux to emulate the behavior in some correlators (e.g. HERA).
        allow_drift : bool
            Option to allow resampling of unprojected or driftscan data. If this is
            False, unprojected or driftscan data will be phased to the ra/dec of zenith
            before resampling and then unprojected or rephased to a driftscan after
            resampling. Note that resampling unprojected or driftscan phased data may
            result in unexpected behavior.
        astrometry_library : str
            Library to use for calculating the LSTs after downsampling. Allowed options
            are 'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas
            library), and 'astropy' (which uses the astropy utilities). Default is erfa
            unless the telescope_location is a MoonLocation object, in which case the
            default is astropy.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If neither or both of `min_int_time` and `n_times_to_avg` are set.
            If there's only one time on the object.
            If `min_int_time` is more than 100 time the max integration time.
            If `n_times_to_avg` is not an integer.

        """
        if min_int_time is None and n_times_to_avg is None:
            raise ValueError("Either min_int_time or n_times_to_avg must be set.")

        if min_int_time is not None and n_times_to_avg is not None:
            raise ValueError("Only one of min_int_time or n_times_to_avg can be set.")

        if self.Ntimes == 1:
            raise ValueError("Only one time in this object, cannot downsample.")

        if min_int_time is not None:
            # check that min_int_time is sensible given integration_time
            max_integration_time = np.amax(self.integration_time)
            sensible_max = 1e2 * max_integration_time
            if min_int_time > sensible_max:
                raise ValueError(
                    "Increasing the integration time by more than a "
                    "factor of 100 is not supported. Also note that "
                    "min_int_time should be in seconds."
                )

            # first figure out where integration_time is shorter than min_int_time
            inds_to_downsample = np.nonzero(
                (self.integration_time < min_int_time)
                & (
                    ~np.isclose(
                        self.integration_time,
                        min_int_time,
                        rtol=self._integration_time.tols[0],
                        atol=self._integration_time.tols[1],
                    )
                )
            )

            if len(inds_to_downsample[0]) == 0:
                warnings.warn(
                    "All values in the integration_time array are already "
                    "longer than the value specified; doing nothing."
                )
                return
        else:
            if not isinstance(n_times_to_avg, int | np.integer):
                raise ValueError("n_times_to_avg must be an integer.")
        # If we're going to do actual work, reorder the baselines to ensure time is
        # monotonically increasing.
        # Default of reorder_blts is baseline major, time minor, which is what we want.
        self.reorder_blts()

        if min_int_time is not None:
            # now re-compute inds_to_downsample, in case things have changed
            inds_to_downsample = np.nonzero(
                (self.integration_time < min_int_time)
                & ~np.isclose(
                    self.integration_time,
                    min_int_time,
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                )
            )
            bls_to_downsample = np.unique(self.baseline_array[inds_to_downsample])
        else:
            bls_to_downsample = np.unique(self.baseline_array)

        # figure out how many baseline times we'll end up with at the end
        n_new_samples = 0
        for bl in bls_to_downsample:
            bl_inds = np.nonzero(self.baseline_array == bl)[0]
            int_times = self.integration_time[bl_inds]

            if min_int_time is not None:
                running_int_time = 0.0
                for itime, int_time in enumerate(int_times):
                    running_int_time += int_time
                    over_min_int_time = running_int_time > min_int_time or np.isclose(
                        running_int_time,
                        min_int_time,
                        rtol=self._integration_time.tols[0],
                        atol=self._integration_time.tols[1],
                    )
                    last_sample = itime == len(bl_inds) - 1
                    # We sum up all the samples found so far if we're over the
                    # target minimum time, or we've hit the end of the time
                    # samples for this baseline.
                    if over_min_int_time or last_sample:
                        if last_sample and not (over_min_int_time or keep_ragged):
                            # don't do anything -- implicitly drop these integrations
                            continue
                        n_new_samples += 1
                        running_int_time = 0.0
            else:
                n_bl_times = self.time_array[bl_inds].size
                nsample_temp = np.sum(n_bl_times / n_times_to_avg)
                if keep_ragged and not np.isclose(nsample_temp, np.floor(nsample_temp)):
                    n_new_samples += np.ceil(nsample_temp).astype(int)
                else:
                    n_new_samples += np.floor(nsample_temp).astype(int)

            # figure out if there are any time gaps in the data
            # meaning that the time differences are larger than the integration times
            # time_array is in JD, need to convert to seconds for the diff
            dtime = np.ediff1d(self.time_array[bl_inds]) * 24 * 3600
            int_times = int_times
            if len(np.unique(int_times)) == 1:
                # this baseline has all the same integration times
                if len(np.unique(dtime)) > 1 and not utils.tools._test_array_constant(
                    dtime, tols=self._integration_time.tols
                ):
                    warnings.warn(
                        "There is a gap in the times of baseline "
                        f"{self.baseline_to_antnums(bl)}. "
                        "The output may include averages across long time gaps."
                    )
                elif not np.isclose(
                    dtime[0],
                    int_times[0],
                    rtol=self._integration_time.tols[0],
                    atol=self._integration_time.tols[1],
                ):
                    warnings.warn(
                        "The time difference between integrations is not the "
                        "same as the integration time for "
                        f"baseline {self.baseline_to_antnums(bl)}. The output "
                        "may average across longer time intervals than expected"
                    )

            else:
                # varying integration times for this baseline, need to be more careful
                expected_dtimes = (int_times[:-1] + int_times[1:]) / 2
                wh_diff = np.nonzero(~np.isclose(dtime, expected_dtimes))
                if wh_diff[0].size > 1:
                    warnings.warn(
                        "The time difference between integrations is different "
                        "than the expected given the integration times for "
                        f"baseline {self.baseline_to_antnums(bl)}. The output "
                        "may include averages across long time gaps."
                    )

        temp_Nblts = n_new_samples

        unprojected_blts = self._check_for_cat_type("unprojected")
        driftscan_blts = self._check_for_cat_type("driftscan")
        initial_driftscan = np.any(driftscan_blts)
        initial_unprojected = np.any(unprojected_blts)
        initial_nphase_ids = np.unique(self.phase_center_id_array).size
        if initial_driftscan:
            initial_phase_catalog = self.phase_center_catalog.copy()
            initial_ids = self.phase_center_id_array
        phased = False
        if initial_unprojected or initial_driftscan:
            if allow_drift:
                print(
                    "Data are unprojected or phased as a driftscan and allow_drift is "
                    "True, so resampling will be done without phasing."
                )
            else:
                phased = True
                # phase to RA/dec of zenith
                print(
                    "Data are unprojected or phased as a driftscan, phasing before "
                    "resampling."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)

        # make temporary arrays
        temp_baseline = np.zeros((temp_Nblts,), dtype=np.uint64)
        temp_id_array = np.zeros((temp_Nblts,), dtype=int)
        temp_time = np.zeros((temp_Nblts,))
        temp_int_time = np.zeros((temp_Nblts,))
        if initial_nphase_ids > 1 and initial_driftscan:
            temp_initial_ids = np.zeros((temp_Nblts,), dtype=int)
        else:
            temp_initial_ids = None
        if initial_nphase_ids > 1 and initial_unprojected:
            temp_unprojected_blts = np.zeros((temp_Nblts,), dtype=bool)
        else:
            temp_unprojected_blts = None
        if self.metadata_only:
            temp_data = None
            temp_flag = None
            temp_nsample = None
        else:
            new_data_shape = (temp_Nblts, self.Nfreqs, self.Npols)
            temp_data = np.zeros(new_data_shape, dtype=self.data_array.dtype)
            temp_flag = np.zeros(new_data_shape, dtype=self.flag_array.dtype)
            temp_nsample = np.zeros(new_data_shape, dtype=self.nsample_array.dtype)

        temp_idx = 0
        for bl in bls_to_downsample:
            bl_inds = np.nonzero(self.baseline_array == bl)[0]
            running_int_time = 0.0
            summing_idx = 0
            n_sum = 0
            for itime, int_time in enumerate(self.integration_time[bl_inds]):
                running_int_time += int_time
                n_sum += 1
                if min_int_time is not None:
                    over_min_int_time = running_int_time > min_int_time or np.isclose(
                        running_int_time,
                        min_int_time,
                        rtol=self._integration_time.tols[0],
                        atol=self._integration_time.tols[1],
                    )
                else:
                    over_min_int_time = n_sum >= n_times_to_avg
                last_sample = itime == len(bl_inds) - 1
                # We sum up all the samples found so far if we're over the
                # target minimum time, or we've hit the end of the time
                # samples for this baseline.
                if over_min_int_time or last_sample:
                    if last_sample and not (over_min_int_time or keep_ragged):
                        # don't do anything -- implicitly drop these integrations
                        continue
                    # sum together that number of samples
                    temp_baseline[temp_idx] = bl
                    # this might be wrong if some of the constituent times are
                    # *totally* flagged
                    averaging_idx = bl_inds[summing_idx : summing_idx + n_sum]
                    if self.Nphase > 1:
                        unique_phase_centers = np.unique(
                            self.phase_center_id_array[averaging_idx]
                        )
                        if unique_phase_centers.size > 1:
                            raise ValueError(
                                "Multiple phase centers included in a downsampling "
                                "window. Use `phase` to phase to a single phase center "
                                "or decrease the `min_int_time` or `n_times_to_avg` "
                                "parameter to avoid multiple phase centers being "
                                "included in a downsampling window."
                            )
                    temp_id_array[temp_idx] = self.phase_center_id_array[
                        averaging_idx[0]
                    ]
                    if initial_nphase_ids > 1:
                        if initial_unprojected:
                            unique_unprojected = np.unique(
                                unprojected_blts[averaging_idx]
                            )
                            temp_unprojected_blts[temp_idx] = unique_unprojected[0]
                        if initial_driftscan:
                            unique_initial_ids = np.unique(initial_ids[averaging_idx])
                            temp_initial_ids[temp_idx] = unique_initial_ids[0]
                    # take potential non-uniformity of integration_time into account
                    temp_time[temp_idx] = np.sum(
                        self.time_array[averaging_idx]
                        * self.integration_time[averaging_idx]
                    ) / np.sum(self.integration_time[averaging_idx])
                    temp_int_time[temp_idx] = running_int_time
                    if not self.metadata_only:
                        # if all inputs are flagged, the flag array should be True,
                        # otherwise it should be False.
                        # The sum below will be zero if it's all flagged and
                        # greater than zero otherwise
                        # Then we use a test against 0 to turn it into a Boolean
                        temp_flag[temp_idx] = (
                            np.sum(~self.flag_array[averaging_idx], axis=0) == 0
                        )

                        mask = self.flag_array[averaging_idx]
                        # need to update mask if a downsampled visibility will
                        # be flagged so that we don't set it to zero
                        if (temp_flag[temp_idx]).any():
                            ax1_inds, ax2_inds = np.nonzero(temp_flag[temp_idx])
                            mask[:, ax1_inds, ax2_inds] = False

                        masked_data = np.ma.masked_array(
                            self.data_array[averaging_idx], mask=mask
                        )

                        # nsample array is the fraction of data that we actually kept,
                        # relative to the amount that went into the sum or average
                        nsample_dtype = self.nsample_array.dtype.type
                        # promote nsample dtype if half-precision
                        if nsample_dtype is np.float16:
                            masked_nsample_dtype = np.float32
                        else:
                            masked_nsample_dtype = nsample_dtype
                        masked_nsample = np.ma.masked_array(
                            self.nsample_array[averaging_idx],
                            mask=mask,
                            dtype=masked_nsample_dtype,
                        )

                        int_time_arr = self.integration_time[
                            averaging_idx, np.newaxis, np.newaxis
                        ]
                        masked_int_time = np.ma.masked_array(
                            np.ones_like(
                                self.data_array[averaging_idx],
                                dtype=self.integration_time.dtype,
                            )
                            * int_time_arr,
                            mask=mask,
                        )
                        if summing_correlator_mode:
                            temp_data[temp_idx] = np.sum(masked_data, axis=0)
                        else:
                            # take potential non-uniformity of integration_time
                            # and nsamples into account
                            weights = masked_nsample * masked_int_time
                            weighted_data = masked_data * weights
                            temp_data[temp_idx] = np.sum(
                                weighted_data, axis=0
                            ) / np.sum(weights, axis=0)

                        # output of masked array calculation should be coerced
                        # to the datatype of temp_nsample (which has the same
                        # precision as the original nsample_array)
                        temp_nsample[temp_idx] = np.sum(
                            masked_nsample * masked_int_time, axis=0
                        ) / np.sum(self.integration_time[averaging_idx])
                    # increment counters and reset values
                    temp_idx += 1
                    summing_idx += n_sum
                    running_int_time = 0.0
                    n_sum = 0

        # make sure we've populated the right number of baseline-times
        assert temp_idx == temp_Nblts, (
            f"Wrong number of baselines. Got {temp_idx:d},  "
            f"expected {temp_Nblts:d}. This is a bug, please make an issue at "
            "https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues"
        )

        # harmonize temporary arrays with existing ones
        if min_int_time is not None:
            bls_not_downsampled = set(self.baseline_array) - set(bls_to_downsample)
            inds_to_keep = []
            for bl in bls_not_downsampled:
                inds_to_keep += np.nonzero(self.baseline_array == bl)[0].tolist()
            inds_to_keep = np.array(inds_to_keep, dtype=np.int64)
        else:
            inds_to_keep = np.array([], dtype=bool)

        # we don't know the order now (because we just messed with it), and in harmonize
        # it sets rectangularity, which gets the wrong behavior if blt_order is wrong.
        # So we set it to None, so that it doesn't say the wrong thing (we reorder
        # properly below).
        self.blt_order = None
        self._harmonize_resample_arrays(
            inds_to_keep=inds_to_keep,
            temp_baseline=temp_baseline,
            temp_id_array=temp_id_array,
            temp_time=temp_time,
            temp_int_time=temp_int_time,
            temp_data=temp_data,
            temp_flag=temp_flag,
            temp_nsample=temp_nsample,
            astrometry_library=astrometry_library,
        )
        if phased:
            print("Undoing phasing.")
            if initial_unprojected:
                if initial_nphase_ids > 1:
                    select_mask = unprojected_blts[inds_to_keep]
                    select_mask = np.concatenate((select_mask, temp_unprojected_blts))
                else:
                    select_mask = None
                self.unproject_phase(select_mask=select_mask)
            if initial_driftscan:
                if initial_nphase_ids > 1:
                    initial_ids = initial_ids[inds_to_keep]
                    initial_ids = np.concatenate((initial_ids, temp_initial_ids))
                for cat_id, cat_dict in initial_phase_catalog.items():
                    if cat_dict["cat_type"] != "driftscan":
                        continue
                    if initial_nphase_ids > 1:
                        select_mask = initial_ids == cat_id
                        if not np.any(select_mask):
                            select_mask = None
                    else:
                        select_mask = None
                    self.phase(
                        lon=cat_dict["cat_lon"],
                        lat=cat_dict["cat_lat"],
                        cat_name=cat_dict["cat_name"],
                        cat_type=cat_dict["cat_type"],
                        phase_frame=cat_dict["cat_frame"],
                        epoch=cat_dict["cat_epoch"],
                        select_mask=select_mask,
                    )

        # reorganize along blt axis
        self.reorder_blts(order=blt_order, minor_order=minor_order)

        # check the resulting object
        self.check()

        # add to the history
        if min_int_time is not None:
            history_update_string = (
                f" Downsampled data to {min_int_time:f} second integration "
                "time using pyuvdata."
            )
        else:
            history_update_string = (
                f" Downsampled data by a factor of {n_times_to_avg} in time "
                "using pyuvdata."
            )
        self.history = self.history + history_update_string

        return

    def resample_in_time(
        self,
        target_time,
        *,
        only_downsample=False,
        only_upsample=False,
        blt_order="time",
        minor_order="baseline",
        keep_ragged=True,
        summing_correlator_mode=False,
        allow_drift=False,
        astrometry_library=None,
    ):
        """
        Intelligently upsample or downsample a UVData object to the target time.

        Parameters
        ----------
        target_time : float
            The target integration time to resample to, in seconds.
        only_downsample : bool
            Option to only call bda_downsample.
        only_upsample : bool
            Option to only call bda_upsample.
        blt_order : str
            Major baseline ordering for output object. Default is "time". See the
            documentation on the `reorder_blts` method for more details.
        minor_order : str
            Minor baseline ordering for output object. Default is "baseline".
        keep_ragged : bool
            When averaging baselines that do not evenly divide into min_int_time,
            keep_ragged controls whether to keep the (summed) integrations
            corresponding to the remaining samples (keep_ragged=True), or
            discard them (keep_ragged=False). Note this option only applies to the
            `bda_downsample` method.
        summing_correlator_mode : bool
            Option to integrate or split the flux from the original samples
            rather than average or duplicate the flux from the original samples
            to emulate the behavior in some correlators (e.g. HERA).
        allow_drift : bool
            Option to allow resampling of unprojected or driftscan data. If this is
            False, unprojected or driftscan data will be phased to the ra/dec of zenith
            before resampling and then unprojected or rephased to a driftscan after
            resampling. Note that resampling unprojected or driftscan phased data may
            result in unexpected behavior.
        astrometry_library : str
            Library to use for calculating the LSTs after resampling. Allowed options
            are 'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas
            library), and 'astropy' (which uses the astropy utilities). Default is erfa
            unless the telescope_location is a MoonLocation object, in which case the
            default is astropy.

        Returns
        -------
        None
        """
        # figure out integration times relative to target time
        min_int_time = np.amin(self.integration_time)
        max_int_time = np.amax(self.integration_time)

        if int(np.floor(target_time / min_int_time)) >= 2 and not only_upsample:
            downsample = True
        else:
            downsample = False

        if int(np.floor(max_int_time / target_time)) >= 2 and not only_downsample:
            upsample = True
        else:
            upsample = False

        if not downsample and not upsample:
            warnings.warn(
                "No resampling will be done because target time is not "
                "a factor of 2 or more off from integration_time. To "
                "force resampling set only_upsample or only_downsample "
                "keywords or call upsample_in_time or downsample_in_time."
            )
            return

        if downsample:
            self.downsample_in_time(
                min_int_time=target_time,
                blt_order=blt_order,
                minor_order=minor_order,
                keep_ragged=keep_ragged,
                summing_correlator_mode=summing_correlator_mode,
                allow_drift=allow_drift,
                astrometry_library=astrometry_library,
            )
        if upsample:
            self.upsample_in_time(
                target_time,
                blt_order=blt_order,
                minor_order=minor_order,
                summing_correlator_mode=summing_correlator_mode,
                allow_drift=allow_drift,
                astrometry_library=astrometry_library,
            )

        return

    def frequency_average(
        self,
        n_chan_to_avg,
        *,
        summing_correlator_mode=False,
        propagate_flags=False,
        respect_spws=True,
        keep_ragged=True,
    ):
        """
        Average in frequency.

        Does a simple average over an integer number of input channels, leaving
        flagged samples out of the average.

        In the future, this method will support setting the frequency
        to the true mean of the averaged non-flagged frequencies rather than
        the simple mean of the input channel frequencies. For now it does not.

        Parameters
        ----------
        n_chan_to_avg : int
            Number of channels to average together. See the keep_ragged parameter for
            the handling if the number of frequencies per spectral window does not
            divide evenly by this number.
        summing_correlator_mode : bool
            Option to integrate the flux from the original samples rather than average
            the flux from the original samples to emulate the behavior in some
            correlators (e.g. HERA).
        propagate_flags: bool
            Option to flag an averaged entry even if some of its contributors
            are not flagged. The averaged result will still leave the flagged
            samples out of the average, except when all contributors are
            flagged.
        respect_spws : bool
            Option to respect spectral window boundaries when averaging. If True, do not
            average across spectral window boundaries. Setting this to False will result
            in the averaged object having a single spectral window.
        keep_ragged : bool
            If the number of frequencies in each spectral window (or Nfreqs if
            respect_spw=False) does not divide evenly by n_chan_to_avg, this
            option controls whether the frequencies at the end of the spectral window
            will be dropped to make it evenly divisable (keep_ragged=False) or will be
            combined into a smaller frequency bin (keep_ragged=True). Default is True.

        """
        if self.Nspws > 1 and not respect_spws:
            # Put everything in one spectral window.
            self.Nspws = 1
            self.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)
            self.spw_array = np.array([0])

        spacing_error, chanwidth_error = self._check_freq_spacing(raise_errors=None)
        if spacing_error:
            warnings.warn(
                "The frequency spacing and/or channel widths vary, so after averaging "
                "they will also vary."
            )
        elif chanwidth_error:
            warnings.warn(
                "The frequency spacing is even but not equal to the channel width, so "
                "after averaging the channel_width will also not match the frequency "
                "spacing."
            )

        if self.eq_coeffs is not None:
            eq_coeff_diff = np.diff(self.eq_coeffs, axis=1)
            if np.abs(np.max(eq_coeff_diff)) > 0:
                warnings.warn(
                    "eq_coeffs vary by frequency. They should be "
                    "applied to the data using `remove_eq_coeffs` "
                    "before frequency averaging."
                )

        # Figure out how many channels are in each spw so we can tell if we have a
        # ragged situation (indicated by the some_uneven variable).
        # While we're at it, build up some useful dicts for later, keyed on spw
        nchans_spw = np.zeros(self.Nspws, dtype=int)
        # final_nchan will hold the number of Nfreqs after averaging.
        final_nchan = 0
        # spw_chans will hold the original channel indices for each spw
        spw_chans = {}
        # final_spw_chans will hold the final channel indices for each spw
        final_spw_chans = {}
        for spw_ind, spw in enumerate(self.spw_array):
            these_inds = np.nonzero(self.flex_spw_id_array == spw)[0]
            spw_chans[spw] = these_inds
            nchans_spw[spw_ind] = these_inds.size
            if keep_ragged:
                this_final_nchan = int(np.ceil(nchans_spw[spw_ind] / n_chan_to_avg))
            else:
                this_final_nchan = int(np.floor(nchans_spw[spw_ind] / n_chan_to_avg))
            final_spw_chans[spw] = np.arange(
                final_nchan, final_nchan + this_final_nchan
            )
            final_nchan += this_final_nchan

        # Since we have to loop through the spws, we cannot do the averaging with a
        # simple reshape and average. So we need to create arrays to hold the
        # various metadata & data after averaging
        final_freq_array = np.zeros(final_nchan, dtype=float)
        final_channel_width = np.zeros(final_nchan, dtype=float)
        final_flex_spw_id_array = np.zeros(final_nchan, dtype=int)
        if self.eq_coeffs is not None:
            final_eq_coeffs = np.zeros((self.telescope.Nants, final_nchan), dtype=float)

        if not self.metadata_only:
            final_shape_tuple = (self.Nblts, final_nchan, self.Npols)
            final_flag_array = np.full(final_shape_tuple, False, dtype=bool)
            final_data_array = np.zeros(final_shape_tuple, dtype=self.data_array.dtype)
            final_nsample_array = np.zeros(
                final_shape_tuple, dtype=self.nsample_array.dtype
            )

        # Now loop through the spws to actually do the averaging
        for spw_ind, spw in enumerate(self.spw_array):
            # n_final_chan_reg is the number of regular (non-ragged) channels after
            # averaging in this spw.
            # For the regular channels, we can average more quickly by reshaping the
            # frequency axis into two axes of lengths (n_final_chan_reg, n_chan_to_avg)
            # followed by an average (or sum) over the axis of length n_chan_to_avg.
            # Then we just have to do one more calculation for the remaining input
            # channels if there are ragged channels.
            n_final_chan_reg = int(np.floor(nchans_spw[spw_ind] / n_chan_to_avg))
            nfreq_mod_navg = nchans_spw[spw_ind] % n_chan_to_avg
            these_inds = spw_chans[spw]
            this_ragged = False
            regular_inds = these_inds
            irregular_inds = np.array([])
            this_final_reg_inds = final_spw_chans[spw]
            if nfreq_mod_navg != 0:
                # not an even number of final channels
                regular_inds = these_inds[0 : n_final_chan_reg * n_chan_to_avg]
                if not keep_ragged:
                    # only use the non-ragged inds
                    these_inds = regular_inds
                else:
                    # find the irregular inds for this spw
                    this_ragged = True
                    irregular_inds = these_inds[n_final_chan_reg * n_chan_to_avg :]
                    this_final_reg_inds = this_final_reg_inds[:-1]

            # Now do the reshaping and combining across the n_chan_to_avg length axis
            final_freq_array[this_final_reg_inds] = (
                self.freq_array[regular_inds]
                .reshape((n_final_chan_reg, n_chan_to_avg))
                .mean(axis=1)
            )
            # take a sum here rather to get final channel width
            final_channel_width[this_final_reg_inds] = (
                self.channel_width[regular_inds]
                .reshape((n_final_chan_reg, n_chan_to_avg))
                .sum(axis=1)
            )
            if this_ragged:
                # deal with the final ragged channel
                final_freq_array[final_spw_chans[spw][-1]] = np.mean(
                    self.freq_array[irregular_inds]
                )
                final_channel_width[final_spw_chans[spw][-1]] = np.sum(
                    self.channel_width[irregular_inds]
                )

            final_flex_spw_id_array[final_spw_chans[spw]] = spw

            if self.eq_coeffs is not None:
                final_eq_coeffs[:, this_final_reg_inds] = (
                    self.eq_coeffs[:, regular_inds]
                    .reshape((self.telescope.Nants, n_final_chan_reg, n_chan_to_avg))
                    .mean(axis=2)
                )
                if this_ragged:
                    final_eq_coeffs[:, final_spw_chans[spw][-1]] = np.mean(
                        self.eq_coeffs[:, irregular_inds], axis=1
                    )

            if not self.metadata_only:
                shape_tuple = (self.Nblts, n_final_chan_reg, n_chan_to_avg, self.Npols)

                reg_mask = self.flag_array[:, regular_inds].reshape(shape_tuple)
                if this_ragged:
                    irreg_mask = self.flag_array[:, irregular_inds]

                if propagate_flags:
                    # if any contributors are flagged, the result should be flagged
                    final_flag_array[:, this_final_reg_inds] = np.any(
                        self.flag_array[:, regular_inds].reshape(shape_tuple), axis=2
                    )
                    if this_ragged:
                        final_flag_array[:, final_spw_chans[spw][-1]] = np.any(
                            self.flag_array[:, irregular_inds], axis=1
                        )
                else:
                    # if all inputs are flagged, the flag array should be True,
                    # otherwise it should be False.
                    final_flag_array[:, this_final_reg_inds] = np.all(
                        self.flag_array[:, regular_inds].reshape(shape_tuple), axis=2
                    )
                    if this_ragged:
                        final_flag_array[:, final_spw_chans[spw][-1]] = np.all(
                            self.flag_array[:, irregular_inds], axis=1
                        )

                # need to update mask if a downsampled visibility will be flagged
                # so that we don't set it to zero
                # This is a common radio astronomy convention that when averaging over
                # entirely flagged channels, you include the flagged channels in the
                # result (so it's not zero) whereas you exclude flagged channels if
                # there are any unflagged channels in the average.
                for chan_ind in np.arange(n_final_chan_reg):
                    this_chan = final_spw_chans[spw][chan_ind]
                    if (final_flag_array[:, this_chan]).any():
                        ax0_inds, ax2_inds = np.nonzero(
                            final_flag_array[:, this_chan, :]
                        )
                        # Only if all entries are masked
                        # May not happen due to propagate_flags keyword
                        # mask should be left alone otherwise
                        fully_flagged = np.all(
                            reg_mask[ax0_inds, this_chan, :, ax2_inds], axis=1
                        )
                        ff_inds = np.nonzero(fully_flagged)
                        reg_mask[ax0_inds[ff_inds], this_chan, :, ax2_inds[ff_inds]] = (
                            False
                        )
                if this_ragged:
                    ax0_inds, ax2_inds = np.nonzero(
                        final_flag_array[:, final_spw_chans[spw][-1], :]
                    )
                    fully_flagged = np.all(irreg_mask[ax0_inds, :, ax2_inds], axis=1)
                    ff_inds = np.nonzero(fully_flagged)
                    irreg_mask[ax0_inds[ff_inds], :, ax2_inds[ff_inds]] = False

                # create a masked data array from the data_array and mask_array
                # (based on the flag_array).
                # This lets numpy handle the averaging with flags.
                masked_reg_data = np.ma.masked_array(
                    self.data_array[:, regular_inds].reshape(shape_tuple), mask=reg_mask
                )
                if this_ragged:
                    masked_irreg_data = np.ma.masked_array(
                        self.data_array[:, irregular_inds], mask=irreg_mask
                    )

                # promote nsample dtype if half-precision
                nsample_dtype = self.nsample_array.dtype.type
                if nsample_dtype is np.float16:
                    masked_nsample_dtype = np.float32
                else:
                    masked_nsample_dtype = nsample_dtype
                # create a masked nsample array from the data_array and mask_array
                masked_reg_nsample = np.ma.masked_array(
                    self.nsample_array[:, regular_inds].reshape(shape_tuple),
                    mask=reg_mask,
                    dtype=masked_nsample_dtype,
                )
                if this_ragged:
                    masked_irreg_nsample = np.ma.masked_array(
                        self.nsample_array[:, irregular_inds],
                        mask=irreg_mask,
                        dtype=masked_nsample_dtype,
                    )

                if summing_correlator_mode:
                    # sum rather than average
                    final_data_array[:, this_final_reg_inds] = np.sum(
                        masked_reg_data, axis=2
                    ).data
                    if this_ragged:
                        final_data_array[:, final_spw_chans[spw][-1]] = np.sum(
                            masked_irreg_data, axis=1
                        ).data
                else:
                    # do a weighted average with the weights given by the nsample_array
                    final_data_array[:, this_final_reg_inds] = (
                        np.sum(masked_reg_data * masked_reg_nsample, axis=2)
                        / np.sum(masked_reg_nsample, axis=2)
                    ).data
                    if this_ragged:
                        final_data_array[:, final_spw_chans[spw][-1]] = (
                            np.sum(masked_irreg_data * masked_irreg_nsample, axis=1)
                            / np.sum(masked_irreg_nsample, axis=1)
                        ).data

                # nsample array is the fraction of data that we actually kept,
                # relative to the amount that went into the sum or average.
                # So it's a sum over the averaged channels divided by the number of
                # averaged channels
                # Need to take care to return precision back to original value.
                final_nsample_array[:, this_final_reg_inds] = (
                    np.sum(masked_reg_nsample, axis=2) / float(n_chan_to_avg)
                ).data.astype(nsample_dtype)
                if this_ragged:
                    final_nsample_array[:, final_spw_chans[spw][-1]] = (
                        np.sum(masked_irreg_nsample, axis=1) / irregular_inds.size
                    ).data.astype(nsample_dtype)

        # Put the final arrays on the object
        self.freq_array = final_freq_array
        self.channel_width = final_channel_width
        self.flex_spw_id_array = final_flex_spw_id_array
        if self.eq_coeffs is not None:
            self.eq_coeffs = final_eq_coeffs

        if not self.metadata_only:
            self.flag_array = final_flag_array
            self.data_array = final_data_array
            self.nsample_array = final_nsample_array

        # update Nfreqs
        self.Nfreqs = final_nchan

    def get_redundancies(
        self,
        *,
        tol=1.0,
        use_antpos=False,
        include_conjugates=None,
        include_autos=True,
        conjugate_bls=False,
        use_grid_alg=True,
    ):
        """
        Get redundant baselines to a given tolerance.

        This can be used to identify redundant baselines present in the data,
        or find all possible redundant baselines given the antenna positions.

        Parameters
        ----------
        tol : float
            Redundancy tolerance in meters (default 1m).
        use_antpos : bool
            Use antenna positions to find all possible redundant groups for this
            telescope (default False).
            The returned baselines are in the 'u>0' convention.
        include_conjugates : bool
            Option to include baselines that are redundant under conjugation.
            Only used if use_antpos is False. Default is currently False but will
            become True in version 3.4.
        include_autos : bool
            Option to include autocorrelations in the full redundancy list.
            Only used if use_antpos is True.
        conjugate_bls : bool
            If using antenna positions, this will conjugate baselines on this
            object to correspond with those in the returned groups.
        use_grid_alg : bool
            Option to use the gridding based algorithm (developed by the HERA team)
            to find redundancies rather than the older clustering algorithm.
            Default is True.

        Returns
        -------
        baseline_groups : list of lists of int
            List of lists of redundant baseline numbers
        vec_bin_centers : list of ndarray of float
            List of vectors describing redundant group uvw centers
        lengths : list of float
            List of redundant group baseline lengths in meters
        conjugates : list of int, or None, optional
            List of indices for baselines that must be conjugated to fit into their
            redundant groups.
            Will return None if use_antpos is True and include_conjugates is True
            Only returned if include_conjugates is True

        Notes
        -----
        If use_antpos is set, then this function will find all redundant baseline groups
        for this telescope, under the u>0 antenna ordering convention.
        If use_antpos is not set, this function will look for redundant groups
        in the data.

        """
        if use_antpos:
            antpos = self.telescope.get_enu_antpos()
            result = utils.redundancy.get_antenna_redundancies(
                self.telescope.antenna_numbers,
                antpos,
                tol=tol,
                include_autos=include_autos,
                use_grid_alg=use_grid_alg,
            )
            if conjugate_bls:
                self.conjugate_bls("u>0", uvw_tol=tol)

            if include_conjugates:
                result = result + (None,)
            return result

        _, unique_inds = np.unique(self.baseline_array, return_index=True)
        unique_inds.sort()
        baselines = np.take(self.baseline_array, unique_inds)

        self.set_rectangularity(force=True)
        if self.blts_are_rectangular or np.all(self._check_for_cat_type("unprojected")):
            # we can just use the uvws to find redundancy
            baseline_vecs = np.take(self.uvw_array, unique_inds, axis=0)
        else:
            # use the antenna positions to get baseline vectors. This ensures
            # that we aren't comparing uvws at different times
            ant1 = np.take(self.ant_1_array, unique_inds)
            ant2 = np.take(self.ant_2_array, unique_inds)
            antpos = self.telescope.get_enu_antpos()

            ant1_inds = np.array(
                [np.nonzero(self.telescope.antenna_numbers == ai)[0][0] for ai in ant1]
            )
            ant2_inds = np.array(
                [np.nonzero(self.telescope.antenna_numbers == ai)[0][0] for ai in ant2]
            )

            baseline_vecs = np.take(antpos, ant2_inds, axis=0) - np.take(
                antpos, ant1_inds, axis=0
            )

        return utils.redundancy.get_baseline_redundancies(
            baselines,
            baseline_vecs,
            tol=tol,
            include_conjugates=include_conjugates,
            use_grid_alg=use_grid_alg,
        )

    def compress_by_redundancy(
        self,
        *,
        method="select",
        tol=1.0,
        inplace=True,
        keep_all_metadata=True,
        use_grid_alg=None,
    ):
        """
        Downselect or average to only have one baseline per redundant group.

        Either select the first baseline in the redundant group or average over
        the baselines in the redundant group. When averaging, only unflagged data are
        averaged and the nsample_array reflects the amount of unflagged data that was
        averaged over. In the case that all the data for a particular visibility to be
        averaged is flagged, all the flagged data is averaged (with an nsample value
        that represents all the data) but the flag array is set to True for that
        visibility.

        Parameters
        ----------
        tol : float
            Redundancy tolerance in meters, default is 1.0 corresponding to 1 meter.
            This specifies what tolerance to use when identifying baselines as
            redundant.
        method : str
            Options are "select", which just keeps the first baseline in each
            redundant group or "average" which averages over the baselines in each
            redundant group and assigns the average to the first baseline in the group.
        inplace : bool
            Option to do selection on current object.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas,
            even those that do not remain after the select option.
        use_grid_alg : bool
            Option to use the gridding based algorithm (developed by the HERA team)
            to find redundancies rather than the older clustering algorithm.

        Returns
        -------
        UVData object or None
            if inplace is False, return the compressed UVData object

        """
        allowed_methods = ["select", "average"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")

        if use_grid_alg is None:
            # This should be removed in v3.4
            warnings.warn(
                "The use_grid_alg parameter is not set. Defaulting to True to "
                "use the new gridding based algorithm (developed by the HERA team) "
                "rather than the older clustering based algorithm. This is change "
                "to the default, to use the clustering algorithm set "
                "use_grid_alg=False."
            )
            use_grid_alg = True

        red_gps, _, _, conjugates = self.get_redundancies(
            tol=tol, include_conjugates=True, use_grid_alg=use_grid_alg
        )
        bl_ants = [self.baseline_to_antnums(gp[0]) for gp in red_gps]

        if method == "average":
            # do a metadata only select to get all the metadata right
            new_obj = self.copy(metadata_only=True)
            new_obj.select(bls=bl_ants, keep_all_metadata=keep_all_metadata)

            if not self.metadata_only:
                # If the baseline is in the "conjugated" list, we will need to
                # fix the conjugation on assignment (since the visibilities are
                # tabulated assuming that the baseline position is on the opposite
                # side of the uvw origin).
                fix_conj = np.isin(new_obj.baseline_array, conjugates)

                # initalize the data like arrays
                new_data_shape = (new_obj.Nblts, new_obj.Nfreqs, new_obj.Npols)
                temp_data_array = np.zeros(new_data_shape, dtype=self.data_array.dtype)
                temp_nsample_array = np.zeros(
                    new_data_shape, dtype=self.nsample_array.dtype
                )
                temp_flag_array = np.zeros(new_data_shape, dtype=self.flag_array.dtype)
            for grp_ind, group in enumerate(red_gps):
                if len(conjugates) > 0:
                    conj_group = set(group).intersection(conjugates)
                    reg_group = list(set(group) - conj_group)
                    conj_group = list(conj_group)
                else:
                    reg_group = group
                    conj_group = []
                group_times = []
                group_inds = []
                conj_group_inds = []
                conj_group_times = []
                for bl in reg_group:
                    bl_inds = np.where(self.baseline_array == bl)[0]
                    group_inds.extend(bl_inds)
                    group_times.extend(self.time_array[bl_inds])
                for bl in conj_group:
                    bl_inds = np.where(self.baseline_array == bl)[0]
                    conj_group_inds.extend(bl_inds)
                    conj_group_times.extend(self.time_array[bl_inds])

                group_inds = np.array(group_inds, dtype=np.int64)
                conj_group_inds = np.array(conj_group_inds, dtype=np.int64)
                # now we have to figure out which times are the same to a tolerance
                # so we can average over them.
                time_inds = np.arange(len(group_times + conj_group_times))
                time_gps = utils.redundancy.find_clusters(
                    location_ids=time_inds,
                    location_vectors=np.array(group_times + conj_group_times),
                    tol=self._time_array.tols[1],
                )

                # average over the same times
                obj_bl = bl_ants[grp_ind]
                obj_inds = new_obj._key2inds(obj_bl)[0]
                obj_times = new_obj.time_array[obj_inds]

                for gp in time_gps:
                    # Note that this average time is just used for identifying the
                    # index to use for the blt axis on the averaged data set.
                    # We do not update the actual time on that data set because it can
                    # result in confusing behavior -- small numerical rounding errors
                    # can result in many more unique times in the final data set than
                    # in the initial data set.
                    avg_time = np.average(np.array(group_times + conj_group_times)[gp])

                    obj_time_ind = np.where(
                        np.abs(obj_times - avg_time) < self._time_array.tols[1]
                    )[0]

                    if obj_time_ind.size == 1:
                        if isinstance(obj_inds, slice):
                            this_obj_ind = (obj_inds.start or 0) + obj_time_ind[0] * (
                                obj_inds.step or 1
                            )
                        else:
                            this_obj_ind = obj_inds[obj_time_ind[0]]
                    else:
                        warnings.warn(
                            "Index baseline in the redundant group does not "
                            "have all the times, compressed object will be "
                            "missing those times."
                        )
                        continue

                    # time_ind contains indices for both regular and conjugated bls
                    # because we needed to group them together in time.
                    # The regular ones are first and extend the length of group_times,
                    # so we use that to split them back up.
                    regular_orientation = np.array(
                        [time_ind for time_ind in gp if time_ind < len(group_times)],
                        dtype=np.int64,
                    )
                    regular_inds = group_inds[np.array(regular_orientation)]
                    conj_orientation = np.array(
                        [
                            time_ind - len(group_times)
                            for time_ind in gp
                            if time_ind >= len(group_times)
                        ],
                        dtype=np.int64,
                    )
                    conj_inds = conj_group_inds[np.array(conj_orientation)]
                    # check that the integration times are all the same
                    int_times = np.concatenate(
                        (
                            self.integration_time[regular_inds],
                            self.integration_time[conj_inds],
                        )
                    )
                    if not np.all(
                        np.abs(int_times - new_obj.integration_time[obj_time_ind])
                        < new_obj._integration_time.tols[1]
                    ):
                        warnings.warn(
                            "Integrations times are not identical in a redundant "
                            "group. Averaging anyway but this may cause unexpected "
                            "behavior."
                        )

                    if not self.metadata_only:
                        vis_to_avg = np.concatenate(
                            (
                                self.data_array[regular_inds],
                                np.conj(self.data_array[conj_inds]),
                            )
                        )
                        nsample_to_avg = np.concatenate(
                            (
                                self.nsample_array[regular_inds],
                                self.nsample_array[conj_inds],
                            )
                        )
                        flags_to_avg = np.concatenate(
                            (self.flag_array[regular_inds], self.flag_array[conj_inds])
                        )
                        # if all data is flagged, average it all as if it were not
                        if np.all(flags_to_avg):
                            mask = np.zeros_like(flags_to_avg)
                        else:
                            mask = flags_to_avg

                        vis_to_avg = np.ma.masked_array(vis_to_avg, mask=mask)

                        nsample_to_avg = np.ma.masked_array(nsample_to_avg, mask=mask)

                        avg_vis = np.ma.average(
                            vis_to_avg, weights=nsample_to_avg, axis=0
                        )
                        avg_nsample = np.sum(nsample_to_avg, axis=0)
                        avg_flag = np.all(flags_to_avg, axis=0)
                        temp_data_array[this_obj_ind] = avg_vis

                        # If we need to flip the conjugation, to that on the
                        # value assignment here.
                        if fix_conj[this_obj_ind]:
                            temp_data_array[this_obj_ind] = np.conj(avg_vis)
                        else:
                            temp_data_array[this_obj_ind] = avg_vis
                        temp_nsample_array[this_obj_ind] = avg_nsample
                        temp_flag_array[this_obj_ind] = avg_flag

            if inplace:
                self.select(bls=bl_ants, keep_all_metadata=keep_all_metadata)
                if not self.metadata_only:
                    self.data_array = temp_data_array
                    self.nsample_array = temp_nsample_array
                    self.flag_array = temp_flag_array
                self.check()
                return
            else:
                if not self.metadata_only:
                    new_obj.data_array = temp_data_array
                    new_obj.nsample_array = temp_nsample_array
                    new_obj.flag_array = temp_flag_array
                new_obj.check()
                return new_obj
        else:
            return self.select(
                bls=bl_ants, inplace=inplace, keep_all_metadata=keep_all_metadata
            )

    def inflate_by_redundancy(
        self, *, tol=1.0, blt_order="time", blt_minor_order=None, use_grid_alg=False
    ):
        """
        Expand data to full size, copying data among redundant baselines.

        Note that this method conjugates baselines to the 'u>0' convention in order
        to inflate the redundancies.

        Parameters
        ----------
        tol : float
            Redundancy tolerance in meters, default is 1.0 corresponding to 1 meter.
        blt_order : str
            string specifying primary order along the blt axis (see `reorder_blts`)
        blt_minor_order : str
            string specifying minor order along the blt axis (see `reorder_blts`)
        use_grid_alg : bool
            Option to use the gridding based algorithm (developed by the HERA team)
            to find redundancies rather than the older clustering algorithm.

        """
        self.conjugate_bls("u>0")
        red_gps, _, _ = self.get_redundancies(
            tol=tol, use_antpos=True, conjugate_bls=True, use_grid_alg=use_grid_alg
        )

        # Stack redundant groups into one array.
        group_index, bl_array_full = zip(
            *[(i, bl) for i, gp in enumerate(red_gps) for bl in gp], strict=True
        )

        # TODO should be an assert that each baseline only ends up in one group

        # Map group index to blt indices in the compressed array.
        bl_array_comp = self.baseline_array
        uniq_bl = np.unique(bl_array_comp)

        group_blti = {}
        Nblts_full = 0
        for i, gp in enumerate(red_gps):
            for bl in gp:
                # First baseline in the group that is also in the compressed
                # baseline array.
                if bl in uniq_bl:
                    group_blti[i] = np.where(bl == bl_array_comp)[0]
                    # add number of blts for this group
                    Nblts_full += group_blti[i].size * len(gp)
                    break

        blt_map = np.zeros(Nblts_full, dtype=int)
        full_baselines = np.zeros(Nblts_full, dtype=int)
        missing = []
        counter = 0
        for bl, gi in zip(bl_array_full, group_index, strict=True):
            try:
                # this makes the time the fastest axis
                blt_map[counter : counter + group_blti[gi].size] = group_blti[gi]
                full_baselines[counter : counter + group_blti[gi].size] = bl
                counter += group_blti[gi].size
            except KeyError:
                missing.append(bl)
                pass

        if np.any(missing):
            warnings.warn("Missing some redundant groups. Filling in available data.")

        # blt_map is an index array mapping compressed blti indices to uncompressed
        self.data_array = self.data_array[blt_map, ...]
        self.nsample_array = self.nsample_array[blt_map, ...]
        self.flag_array = self.flag_array[blt_map, ...]
        self.time_array = self.time_array[blt_map]
        self.lst_array = self.lst_array[blt_map]
        self.integration_time = self.integration_time[blt_map]
        self.uvw_array = self.uvw_array[blt_map, ...]

        self.baseline_array = full_baselines
        self.ant_1_array, self.ant_2_array = self.baseline_to_antnums(
            self.baseline_array
        )
        self.Nants_data = self._calc_nants_data()
        self.Nbls = np.unique(self.baseline_array).size
        self.Nblts = Nblts_full

        self.phase_center_app_ra = self.phase_center_app_ra[blt_map]
        self.phase_center_app_dec = self.phase_center_app_dec[blt_map]
        self.phase_center_frame_pa = self.phase_center_frame_pa[blt_map]
        self.phase_center_id_array = self.phase_center_id_array[blt_map]

        self.reorder_blts(order=blt_order, minor_order=blt_minor_order)

        self.check()

    def _convert_from_filetype(self, other):
        """
        Convert from a file-type specific object to a UVData object.

        Used in reads.

        Parameters
        ----------
        other : object that inherits from UVData
            File type specific object to convert to UVData
        """
        for p in other:
            param = getattr(other, p)
            setattr(self, p, param)

    def _convert_to_filetype(self, filetype):
        """
        Convert from a UVData object to a file-type specific object.

        Used in writes.

        Parameters
        ----------
        filetype : str
            Specifies what file type object to convert to. Options are: 'uvfits',
            'fhd', 'miriad', 'uvh5', 'mir', 'ms'

        Raises
        ------
        ValueError
            if filetype is not a known type
        """
        if filetype == "uvfits":
            from . import uvfits

            other_obj = uvfits.UVFITS()
        elif filetype == "fhd":
            from . import fhd

            other_obj = fhd.FHD()
        elif filetype == "miriad":
            from . import miriad

            other_obj = miriad.Miriad()
        elif filetype == "uvh5":
            from . import uvh5

            other_obj = uvh5.UVH5()
        elif filetype == "mir":
            from . import mir

            other_obj = mir.Mir()
        elif filetype == "ms":
            from . import ms

            other_obj = ms.MS()
        else:
            raise ValueError("filetype must be uvfits, mir, miriad, ms, fhd, or uvh5")

        for par in self:
            setattr(other_obj, par, getattr(self, par))

        return other_obj

    def read_fhd(self, vis_files, *, params_file, **kwargs):
        """
        Read in data from a list of FHD files.

        Parameters
        ----------
        vis_files : array_like of str
            The FHD data (or model) visibility save files. Can be None if `read_data` is
            False.
        params_file : str
            The FHD params save file. Required.
        obs_file : str
            The FHD obs save file. Required if `read_data` is False.
        flags_file : str
            The FHD data (or model) flag save file. Required if `read_data` is True.
        layout_file : str
            The FHD layout save file. Required for correct antenna metadata.
        settings_file : str
            The FHD settings text file. Required for full history information.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object. If read_data is False, an obs file must be
            included in the filelist. Note that if read_data is False, Npols is
            derived from the obs file and reflects the number of polarizations
            used in the FHD run. If read_data is True, Npols is given by the
            number of visibility data files provided in `filelist`.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        default_mount_type : str
            If not recorded in the data set or telescope is unknown to pyuvdata, the
            `Telescope.mount_type` parameter is automatically set to "other". However,
            users can specify a different default by passing an argument here.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Raises
        ------
        IOError
            If root file directory doesn't exist.
        ValueError
            If required files are missing or multiple files for any polarization
            are included in filelist.
            If there is no recognized key for visibility weights in the flags_file.

        """
        from . import fhd

        if isinstance(vis_files, list | tuple | np.ndarray) and isinstance(
            vis_files[0], list | tuple | np.ndarray
        ):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        fhd_obj = fhd.FHD()
        fhd_obj.read_fhd(vis_files, params_file=params_file, **kwargs)
        self._convert_from_filetype(fhd_obj)
        del fhd_obj

    def read_mir(self, filepath, **kwargs):
        """
        Read in data from an SMA MIR file.

        Note that with the exception of filepath, most of the remaining parameters are
        used to sub-select a range of data.

        Parameters
        ----------
        filepath : str
            The file path to the MIR folder to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) specifying baselines
            to include when reading data in to the object.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
        catalog_names : str or array-like of str
            The names of the phase centers (sources) to include when reading data into
            the object, which should match exactly in spelling and capitalization.
        corrchunk : int or array-like of int
            Correlator "chunk" (spectral window) to include when reading data into the
            object, where 0 corresponds to the pseudo-continuum channel.
        receivers : str or array-like of str
            The names of the receivers ("230", "240", "345", "400") to include when
            reading data into the object.
        sidebands : str or array-like of str
            The names of the sidebands ("l" for lower, "u" for upper) to include when
            reading data into the object.
        select_where : tuple or list of tuples, optional
            Argument to pass to the `MirParser.select` method, which will downselect
            which data is read into the object.
        apply_flags : bool
            If set to True, apply "wideband" flags to the visibilities, which are
            recorded by the realtime system to denote when data are expected to be bad
            (e.g., antennas not on source, dewar warm). Default it true.
        apply_tsys : bool
            If set to False, data are returned as correlation coefficients (normalized
            by the auto-correlations). Default is True, which instead scales the raw
            visibilities and forward-gain of the antenna to produce values in Jy
            (uncalibrated).
        apply_dedoppler : bool
            If set to True, data will be corrected for any doppler-tracking performed
            during observations, and brought into the topocentric rest frame (default
            for UVData objects). Default is False.
        pseudo_cont : boolean
            Read in only pseudo-continuum values. Default is false.
        rechunk : int
            Number of channels to average over when reading in the dataset. Optional
            argument, typically required to be a power of 2.
        compass_soln : str
            Optional argument, specifying the path of COMPASS-derived flagging and
            bandpass gains solutions, which are applied prior to any potential spectral
            averaging (as triggered by using the `rechunk` keyword).
        swarm_only : bool
            By default, only SMA SWARM data is loaded. If set to false, this will also
            enable loading of older ASIC data.
        codes_check : bool
            Option to check the cross-check the internal metadata codes, and deselect
            data without valid matches, useful for automatically handling various data
            recording issues. Default is True.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvw coordinates match antenna positions does not pass.
        allow_flex_pol : bool
            If only one polarization per spectral window is read (and the polarization
            differs from window to window), allow for the `UVData` object to use
            "flexible polarization", which compresses the polarization-axis of various
            attributes to be of length 1, sets the `flex_spw_polarization_array`
            attribute to define the polarization per spectral window. Default is True.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array.  Default is True.

        """
        from . import mir

        mir_obj = mir.Mir()
        mir_obj.read_mir(filepath, **kwargs)
        self._convert_from_filetype(mir_obj)
        del mir_obj

    def read_miriad(self, filepath, **kwargs):
        """
        Read in data from a miriad file.

        Parameters
        ----------
        filepath : str
            The miriad root directory to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to read into the object.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when reading data into the object. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when reading data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `bls` or `polarizations` parameters, if it is a ValueError will be raised.
        polarizations : array_like of int or str, optional
            List of polarization integers or strings to read-in. e.g. ['xx', 'yy', ...]
        time_range : list of float, optional
            len-2 list containing min and max range of times in Julian Date to
            include when reading data into the object. e.g. [2458115.20, 2458115.40]
        read_data : bool
            Read in the uvws, times, visibility and flag data. If set to False,
            only the metadata that can be read quickly (without reading the data)
            will be read in. For Miriad, some of the normally required metadata
            are not fast to read in (e.g. uvws, times) so will not be read in
            if this keyword is False. Therefore, setting read_data to False
            results in an incompletely defined object (check will not pass).
        projected : bool or None
            Option to force the dataset to be labelled as projected or unprojected
            regardless of the evidence in the file. The default is None which means that
            the projection will be set based on the file contents. Be careful setting
            this keyword unless you are confident about the contents of the file.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing. Default is True.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        calc_lst : bool
            Recalculate the LST values that are present within the file, useful in
            cases where the "online" calculate values have precision or value errors.
            Default is True.
        fix_old_proj : bool
            Applies a fix to uvw-coordinates and phasing, assuming that the old `phase`
            method was used prior to writing the data, which had errors of the order of
            one part in 1e4 - 1e5. See the phasing memo for more details.
        fix_use_ant_pos : bool
            If setting `fix_old_proj` to True, use the antenna positions to derive the
            correct uvw-coordinates rather than using the baseline vectors. Default is
            True.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        default_mount_type : str
            If not recorded in the data set or telescope is unknown to pyuvdata, the
            `Telescope.mount_type` parameter is automatically set to "other". However,
            users can specify a different default by passing an argument here.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Raises
        ------
        IOError
            If root file directory doesn't exist.
        ValueError
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.
            If the metadata are not internally consistent.

        """
        from . import miriad

        if isinstance(filepath, list | tuple | np.ndarray):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        miriad_obj = miriad.Miriad()
        miriad_obj.read_miriad(filepath, **kwargs)
        self._convert_from_filetype(miriad_obj)
        del miriad_obj

    def read_ms(self, filepath, **kwargs):
        """
        Read in a casa measurement set.

        Parameters
        ----------
        filepath : str
            The measurement set root directory to read from.
        data_column : str
            name of CASA data column to read into data_array. Options are:
            "DATA", "MODEL", or "CORRECTED_DATA". Default is "DATA".
        pol_order : str or None
            Option to specify polarizations order convention, options are
            "CASA", "AIPS", or None (no reordering). Default is "AIPS".
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        ignore_single_chan : bool
            Some measurement sets (e.g., those from ALMA) use single channel spectral
            windows for recording pseudo-continuum channels or storing other metadata
            in the track when the telescopes are not on source. Because of the way
            the object is strutured (where all spectral windows are assumed to be
            simultaneously recorded), this can significantly inflate the size and memory
            footprint of UVData objects. By default, single channel windows are ignored
            to avoid this issue, although they can be included if setting this parameter
            equal to True.
        raise_error : bool
            The measurement set format allows for different spectral windows and
            polarizations to have different metdata for the same time-baseline
            combination, but UVData objects do not. It also allows for timescales that
            are not supported by astropy. If any of these problems are detected, by
            default the reader will throw an error. However, if set to False, the reader
            will simply give a warning and try to do the best it can. If the problem is
            with differing metadata, it will use the first value read in the file as the
            "correct" metadata in the UVData object. If the problem is with the
            timescale, it will just assume UTC.
        read_weights : bool
            Read in the weights from the MS file, default is True. If false, the method
            will set the `nsamples_array` to the same uniform value (namely 1.0).
        allow_flex_pol : bool
            If only one polarization per spectral window is read (and the polarization
            differs from window to window), allow for the `UVData` object to use
            "flexible polarization", which compresses the polarization-axis of various
            attributes to be of length 1, sets the `flex_spw_polarization_array`
            attribute to define the polarization per spectral window.  Default is True.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        default_mount_type : str
            If not recorded in the data set or telescope is unknown to pyuvdata, the
            `Telescope.mount_type` parameter is automatically set to "other". However,
            users can specify a different default by passing an argument here.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Raises
        ------
        IOError
            If root file directory doesn't exist.
        ValueError
            If the `data_column` is not set to an allowed value.
            If the data are have multiple subarrays or are multi source or have
            multiple spectral windows.
            If the data have multiple data description ID values.

        """
        if isinstance(filepath, list | tuple | np.ndarray):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        from . import ms

        ms_obj = ms.MS()
        ms_obj.read_ms(filepath, **kwargs)
        self._convert_from_filetype(ms_obj)
        del ms_obj

    def read_mwa_corr_fits(self, filelist, **kwargs):
        """
        Read in MWA correlator gpu box files.

        The default settings remove some of the instrumental effects in the bandpass
        by dividing out the coarse band shape (for legacy data only) and the digital
        gains, and applying a cable length correction.
        If the desired output is raw correlator data, set remove_dig_gains=False,
        remove_coarse_band=False, correct_cable_len=False, and
        phase_to_pointing_center=False.

        Parameters
        ----------
        filelist : list of str
            The list of MWA correlator files to read from. Must include at
            least one fits file and only one metafits file per data set.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided. Ignored if read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided. Ignored if read_data is False.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]). The
            ordering of the numbers within the tuple does not matter.
            Note that this is different than what can be passed to the parameter
            of the same name on `select` and other read methods -- this parameter
            does not accept 3-tuples or baseline numbers.
            Ignored if read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array. Ignored if
            read_data is False.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the object.
        spws : array_like of int, optional
            The spectral window numbers to keep in the object.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`, `lsts`, or `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
            Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do not have data associated with them after the select option.
        use_aoflagger_flags : bool
            Option to use aoflagger mwaf flag files. Defaults to true if aoflagger
            flag files are submitted, False otherwise.
        remove_dig_gains : bool
            Option to divide out digital gains, default is True.
        remove_coarse_band : bool
            Option to divide out coarse band shape, default is True.
        correct_cable_len : bool
            Option to apply a cable delay correction, default is True.
        correct_van_vleck : bool
            Option to apply a van vleck correction, default is False.
        cheby_approx : bool
            Only used if correct_van_vleck is True. Option to implement the van
            vleck correction with a chebyshev polynomial approximation, default is True.
        flag_small_auto_ants : bool
            Only used if correct_van_vleck is True. Option to completely flag any
            antenna for which the autocorrelation falls below a threshold found by
            the Van Vleck correction to indicate bad data. Specifically, the
            threshold used is 0.5 * integration_time * channel_width. If set to False,
            only the times and frequencies at which the auto is below the
            threshold will be flagged for the antenna, default is True.
        phase_to_pointing_center : bool
            Option to phase to the observation pointing center, default is False.
        propagate_coarse_flags : bool
            Option to propagate flags for missing coarse channel integrations
            across frequency, default is True.
        flag_init: bool
            Set to True in order to do routine flagging of coarse channel edges,
            start or end integrations, or the center fine channel of each coarse
            channel. See associated keywords, default is True.
        edge_width: float
            Only used if flag_init is True. The width to flag on the edge of
            each coarse channel, in hz. Errors if not equal to integer multiple
            of channel_width. Set to 0 for no edge flagging. Default is 80 kHz (80e3).
        start_flag: float or str
            Only used if flag_init is True. The number of seconds to flag at the
            beginning of the observation. Set to 0 for no flagging. Default is
            "goodtime", which uses information in the metafits file to determine
            the length of time that should be flagged. Errors if input is not a
            float or "goodtime". Errors if float input is not equal to an
            integer multiple of the integration time.
        end_flag: floats
            Only used if flag_init is True. The number of seconds to flag at the
            end of the observation. Set to 0 for no flagging, which is the default.
            Errors if not equal to an integer multiple of the integration time.
        flag_dc_offset: bool
            Only used if flag_init is True. Set to True to flag the center fine
            channel of each coarse channel, default is True.
        remove_flagged_ants : bool
            Option to perform a select to remove antennas flagged in the metafits
            file. If correct_van_vleck and flag_small_auto_ants are both True then
            antennas flagged by the Van Vleck correction are also removed.
            Default is True.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object. Default is True.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128
            (double-precision real and imaginary). Default is np.complex64.
        nsample_array_dtype : numpy dtype
            Datatype to store the output nsample_array as. Must be either
            np.float64 (double-precision), np.float32 (single-precision), or
            np.float16 (half-precision). Half-precision is only recommended for
            cases where no sampling or averaging of baselines will occur,
            because round-off errors can be quite large (~1e-3). Default is np.float32.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run).
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass. Default is False.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Raises
        ------
        ValueError
            If required files are missing or multiple files metafits files are
            included in filelist.
            If files from different observations are included in filelist.
            If files in fileslist have different fine channel widths
            If file types other than fits, metafits, and mwaf files are included
            in filelist.

        """
        from . import mwa_corr_fits

        if isinstance(filelist[0], list | tuple | np.ndarray):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        corr_obj = mwa_corr_fits.MWACorrFITS()
        corr_obj.read_mwa_corr_fits(filelist, **kwargs)
        self._convert_from_filetype(corr_obj)
        del corr_obj

    def read_uvfits(self, filename, **kwargs):
        """
        Read in header, metadata and data from a uvfits file.

        Parameters
        ----------
        filename : str
            The uvfits file to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided. Ignored if read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided. Ignored if read_data is False.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when reading data into the object. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None. Ignored if read_data is False.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when reading data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised. Ignored if read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the object.
        spws : array_like of int, optional
            The spectral window numbers to keep in the object.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`, `lsts`, or `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to keep in the object, must be
            length 2. Some of the times in the object should fall between the
            first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when reading data into the
            object. This is not commonly used. Ignored if read_data is False.
        phase_center_ids : array_like of int, optional
            Phase center IDs to include when reading data into the object (effectively
            a selection on baseline-times). Cannot be used with catalog_names.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to include when reading data into
            the object, which should match exactly in spelling and capitalization.
            Cannot be used with phase_center_ids.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do not have data associated with them after the select option.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        fix_old_proj : bool
            Applies a fix to uvw-coordinates and phasing, assuming that the old `phase`
            method was used prior to writing the data, which had errors of the order of
            one part in 1e4 - 1e5. See the phasing memo for more details. Default is
            False.
        fix_use_ant_pos : bool
            If setting `fix_old_proj` to True, use the antenna positions to derive the
            correct uvw-coordinates rather than using the baseline vectors. Default is
            True.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        default_mount_type : str
            If not recorded in the data set or telescope is unknown to pyuvdata, the
            `Telescope.mount_type` parameter is automatically set to "other". However,
            users can specify a different default by passing an argument here.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Raises
        ------
        IOError
            If filename doesn't exist.
        ValueError
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.
            If the metadata are not internally consistent or missing.

        """
        from . import uvfits

        if isinstance(filename, list | tuple | np.ndarray):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        uvfits_obj = uvfits.UVFITS()
        uvfits_obj.read_uvfits(filename, **kwargs)
        self._convert_from_filetype(uvfits_obj)
        del uvfits_obj

    def read_uvh5(self, filename, **kwargs):
        """
        Read in data from a UVH5 file.

        Parameters
        ----------
        filename : str
             The UVH5 file to read from.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided. Ignored if read_data is False.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided. Ignored if read_data is False.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when reading data into the object. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None. Ignored if read_data is False.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when reading data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised. Ignored if read_data is False.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array. Ignored if
            read_data is False.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the object.
        spws : array_like of int, optional
            The spectral window numbers to keep in the object.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`, `lsts`, or `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
            Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
            Ignored if read_data is False.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when reading data into the
            object. This is not commonly used. Ignored if read_data is False.
        phase_center_ids : array_like of int, optional
            Phase center IDs to include when reading data into the object (effectively
            a selection on baseline-times). Cannot be used with catalog_names.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to include when reading data into
            the object, which should match exactly in spelling and capitalization.
            Cannot be used with phase_center_ids.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do not have data associated with them after the select option.
        read_data : bool
            Read in the visibility, nsample and flag data. If set to False, only
            the metadata will be read in. Setting read_data to False results in
            a metadata only object.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128 (double-
            precision real and imaginary). Only used if the datatype of the visibility
            data on-disk is not 'c8' or 'c16'.
        multidim_index : bool
            If True, attempt to index the HDF5 dataset simultaneously along all
            data axes. Otherwise index one axis at-a-time. This only works if
            data selection is sliceable along all but one axis. If indices are
            not well-matched to data chunks, this can be slow. Default is False.
        remove_flex_pol : bool
            If True and if the file is a flex_pol file, convert back to a standard
            UVData object. Default is True.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        fix_old_proj : bool
            Applies a fix to uvw-coordinates and phasing, assuming that the old `phase`
            method was used prior to writing the data, which had errors of the order of
            one part in 1e4 - 1e5. See the phasing memo for more details. Default is
            to apply the correction if the attributes `phase_center_app_ra` and
            `phase_center_app_dec` are missing (as they were introduced alongside the
            new phasing method).
        fix_use_ant_pos : bool
            If setting `fix_old_proj` to True, use the antenna positions to derive the
            correct uvw-coordinates rather than using the baseline vectors. Default is
            True.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        default_mount_type : str
            If not recorded in the data set or telescope is unknown to pyuvdata, the
            `Telescope.mount_type` parameter is automatically set to "other". However,
            users can specify a different default by passing an argument here.
        blt_order : tuple of str or "determine", optional
            The order of the baseline-time axis *in the file*. This can be determined,
            or read directly from file, however since it has been optional in the past,
            many existing files do not contain it in the metadata.
            Some reading operations are significantly faster if this is known, so
            providing it here can provide a speedup. Default is to try and read it from
            file, and if not there, just leave it as None. Set to "determine" to
            auto-detect the blt_order from the metadata (takes extra time to do so).
        blts_are_rectangular : bool, optional
            Whether the baseline-time axis is rectangular. This can be read from
            metadata in new files, but many old files do not contain it. If not
            provided, the rectangularity will be determined from the data. This is a
            non-negligible operation, so if you know it, it can be provided here to
            speed up reading.
        time_axis_faster_than_bls : bool, optional
            If blts are rectangular, this variable specifies whether the time axis is
            the fastest-moving virtual axis. Various reading functions benefit from
            knowing this, so if it is known, it can be provided here to speed up
            reading. It will be determined from the data if not provided.
        recompute_nbls : bool, optional
            Whether to recompute the number of unique baselines from the data. Before
            v1.2 of the UVH5 spec, it was possible to have an incorrect number of
            baselines in the header without error, so this provides an opportunity to
            rectify it. Old HERA files (< March 2023) may have this issue, but in this
            case the correct number of baselines can be computed more quickly than by
            fully re=computing, and so we do this.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If filename doesn't exist.
        ValueError
            If the data_array_dtype is not a complex dtype.
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.

        """
        from . import uvh5

        if isinstance(filename, list | tuple | np.ndarray):
            raise ValueError(
                "Reading multiple files from class specific "
                "read functions is no longer supported. "
                "Use the generic `uvdata.read` function instead."
            )

        uvh5_obj = uvh5.UVH5()
        uvh5_obj.read_uvh5(filename, **kwargs)
        self._convert_from_filetype(uvh5_obj)
        del uvh5_obj

    def read(
        self,
        filename,
        *,
        axis=None,
        file_type=None,
        read_data=True,
        skip_bad_files=False,
        background_lsts=True,
        astrometry_library=None,
        ignore_name=False,
        # selecting parameters
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        catalog_names=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        blt_inds=None,
        phase_center_ids=None,
        keep_all_metadata=True,
        # checking parameters
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=True,
        default_mount_type="other",
        # file-type specific parameters
        # miriad
        projected=None,
        correct_lat_lon=True,
        calc_lst=True,
        # Miriad, UVFITS & UVH5
        fix_old_proj=None,
        fix_use_ant_pos=True,
        # FHD
        params_file=None,
        obs_file=None,
        flags_file=None,
        layout_file=None,
        settings_file=None,
        # MS
        data_column="DATA",
        pol_order="AIPS",
        ignore_single_chan=True,
        raise_error=True,
        read_weights=True,
        # MS & MIR
        allow_flex_pol=True,
        # uvh5
        multidim_index=False,
        remove_flex_pol=True,
        blt_order: tuple[str] | Literal["determine"] | None = None,
        blts_are_rectangular: bool | None = None,
        time_axis_faster_than_bls: bool | None = None,
        # uvh5 & mwa_corr_fits
        data_array_dtype=np.complex128,
        # mwa_corr_fits
        use_aoflagger_flags=None,
        remove_dig_gains=True,
        remove_coarse_band=True,
        correct_cable_len=True,
        correct_van_vleck=False,
        cheby_approx=True,
        flag_small_auto_ants=True,
        propagate_coarse_flags=True,
        flag_init=True,
        edge_width=80e3,
        start_flag="goodtime",
        end_flag=0.0,
        flag_dc_offset=True,
        remove_flagged_ants=True,
        phase_to_pointing_center=False,
        nsample_array_dtype=np.float32,
        # MIR
        corrchunk=None,
        receivers=None,
        sidebands=None,
        mir_select_where=None,
        apply_tsys=True,
        apply_flags=True,
        apply_dedoppler=False,
        pseudo_cont=False,
        rechunk=None,
        compass_soln=None,
        swarm_only=True,
        codes_check=True,
        recompute_nbls: bool | None = None,
    ):
        """
        Read a generic file into a UVData object.

        This method supports a number of different types of files.
        Universal parameters (required and optional) are listed directly below,
        followed by parameters used by all file types related to phasing, selecting on
        read (partial read) and checking. Each file type also has its own set of
        optional parameters that are listed at the end of this docstring.

        Note that select on read (partial reading) is not always faster than
        reading an entire file and then downselecting. Which approach is faster
        depends on the fraction of data that is selected as well on the relationship
        between the selection and the internal data ordering in the file. When
        the select is on a small area of the file or has a regular stride it can
        be much faster to do the select on read, but in other cases it can be slower.
        Select on read does generally reduce the memory footprint.

        Parameters
        ----------
        filename : str or array_like of str
            The file(s) or list(s) (or array(s)) of data files to read from.
        file_type : str
            One of ['uvfits', 'miriad', 'ms', 'uvh5', 'fhd', 'mwa_corr_fits', 'mir']
            or None. If None, the code attempts to guess what the file type is.
            For miriad and ms types, this is based on the standard directory
            structure. For FHD, uvfits and uvh5 files it's based on file
            extensions (FHD: .sav, .txt; uvfits: .uvfits; uvh5: .uvh5).
            Note that if a list of datasets is passed, the file type is
            determined from the first dataset.
        axis : str
            Axis to concatenate files along. This enables fast concatenation
            along the specified axis without the normal checking that all other
            metadata agrees. This method does not guarantee correct resulting
            objects. Please see the docstring for fast_concat for details.
            Allowed values are: 'blt', 'freq', 'polarization'. Only used if
            multiple files are passed.
        read_data : bool
            Read in the data. Not used if file_type is 'ms' or 'mir'.
            If set to False, only the metadata will be read in. Setting read_data to
            False results in a metdata only object.
        skip_bad_files : bool
            Option when reading multiple files to catch read errors such that
            the read continues even if one or more files are corrupted. Files
            that produce errors will be printed. Default is False (files will
            not be skipped).
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        astrometry_library : str
            Library used for calculating LSTs. Allowed options are 'erfa' (which uses
            the pyERFA), 'novas' (which uses the python-novas library), and 'astropy'
            (which uses the astropy utilities). Default is erfa unless the
            telescope_location frame is MCMF (on the moon), in which case the default
            is astropy.
        ignore_name : bool
            Only relevant when reading in multiple files, which are concatenated into a
            single UVData object. Option to ignore the name of the phase center when
            combining multiple files, which would otherwise result in an error being
            raised because of attributes not matching. Doing so effectively adopts the
            name found in the first file read in. Default is False.

        Selecting
        ---------
        antenna_nums : array_like of int, optional
            The antennas numbers to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when reading data into the object
            (antenna positions and names for the removed antennas will be retained
            unless `keep_all_metadata` is False). This cannot be provided if
            `antenna_nums` is also provided.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include when reading data into the object.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
            Note that this keyword is not supported for MWA correlator FITS files.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when reading data into the object. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None. Note that for MWA correlator FITS files, this can only be a
            list of antenna number 2-tuples.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to include when reading data into
            the object, which should match exactly in spelling and capitalization.
            Cannot be used with phase_center_ids.
        frequencies : array_like of float, optional
            The frequencies to include when reading data into the object, each
            value passed here should exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include when reading data into the object.
        spws : array_like of int, optional
            The spectral window numbers to keep in the object.
        times : array_like of float, optional
            The times to include when reading data into the object, each value
            passed here should exist in the time_array in the file. Cannot be used with
            `time_range`, `lsts`, or `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when reading data into
            the object, must be length 2. Some of the times in the file should
            fall between the first and last elements.
            Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int, optional
            The polarizations numbers to include when reading data into the
            object, each value passed here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when reading data into the
            object. This is not commonly used.
        phase_center_ids : array_like of int, optional
            Phase center IDs to include when reading data into the object (effectively
            a selection on baseline-times). Cannot be used with catalog_names.
        keep_all_metadata : bool
            Option to keep all the metadata associated with antennas, even those
            that do not have data associated with them after the select option.

        Checking
        --------
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            after after reading in the file (the default is True,
            meaning the check will be run). Ignored if read_data is False.
        check_extra : bool
            Option to check optional parameters as well as required ones (the
            default is True, meaning the optional parameters will be checked).
            Ignored if read_data is False.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters after
            reading in the file (the default is True, meaning the acceptable
            range check will be done). Ignored if read_data is False.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass. Default is False.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is True.
        default_mount_type : str
            If not recorded in the data set or telescope is unknown to pyuvdata, the
            `Telescope.mount_type` parameter is automatically set to "other". However,
            users can specify a different default by passing an argument here. Note that
            this parameter is not used for reading in MIR and MWACorrFITS files, since
            those only originate from a single known telescope per format.

        Miriad
        ------
        projected : bool or None
            Option to force the dataset to be labelled as projected or unprojected
            regardless of the evidence in the file. The default is None which means that
            the projection will be set based on the file contents. Be careful setting
            this keyword unless you are confident about the contents of the file.
        correct_lat_lon : bool
            Option to update the latitude and longitude from the known_telescopes
            list if the altitude is missing. Default is True.
        calc_lst : bool
            Recalculate the LST values that are present within the file, useful in
            cases where the "online" calculate values have precision or value errors.
            Default is True.
        fix_old_proj : bool
            Applies a fix to uvw-coordinates and phasing, assuming that the old `phase`
            method was used prior to writing the data, which had errors of the order of
            one part in 1e4 - 1e5. See the phasing memo for more details. Default is
            False.
        fix_use_ant_pos : bool
            If setting `fix_old_proj` to True, use the antenna positions to derive the
            correct uvw-coordinates rather than using the baseline vectors. Default is
            True.

        UVFITS
        ------
        fix_old_proj : bool
            Applies a fix to uvw-coordinates and phasing, assuming that the old `phase`
            method was used prior to writing the data, which had errors of the order of
            one part in 1e4 - 1e5. See the phasing memo for more details. Default is
            False.
        fix_use_ant_pos : bool
            If setting `fix_old_proj` to True, use the antenna positions to derive the
            correct uvw-coordinates rather than using the baseline vectors. Default is
            True.

        FHD
        ---
        params_file : str
            The FHD params save file. Required.
        obs_file : str
            The FHD obs save file. Required if `read_data` is False.
        flags_file : str
            The FHD data (or model) flag save file. Required if `read_data` is True.
        layout_file : str
            The FHD layout save file. Required for correct antenna metadata.
        settings_file : str
            The FHD settings text file. Required for full history information.

        MS
        --
        data_column : str
            name of CASA data column to read into data_array. Options are:
            'DATA', 'MODEL', or 'CORRECTED_DATA'. Default is "DATA".
        pol_order : str or None
            Option to specify polarizations order convention, options are
            "CASA", "AIPS", or None (no reordering). Default is "AIPS".
        ignore_single_chan : bool
            Option to ignore single channel spectral windows in measurement sets to
            limit object size. Some measurement sets (e.g., those from ALMA) use single
            channel spectral windows for recording pseudo-continuum channels or storing
            other metadata in the track when the telescopes are not on source. Because
            of the way the object is strutured (where all spectral windows are assumed
            to be simultaneously recorded), this can significantly inflate the size and
            memory footprint of UVData objects. By default, single channel windows are
            ignored to avoid this issue, they can be included by setting this parameter
            to True.
        raise_error : bool
            The measurement set format allows for different spectral windows and
            polarizations to have different metdata for the same time-baseline
            combination, but UVData objects do not. It also allows for timescales that
            are not supported by astropy. If any of these problems are detected, by
            default the reader will throw an error. However, if set to False, the reader
            will simply give a warning and try to do the best it can. If the problem is
            with differing metadata, it will use the first value read in the file as the
            "correct" metadata in the UVData object. If the problem is with the
            timescale, it will just assume UTC.
        read_weights : bool
            Read in the weights from the MS file, default is True. If false, the method
            will set the `nsamples_array` to the same uniform value (namely 1.0).
        allow_flex_pol : bool
            If only one polarization per spectral window is read (and the polarization
            differs from window to window), allow for the `UVData` object to use
            "flexible polarization", which compresses the polarization-axis of various
            attributes to be of length 1, sets the `flex_spw_polarization_array`
            attribute to define the polarization per spectral window. Default is True.

        UVH5
        ----
        multidim_index : bool
            If True, attempt to index the HDF5 dataset simultaneously along all data
            axes. Otherwise index one axis at-a-time. This only works if data selection
            is sliceable along all but one axis. If indices are not well-matched to
            data chunks, this can be slow. Default is False.
        remove_flex_pol : bool
            If True and if the file is a flex_pol file, convert back to a standard
            UVData object. Default is True.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128 (double-
            precision real and imaginary). Only used if the datatype of the visibility
            data on-disk is not 'c8' or 'c16'. Default is np.complex128.
        blt_order : tuple of str or "determine", optional
            The order of the baseline-time axis *in the file*. This can be determined,
            or read directly from file, however since it has been optional in the past,
            many existing files do not contain it in the metadata.
            Some reading operations are significantly faster if this is known, so
            providing it here can provide a speedup. Default is to try and read it from
            file, and if not there, just leave it as None. Set to "determine" to
            auto-detect the blt_order from the metadata (takes extra time to do so).
        blts_are_rectangular : bool, optional
            Whether the baseline-time axis is rectangular. This can be read from
            metadata in new files, but many old files do not contain it. If not
            provided, the rectangularity will be determined from the data. This is a
            non-negligible operation, so if you know it, it can be provided here to
            speed up reading.
        time_axis_faster_than_bls : bool, optional
            If blts are rectangular, this variable specifies whether the time axis is
            the fastest-moving virtual axis. Various reading functions benefit from
            knowing this, so if it is known, it can be provided here to speed up
            reading. It will be determined from the data if not provided.
        recompute_nbls : bool, optional
            Whether to recompute the number of unique baselines from the data. Before
            v1.2 of the UVH5 spec, it was possible to have an incorrect number of
            baselines in the header without error, so this provides an opportunity to
            rectify it. Old HERA files (< March 2023) may have this issue, but in this
            case the correct number of baselines can be computed more quickly than by
            fully re=computing, and so we do this.
        fix_old_proj : bool
            Applies a fix to uvw-coordinates and phasing, assuming that the old `phase`
            method was used prior to writing the data, which had errors of the order of
            one part in 1e4 - 1e5. See the phasing memo for more details. Default is
            False, unless reading a UVH5 file that is missing the `phase_center_app_ra`
            and `phase_center_app_dec` attributes (as these were introduced at the same
            time as the new `phase` method), in which case the default is True.
        fix_use_ant_pos : bool
            If setting `fix_old_proj` to True, use the antenna positions to derive the
            correct uvw-coordinates rather than using the baseline vectors. Default is
            True.

        MWA FITS
        --------
        use_aoflagger_flags : bool
            Option to use aoflagger mwaf flag files. Defaults to true if aoflagger
            flag files are submitted, False otherwise.
        remove_dig_gains : bool
            Option to divide out digital gains, default is True.
        remove_coarse_band : bool
            Option to divide out coarse band shape, default is True.
        correct_cable_len : bool
            Option to apply a cable delay correction, default is True.
        correct_van_vleck : bool
            Option to apply a van vleck correction, default is False.
        cheby_approx : bool
            Only used if correct_van_vleck is True. Option to implement the van
            vleck correction with a chebyshev polynomial approximation, default is True.
        flag_small_auto_ants : bool
            Only used if correct_van_vleck is True. Option to completely flag any
            antenna for which the autocorrelation falls below a threshold found by
            the Van Vleck correction to indicate bad data. Specifically, the
            threshold used is 0.5 * integration_time * channel_width. If set to False,
            only the times and frequencies at which the auto is below the
            threshold will be flagged for the antenna, default is True.
        phase_to_pointing_center : bool
            Option to phase to the observation pointing center, default is False.
        propagate_coarse_flags : bool
            Option to propagate flags for missing coarse channel integrations
            across frequency, default is True.
        flag_init: bool
            Set to True in order to do routine flagging of coarse channel edges,
            start or end integrations, or the center fine channel of each coarse
            channel. See associated keywords, default is True.
        edge_width: float
            Only used if flag_init is True. The width to flag on the edge of
            each coarse channel, in hz. Errors if not equal to integer multiple
            of channel_width. Set to 0 for no edge flagging. Default is 80 kHz (80e3).
        start_flag: float or str
            Only used if flag_init is True. The number of seconds to flag at the
            beginning of the observation. Set to 0 for no flagging. Default is
            "goodtime", which uses information in the metafits file to determine
            the length of time that should be flagged. Errors if input is not a
            float or "goodtime". Errors if float input is not equal to an
            integer multiple of the integration time.
        end_flag: floats
            Only used if flag_init is True. The number of seconds to flag at the
            end of the observation. Set to 0 for no flagging, which is the default.
            Errors if not equal to an integer multiple of the integration time.
        flag_dc_offset: bool
            Only used if flag_init is True. Set to True to flag the center fine
            channel of each coarse channel, default is True.
        remove_flagged_ants : bool
            Option to perform a select to remove antennas flagged in the metafits
            file. If correct_van_vleck and flag_small_auto_ants are both True then
            antennas flagged by the Van Vleck correction are also removed.
            Default is True.
        data_array_dtype : numpy dtype
            Datatype to store the output data_array as. Must be either
            np.complex64 (single-precision real and imaginary) or np.complex128
            (double-precision real and imaginary). Default is np.complex128,
            **note that this is a different default than on `read_mwa_corr_fits`**.
        nsample_array_dtype : numpy dtype
            Datatype to store the output nsample_array as. Must be either
            np.float64 (double-precision), np.float32 (single-precision), or
            np.float16 (half-precision). Half-precision is only recommended for
            cases where no sampling or averaging of baselines will occur,
            because round-off errors can be quite large (~1e-3). Default is np.float32.

        MIR
        ---
        corrchunk : int or array-like of int, optional
            Correlator "chunk" (spectral window) to include when reading data into the
            object, where 0 corresponds to the pseudo-continuum channel.
        receivers : str or array-like of str, optional
            The names of the receivers ("230", "240", "345", "400") to include when
            reading data into the object.
        sidebands : str or array-like of str, optional
            The names of the sidebands ("l" for lower, "u" for upper) to include when
            reading data into the object.
        mir_select_where : tuple or list of tuples, optional
            Argument to pass to the `MirParser.select` method, which will downselect
            which data is read into the object.
        apply_flags : bool
            If set to True, apply "wideband" flags to the visibilities, which are
            recorded by the realtime system to denote when data are expected to be bad
            (e.g., antennas not on source, dewar warm). Default is True.
        apply_tsys : bool
            If set to False, data are returned as correlation coefficients (normalized
            by the auto-correlations). Default is True, which instead scales the raw
            visibilities and forward-gain of the antenna to produce values in Jy
            (uncalibrated).
        apply_dedoppler : bool
            If set to True, data will be corrected for any doppler-tracking performed
            during observations, and brought into the topocentric rest frame (default
            for UVData objects). Default is False.
        allow_flex_pol : bool
            If only one polarization per spectral window is read (and the polarization
            differs from window to window), allow for the `UVData` object to use
            "flexible polarization", which compresses the polarization-axis of various
            attributes to be of length 1, sets the `flex_spw_polarization_array`
            attribute to define the polarization per spectral window. Default is True.
        swarm_only : bool
            By default, only SMA SWARM data is loaded. If set to false, this will also
            enable loading of older ASIC data.
        compass_soln : str, optional
            Optional argument, specifying the path of COMPASS-derived flagging and
            bandpass gains solutions, which are applied prior to any potential spectral
            averaging (as triggered by using the `rechunk` keyword).
        codes_check : bool
            Option to check the cross-check the internal MIR metadata, and deselect
            data without valid matches, useful for automatically handling various data
            recording issues. Default is True. Note this is different than the various
            checks done on the UVData object itself (controlled by other keywords listed
            here).

        Raises
        ------
        ValueError
            If the file_type is not set and cannot be determined from the file name.
            If incompatible select keywords are set (e.g. `ant_str` with other
            antenna selectors, `times` and `time_range`) or select keywords
            exclude all data or if keywords are set to the wrong type.
            If phase_center_radec is not None and is not length 2.

        """
        if isinstance(filename, list | tuple | np.ndarray):
            # this is either a list of separate files to read or a list of
            # FHD files or MWA correlator FITS files
            if isinstance(filename[0], list | tuple | np.ndarray):
                if file_type is None:
                    # this must be a list of lists of FHD or MWA correlator FITS
                    _, extension = os.path.splitext(filename[0][0])
                    if extension in [".sav", ".txt"]:
                        file_type = "fhd"
                    elif (
                        extension == ".fits"
                        or extension == ".metafits"
                        or extension == ".mwaf"
                    ):
                        file_type = "mwa_corr_fits"
                multi = True
            else:
                if file_type is None:
                    _, extension = os.path.splitext(filename[0])
                    if extension in [".sav", ".txt"]:
                        file_type = "fhd"
                    elif (
                        extension == ".fits"
                        or extension == ".metafits"
                        or extension == ".mwaf"
                    ):
                        file_type = "mwa_corr_fits"

                if file_type == "fhd" or file_type == "mwa_corr_fits":
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

            if isinstance(file_test, str) and not os.path.exists(file_test):
                # Adding this b/c otherwise you get a "filetype not determined" error
                # that I (Karto) have dumbly lost a lot of time to.
                raise FileNotFoundError(f"File not found, check path for: {file_test}")

            if os.path.isdir(file_test):
                # it's a directory, so it's either miriad, mir, or ms file type
                if os.path.exists(os.path.join(file_test, "vartable")):
                    # It's miriad.
                    file_type = "miriad"
                elif os.path.exists(os.path.join(file_test, "OBSERVATION")):
                    # It's a measurement set.
                    file_type = "ms"
                elif os.path.exists(os.path.join(file_test, "sch_read")):
                    # It's Submillimeter Array mir format.
                    file_type = "mir"

            else:
                _, extension = os.path.splitext(file_test)
                if extension == ".uvfits":
                    file_type = "uvfits"
                elif extension == ".uvh5":
                    file_type = "uvh5"
                elif extension == ".sav":
                    file_type = "fhd"

        if file_type is None:
            raise ValueError(
                "File type could not be determined, use the "
                "file_type keyword to specify the type."
            )
        if file_type == "fhd" and params_file is None:
            raise ValueError("The params_file must be passed for FHD files.")

        if time_range is not None and times is not None:
            raise ValueError("Only one of times and time_range can be provided.")

        if antenna_names is not None and antenna_nums is not None:
            raise ValueError(
                "Only one of antenna_nums and antenna_names can be provided."
            )

        if multi:
            file_num = 0
            file_warnings = ""
            unread = True
            f = filename[file_num]
            params_file_use = None
            obs_file_use = None
            flags_file_use = None
            layout_file_use = None
            settings_file_use = None
            if file_type == "fhd":
                n_files = len(filename)
                if (
                    not isinstance(params_file, list | tuple | np.ndarray)
                    or len(params_file) != n_files
                ):
                    raise ValueError(
                        "For multiple FHD files, the number of params_file values must "
                        "match the number of data file sets."
                    )
                if obs_file is not None:
                    if (
                        not isinstance(obs_file, list | tuple | np.ndarray)
                        or len(obs_file) != n_files
                    ):
                        raise ValueError(
                            "For multiple FHD files, if obs_file is passed, the number "
                            "of obs_file values must match the number of data file "
                            "sets."
                        )
                else:
                    obs_file = [None] * n_files

                if flags_file is not None:
                    if (
                        not isinstance(flags_file, list | tuple | np.ndarray)
                        or len(flags_file) != n_files
                    ):
                        raise ValueError(
                            "For multiple FHD files, if flags_file is passed, the "
                            "number of flags_file values must match the number of data "
                            "file sets."
                        )
                else:
                    flags_file = [None] * n_files

                if layout_file is not None:
                    if (
                        not isinstance(layout_file, list | tuple | np.ndarray)
                        or len(layout_file) != n_files
                    ):
                        raise ValueError(
                            "For multiple FHD files, if layout_file is passed, the "
                            "number of layout_file values must match the number of "
                            "data file sets."
                        )
                else:
                    layout_file = [None] * n_files

                if settings_file is not None:
                    if (
                        not isinstance(settings_file, list | tuple | np.ndarray)
                        or len(settings_file) != n_files
                    ):
                        raise ValueError(
                            "For multiple FHD files, if settings_file is passed, the "
                            "number of settings_file values must match the number of "
                            "data file sets."
                        )
                else:
                    settings_file = [None] * n_files

                params_file_use = params_file[file_num]
                obs_file_use = obs_file[file_num]
                flags_file_use = flags_file[file_num]
                layout_file_use = layout_file[file_num]
                settings_file_use = settings_file[file_num]

            while unread and file_num < len(filename):
                try:
                    self.read(
                        filename[file_num],
                        file_type=file_type,
                        read_data=read_data,
                        skip_bad_files=skip_bad_files,
                        background_lsts=background_lsts,
                        astrometry_library=astrometry_library,
                        # phasing parameters
                        fix_old_proj=fix_old_proj,
                        fix_use_ant_pos=fix_use_ant_pos,
                        # selecting parameters
                        antenna_nums=antenna_nums,
                        antenna_names=antenna_names,
                        ant_str=ant_str,
                        bls=bls,
                        frequencies=frequencies,
                        freq_chans=freq_chans,
                        spws=spws,
                        times=times,
                        time_range=time_range,
                        lsts=lsts,
                        lst_range=lst_range,
                        polarizations=polarizations,
                        blt_inds=blt_inds,
                        phase_center_ids=phase_center_ids,
                        keep_all_metadata=keep_all_metadata,
                        # checking parameters
                        run_check=run_check,
                        check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability,
                        strict_uvw_antpos_check=strict_uvw_antpos_check,
                        check_autos=check_autos,
                        fix_autos=fix_autos,
                        # file-type specific parameters
                        # multiple
                        default_mount_type=default_mount_type,
                        # miriad
                        projected=projected,
                        correct_lat_lon=correct_lat_lon,
                        calc_lst=calc_lst,
                        # FHD
                        params_file=params_file_use,
                        obs_file=obs_file_use,
                        flags_file=flags_file_use,
                        layout_file=layout_file_use,
                        settings_file=settings_file_use,
                        # MS
                        data_column=data_column,
                        pol_order=pol_order,
                        ignore_single_chan=ignore_single_chan,
                        raise_error=raise_error,
                        read_weights=read_weights,
                        # MS & MIR
                        allow_flex_pol=allow_flex_pol,
                        # uvh5
                        multidim_index=multidim_index,
                        remove_flex_pol=remove_flex_pol,
                        # uvh5 & mwa_corr_fits
                        data_array_dtype=data_array_dtype,
                        # mwa_corr_fits
                        use_aoflagger_flags=use_aoflagger_flags,
                        remove_dig_gains=remove_dig_gains,
                        remove_coarse_band=remove_coarse_band,
                        correct_cable_len=correct_cable_len,
                        correct_van_vleck=correct_van_vleck,
                        cheby_approx=cheby_approx,
                        flag_small_auto_ants=flag_small_auto_ants,
                        propagate_coarse_flags=propagate_coarse_flags,
                        flag_init=flag_init,
                        edge_width=edge_width,
                        start_flag=start_flag,
                        end_flag=end_flag,
                        flag_dc_offset=flag_dc_offset,
                        remove_flagged_ants=remove_flagged_ants,
                        phase_to_pointing_center=phase_to_pointing_center,
                        nsample_array_dtype=nsample_array_dtype,
                        # mir
                        corrchunk=corrchunk,
                        receivers=receivers,
                        sidebands=sidebands,
                        mir_select_where=mir_select_where,
                        apply_tsys=apply_tsys,
                        apply_flags=apply_flags,
                        apply_dedoppler=apply_dedoppler,
                        pseudo_cont=pseudo_cont,
                        rechunk=rechunk,
                        compass_soln=compass_soln,
                        swarm_only=swarm_only,
                        codes_check=codes_check,
                        # other
                        recompute_nbls=recompute_nbls,
                        time_axis_faster_than_bls=time_axis_faster_than_bls,
                        blts_are_rectangular=blts_are_rectangular,
                    )
                    unread = False
                except KeyError as err:
                    file_warnings = (
                        file_warnings + f"Failed to read {f} due to KeyError: {err}\n"
                    )
                    file_num += 1
                    if skip_bad_files is False:
                        raise
                except ValueError as err:
                    file_warnings = (
                        file_warnings + f"Failed to read {f} due to ValueError: {err}\n"
                    )
                    file_num += 1
                    if skip_bad_files is False:
                        raise
                except OSError as err:  # pragma: nocover
                    file_warnings = (
                        file_warnings + f"Failed to read {f} due to OSError: {err}\n"
                    )
                    file_num += 1
                    if skip_bad_files is False:
                        raise

            if unread is True:
                warnings.warn(
                    "########################################################\n"
                    "ALL FILES FAILED ON READ - NO READABLE FILES IN FILENAME\n"
                    "########################################################"
                )
                return

            uv_list = []
            if len(filename) > file_num + 1:
                for f in filename[file_num + 1 :]:
                    if file_type == "fhd":
                        params_file_use = params_file[file_num]
                        obs_file_use = obs_file[file_num]
                        flags_file_use = flags_file[file_num]
                        layout_file_use = layout_file[file_num]
                        settings_file_use = settings_file[file_num]

                    uv2 = UVData()
                    try:
                        uv2.read(
                            f,
                            file_type=file_type,
                            read_data=read_data,
                            skip_bad_files=skip_bad_files,
                            background_lsts=background_lsts,
                            astrometry_library=astrometry_library,
                            # phasing parameters
                            fix_old_proj=fix_old_proj,
                            fix_use_ant_pos=fix_use_ant_pos,
                            # selecting parameters
                            antenna_nums=antenna_nums,
                            antenna_names=antenna_names,
                            ant_str=ant_str,
                            bls=bls,
                            catalog_names=catalog_names,
                            frequencies=frequencies,
                            freq_chans=freq_chans,
                            spws=spws,
                            times=times,
                            time_range=time_range,
                            lsts=lsts,
                            lst_range=lst_range,
                            polarizations=polarizations,
                            blt_inds=blt_inds,
                            phase_center_ids=phase_center_ids,
                            keep_all_metadata=keep_all_metadata,
                            # checking parameters
                            run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                            strict_uvw_antpos_check=strict_uvw_antpos_check,
                            check_autos=check_autos,
                            fix_autos=fix_autos,
                            # file-type specific parameters
                            # multiple
                            default_mount_type=default_mount_type,
                            # miriad
                            projected=projected,
                            correct_lat_lon=correct_lat_lon,
                            calc_lst=calc_lst,
                            # FHD
                            params_file=params_file_use,
                            obs_file=obs_file_use,
                            flags_file=flags_file_use,
                            layout_file=layout_file_use,
                            settings_file=settings_file_use,
                            # MS
                            data_column=data_column,
                            pol_order=pol_order,
                            ignore_single_chan=ignore_single_chan,
                            raise_error=raise_error,
                            read_weights=read_weights,
                            # MS & MIR
                            allow_flex_pol=allow_flex_pol,
                            # uvh5
                            multidim_index=multidim_index,
                            remove_flex_pol=remove_flex_pol,
                            blts_are_rectangular=blts_are_rectangular,
                            time_axis_faster_than_bls=time_axis_faster_than_bls,
                            recompute_nbls=recompute_nbls,
                            # uvh5 & mwa_corr_fits
                            data_array_dtype=data_array_dtype,
                            # mwa_corr_fits
                            use_aoflagger_flags=use_aoflagger_flags,
                            remove_dig_gains=remove_dig_gains,
                            remove_coarse_band=remove_coarse_band,
                            correct_cable_len=correct_cable_len,
                            correct_van_vleck=correct_van_vleck,
                            cheby_approx=cheby_approx,
                            flag_small_auto_ants=flag_small_auto_ants,
                            propagate_coarse_flags=propagate_coarse_flags,
                            flag_init=flag_init,
                            edge_width=edge_width,
                            start_flag=start_flag,
                            end_flag=end_flag,
                            flag_dc_offset=flag_dc_offset,
                            remove_flagged_ants=remove_flagged_ants,
                            phase_to_pointing_center=phase_to_pointing_center,
                            nsample_array_dtype=nsample_array_dtype,
                            # MIR
                            mir_select_where=mir_select_where,
                            apply_tsys=apply_tsys,
                            apply_flags=apply_flags,
                            apply_dedoppler=apply_dedoppler,
                            pseudo_cont=pseudo_cont,
                            rechunk=rechunk,
                            compass_soln=compass_soln,
                            swarm_only=swarm_only,
                            codes_check=codes_check,
                        )

                        uv_list.append(uv2)
                    except KeyError as err:
                        file_warnings = (
                            file_warnings
                            + f"Failed to read {f} due to KeyError: {err}\n"
                        )
                        if skip_bad_files:
                            continue
                        else:
                            raise
                    except ValueError as err:
                        file_warnings = (
                            file_warnings
                            + f"Failed to read {f} due to ValueError: {err}\n"
                        )
                        if skip_bad_files:
                            continue
                        else:
                            raise
                    except OSError as err:  # pragma: nocover
                        file_warnings = (
                            file_warnings
                            + f"Failed to read {f} due to OSError: {err}\n"
                        )
                        if skip_bad_files:
                            continue
                        else:
                            raise
            if len(file_warnings) > 0:
                warnings.warn(file_warnings)

            # Concatenate once at end
            if axis is not None:
                # Rewrote fast_concat to operate on lists
                self.fast_concat(
                    uv_list,
                    axis,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    inplace=True,
                    ignore_name=ignore_name,
                )
            else:
                # Too much work to rewrite __add__ to operate on lists
                # of files, so instead doing a binary tree merge
                uv_list = [self] + uv_list
                while len(uv_list) > 1:
                    # for an odd number of files, the second argument will be shorter
                    # so the last element in the first list won't be combined, but it
                    # will not be lost, so it's ok.
                    for uv1, uv2 in zip(uv_list[0::2], uv_list[1::2], strict=False):
                        uv1.__iadd__(
                            uv2,
                            run_check=run_check,
                            check_extra=check_extra,
                            run_check_acceptability=run_check_acceptability,
                            ignore_name=ignore_name,
                        )
                    uv_list = uv_list[0::2]
                # Because self was at the beginning of the list,
                # everything is merged into it at the end of this loop

        else:
            if file_type in ["fhd", "ms"]:
                if (
                    antenna_nums is not None
                    or antenna_names is not None
                    or ant_str is not None
                    or bls is not None
                    or frequencies is not None
                    or freq_chans is not None
                    or spws is not None
                    or times is not None
                    or lsts is not None
                    or time_range is not None
                    or lst_range is not None
                    or polarizations is not None
                    or blt_inds is not None
                    or phase_center_ids is not None
                ):
                    select = True
                    warnings.warn(
                        "Warning: select on read keyword set, but "
                        f'file_type is "{file_type}" which does not support select '
                        "on read. Entire file will be read and then select "
                        "will be performed"
                    )
                    # these file types do not have select on read, so set all
                    # select parameters
                    select_antenna_nums = antenna_nums
                    select_antenna_names = antenna_names
                    select_ant_str = ant_str
                    select_bls = bls
                    select_frequencies = frequencies
                    select_freq_chans = freq_chans
                    select_spws = spws
                    select_times = times
                    select_lsts = lsts
                    select_time_range = time_range
                    select_lst_range = lst_range
                    select_polarizations = polarizations
                    select_blt_inds = blt_inds
                    select_phase_center_ids = phase_center_ids
                else:
                    select = False
            elif file_type in ["uvfits", "uvh5"]:
                select = False
            elif file_type in ["miriad"]:
                if (
                    antenna_names is not None
                    or frequencies is not None
                    or freq_chans is not None
                    or spws is not None
                    or times is not None
                    or blt_inds is not None
                    or phase_center_ids is not None
                ):
                    if blt_inds is not None:
                        if (
                            antenna_nums is not None
                            or ant_str is not None
                            or bls is not None
                            or time_range is not None
                        ):
                            warnings.warn(
                                "Warning: blt_inds is set along with select "
                                "on read keywords that are supported by "
                                "read_miriad and may downselect blts. "
                                "This may result in incorrect results "
                                "because the select on read will happen "
                                "before the blt_inds selection so the indices "
                                "may not match the expected locations."
                            )
                    else:
                        warnings.warn(
                            "Warning: a select on read keyword is set that is "
                            "not supported by read_miriad. This select will "
                            "be done after reading the file."
                        )
                    select = True
                    # these are all done by partial read, so set to None
                    select_antenna_nums = None
                    select_ant_str = None
                    select_bls = None
                    select_time_range = None
                    select_polarizations = None

                    # these aren't supported by partial read, so do it in select
                    select_antenna_names = antenna_names
                    select_frequencies = frequencies
                    select_freq_chans = freq_chans
                    select_spws = spws
                    select_times = times
                    select_lsts = lsts
                    select_lst_range = lst_range
                    select_blt_inds = blt_inds
                    select_phase_center_ids = phase_center_ids
                else:
                    select = False
            elif file_type in ["mir"]:
                select = True
                # these are all done by partial read, so set to None
                select_antenna_nums = None
                select_antenna_names = None
                select_ant_str = None
                select_bls = None
                select_lst_range = None
                select_time_range = None
                select_polarizations = None

                # MIR can handle length-two bls tuples, so if any three element tuples
                # are seen, we need to deal w/ that via select.
                if bls is not None:
                    select_bls = bls if any(len(item) == 3 for item in bls) else None

                # these aren't supported by partial read, so do it in select
                select_frequencies = frequencies
                select_freq_chans = freq_chans
                select_spws = spws
                select_blt_inds = blt_inds
                select_phase_center_ids = phase_center_ids
                select_times = times
                select_lsts = lsts

                if all(
                    item is None
                    for item in [
                        select_bls,
                        frequencies,
                        freq_chans,
                        blt_inds,
                        phase_center_ids,
                        times,
                        lsts,
                    ]
                ):
                    # If there's nothing to select, just bypass that operation.
                    select = False
                else:
                    warnings.warn(
                        "Warning: a select on read keyword is set that is "
                        "not supported by read_mir. This select will "
                        "be done after reading the file."
                    )
            elif file_type == "mwa_corr_fits":
                select = True
                # these are all done by partial read, so set to None
                select_antenna_nums = None
                select_antenna_names = None
                select_bls = None
                select_lst_range = None
                select_time_range = None
                select_times = None
                select_lsts = None
                select_polarizations = None
                select_frequencies = None
                select_freq_chans = None
                select_spws = None

                # MWA corr fits can only handle length-two bls tuples, anything
                # else needs to be handled via select.
                bls_use = copy.deepcopy(bls)
                if bls is not None and not all(len(item) == 2 for item in bls):
                    select_bls = bls
                    bls_use = None

                # these aren't supported by partial read, so do it in select
                select_ant_str = ant_str
                select_blt_inds = blt_inds
                select_phase_center_ids = phase_center_ids

                if all(
                    item is None
                    for item in [select_bls, blt_inds, phase_center_ids, ant_str]
                ):
                    # If there's nothing to select, just bypass that operation.
                    select = False
                else:
                    warnings.warn(
                        "Warning: a select on read keyword is set that is "
                        "not supported by read_mwa_corr_fits. This select will "
                        "be done after reading the file."
                    )
            # reading a single "file". Call the appropriate file-type read
            if file_type == "uvfits":
                self.read_uvfits(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    ant_str=ant_str,
                    bls=bls,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    spws=spws,
                    times=times,
                    time_range=time_range,
                    lsts=lsts,
                    lst_range=lst_range,
                    polarizations=polarizations,
                    blt_inds=blt_inds,
                    phase_center_ids=phase_center_ids,
                    catalog_names=catalog_names,
                    read_data=read_data,
                    keep_all_metadata=keep_all_metadata,
                    background_lsts=background_lsts,
                    default_mount_type=default_mount_type,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    fix_old_proj=fix_old_proj,
                    fix_use_ant_pos=fix_use_ant_pos,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                    astrometry_library=astrometry_library,
                )

            elif file_type == "mir":
                self.read_mir(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    bls=bls,
                    time_range=time_range,
                    lst_range=lst_range,
                    polarizations=polarizations,
                    catalog_names=catalog_names,
                    corrchunk=corrchunk,
                    receivers=receivers,
                    sidebands=sidebands,
                    select_where=mir_select_where,
                    apply_dedoppler=apply_dedoppler,
                    apply_tsys=apply_tsys,
                    apply_flags=apply_flags,
                    pseudo_cont=pseudo_cont,
                    rechunk=rechunk,
                    compass_soln=compass_soln,
                    swarm_only=swarm_only,
                    codes_check=codes_check,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    allow_flex_pol=allow_flex_pol,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                )
            elif file_type == "miriad":
                self.read_miriad(
                    filename,
                    antenna_nums=antenna_nums,
                    ant_str=ant_str,
                    bls=bls,
                    polarizations=polarizations,
                    time_range=time_range,
                    read_data=read_data,
                    projected=projected,
                    correct_lat_lon=correct_lat_lon,
                    background_lsts=background_lsts,
                    default_mount_type=default_mount_type,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    calc_lst=calc_lst,
                    fix_old_proj=fix_old_proj,
                    fix_use_ant_pos=fix_use_ant_pos,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                    astrometry_library=astrometry_library,
                )

            elif file_type == "mwa_corr_fits":
                self.read_mwa_corr_fits(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    bls=bls_use,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    spws=spws,
                    times=times,
                    time_range=time_range,
                    lsts=lsts,
                    lst_range=lst_range,
                    polarizations=polarizations,
                    keep_all_metadata=keep_all_metadata,
                    use_aoflagger_flags=use_aoflagger_flags,
                    remove_dig_gains=remove_dig_gains,
                    remove_coarse_band=remove_coarse_band,
                    correct_cable_len=correct_cable_len,
                    correct_van_vleck=correct_van_vleck,
                    cheby_approx=cheby_approx,
                    flag_small_auto_ants=flag_small_auto_ants,
                    propagate_coarse_flags=propagate_coarse_flags,
                    flag_init=flag_init,
                    edge_width=edge_width,
                    start_flag=start_flag,
                    end_flag=end_flag,
                    flag_dc_offset=flag_dc_offset,
                    remove_flagged_ants=remove_flagged_ants,
                    phase_to_pointing_center=phase_to_pointing_center,
                    read_data=read_data,
                    data_array_dtype=data_array_dtype,
                    nsample_array_dtype=nsample_array_dtype,
                    background_lsts=background_lsts,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                    astrometry_library=astrometry_library,
                )

            elif file_type == "fhd":
                self.read_fhd(
                    vis_files=filename,
                    params_file=params_file,
                    obs_file=obs_file,
                    flags_file=flags_file,
                    layout_file=layout_file,
                    settings_file=settings_file,
                    background_lsts=background_lsts,
                    read_data=read_data,
                    default_mount_type=default_mount_type,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                    astrometry_library=astrometry_library,
                )

            elif file_type == "ms":
                self.read_ms(
                    filename,
                    data_column=data_column,
                    pol_order=pol_order,
                    background_lsts=background_lsts,
                    ignore_single_chan=ignore_single_chan,
                    raise_error=raise_error,
                    read_weights=read_weights,
                    allow_flex_pol=allow_flex_pol,
                    default_mount_type=default_mount_type,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                    astrometry_library=astrometry_library,
                )

            elif file_type == "uvh5":
                self.read_uvh5(
                    filename,
                    antenna_nums=antenna_nums,
                    antenna_names=antenna_names,
                    ant_str=ant_str,
                    bls=bls,
                    frequencies=frequencies,
                    freq_chans=freq_chans,
                    spws=spws,
                    times=times,
                    time_range=time_range,
                    lsts=lsts,
                    lst_range=lst_range,
                    polarizations=polarizations,
                    blt_inds=blt_inds,
                    phase_center_ids=phase_center_ids,
                    catalog_names=catalog_names,
                    read_data=read_data,
                    data_array_dtype=data_array_dtype,
                    keep_all_metadata=keep_all_metadata,
                    multidim_index=multidim_index,
                    remove_flex_pol=remove_flex_pol,
                    background_lsts=background_lsts,
                    default_mount_type=default_mount_type,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                    fix_old_proj=fix_old_proj,
                    fix_use_ant_pos=fix_use_ant_pos,
                    check_autos=check_autos,
                    fix_autos=fix_autos,
                    time_axis_faster_than_bls=time_axis_faster_than_bls,
                    blts_are_rectangular=blts_are_rectangular,
                    recompute_nbls=recompute_nbls,
                    astrometry_library=astrometry_library,
                )
                select = False

            if select:
                self.select(
                    antenna_nums=select_antenna_nums,
                    antenna_names=select_antenna_names,
                    ant_str=select_ant_str,
                    bls=select_bls,
                    frequencies=select_frequencies,
                    freq_chans=select_freq_chans,
                    spws=select_spws,
                    times=select_times,
                    lsts=select_lsts,
                    time_range=select_time_range,
                    lst_range=select_lst_range,
                    polarizations=select_polarizations,
                    blt_inds=select_blt_inds,
                    phase_center_ids=select_phase_center_ids,
                    keep_all_metadata=keep_all_metadata,
                    run_check=run_check,
                    check_extra=check_extra,
                    run_check_acceptability=run_check_acceptability,
                    strict_uvw_antpos_check=strict_uvw_antpos_check,
                )

    @classmethod
    @copy_replace_short_description(read, style=DocstringStyle.NUMPYDOC)
    def from_file(cls, filename, **kwargs):
        """Initialize a new UVData object by reading the input file."""
        uvd = cls()
        uvd.read(filename, **kwargs)
        return uvd

    def write_miriad(
        self,
        filepath,
        *,
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
            than their channel width or if the UVData object is a metadata only object.
        TypeError
            If any entry in extra_keywords is not a single string or number.

        """
        if self.metadata_only:
            raise ValueError("Cannot write out metadata only objects to a miriad file.")

        miriad_obj = self._convert_to_filetype("miriad")

        # If we have a flex-pol dataset, remove it so that we can pass the data to the
        # writer without issue. We create a copy because otherwise we run the risk of
        # messing w/ the metadata of the original object, and because the overhead of
        # doing so is generally smaller than that from removing flex-pol.
        # TODO: reconsider this. The overhead is not smaller if `convert_to_flex_pol`
        # was used.
        if self.flex_spw_polarization_array is not None:
            miriad_obj = miriad_obj.copy()
            miriad_obj.remove_flex_pol()

        miriad_obj.write_miriad(
            filepath,
            clobber=clobber,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
            no_antnums=no_antnums,
            calc_lst=calc_lst,
            check_autos=check_autos,
            fix_autos=fix_autos,
        )
        del miriad_obj

    def write_mir(self, filepath):
        """
        Write the data to a mir file.

        Parameters
        ----------
        filename : str
            The mir root directory to write to.

        Raises
        ------
        ValueError
            If the UVData object is a metadata only object.
        NotImplementedError
            Method is not fully implemented yet, and thus will raise an error

        """
        if self.metadata_only:
            raise ValueError("Cannot write out metadata only objects to a mir file.")

        mir_obj = self._convert_to_filetype("mir")
        mir_obj.write_mir(filepath)
        del mir_obj  # pragma: nocover

    def write_ms(
        self,
        filename,
        *,
        force_phase=False,
        model_data=None,
        corrected_data=None,
        flip_conj=None,
        clobber=False,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=False,
    ):
        """
        Write a CASA measurement set (MS).

        Parameters
        ----------
        filename : str
            The measurement set file path to write to (a measurement set is really
            a folder with many files).
        force_phase : bool
            Option to automatically phase unprojected data to zenith of the first
            timestamp.
        model_data : ndarray
            Optional argument, which contains data to be written into the MODEL_DATA
            column of the measurement set (along with the data, which is written into
            the DATA column). Must contain the same dimensions as `data_array`.
        corrected_data : ndarray
            Optional argument, which contains data to be written into the CORRECTED_DATA
            column of the measurement set (along with the data, which is written into
            the DATA column). Must contain the same dimensions as `data_array`.
        flip_conj : bool
            If set to True, and the UVW coordinates are flipped (i.e., multiplied by
            -1) and the visibilities are complex conjugated prior to write, such that
            the data are written with the "opposite" conjugation scheme to what UVData
            normally uses. If set to False, no baseline conjugation is performed. By
            default, the conjugation scheme is automatically determined by baseline
            conjugation (e.g., "ant1>ant2" or "ant1<ant2"), see UVData.conjugate_bls for
            further details). Note that this is only needed for specific subset of
            applications that read MS-formatted data, and should only be modified by
            expert users.
        clobber : bool
            Option to overwrite the file if it already exists.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is False.

        Raises
        ------
        ValueError
            If the UVData object is a metadata only object.

        """
        if self.metadata_only:
            raise ValueError(
                "Cannot write out metadata only objects to a measurement set file."
            )

        ms_obj = self._convert_to_filetype("ms")
        ms_obj.write_ms(
            filename,
            force_phase=force_phase,
            model_data=model_data,
            corrected_data=corrected_data,
            flip_conj=flip_conj,
            clobber=clobber,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
            check_autos=check_autos,
            fix_autos=fix_autos,
        )
        del ms_obj

    def write_uvfits(self, filename: str, **kwargs):
        """
        Write the data to a uvfits file.

        If using this method to write out a data set for import into CASA, users should
        be aware that the `importuvifts` task does not currently support reading in
        data sets where the number of antennas is > 255. If writing out such a data set
        for use in CASA, we suggest using the measurement set writer (`UVData.write_ms`)
        instead, as the `importuvfits` has some hard-coded behaviors that are telescope
        dependent and not consistent with documented standards for UVFITS (including
        antenna position handling, among other metadata).

        Parameters
        ----------
        filename : str
            The uvfits file to write to.
        write_lst : bool
            Option to write the LSTs to the metadata (random group parameters).
            Default is True.
        force_phase : bool
            Option to automatically phase unprojected data to zenith of the first
            timestamp. Default is False.
        uvw_double : bool
            Option to write uvws at double precision if data array is single
            precision (if data array is double precision uvws are always written
            at double precision). This requires writing the uvws out into two
            identically named parameters to be added together on read (the same
            mechanism that is used for times in uvfits). Default is True.
        use_miriad_convention : bool
            Option to use the MIRIAD baseline convention, and write to BASELINE column.
            This mode is required for UVFITS files with >256 antennas to be
            readable by MIRIAD, and supports up to 2048 antennas.
            The MIRIAD baseline ID is given by
            `bl = 256 * ant1 + ant2` if `ant2 < 256`, otherwise
            `bl = 2048 * ant1 + ant2 + 2**16`.
            Note MIRIAD uses 1-indexed antenna IDs, but this code accepts 0-based.
            Default is False.
        run_check : bool
            Option to check for the existence and proper shapes of parameters
            before writing the file. Default is True.
        check_extra : bool
            Option to check optional parameters as well as required ones.
            Default is True.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file. Default is True.
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass. Default is False.
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is False.

        Raises
        ------
        ValueError
            The object contains unprojected data and `force_phase` keyword is not set.
            If the frequencies are not evenly spaced or are separated by more
            than their channel width.
            The polarization values are not evenly spaced.
            If the UVData object is a metadata only object.
            If the `timesys` parameter is set to anything other than "UTC" or None.
        TypeError
            If any entry in extra_keywords is not a single string or number.

        """
        if self.metadata_only:
            raise ValueError("Cannot write out metadata only objects to a uvfits file.")

        uvfits_obj = self._convert_to_filetype("uvfits")

        # If we have a flex-pol dataset, remove it so that we can pass the data to the
        # writer without issue. We create a copy because otherwise we run the risk of
        # messing w/ the metadata of the original object, and because the overhead of
        # doing so is generally smaller than that from removing flex-pol.
        # TODO: reconsider this. The overhead is not smaller if `convert_to_flex_pol`
        # was used.
        if self.flex_spw_polarization_array is not None:
            uvfits_obj = uvfits_obj.copy()
            uvfits_obj.remove_flex_pol()

        uvfits_obj.write_uvfits(filename, **kwargs)
        del uvfits_obj

    def write_uvh5(
        self,
        filename,
        *,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        check_autos=True,
        fix_autos=False,
    ):
        """
        Write a completely in-memory UVData object to a UVH5 file.

        Parameters
        ----------
        filename : str
             The UVH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression). In addition to the normal HDF5 filter values, the
            user may specify "bitshuffle" which will set the compression to `32008` for
            bitshuffle and will set the `compression_opts` to `(0, 2)` to allow
            bitshuffle to automatically determine the block size and to use the LZF
            filter after bitshuffle. Using `bitshuffle` requires having the
            `hdf5plugin` package installed.  Dataset must be chunked to use compression.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        data_write_dtype : numpy dtype
            datatype of output visibility data. If 'None', then the same datatype
            as data_array will be used. Otherwise, a numpy dtype object must be
            specified with an 'r' field and an 'i' field for real and imaginary
            parts, respectively. See uvh5.py for an example of defining such a datatype.
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
        check_autos : bool
            Check whether any auto-correlations have non-zero imaginary values in
            data_array (which should not mathematically exist). Default is True.
        fix_autos : bool
            If auto-correlations with imaginary values are found, fix those values so
            that they are real-only in data_array. Default is False.

        Raises
        ------
        ValueError
            If the UVData object is a metadata only object.

        """
        if self.metadata_only:
            raise ValueError(
                "Cannot write out metadata only objects to a uvh5 file. To initialize "
                "a uvh5 file for partial writing, use the `initialize_uvh5_file` "
                "method."
            )

        uvh5_obj = self._convert_to_filetype("uvh5")
        uvh5_obj.write_uvh5(
            filename,
            clobber=clobber,
            chunks=chunks,
            data_compression=data_compression,
            flags_compression=flags_compression,
            nsample_compression=nsample_compression,
            data_write_dtype=data_write_dtype,
            run_check=run_check,
            check_extra=check_extra,
            run_check_acceptability=run_check_acceptability,
            strict_uvw_antpos_check=strict_uvw_antpos_check,
            check_autos=check_autos,
            fix_autos=fix_autos,
        )
        del uvh5_obj

    def initialize_uvh5_file(
        self,
        filename,
        *,
        clobber=False,
        chunks=True,
        data_compression=None,
        flags_compression="lzf",
        nsample_compression="lzf",
        data_write_dtype=None,
    ):
        """
        Initialize a UVH5 file on disk with the header metadata and empty data arrays.

        Parameters
        ----------
        filename : str
             The UVH5 file to write to.
        clobber : bool
            Option to overwrite the file if it already exists.
        chunks : tuple or bool
            h5py.create_dataset chunks keyword. Tuple for chunk shape,
            True for auto-chunking, None for no chunking. Default is True.
        data_compression : str
            HDF5 filter to apply when writing the data_array. Default is None
            (no filter/compression). In addition to the normal HDF5 filter values, the
            user may specify "bitshuffle" which will set the compression to `32008` for
            bitshuffle and will set the `compression_opts` to `(0, 2)` to allow
            bitshuffle to automatically determine the block size and to use the LZF
            filter after bitshuffle. Using `bitshuffle` requires having the
            `hdf5plugin` package installed.  Dataset must be chunked to use compression.
        flags_compression : str
            HDF5 filter to apply when writing the flags_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        nsample_compression : str
            HDF5 filter to apply when writing the nsample_array. Default is "lzf"
            for the LZF filter. Dataset must be chunked.
        data_write_dtype : numpy dtype
            datatype of output visibility data. If 'None', then the same datatype
            as data_array will be used. Otherwise, a numpy dtype object must be
            specified with an 'r' field and an 'i' field for real and imaginary
            parts, respectively. See uvh5.py for an example of defining such a datatype.

        Notes
        -----
        When partially writing out data, this function should be called first
        to initialize the file on disk. The data is then actually written by
        calling the write_uvh5_part method, with the same filename as the one
        specified in this function. See the tutorial for a worked example.

        """
        uvh5_obj = self._convert_to_filetype("uvh5")
        uvh5_obj.initialize_uvh5_file(
            filename,
            clobber=clobber,
            chunks=chunks,
            data_compression=data_compression,
            flags_compression=flags_compression,
            nsample_compression=nsample_compression,
            data_write_dtype=data_write_dtype,
        )
        del uvh5_obj

    def write_uvh5_part(
        self,
        filename,
        *,
        data_array,
        flag_array,
        nsample_array,
        check_header=True,
        antenna_nums=None,
        antenna_names=None,
        ant_str=None,
        bls=None,
        frequencies=None,
        freq_chans=None,
        spws=None,
        times=None,
        time_range=None,
        lsts=None,
        lst_range=None,
        polarizations=None,
        blt_inds=None,
        phase_center_ids=None,
        catalog_names=None,
        add_to_history=None,
        run_check_acceptability=True,
        fix_autos=False,
    ):
        """
        Write data to a UVH5 file that has already been initialized.

        Parameters
        ----------
        filename : str
            The UVH5 file to write to. It must already exist, and is assumed to
            have been initialized with initialize_uvh5_file.
        data_array : ndarray
            The data to write to disk. A check is done to ensure that the
            dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        flags_array : ndarray
            The flags array to write to disk. A check is done to ensure that the
            dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        nsample_array : ndarray
            The nsample array to write to disk. A check is done to ensure that the
            dimensions of the data passed in conform to the ones specified by
            the "selection" arguments.
        check_header : bool
            Option to check that the metadata present in the header on disk
            matches that in the object.
        antenna_nums : array_like of int, optional
            The antennas numbers to include when writing data into the file
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_names` is also provided.
        antenna_names : array_like of str, optional
            The antennas names to include when writing data into the file
            (antenna positions and names for the removed antennas will be retained).
            This cannot be provided if `antenna_nums` is also provided.
        bls : list of tuple, optional
            A list of antenna number tuples (e.g. [(0, 1), (3, 2)]) or a list of
            baseline 3-tuples (e.g. [(0, 1, 'xx'), (2, 3, 'yy')]) specifying baselines
            to include when writing data into the file. For length-2 tuples,
            the ordering of the numbers within the tuple does not matter. For
            length-3 tuples, the polarization string is in the order of the two
            antennas. If length-3 tuples are provided, `polarizations` must be
            None.
        ant_str : str, optional
            A string containing information about what antenna numbers
            and polarizations to include writing data into the file.
            Can be 'auto', 'cross', 'all', or combinations of antenna numbers
            and polarizations (e.g. '1', '1_2', '1x_2y').  See tutorial for more
            examples of valid strings and the behavior of different forms for ant_str.
            If '1x_2y,2y_3y' is passed, both polarizations 'xy' and 'yy' will
            be kept for both baselines (1, 2) and (2, 3) to return a valid
            pyuvdata object.
            An ant_str cannot be passed in addition to any of `antenna_nums`,
            `antenna_names`, `bls` args or the `polarizations` parameters,
            if it is a ValueError will be raised.
        frequencies : array_like of float, optional
            The frequencies to include when writing data into the file, each
            value passed here should exist in the freq_array.
        freq_chans : array_like of int, optional
            The frequency channel numbers to include writing data into the file.
        spws : array_like of int, optional
            The spectral window numbers to include writing data into the file.
        times : array_like of float, optional
            The times to include when writing data into the file, each value
            passed here should exist in the time_array. Cannot be used with
            `time_range`, `lsts`, or `lst_array`.
        time_range : array_like of float, optional
            The time range in Julian Date to include when writing data to the
            file, must be length 2. Some of the times in the object should fall
            between the first and last elements. Cannot be used with `times`.
        lsts : array_like of float, optional
            The local sidereal times (LSTs) to keep in the object, each value
            passed here should exist in the lst_array. Cannot be used with
            `times`, `time_range`, or `lst_range`.
        lst_range : array_like of float, optional
            The local sidereal time (LST) range in radians to keep in the
            object, must be of length 2. Some of the LSTs in the object should
            fall between the first and last elements. If the second value is
            smaller than the first, the LSTs are treated as having phase-wrapped
            around LST = 2*pi = 0, and the LSTs kept on the object will run from
            the larger value, through 0, and end at the smaller value.
        polarizations : array_like of int, optional
            The polarizations numbers to include when writing data into the file,
            each value passed here should exist in the polarization_array.
        blt_inds : array_like of int, optional
            The baseline-time indices to include when writing data into the file.
            This is not commonly used.
        phase_center_ids : array_like of int, optional
            Phase center IDs to include when writing data into the file (effectively
            a selection on baseline-times). Cannot be used with catalog_names.
        catalog_names : str or array-like of str, optional
            The names of the phase centers (sources) to include when writing data to
            the file, which should match exactly in spelling and capitalization.
            Cannot be used with phase_center_ids.
        add_to_history : str
            String to append to history before write out. Default is no appending.
        run_check_acceptability : bool
            Option to check acceptable range of the values of parameters before
            writing the file (the default is True, meaning the acceptable
            range check will be done).
        strict_uvw_antpos_check : bool
            Option to raise an error rather than a warning if the check that
            uvws match antenna positions does not pass.
        fix_autos : bool
            Force the auto-correlations to be real-only values in data_array.
            Default is False.

        """
        if fix_autos:
            self._fix_autos()

        uvh5_obj = self._convert_to_filetype("uvh5")
        uvh5_obj.write_uvh5_part(
            filename,
            data_array=data_array,
            flag_array=flag_array,
            nsample_array=nsample_array,
            check_header=check_header,
            antenna_nums=antenna_nums,
            antenna_names=antenna_names,
            bls=bls,
            ant_str=ant_str,
            frequencies=frequencies,
            freq_chans=freq_chans,
            spws=spws,
            times=times,
            time_range=time_range,
            lsts=lsts,
            lst_range=lst_range,
            polarizations=polarizations,
            blt_inds=blt_inds,
            phase_center_ids=phase_center_ids,
            catalog_names=catalog_names,
            add_to_history=add_to_history,
            run_check_acceptability=run_check_acceptability,
        )
        del uvh5_obj

    def normalize_by_autos(self, *, skip_autos=True, invert=False):
        """
        Normalize cross-correlations by auto-correlation data.

        Normalizes the cross-correlations by the geometric mean of the autocorrelations
        that make up the antenna pair for a given baseline. Useful for converting
        arbitrarily-scaled data into correlation coefficients (which can sometimes be
        more readily converted to a flux scale).

        Parameters
        ----------
        skip_autos : bool
            If set to True, the method will skip over records which correspond to the
            auto-correlations (which would otherwise simply be equal to all ones), which
            can be useful if intending to undo or redo the normalization at some point
            later. Default is True.
        invert : bool
            If set to True, will multiply by the geometric mean of the autos instead of
            dividing by it. Useful for undoing previous normalizations (which also
            requires setting `skip_autos=True` on previous calls).
        """
        # First up, let's double-check to make sure that we can actually normalize this
        # data set, and if not, bail early.
        if not np.any(self.ant_1_array == self.ant_2_array):
            raise ValueError(
                "No autos available in this data set to do normalization with."
            )

        # Now figure out how the different polarizations map to the autos
        pol_groups = []
        pol_list = list(self.polarization_array)
        for pol in pol_list:
            try:
                feed_pols = utils.pol.POL_TO_FEED_DICT[utils.POL_NUM2STR_DICT[pol]]
                pol_groups.append(
                    [
                        pol_list.index(utils.POL_STR2NUM_DICT[item + item])
                        for item in feed_pols
                    ]
                )
            except KeyError:
                # If we run into a key error, it means that one of the dicts above does
                # not have a match to the given polarization, in which case assume that
                # this pol its it's own auto.
                pol_groups.append([pol_list.index(pol)] * 2)
            except ValueError as err:
                # If we have an value error, it means that the pol that _would_ be the
                # auto is not found in the data, in which case we  throw an error.
                raise ValueError(
                    f"Cannot normalize {utils.POL_NUM2STR_DICT[pol]}, matching "
                    "pols for autos not found."
                ) from err

        # Each pol group contains the index positions for the "auto" polarizations,
        # so we can grab all of the auto pols by searching for the unique values
        auto_pols = list(np.unique(pol_groups))

        # Grab references to data and flags, to manipulate later
        data_arr = self.data_array
        flag_arr = self.flag_array

        # We need to match baselines in a single integration, so figure out how to
        # group the data by time.
        time_ordered = not np.any(self.time_array[:-1] > self.time_array[1:])
        blt_idx = np.arange(self.Nblts) if time_ordered else np.argsort(self.time_array)
        ordered_time = self.time_array[blt_idx if time_ordered else ...]

        # Normally time has a fixed atol, but verify that this is the case
        time_tol = self._time_array.tols[1]
        assert self._time_array.tols[0] == 0

        # Start searching through time, keeping in mind that the data are time ordered.
        time_groups = []
        start_idx = end_idx = 0
        while self.Nblts != start_idx:
            # Search sorted will find where one would insert the value of the current
            # time + the time tol, which will give us the ending index of the group.
            end_idx += np.searchsorted(
                ordered_time[start_idx:], ordered_time[start_idx] + time_tol, "right"
            )
            # Create this new grouping of blts
            time_groups.append(blt_idx[start_idx:end_idx])
            start_idx = end_idx

        # Not start the heavy lifting
        for group in time_groups:
            # Now start going through the groups
            ant_1_arr = self.ant_1_array[group]
            ant_2_arr = self.ant_2_array[group]
            norm_dict = {}
            flag_dict = {}
            for grp_idx, ant1, ant2 in zip(group, ant_1_arr, ant_2_arr, strict=True):
                # Tabulate up front the normalization for each auto-correlation
                # spectrum, which will save some work downstream
                if ant1 != ant2:
                    continue
                norm_dict[ant1] = {}
                flag_dict[ant1] = {}
                for pol in auto_pols:
                    # Autos _should_ be real only, extracting them out like this will
                    # make the multipliction later a bit faster.
                    auto_data = data_arr[grp_idx, :, pol].real
                    auto_flag = flag_arr[grp_idx, :, pol] | ~(auto_data > 0)
                    norm_data = np.zeros_like(auto_data)
                    if invert:
                        norm_data = np.sqrt(auto_data, where=~auto_flag, out=norm_data)
                    else:
                        norm_data = np.reciprocal(
                            auto_data, where=~auto_flag, out=norm_data
                        )
                        norm_data = np.sqrt(norm_data, out=norm_data)
                    norm_dict[ant1][pol] = norm_data
                    flag_dict[ant1][pol] = auto_flag

            # Now that we have the autos "normalization-ready", we can get to
            # actually normalizing the crosses.
            for grp_idx, ant1, ant2 in zip(group, ant_1_arr, ant_2_arr, strict=True):
                try:
                    for jdx, (pol1, pol2) in enumerate(pol_groups):
                        # Proceed pol by pol
                        if skip_autos and (ant1 == ant2) and (pol1 == pol2):
                            continue
                        data_arr[grp_idx, :, jdx] *= (
                            norm_dict[ant1][pol1] * norm_dict[ant2][pol2]
                        )
                        flag_arr[grp_idx, :, jdx] |= (
                            flag_dict[ant1][pol1] | flag_dict[ant2][pol2]
                        )
                except KeyError:
                    # If no data found for this antenna, then flag the whole blt
                    flag_arr[grp_idx] = True
