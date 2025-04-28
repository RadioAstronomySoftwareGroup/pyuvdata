# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing uvfits files."""

import copy
import os
import warnings

import numpy as np
from astropy import constants as const, units
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.time import Time
from docstring_parser import DocstringStyle

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import fits as fits_utils
from . import UVData

__all__ = ["UVFITS"]


class UVFITS(UVData):
    """
    Defines a uvfits-specific subclass of UVData for reading and writing uvfits.

    This class should not be interacted with directly, instead use the read_uvfits
    and write_uvfits methods on the UVData class.
    """

    def _get_parameter_data(
        self,
        vis_hdu,
        *,
        read_source,
        run_check_acceptability,
        background_lsts=True,
        astrometry_library=None,
    ):
        """
        Read just the random parameters portion of the uvfits file ("metadata").

        Separated from full read so that header, metadata and data can be read
        independently.
        """
        # astropy.io fits reader scales date according to relevant PZER0 (?)
        # uvfits standard is to have 2 DATE parameters, both floats:
        # DATE (full day) and _DATE (fractional day)
        # cotter uvfits files have one DATE that is a double
        # using data.par('date') is general -- it will add them together if there are 2
        self.time_array = vis_hdu.data.par("date")

        self.Ntimes = len(np.unique(self.time_array))

        # check if lst array is saved. It's not a standard metadata item in uvfits,
        # but if the file was written with pyuvdata it may be present
        # (depending on pyuvdata version)
        proc = None
        if "LST" in vis_hdu.data.parnames:
            # angles in uvfits files are stored in degrees, so convert to radians
            self.lst_array = np.deg2rad(vis_hdu.data.par("lst"))
            if run_check_acceptability:
                utils.times.check_lsts_against_times(
                    jd_array=self.time_array,
                    lst_array=self.lst_array,
                    telescope_loc=self.telescope.location,
                    lst_tols=(0, utils.LST_RAD_TOL),
                )
        else:
            proc = self.set_lsts_from_time_array(
                background=background_lsts, astrometry_library=astrometry_library
            )

        # if antenna arrays are present, use them. otherwise use baseline array
        if "ANTENNA1" in vis_hdu.data.parnames and "ANTENNA2" in vis_hdu.data.parnames:
            # Note: we no longer subtract 1 from the antenna arrays
            # The antanna arrays are not indices but rather are numbers
            # that correspond to particular antennas
            self.ant_1_array = np.int32(vis_hdu.data.par("ANTENNA1"))
            self.ant_2_array = np.int32(vis_hdu.data.par("ANTENNA2"))
            # for instruments with a single subarray, the subarray should be set to 1s
            subarray = np.int32(vis_hdu.data.par("SUBARRAY"))
            # error on files with multiple subarrays
            if len(set(subarray)) > 1:
                raise ValueError(
                    "This file appears to have multiple subarray "
                    "values; only files with one subarray are "
                    "supported."
                )
        else:
            # cannot set this to be the baseline array because it uses the
            # 256 convention, not our 2048 convention
            bl_input_array = np.int64(vis_hdu.data.par("BASELINE"))

            # get antenna arrays based on uvfits baseline array
            self.ant_1_array, self.ant_2_array = self.baseline_to_antnums(
                bl_input_array
            )

        if read_source:
            source = vis_hdu.data.par("SOURCE")
            self.phase_center_id_array = source.astype(int)

        # get self.baseline_array using our convention
        self.baseline_array = self.antnums_to_baseline(
            self.ant_1_array, self.ant_2_array
        )
        self.Nbls = len(np.unique(self.baseline_array))

        # initialize internal variables based on the antenna lists
        self.Nants_data = int(np.union1d(self.ant_1_array, self.ant_2_array).size)

        # check for suffixes in the baseline coordinate names indicating the
        # baseline coordinate system
        if (
            "UU" in vis_hdu.data.parnames
            and "VV" in vis_hdu.data.parnames
            and "WW" in vis_hdu.data.parnames
        ):
            uvw_names = ["UU", "VV", "WW"]
        elif (
            "UU--" in vis_hdu.data.parnames
            and "VV--" in vis_hdu.data.parnames
            and "WW--" in vis_hdu.data.parnames
        ):
            uvw_names = ["UU--", "VV--", "WW--"]
        elif (
            "UU---SIN" in vis_hdu.data.parnames
            and "VV---SIN" in vis_hdu.data.parnames
            and "WW---SIN" in vis_hdu.data.parnames
        ):
            uvw_names = ["UU---SIN", "VV---SIN", "WW---SIN"]
        elif (
            "UU---NCP" in vis_hdu.data.parnames
            and "VV---NCP" in vis_hdu.data.parnames
            and "WW---NCP" in vis_hdu.data.parnames
        ):
            uvw_names = ["UU---NCP", "VV---NCP", "WW---NCP"]
            warnings.warn(
                "The baseline coordinates (uvws) in this file are specified in the "
                "---NCP coordinate system, which is does not agree with our baseline "
                "coordinate conventions. Rotating the uvws to match our convention "
                "(Note that this rotation has not been widely tested)."
            )
        else:
            raise ValueError(
                "There is no consistent set of baseline coordinates in this file. "
                "The UU, VV and WW coordinate must have no suffix or the '---SIN' or "
                "'---NCP' suffix and the suffixes must match on all three baseline "
                "coordinate parameters."
            )

        # read baseline vectors in units of seconds, return in meters
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        # uvfits files often have uvws in single precision rather than double precision.
        # setting the dtype below enforces double precision
        self.uvw_array = (-1) * (
            np.array(
                np.stack(
                    (
                        vis_hdu.data.par(uvw_names[0]),
                        vis_hdu.data.par(uvw_names[1]),
                        vis_hdu.data.par(uvw_names[2]),
                    )
                ),
                dtype=self._uvw_array.expected_type,
            )
            * const.c.to_value("m/s")
        ).T

        if "INTTIM" in vis_hdu.data.parnames:
            self.integration_time = np.asarray(
                vis_hdu.data.par("INTTIM"), dtype=np.float64
            )
        else:
            if self.Ntimes > 1:
                # assume that all integration times in the file are the same
                int_time = self._calc_single_integration_time()
                self.integration_time = (
                    np.ones_like(self.time_array, dtype=np.float64) * int_time
                )
            else:
                warnings.warn(
                    "The integration time is not specified and only one time is "
                    "present so it cannot be calculated from the difference between "
                    "integration times. Setting to None which will cause the check to "
                    "error. Set `run_check` to False to read in the file without "
                    "checking. Then set the integration_time (to an array of length "
                    "Nblts) directly on the object to allow futher processing."
                )

        if proc is not None:
            proc.join()

    def _get_data(
        self,
        vis_hdu,
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
        keep_all_metadata,
        fix_old_proj,
        fix_use_ant_pos,
    ):
        """
        Read just the visibility and flag data of the uvfits file.

        Separated from full read so header and metadata can be read without data.
        """
        # figure out what data to read in
        blt_inds, freq_inds, spw_inds, pol_inds, history_update_string = (
            self._select_preprocess(
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
            )
        )

        if blt_inds is not None:
            blt_frac = len(blt_inds) / float(self.Nblts)
        else:
            blt_frac = 1

        if freq_inds is not None:
            freq_frac = len(freq_inds) * float(self.Nspws) / float(self.Nfreqs)
        else:
            freq_frac = 1

        if pol_inds is not None:
            pol_frac = len(pol_inds) / float(self.Npols)
        else:
            pol_frac = 1

        min_frac = np.min([blt_frac, freq_frac, pol_frac])

        if min_frac == 1:
            # no select, read in all the data
            if vis_hdu.header["NAXIS"] == 7:
                raw_data_array = vis_hdu.data.data[:, 0, 0, :, :, :, :]
                assert self.Nspws == raw_data_array.shape[1]

            else:
                # in many uvfits files the spw axis is left out,
                # here we put it back in so the dimensionality stays the same
                raw_data_array = vis_hdu.data.data[:, 0, 0, :, :, :]
                raw_data_array = raw_data_array[:, np.newaxis, :, :]
        else:
            # do select operations on everything except data_array, flag_array
            # and nsample_array
            self._select_by_index(
                blt_inds=blt_inds,
                freq_inds=freq_inds,
                spw_inds=spw_inds,
                pol_inds=pol_inds,
                history_update_string=history_update_string,
                keep_all_metadata=keep_all_metadata,
            )

            # just read in the right portions of the data and flag arrays
            if blt_frac == min_frac:
                if vis_hdu.header["NAXIS"] == 7:
                    raw_data_array = vis_hdu.data.data[blt_inds, :, :, :, :, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert self.Nspws == raw_data_array.shape[1]
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[blt_inds, :, :, :, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]
                if freq_frac < 1:
                    raw_data_array = raw_data_array[:, :, freq_inds, :, :]
                if pol_frac < 1:
                    raw_data_array = raw_data_array[:, :, :, pol_inds, :]
            elif freq_frac == min_frac:
                if vis_hdu.header["NAXIS"] == 7:
                    raw_data_array = vis_hdu.data.data[:, :, :, :, freq_inds, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert self.Nspws == raw_data_array.shape[1]
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[:, :, :, freq_inds, :, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]

                if blt_frac < 1:
                    raw_data_array = raw_data_array[blt_inds, :, :, :, :]
                if pol_frac < 1:
                    raw_data_array = raw_data_array[:, :, :, pol_inds, :]
            else:
                if vis_hdu.header["NAXIS"] == 7:
                    raw_data_array = vis_hdu.data.data[:, :, :, :, :, pol_inds, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :, :]
                    assert self.Nspws == raw_data_array.shape[1]
                else:
                    # in many uvfits files the spw axis is left out,
                    # here we put it back in so the dimensionality stays the same
                    raw_data_array = vis_hdu.data.data[:, :, :, :, pol_inds, :]
                    raw_data_array = raw_data_array[:, 0, 0, :, :, :]
                    raw_data_array = raw_data_array[:, np.newaxis, :, :, :]

                if blt_frac < 1:
                    raw_data_array = raw_data_array[blt_inds, :, :, :, :]
                if freq_frac < 1:
                    raw_data_array = raw_data_array[:, :, freq_inds, :, :]

        assert len(raw_data_array.shape) == 5

        # Reshape the data array to be the right size if we are working w/ multiple
        # spectral windows
        raw_data_array = np.reshape(
            raw_data_array,
            (self.Nblts, self.Nfreqs, self.Npols, raw_data_array.shape[4]),
        )

        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        self.data_array = raw_data_array[:, :, :, 0] - 1j * raw_data_array[:, :, :, 1]
        self.flag_array = raw_data_array[:, :, :, 2] <= 0
        self.nsample_array = np.abs(raw_data_array[:, :, :, 2])

        if fix_old_proj:
            self.fix_phase(use_ant_pos=fix_use_ant_pos)

    @copy_replace_short_description(UVData.read_uvfits, style=DocstringStyle.NUMPYDOC)
    def read_uvfits(
        self,
        filename,
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
        keep_all_metadata=True,
        read_data=True,
        background_lsts=True,
        default_mount_type="other",
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        strict_uvw_antpos_check=False,
        fix_old_proj=False,
        fix_use_ant_pos=True,
        check_autos=True,
        fix_autos=True,
        astrometry_library=None,
    ):
        """Read in header, metadata and data from a uvfits file."""
        # update filename attribute
        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

        with fits.open(filename, memmap=True) as hdu_list:
            vis_hdu = hdu_list[0]  # assumes the visibilities are in the primary hdu
            vis_hdr = vis_hdu.header.copy()
            hdunames = fits_utils._indexhdus(hdu_list)  # find the rest of the tables

            # First get everything we can out of the header.

            # check for multi source files. NOW SUPPORTED, W00T!
            read_source = False
            if "SOURCE" in vis_hdu.data.parnames:
                if "AIPS SU" in hdunames:
                    read_source = True
                else:
                    warnings.warn(
                        "UVFITS file is missing AIPS SU table, which is required when "
                        "SOURCE is one of the `random paramters` in the main binary "
                        "table. Bypassing for now, but note that this file _may_ not "
                        "work correctly in UVFITS-based programs (e.g., AIPS, CASA)."
                    )

            # check if we have an spw dimension
            if vis_hdr["NAXIS"] == 7:
                self.Nspws = vis_hdr.pop("NAXIS5")
                # By UVFITS definition, this _has_ to be the way this axis is organized
                self.spw_array = np.arange(self.Nspws)
                ra_axis = 6
                dec_axis = 7
            else:
                self.Nspws = 1
                self.spw_array = np.array([np.int64(0)])
                ra_axis = 5
                dec_axis = 6

            # get shapes
            self.Npols = vis_hdr.pop("NAXIS3")
            self.Nblts = vis_hdr.pop("GCOUNT")

            if "AIPS FQ" in hdunames:
                # If this is multi-spw, get details from the FQ table
                uvfits_nchan = vis_hdr.pop("NAXIS4")
                self.Nfreqs = uvfits_nchan * self.Nspws
                self.flex_spw_id_array = np.transpose(
                    np.tile(np.arange(self.Nspws), (uvfits_nchan, 1))
                ).flatten()
                fq_hdu = hdu_list[hdunames["AIPS FQ"]]
                assert self.Nspws == fq_hdu.header["NO_IF"]

                # TODO: This is fine for now, although I (karto) think that this
                # is relative to the ref_freq, which can be specified as part of
                # the AIPS SU table.

                # Get rest freq value
                ref_freq = fits_utils._gethduaxis(vis_hdu, 4)[0]
                self.channel_width = np.transpose(
                    np.tile(abs(fq_hdu.data["CH WIDTH"]), (uvfits_nchan, 1))
                ).flatten()
                self.freq_array = np.transpose(
                    ref_freq
                    + fq_hdu.data["IF FREQ"]
                    + np.outer(np.arange(uvfits_nchan), fq_hdu.data["CH WIDTH"])
                ).flatten()
            else:
                # If there's only one window, then the UVFITS file may not have an
                # FQ table, in which case pull the info from the main table
                self.Nfreqs = vis_hdr.pop("NAXIS4")
                self.freq_array = fits_utils._gethduaxis(vis_hdu, 4)
                self.channel_width = np.full(self.Nfreqs, vis_hdr.pop("CDELT4"))
                self.flex_spw_id_array = np.zeros(self.Nfreqs, dtype=int)

            self.polarization_array = np.int32(fits_utils._gethduaxis(vis_hdu, 3))
            # other info -- not required but frequently used
            self.telescope.name = vis_hdr.pop("TELESCOP", None)
            self.telescope.instrument = vis_hdr.pop("INSTRUME", None)
            latitude_degrees = vis_hdr.pop("LAT", None)
            longitude_degrees = vis_hdr.pop("LON", None)
            altitude = vis_hdr.pop("ALT", None)
            x_orientation = vis_hdr.pop("XORIENT", None)
            blt_order_str = vis_hdr.pop("BLTORDER", None)
            if blt_order_str is not None:
                self.blt_order = tuple(blt_order_str.split(", "))
                if self.blt_order == ("bda",):
                    self._blt_order.form = (1,)
            self.history = str(vis_hdr.get("HISTORY", ""))
            pyuvdata_written = "pyuvdata" in self.history
            if not utils.history._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.vis_units = vis_hdr.pop("BUNIT", "uncalib")
            # Added here as a fix since some previous versions of UVData allowed for
            # all caps versions of UNCALIB.
            if self.vis_units == "UNCALIB":
                self.vis_units = "uncalib"

            self.pol_convention = vis_hdr.pop("POLCONV", None)

            # PHSFRAME is not a standard UVFITS keyword, but was used by older
            # versions of pyuvdata. To ensure backwards compatibility, we look
            # for it first to determine the coordinate frame for the data
            phase_center_frame = vis_hdr.pop("PHSFRAME", None)
            # If we don't find the special keyword PHSFRAME, try for the more
            # FITS-standard RADESYS
            if phase_center_frame is None:
                phase_center_frame = vis_hdr.pop("RADESYS", None)
            # If we still don't find anything, try the two 'special' variant
            # names for the coordinate frame that seem to have been documented
            if phase_center_frame is None:
                phase_center_frame = vis_hdr.pop("RADESYSA", None)
            if phase_center_frame is None:
                phase_center_frame = vis_hdr.pop("RADESYSa", None)

            if not read_source:
                # the axis number for phase center depends on if the spw exists

                phase_center_ra_degrees = float(vis_hdr.pop("CRVAL" + str(ra_axis)))
                phase_center_dec_degrees = float(vis_hdr.pop("CRVAL" + str(dec_axis)))

                phase_center_epoch = vis_hdr.pop("EPOCH", None)
                object_name = vis_hdr.pop("OBJECT", None)

                # If we _still_ didn't find the phase frame, take a guess based on the
                # value listed in the EPOCH. The behavior listed here is based off of
                # the AIPS task REGRD (http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?REGRD)
                if phase_center_frame is None:
                    if phase_center_epoch is None:
                        phase_center_frame = "icrs"
                    else:
                        if phase_center_epoch == 1950.0:
                            phase_center_frame = "fk4"
                        else:
                            phase_center_frame = "fk5"
                cat_id = self._add_phase_center(
                    object_name,
                    cat_type="sidereal",
                    cat_lon=np.deg2rad(phase_center_ra_degrees),
                    cat_lat=np.deg2rad(phase_center_dec_degrees),
                    cat_frame=phase_center_frame,
                    cat_epoch=phase_center_epoch,
                )
                self.phase_center_id_array = np.zeros(self.Nblts, dtype=int) + cat_id

            self.extra_keywords = fits_utils._get_extra_keywords(
                vis_hdr,
                keywords_to_skip=[
                    "DATE-OBS",
                    "EPOCH",
                    "OBJECT",
                    "PHSFRAME",
                    "RADESYS",
                    "RADESYSA",
                    "RADESYSa",
                ],
            )

            # Next read the antenna table
            ant_hdu = hdu_list[hdunames["AIPS AN"]]

            # stuff in the header
            if self.telescope.name is None:
                self.telescope.name = ant_hdu.header["ARRNAM"]

            self.gst0 = ant_hdu.header["GSTIA0"]
            self.rdate = ant_hdu.header["RDATE"]
            self.earth_omega = ant_hdu.header["DEGPDY"]
            self.dut1 = ant_hdu.header["UT1UTC"]
            if "TIMESYS" in ant_hdu.header:
                self.timesys = ant_hdu.header["TIMESYS"]
            else:
                # CASA misspells this one
                self.timesys = ant_hdu.header["TIMSYS"]

            prefer_lat_lon_alt = False
            ellipsoid = None
            if "FRAME" in ant_hdu.header:
                if ant_hdu.header["FRAME"] == "ITRF":
                    # uvfits uses ITRF, astropy uses itrs. They are the same.
                    telescope_frame = "itrs"
                elif all(key == "?" for key in ant_hdu.header["FRAME"]):
                    # default to itrs, but use the lat/lon/alt to set the location
                    # if they are available. Note the above code handles all question
                    # marks no matter the length.
                    frame_name = ant_hdu.header["FRAME"]
                    warnings.warn(
                        f"The telescope frame is set to '{frame_name}', "
                        "which generally indicates ignorance. Defaulting the frame to "
                        "'itrs', but this may lead to other warnings or errors."
                    )
                    prefer_lat_lon_alt = True
                    telescope_frame = "itrs"
                else:
                    telescope_frame = ant_hdu.header["FRAME"].lower()
                    if telescope_frame not in ["itrs", "mcmf"]:
                        raise ValueError(
                            f"Telescope frame in file is {telescope_frame}. "
                            "Only 'itrs' and 'mcmf' are currently supported."
                        )
                    if telescope_frame == "mcmf":
                        try:
                            from lunarsky import MoonLocation
                        except ImportError as ie:
                            raise ValueError(
                                "Need to install `lunarsky` package to work with "
                                "MCMF frames."
                            ) from ie

                        ellipsoid = ant_hdu.header.get("ELLIPSOI", "SPHERE")

            else:
                warnings.warn(
                    "Required Antenna keyword 'FRAME' not set; "
                    "Assuming frame is 'ITRF'."
                )
                telescope_frame = "itrs"

            # get telescope location and antenna positions.
            # VLA incorrectly sets ARRAYX/ARRAYY/ARRAYZ to 0, and puts array center
            # in the antenna positions themselves
            if (
                np.isclose(ant_hdu.header["ARRAYX"], 0)
                and np.isclose(ant_hdu.header["ARRAYY"], 0)
                and np.isclose(ant_hdu.header["ARRAYZ"], 0)
            ):
                x_telescope = np.mean(ant_hdu.data["STABXYZ"][:, 0])
                y_telescope = np.mean(ant_hdu.data["STABXYZ"][:, 1])
                z_telescope = np.mean(ant_hdu.data["STABXYZ"][:, 2])
                self.telescope.antenna_positions = ant_hdu.data.field(
                    "STABXYZ"
                ) - np.array([x_telescope, y_telescope, z_telescope])

            else:
                x_telescope = ant_hdu.header["ARRAYX"]
                y_telescope = ant_hdu.header["ARRAYY"]
                z_telescope = ant_hdu.header["ARRAYZ"]
                # AIPS memo #117 says that antenna_positions should be relative to
                # the array center, but in a rotated ECEF frame so that the x-axis
                # goes through the local meridian.
                rot_ecef_positions = ant_hdu.data.field("STABXYZ")
                _, longitude, altitude = utils.LatLonAlt_from_XYZ(
                    np.array([x_telescope, y_telescope, z_telescope]),
                    frame=telescope_frame,
                    ellipsoid=ellipsoid,
                    check_acceptability=run_check_acceptability,
                )
                self.telescope.antenna_positions = utils.ECEF_from_rotECEF(
                    rot_ecef_positions, longitude
                )

            if prefer_lat_lon_alt and (
                latitude_degrees is not None
                and longitude_degrees is not None
                and altitude is not None
            ):
                # only get here if the frame is set to "????", default to ITRS
                self.telescope.location = EarthLocation.from_geodetic(
                    lat=latitude_degrees * units.deg,
                    lon=longitude_degrees * units.deg,
                    height=altitude * units.m,
                )
            else:
                if telescope_frame == "itrs":
                    self.telescope.location = EarthLocation.from_geocentric(
                        x=x_telescope * units.m,
                        y=y_telescope * units.m,
                        z=z_telescope * units.m,
                    )
                else:
                    self.telescope.location = MoonLocation.from_selenocentric(
                        x=x_telescope * units.m,
                        y=y_telescope * units.m,
                        z=z_telescope * units.m,
                    )
                    self.telescope.location.ellipsoid = ellipsoid

            # stuff in columns
            ant_names = ant_hdu.data.field("ANNAME").tolist()
            self.telescope.antenna_names = []
            for name in ant_names:
                # Sometimes CASA writes antnames as bytes not strings.
                # If the ant name is shorter than 8 characters, the trailing
                # characters may be non-ascii.
                # This is technically a FITS violation as FITS requires ascii.
                # So we just ignore any non-ascii bytes in the decode.
                if isinstance(name, bytes):
                    ant_name_str = str(name.decode("utf-8", "ignore"))
                else:
                    ant_name_str = name
                # remove non-printing ascii characters and exclamation points
                ant_name_str = (
                    ant_name_str.replace("\x00", "")
                    .replace("\x07", "")
                    .replace("!", "")
                )
                self.telescope.antenna_names.append(ant_name_str)

            # Note: we no longer subtract one to get to 0-indexed values
            # rather than 1-indexed values. Antenna numbers are not indices
            # but are unique to each antenna.
            self.telescope.antenna_numbers = ant_hdu.data.field("NOSTA")

            self.telescope.Nants = len(self.telescope.antenna_numbers)

            if "DIAMETER" in ant_hdu.columns.names:
                self.telescope.antenna_diameters = ant_hdu.data.field("DIAMETER")

            # If written by older versions of pyuvdata, we don't neccessarily want to
            # trust the mount information, otherwise read it in.
            if ant_hdu.header.get("HASMNT", not pyuvdata_written):
                ref_dict = utils.antenna.MOUNT_NUM2STR_DICT
                if not pyuvdata_written:
                    # Standard UVFITS only supports codes 0-6, so remove the other
                    # codes so as to prevent mislabeling
                    ref_dict = {k: v for k, v in ref_dict.items() if k in range(7)}
                # Default to other if mount code isn't found
                self.telescope.mount_type = [
                    ref_dict.get(mount, "other") for mount in ant_hdu.data["MNTSTA"]
                ]

            if ant_hdu.header.get("HASFEED", not pyuvdata_written):
                # Tranpose here so that the shape is (Nants, Nfeeds)
                # NB: Older versions of the fits reader produce " " here, so we use
                # strip so that the output across versions is a uniform ""
                self.telescope.feed_array = np.array(
                    [
                        [item.strip().lower() for item in ant_hdu.data["POLTYA"]],
                        [item.strip().lower() for item in ant_hdu.data["POLTYB"]],
                    ]
                ).T
                self.telescope.Nfeeds = 2
                self.telescope.feed_angle = np.radians(
                    [ant_hdu.data["POLAA"], ant_hdu.data["POLAB"]]
                ).T

                # If POLYTB is all missing, assuming this is a single-feed setup
                if all(self.telescope.feed_array[:, 1] == ""):
                    self.telescope.feed_array = self.telescope.feed_array[:, :1]
                    self.telescope.feed_angle = self.telescope.feed_angle[:, :1]
                    self.telescope.Nfeeds = 1

            # This will not error because uvfits required keywords ensure we
            # have everything that is required for this method.
            self.set_telescope_params(
                x_orientation=x_orientation,
                run_check=run_check,
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                mount_type=default_mount_type,
            )

            # Now read in the random parameter info
            self._get_parameter_data(
                vis_hdu,
                read_source=read_source,
                run_check_acceptability=run_check_acceptability,
                background_lsts=background_lsts,
                astrometry_library=astrometry_library,
            )

            if read_source:
                su_hdu = hdu_list[hdunames["AIPS SU"]]

                # Set up these arrays so we can assign values to them
                self.phase_center_app_ra = np.zeros(self.Nblts)
                self.phase_center_app_dec = np.zeros(self.Nblts)
                self.phase_center_app_pa = np.zeros(self.Nblts)

                # Alright, we are off to the races!
                for idx in range(len(su_hdu.data)):
                    # Grab the indv source entry
                    sou_info = su_hdu.data[idx]
                    sou_id = sou_info["ID. NO."]
                    sou_name = sou_info["SOURCE"]
                    sou_ra = sou_info["RAEPO"] * (np.pi / 180.0)
                    sou_dec = sou_info["DECEPO"] * (np.pi / 180.0)
                    sou_epoch = sou_info["EPOCH"]

                    if phase_center_frame is not None:
                        sou_frame = phase_center_frame
                    else:
                        # Note that this can result in different frames for different
                        # phase centers, which we disallow on write. Only get here if
                        # the RADESYS header parameter is not in file
                        # (we always write it)
                        if sou_epoch == 1950.0:
                            sou_frame = "fk4"
                        else:
                            sou_frame = "fk5"

                    self._add_phase_center(
                        sou_name,
                        cat_id=sou_id,
                        cat_type="sidereal",
                        cat_lon=sou_ra,
                        cat_lat=sou_dec,
                        cat_frame=sou_frame,
                        cat_epoch=sou_epoch,
                        info_source="uvfits file",
                    )

                # All entries in phase_center_id_array should in in the catalog, so
                # check that now before we continue reading through the file.
                if not all(
                    np.isin(self.phase_center_id_array, list(self.phase_center_catalog))
                ):
                    raise RuntimeError(
                        "The UVFITS file has a malformed AIPS SU table - entries in "
                        "primary data header have no corresponding match."
                    )  # pragma: no cover

            # Calculate the apparent coordinate values
            self._set_app_coords_helper()

            # fix up the uvws if in the NCP baseline coordinate frame.
            # Must be done here because it requires the phase_center_app_dec
            if "UU---NCP" in vis_hdu.data.parnames:
                self.uvw_array = utils.phasing._rotate_one_axis(
                    xyz_array=self.uvw_array[:, :, None],
                    rot_amount=self.phase_center_app_dec - np.pi / 2,
                    rot_axis=0,
                )[:, :, 0]

            if read_data:
                # Now read in the data
                self._get_data(
                    vis_hdu,
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
                    keep_all_metadata=keep_all_metadata,
                    fix_old_proj=fix_old_proj,
                    fix_use_ant_pos=fix_use_ant_pos,
                )

        # check if object has all required UVParameters set
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                allow_flip_conj=True,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

    @copy_replace_short_description(UVData.write_uvfits, style=DocstringStyle.NUMPYDOC)
    def write_uvfits(
        self,
        filename: str,
        *,
        write_lst: bool = True,
        force_phase: bool = False,
        uvw_double: bool = True,
        use_miriad_convention: bool = False,
        run_check: bool = True,
        check_extra: bool = True,
        run_check_acceptability: bool = True,
        strict_uvw_antpos_check: bool = False,
        check_autos: bool = True,
        fix_autos: bool = False,
    ):
        """Write data to a uvfits file."""
        if any(
            entry.get("cat_type") == "near_field"
            for entry in self.phase_center_catalog.values()
        ):
            raise NotImplementedError(
                "Writing near-field phased data to uvfits format is not yet supported."
            )

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_freq_spacing=True,
                check_pol_spacing=True,
                strict_uvw_antpos_check=strict_uvw_antpos_check,
                check_autos=check_autos,
                fix_autos=fix_autos,
            )

        if np.any(~self._check_for_cat_type(["sidereal"])):
            if force_phase:
                print(
                    "The data are not all phased to a sidereal source. Phasing to "
                    "zenith of the first timestamp."
                )
                phase_time = Time(self.time_array[0], format="jd")
                self.phase_to_time(phase_time)
            else:
                raise ValueError(
                    "The data are not all phased to a sidereal source. "
                    "Set force_phase to true to phase the data "
                    "to zenith of the first timestamp before "
                    "writing a uvfits file."
                )

        nchan_list = []
        start_freq_array = []
        delta_freq_array = []
        for idx in self.spw_array:
            chan_mask = self.flex_spw_id_array == idx
            nchan_list += [np.sum(chan_mask)]
            start_freq_array += [self.freq_array[chan_mask][0]]
            # Need the array direction here since channel_width is always supposed
            # to be > 0, but channels can be in decending freq order
            freq_dir = 1.0
            if nchan_list[-1] > 1:
                freq_dir = np.sign(np.median(np.diff(self.freq_array[chan_mask])))
            delta_freq_array += [np.median(self.channel_width[chan_mask]) * freq_dir]

        start_freq_array = np.reshape(np.array(start_freq_array), (1, -1)).astype(
            np.float64
        )

        delta_freq_array = np.reshape(np.array(delta_freq_array), (1, -1)).astype(
            np.float64
        )

        # We've constructed a couple of lists with relevant values, now time to
        # check them to make sure that the data will write correctly

        # Make sure that all the windows are of the same size
        if len(np.unique(nchan_list)) != 1:
            raise IndexError(
                "UVFITS format cannot handle spectral windows of different sizes!"
            )

        # Make sure freq values are greater zero. Note that I think _technically
        # one could write negative frequencies into the dataset, but I am pretty
        # sure that reduction packages may balk hard.
        if np.any(start_freq_array <= 0):
            raise ValueError("Frequency values must be > 0 for UVFITS!")

        # Make sure the delta values are non-zero
        if np.any(delta_freq_array == 0):
            raise ValueError("Something is wrong, frequency values not unique!")

        # If we passed all the above checks, then it's time to fill some extra
        # array values. Note that 'ref_freq' is something of a placeholder for
        # other exciting things...
        ref_freq = start_freq_array[0, 0]

        if self.Npols > 1:
            pol_indexing = np.argsort(np.abs(self.polarization_array))
            polarization_array = self.polarization_array[pol_indexing]
            pol_spacing = polarization_array[1] - polarization_array[0]
        else:
            pol_indexing = np.asarray([0])
            polarization_array = self.polarization_array
            pol_spacing = 1

        # check for unflagged data with nsample = 0. Warn if any found
        wh_nsample0 = np.where(self.nsample_array == 0)
        if np.any(~self.flag_array[wh_nsample0]):
            warnings.warn(
                "Some unflagged data has nsample = 0. Flags and "
                "nsamples are combined in uvfits files such that "
                "these data will appear to be flagged."
            )

        uvfits_data_shape = (
            self.Nblts,
            1,
            1,
            self.Nspws,
            self.Nfreqs // self.Nspws,
            self.Npols,
            1,
        )

        # Reshape the arrays so that they match the uvfits conventions
        # FITS uvw direction convention is opposite ours and Miriad's.
        # So conjugate the visibilities and flip the uvws:
        data_array = np.reshape(np.conj(self.data_array), uvfits_data_shape)
        weights_array = np.reshape(
            self.nsample_array * np.where(self.flag_array, -1, 1), uvfits_data_shape
        )
        data_array = data_array[:, :, :, :, :, pol_indexing, :]
        weights_array = weights_array[:, :, :, :, :, pol_indexing, :]

        uvfits_array_data = np.concatenate(
            [data_array.real, data_array.imag, weights_array], axis=6
        )
        # convert to seconds units
        uvw_array_sec = -1 * self.uvw_array / const.c.to_value("m/s")

        if self.data_array.dtype == "complex128":
            write_precision = 64
        else:
            write_precision = 32

        # uvfits convention is that there are two float32 time_arrays and the
        # float64 sum of them + relevant PZERO = actual JD
        # a common practice is to set the PZERO to the JD at midnight of the first time
        jd_midnight = np.floor(self.time_array[0] - 0.5) + 0.5
        if write_precision == 32:
            time_array1 = (self.time_array - jd_midnight).astype(np.float32)
            time_array2 = (
                self.time_array - jd_midnight - time_array1.astype(np.float64)
            ).astype(np.float32)

            uvw_array1 = np.float32(uvw_array_sec)
            uvw_array2 = np.float32(uvw_array_sec - np.float64(uvw_array1))
        else:
            time_array1 = self.time_array - jd_midnight
            uvw_array1 = uvw_array_sec

        int_time_array = self.integration_time

        # If using MIRIAD convention, we need 1-indexed data
        ant_nums_use = copy.copy(self.telescope.antenna_numbers)
        ant1_array_use = copy.copy(self.ant_1_array)
        ant2_array_use = copy.copy(self.ant_2_array)
        if use_miriad_convention and np.min(ant_nums_use) == 0:
            ant_nums_use += 1
            ant1_array_use += 1
            ant2_array_use += 1

        # Generate baseline IDs
        attempt256 = not use_miriad_convention
        baselines_use = self.antnums_to_baseline(
            ant1_array_use,
            ant2_array_use,
            attempt256=attempt256,
            use_miriad_convention=use_miriad_convention,
        )
        # Set up dictionaries for populating hdu
        # Antenna arrays are populated with actual antenna numbers,
        # that is, numbers that are unique to each antenna.
        # We no longer add 1 to the arrays as the arrays are not indices.

        group_parameter_dict = {
            "UU      ": uvw_array_sec[:, 0],
            "VV      ": uvw_array_sec[:, 1],
            "WW      ": uvw_array_sec[:, 2],
            "DATE    ": time_array1,
            "BASELINE": baselines_use,
            "SOURCE  ": None,
            "ANTENNA1": ant1_array_use,
            "ANTENNA2": ant2_array_use,
            "SUBARRAY": np.ones_like(ant1_array_use),
            "INTTIM  ": int_time_array,
        }

        id_offset = int(0 in self.phase_center_catalog)
        group_parameter_dict["SOURCE  "] = self.phase_center_id_array + id_offset

        pscal_dict = {
            "UU      ": 1.0,
            "VV      ": 1.0,
            "WW      ": 1.0,
            "DATE    ": 1.0,
            "BASELINE": 1.0,
            "SOURCE  ": 1.0,
            "ANTENNA1": 1.0,
            "ANTENNA2": 1.0,
            "SUBARRAY": 1.0,
            "INTTIM  ": 1.0,
        }
        pzero_dict = {
            "UU      ": 0.0,
            "VV      ": 0.0,
            "WW      ": 0.0,
            "DATE    ": jd_midnight,
            "BASELINE": 0.0,
            "SOURCE  ": 0.0,
            "ANTENNA1": 0.0,
            "ANTENNA2": 0.0,
            "SUBARRAY": 0.0,
            "INTTIM  ": 0.0,
        }

        if write_lst:
            # lst is a non-standard entry (it's not in the AIPS memo)
            # but storing it can be useful (e.g. can avoid recalculating it on read)
            # need to store it in 2 parts to get enough accuracy
            # angles in uvfits files are stored in degrees, so first convert to degrees
            lst_array_deg = np.rad2deg(self.lst_array)
            if write_precision == 32:
                lst_array_1 = lst_array_deg.astype(np.float32)
                lst_array_2 = (lst_array_deg - lst_array_1.astype(np.float64)).astype(
                    np.float32
                )
            else:
                lst_array_1 = lst_array_deg
            group_parameter_dict["LST     "] = lst_array_1
            pscal_dict["LST     "] = 1.0
            pzero_dict["LST     "] = 0.0
            if write_precision == 32:
                pscal_dict["LST2    "] = 1.0
                pzero_dict["LST2    "] = 0.0

        # list contains arrays of [u,v,w,date,baseline];
        # each array has shape (Nblts)
        parnames_use = ["UU      ", "VV      ", "WW      ", "DATE    "]
        if write_precision == 32:
            group_parameter_dict["DATE2   "] = time_array2
            pscal_dict["DATE2   "] = 1.0
            pzero_dict["DATE2   "] = 0.0
            parnames_use.append("DATE2   ")
            if uvw_double:
                for ind, name in enumerate(["UU", "VV", "WW"]):
                    keyname = name + "2     "
                    group_parameter_dict[keyname] = uvw_array2[:, ind]
                    pscal_dict[keyname] = 1.0
                    pzero_dict[keyname] = 0.0
                    parnames_use.append(keyname)

        if use_miriad_convention or (
            np.max(ant1_array_use) < 255 and np.max(ant2_array_use) < 255
        ):
            # if the number of antennas is less than 256 then include both the
            # baseline array and the antenna arrays in the group parameters.
            # Otherwise just use the antenna arrays unless writing for mirad.
            # MIRIAD requires the BASELINE column.
            parnames_use.append("BASELINE")
        else:
            warnings.warn(
                "Found antenna numbers > 255 in this data set. This is permitted by "
                "UVFITS standards, but may cause the `importuvfits` utility within "
                "CASA to crash. If attempting to use this data set in CASA, consider "
                "using the measurement set writer method (`write_ms`) instead."
            )

        parnames_use += ["SOURCE  ", "ANTENNA1", "ANTENNA2", "SUBARRAY", "INTTIM  "]

        if write_lst:
            parnames_use.append("LST     ")

        group_parameter_list = [
            group_parameter_dict[parname] for parname in parnames_use
        ]

        if write_precision == 32:
            # add second date part
            parnames_write = copy.deepcopy(parnames_use)
            parnames_write[parnames_write.index("DATE2   ")] = "DATE    "
            if uvw_double:
                parnames_write[parnames_write.index("UU2     ")] = "UU      "
                parnames_write[parnames_write.index("VV2     ")] = "VV      "
                parnames_write[parnames_write.index("WW2     ")] = "WW      "
            if write_lst:
                # add second LST array part
                parnames_use.append("LST2    ")
                parnames_write.append("LST     ")
                group_parameter_list.append(lst_array_2)
        else:
            parnames_write = copy.deepcopy(parnames_use)

        hdu = fits.GroupData(
            uvfits_array_data,
            parnames=parnames_write,
            pardata=group_parameter_list,
            bitpix=(-1 * write_precision),
        )
        hdu = fits.GroupsHDU(hdu)

        for i, key in enumerate(parnames_use):
            hdu.header["PSCAL" + str(i + 1) + "  "] = pscal_dict[key]
            hdu.header["PZERO" + str(i + 1) + "  "] = pzero_dict[key]

        # Create an astropy.time.Time object from first timestamp
        # obs_date0:           Time object in JD from first timestamp
        # obs_date0_rdate_str: YYYY-MM-DD string for reference date (at 00:00 hours)
        #                      This is used to generate DATE-OBS and RDATE keywords
        # obs_date0_zerohrs:   Time object for reference date at 00:00 hours
        #                      Used for GSTIA0 and UT1UTC calculations
        obs_date0 = Time(self.time_array[0], format="jd", scale="utc")
        obs_date0_rdate_str = obs_date0.strftime("%Y-%m-%d")
        obs_date0_zerohrs = Time(obs_date0_rdate_str, format="iso", scale="utc")

        # Per AIPS memo 117, DATE-OBS is the YYYY-MM-DD string
        hdu.header["DATE-OBS"] = obs_date0_rdate_str
        hdu.header["CTYPE2  "] = "COMPLEX "
        hdu.header["CRVAL2  "] = 1.0
        hdu.header["CRPIX2  "] = 1.0
        hdu.header["CDELT2  "] = 1.0

        # Note: This axis is called STOKES to comply with the AIPS memo 117
        # However, this confusing because it is NOT a true Stokes axis,
        #   it is really the polarization axis.
        hdu.header["CTYPE3  "] = "STOKES  "
        hdu.header["CRVAL3  "] = float(polarization_array[0])
        hdu.header["CRPIX3  "] = 1.0
        hdu.header["CDELT3  "] = float(pol_spacing)

        hdu.header["CTYPE4  "] = "FREQ    "
        hdu.header["CRVAL4  "] = ref_freq
        hdu.header["CRPIX4  "] = 1.0
        hdu.header["CDELT4  "] = delta_freq_array[0, 0]

        hdu.header["CTYPE5  "] = "IF      "
        hdu.header["CRVAL5  "] = 1.0
        hdu.header["CRPIX5  "] = 1.0
        hdu.header["CDELT5  "] = 1.0

        if self.Nphase > 1:
            ra_use = 0.0
            dec_use = 0.0
            name_use = "MULTI"
        else:
            phase_dict = list(self.phase_center_catalog.values())[0]
            ra_use = np.rad2deg(phase_dict["cat_lon"])
            dec_use = np.rad2deg(phase_dict["cat_lat"])
            name_use = phase_dict["cat_name"]

        hdu.header["CTYPE6  "] = "RA"
        hdu.header["CRVAL6  "] = ra_use

        hdu.header["CTYPE7  "] = "DEC"
        hdu.header["CRVAL7  "] = dec_use

        hdu.header["BUNIT   "] = self.vis_units
        hdu.header["BSCALE  "] = 1.0
        hdu.header["BZERO   "] = 0.0

        hdu.header["OBJECT  "] = name_use
        hdu.header["TELESCOP"] = self.telescope.name
        lat, lon, alt = self.telescope.location_lat_lon_alt_degrees
        hdu.header["LAT     "] = lat
        hdu.header["LON     "] = lon
        hdu.header["ALT     "] = alt
        hdu.header["INSTRUME"] = self.telescope.instrument
        if self.Nphase == 1:
            hdu.header["EPOCH   "] = float(phase_dict["cat_epoch"])
        # TODO: This is a keyword that should at some point get added for velocity
        # reference stuff, although for right now pyuvdata doesn't do any sort of
        # handling of this, so stub this out for now.
        # hdu.header["SPECSYS "] = "TOPOCENT"

        if self.Nphase == 1:
            hdu.header["RADESYS"] = phase_dict["cat_frame"]
        else:
            frames = [
                ph_dict["cat_frame"] for ph_dict in self.phase_center_catalog.values()
            ]
            unique_frames, frame_count = np.unique(np.array(frames), return_counts=True)
            if unique_frames.size == 1:
                hdu.header["RADESYS"] = unique_frames[0]
            else:
                most_common_frames = unique_frames[frame_count == np.max(frame_count)]
                if most_common_frames.size == 1:
                    hdu.header["RADESYS"] = most_common_frames[0]
                else:
                    preferred_frames = ["fk5", "icrs"] + most_common_frames.tolist()
                    # this is guaranteed to find a match because most_common_frames are
                    # in the list, but it puts more preferred frames first
                    for frame in preferred_frames:
                        if frame in most_common_frames:
                            hdu.header["RADESYS"] = frame
                            break

        if self.pol_convention is not None:
            hdu.header["POLCONV"] = self.pol_convention

        if self.blt_order is not None:
            blt_order_str = ", ".join(self.blt_order)
            hdu.header["BLTORDER"] = blt_order_str

        for line in self.history.splitlines():
            hdu.header.add_history(line)

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.items():
            # header keywords have to be 8 characters or less
            if len(str(key)) > 8:
                warnings.warn(
                    f"key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 as required "
                    "by the uvfits file format."
                )
            keyword = key[:8].upper()
            if isinstance(value, dict | list | np.ndarray):
                raise TypeError(
                    f"Extra keyword {key} is of {type(value)}. "
                    "Only strings and numbers are "
                    "supported in uvfits."
                )

            if keyword == "COMMENT":
                for line in value.splitlines():
                    hdu.header.add_comment(line)
            else:
                hdu.header[keyword] = value

        # ADD the ANTENNA table
        staxof = np.zeros(self.telescope.Nants)

        # If unknown, default to alt-az (0) as the default (as it was before)
        mntsta = np.zeros(self.telescope.Nants)
        if self.telescope.mount_type is not None:
            mntsta = np.array(
                [
                    utils.antenna.MOUNT_STR2NUM_DICT[mount]
                    for mount in self.telescope.mount_type
                ]
            )

        polaa = np.full(self.telescope.Nants, 0.0)
        polab = np.full(self.telescope.Nants, 0.0)
        poltya = np.full(self.telescope.Nants, "", dtype=np.object_)
        poltyb = np.full(self.telescope.Nants, "", dtype=np.object_)

        if self.telescope.feed_array is not None:
            for idx, feeds in enumerate(self.telescope.feed_array):
                # Format everything uppercase as expected by
                feeds = [feed.upper() for feed in feeds]
                if any(item not in ["X", "Y", "L", "R", ""] for item in feeds):
                    raise ValueError(
                        "UVFITS only supports x/y or l/r polarized feed information."
                    )
                # See if we need to flip to UVFITS convention, which wants XY/RL
                feed_a, feed_b = (1, 0) if feeds in [["Y", "X"], ["L", "R"]] else (0, 1)
                poltya[idx] = feeds[feed_a]
                polaa[idx] = np.degrees(self.telescope.feed_angle[idx, feed_a])
                if len(feeds) == 2:
                    poltyb[idx] = feeds[feed_b]
                    polab[idx] = np.degrees(self.telescope.feed_angle[idx, feed_b])

        col1 = fits.Column(
            name="ANNAME", format="8A", array=self.telescope.antenna_names
        )
        # AIPS memo #117 says that antenna_positions should be relative to
        # the array center, but in a rotated ECEF frame so that the x-axis
        # goes through the local meridian.
        rot_ecef_positions = utils.rotECEF_from_ECEF(
            self.telescope.antenna_positions, self.telescope.location.lon.rad
        )
        col2 = fits.Column(name="STABXYZ", format="3D", array=rot_ecef_positions)
        # col3 = fits.Column(name="ORBPARAM", format="0D", array=Norb)
        col4 = fits.Column(name="NOSTA", format="1J", array=ant_nums_use)
        col5 = fits.Column(name="MNTSTA", format="1J", array=mntsta)
        col6 = fits.Column(name="STAXOF", format="1E", array=staxof)
        col7 = fits.Column(name="POLTYA", format="1A", array=poltya)
        col8 = fits.Column(name="POLAA", format="1E", array=polaa)
        # col9 = fits.Column(name='POLCALA', format='0E', array=Npcal, Nspws)
        col10 = fits.Column(name="POLTYB", format="1A", array=poltyb)
        col11 = fits.Column(name="POLAB", format="1E", array=polab)
        # col12 = fits.Column(name='POLCALB', format='0E', array=Npcal, Nspws)
        col_list = [col1, col2, col4, col5, col6, col7, col8, col10, col11]
        # The commented out entires are up above to help check for consistency with the
        # UVFITS format. ORBPARAM, POLCALA, and POLCALB are all technically required,
        # but are all of zero length. Added here to help with debugging.
        if self.telescope.antenna_diameters is not None:
            col12 = fits.Column(
                name="DIAMETER", format="1E", array=self.telescope.antenna_diameters
            )
            col_list.append(col12)

        cols = fits.ColDefs(col_list)

        ant_hdu = fits.BinTableHDU.from_columns(cols)

        ant_hdu.header["EXTNAME"] = "AIPS AN"
        ant_hdu.header["EXTVER"] = 1

        # write XYZ coordinates
        tel_x, tel_y, tel_z = self.telescope._location.xyz()
        ant_hdu.header["ARRAYX"] = tel_x
        ant_hdu.header["ARRAYY"] = tel_y
        ant_hdu.header["ARRAYZ"] = tel_z
        if self.telescope._location.frame == "itrs":
            # uvfits uses "ITRF" rather than "ITRS". They are the same thing.
            ant_hdu.header["FRAME"] = "ITRF"
        else:
            ant_hdu.header["FRAME"] = self.telescope._location.frame.upper()
            # use ELLIPSOI because of FITS 8 character limit for header items
            ant_hdu.header["ELLIPSOI"] = self.telescope._location.ellipsoid

        # TODO Karto: Do this more intelligently in the future
        ant_hdu.header["FREQ"] = self.freq_array[0]

        # RDATE: Reference date when obs started, YYYY-MM-DD
        if (self.rdate is None) or (self.rdate == ""):
            ant_hdu.header["RDATE"] = obs_date0_rdate_str
        else:
            ant_hdu.header["RDATE"] = self.rdate

        # AIPS Memo 117: The value of the GSTIA0 keyword shall be the
        # Greenwich sidereal time in degrees at zero hours on RDATE
        if self.gst0 is None:
            ant_hdu.header["GSTIA0"] = obs_date0_zerohrs.sidereal_time(
                "apparent", "greenwich"
            ).deg
        else:
            ant_hdu.header["GSTIA0"] = self.gst0

        # UT1UTC: the difference between UT1 and UTC in seconds on RDATE
        if self.dut1 is None:
            ant_hdu.header["UT1UTC"] = float(obs_date0_zerohrs.delta_ut1_utc)
        else:
            ant_hdu.header["UT1UTC"] = self.dut1

        if not (self.timesys is None or self.timesys == "UTC"):
            raise ValueError(
                f"This file has a time system {self.timesys}. Only "
                '"UTC" time system files are supported'
            )
        ant_hdu.header["TIMESYS"] = "UTC"
        ant_hdu.header["ARRNAM"] = self.telescope.name
        ant_hdu.header["NO_IF"] = self.Nspws
        # Note the value below is basically 360 deg x num of sidereal days in a year /
        # num of soalr days in a year.
        ant_hdu.header["DEGPDY"] = (
            360.9856438593 if self.earth_omega is None else self.earth_omega
        )
        # This is just a statically defined value
        ant_hdu.header["IATUTC"] = 37.0

        # set mandatory parameters which are not supported by this object
        # (or that we just don't understand)
        ant_hdu.header["NUMORB"] = 0

        # note: Bart had this set to 3. We've set it 0 after aips 117. -jph
        ant_hdu.header["NOPCAL"] = 0

        # TODO: evaluate poltype if/when leakages ever get propagated here
        ant_hdu.header["POLTYPE"] = "X-Y LIN"
        ant_hdu.header["HASMNT"] = self.telescope.mount_type is not None
        ant_hdu.header["HASFEED"] = self.telescope.feed_array is not None

        ant_hdu.header["FREQID"] = 1

        # if there are offsets in images, this could be the culprit
        ant_hdu.header["POLARX"] = 0.0
        ant_hdu.header["POLARY"] = 0.0

        ant_hdu.header["DATUTC"] = 0  # ONLY UTC SUPPORTED

        # we always output right handed coordinates
        ant_hdu.header["XYZHAND"] = "RIGHT"

        fits_tables = [hdu, ant_hdu]
        # If needed, add the FQ table
        if self.Nspws > 1:
            fmt_d = f"{self.Nspws}D"
            fmt_e = f"{self.Nspws}E"
            fmt_j = f"{self.Nspws}J"

            # TODO Karto: Temp implementation until we fix some other things in UVData
            if_freq = start_freq_array - ref_freq
            ch_width = delta_freq_array
            tot_bw = (self.Nfreqs // self.Nspws) * np.abs(delta_freq_array)
            sideband = np.sign(delta_freq_array) * np.ones((1, self.Nspws))

            # FRQSEL is hardcoded at the moment, could think about doing this
            # at least somewhat more intelligently...
            col_list = [
                fits.Column(name="FRQSEL", format="1J", array=[1]),
                fits.Column(name="IF FREQ", unit="HZ", format=fmt_d, array=if_freq),
                fits.Column(name="CH WIDTH", unit="HZ", format=fmt_e, array=ch_width),
                fits.Column(
                    name="TOTAL BANDWIDTH", unit="HZ", format=fmt_e, array=tot_bw
                ),
                fits.Column(name="SIDEBAND", format=fmt_j, array=sideband),
            ]

            fq_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(col_list))

            fq_hdu.header["EXTNAME"] = "AIPS FQ"
            fq_hdu.header["NO_IF"] = self.Nspws
            fits_tables.append(fq_hdu)

        # Always write the SU table
        fmt_d = f"{self.Nspws}D"
        fmt_e = f"{self.Nspws}E"
        fmt_j = f"{self.Nspws}J"

        int_zeros = np.zeros(self.Nphase, dtype=int)
        flt_zeros = np.zeros(self.Nphase, dtype=np.float64)
        zero_arr = np.zeros((self.Nphase, self.Nspws))
        sou_ids = np.zeros(self.Nphase)
        name_arr = np.array(
            [ps_dict["cat_name"] for ps_dict in self.phase_center_catalog.values()]
        )
        cal_code = ["    "] * self.Nphase
        # These are things we need to flip through on a source-by-source basis
        ra_arr = np.zeros(self.Nphase, dtype=np.float64)
        app_ra = np.zeros(self.Nphase, dtype=np.float64)
        dec_arr = np.zeros(self.Nphase, dtype=np.float64)
        app_dec = np.zeros(self.Nphase, dtype=np.float64)
        epo_arr = np.zeros(self.Nphase, dtype=np.float64)
        pm_ra = np.zeros(self.Nphase, dtype=np.float64)
        pm_dec = np.zeros(self.Nphase, dtype=np.float64)
        rest_freq = np.zeros((self.Nphase, self.Nspws), dtype=np.float64)
        for idx, cat_id in enumerate(self.phase_center_catalog):
            phase_dict = self.phase_center_catalog[cat_id]
            # This is a stub for something smarter in the future
            sou_ids[idx] = cat_id + id_offset
            rest_freq[idx][:] = np.mean(self.freq_array)
            pm_ra[idx] = 0.0
            pm_dec[idx] = 0.0
            # So here's the deal -- we need all the objects to be in the same
            # coordinate frame, although nothing in phase_center_catalog forces
            # objects to share the same frame. So we want to make sure that
            # everything lines up with the coordinate frame listed.
            new_ra, new_dec = utils.phasing.transform_sidereal_coords(
                longitude=phase_dict["cat_lon"],
                latitude=phase_dict["cat_lat"],
                in_coord_frame=phase_dict["cat_frame"],
                out_coord_frame=hdu.header["RADESYS"],
                in_coord_epoch=phase_dict.get("cat_epoch"),
                out_coord_epoch=phase_dict.get("cat_epoch"),
                time_array=np.mean(self.time_array),
            )
            ra_arr[idx] = new_ra[0]
            dec_arr[idx] = new_dec[0]

            epo_arr[idx] = phase_dict.get("cat_epoch", 2000.0)

            if any(self.phase_center_id_array == cat_id):
                # Fill in these values if we calculated them for the phase centers,
                # otherwise let them just be zero, since there's no clear reference
                # time to tie them to (and this metadata isn't used by pyuvdata anyway).
                app_ra[idx] = np.median(
                    self.phase_center_app_ra[self.phase_center_id_array == cat_id]
                )

                app_dec[idx] = np.median(
                    self.phase_center_app_dec[self.phase_center_id_array == cat_id]
                )

        ra_arr *= 180.0 / np.pi
        dec_arr *= 180.0 / np.pi
        app_ra *= 180.0 / np.pi
        app_dec *= 180.0 / np.pi

        col_list = [
            fits.Column(name="ID. NO.", format="1J", array=sou_ids),
            fits.Column(name="SOURCE", format="20A", array=name_arr),
            fits.Column(name="QUAL", format="1J", array=int_zeros),
            fits.Column(name="CALCODE", format="4A", array=cal_code),
            fits.Column(name="IFLUX", format=fmt_e, unit="JY", array=zero_arr),
            fits.Column(name="QFLUX", format=fmt_e, unit="JY", array=zero_arr),
            fits.Column(name="UFLUX", format=fmt_e, unit="JY", array=zero_arr),
            fits.Column(name="VFLUX", format=fmt_e, unit="JY", array=zero_arr),
            fits.Column(name="FREQOFF", format=fmt_d, unit="HZ", array=zero_arr),
            fits.Column(name="BANDWIDTH", format="1D", unit="HZ", array=flt_zeros),
            fits.Column(name="RAEPO", format="1D", unit="DEGREES", array=ra_arr),
            fits.Column(name="DECEPO", format="1D", unit="DEGREES", array=dec_arr),
            fits.Column(name="EPOCH", format="1D", unit="YEARS", array=epo_arr),
            fits.Column(name="RAAPP", format="1D", unit="DEGREES", array=app_ra),
            fits.Column(name="DECAPP", format="1D", unit="DEGREES", array=app_dec),
            fits.Column(name="LSRVEL", format=fmt_d, unit="M/SEC", array=zero_arr),
            fits.Column(name="RESTFREQ", format=fmt_d, unit="HZ", array=rest_freq),
            fits.Column(name="PMRA", format="1D", unit="DEG/DAY", array=pm_ra),
            fits.Column(name="PMDEC", format="1D", unit="DEG/DAY", array=pm_dec),
        ]

        su_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(col_list))
        su_hdu.header["EXTNAME"] = "AIPS SU"
        su_hdu.header["NO_IF"] = self.Nspws
        su_hdu.header["FREQID"] = 1
        su_hdu.header["VELDEF"] = "RADIO"
        # TODO: Eventually we want to not have this hardcoded, but pyuvdata at
        # present does not carry around any velocity information. As per usual,
        # I (Karto) am tipping my hand on what I might be working on next...
        su_hdu.header["VELTYP"] = "LSR"
        fits_tables.append(su_hdu)

        # write the file
        hdulist = fits.HDUList(hdus=fits_tables)
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
