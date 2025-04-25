# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading and writing calibration FITS files."""

import os
import warnings

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation
from astropy.io import fits
from docstring_parser import DocstringStyle

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import fits as fits_utils
from . import UVCal

__all__ = ["CALFITS"]


class CALFITS(UVCal):
    """
    Defines a calfits-specific class for reading and writing calfits files.

    This class should not be interacted with directly, instead use the read_calfits
    and write_calfits methods on the UVCal class.

    """

    def write_calfits(
        self,
        filename,
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        clobber=False,
    ):
        """
        Write the data to a calfits file.

        Parameters
        ----------
        filename : str
            The calfits file to write to.
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters before writing the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters before writing the file.
        clobber : bool
            Option to overwrite the filename if the file already exists.

        """
        if (
            self.phase_center_catalog is not None
            or self.phase_center_id_array is not None
            or self.scan_number_array is not None
            or self.ref_antenna_array is not None
        ):
            warnings.warn(
                "The calfits format does not support recording optional phase center, "
                "scan number, or time-varying reference antenna information, and "
                "these information will not be included in the written file."
            )

        if self.Nspws > 1:
            raise ValueError(
                "The calfits format does not support multiple spectral windows"
            )

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_freq_spacing=True,
                check_jones_spacing=True,
                check_time_spacing=True,
            )

        # calfits allows for frequency spacing to not equal channel widths as long as
        # the frequencies are evenly spaced, so only raise spacing error
        if self.freq_array is None:
            ref_freq = self.freq_range[0, 0]
        else:
            ref_freq = self.freq_array[0]
            _, chanwidth_error = self._check_freq_spacing(raise_errors=None)

        if self.cal_type == "gain" and self.Nfreqs > 1:
            if chanwidth_error:
                # this means that the frequencies are evenly spaced but do not
                # match our channel widths. Use some rounding to get a good delta.
                freq_arr_use = self.freq_array
                rounded_spacing = np.around(
                    np.diff(freq_arr_use),
                    int(np.ceil(np.log10(self._freq_array.tols[1]) * -1)),
                )
                delta_freq_array = rounded_spacing[0]
            else:
                delta_freq_array = np.median(self.channel_width)
        else:
            if self.channel_width is not None:
                delta_freq_array = self.channel_width[0]
            else:
                # default to 1 Hz for wide-band cals with Nfreqs=1 and no channel
                # width info
                delta_freq_array = 1.0

        if self.Ntimes > 1:
            time_spacing = np.diff(self.time_array)
            median_int_time = np.median(self.integration_time)
            if np.isclose(time_spacing[0], median_int_time / (24.0 * 60.0**2)):
                time_spacing = median_int_time / (24.0 * 60.0**2)
            else:
                rounded_spacing = np.around(
                    time_spacing,
                    int(
                        np.ceil(np.log10(self._time_array.tols[1] / self.Ntimes) * -1)
                        + 1
                    ),
                )
                time_spacing = rounded_spacing[0]
        else:
            time_spacing = self.integration_time[0] / (24.0 * 60.0**2)
        if self.time_array is None:
            time_zero = np.mean(self.time_range)
        else:
            time_zero = self.time_array[0]

        if self.Njones > 1:
            jones_spacing = self.jones_array[1] - self.jones_array[0]
        else:
            jones_spacing = -1

        prihdr = fits.Header()
        if self.total_quality_array is not None:
            totqualhdr = fits.Header()
            totqualhdr["EXTNAME"] = "TOTQLTY"
        if self.cal_type != "gain":
            sechdr = fits.Header()
            sechdr["EXTNAME"] = "FLAGS"
        # Conforming to fits format
        prihdr["SIMPLE"] = True
        prihdr["TELESCOP"] = self.telescope.name
        tel_x, tel_y, tel_z = self.telescope._location.xyz()
        prihdr["ARRAYX"] = tel_x
        prihdr["ARRAYY"] = tel_y
        prihdr["ARRAYZ"] = tel_z
        prihdr["FRAME"] = self.telescope._location.frame
        if self.telescope._location.ellipsoid is not None:
            # use ELLIPSOI because of FITS 8 character limit for header items
            prihdr["ELLIPSOI"] = self.telescope._location.ellipsoid
        prihdr["LAT"] = self.telescope.location.lat.rad
        prihdr["LON"] = self.telescope.location.lon.rad
        prihdr["ALT"] = self.telescope.location.height.to_value("m")
        prihdr["GNCONVEN"] = self.gain_convention
        prihdr["CALTYPE"] = self.cal_type
        prihdr["CALSTYLE"] = self.cal_style
        if self.sky_catalog is not None:
            prihdr["CATALOG"] = self.sky_catalog
        if self.ref_antenna_name is not None:
            prihdr["REFANT"] = self.ref_antenna_name
        if self.Nsources is not None:
            prihdr["NSOURCES"] = self.Nsources
        if self.baseline_range is not None:
            prihdr["BL_RANGE"] = (
                "[" + ", ".join([str(b) for b in self.baseline_range]) + "]"
            )
        if self.diffuse_model is not None:
            prihdr["DIFFUSE"] = self.diffuse_model
        if self.gain_scale is not None:
            prihdr["GNSCALE"] = self.gain_scale
        if self.pol_convention is not None:
            prihdr["POLCONV"] = self.pol_convention
        if self.Ntimes > 1:
            prihdr["INTTIME"] = median_int_time
        else:
            prihdr["INTTIME"] = self.integration_time[0]

        if self.cal_type == "gain" and self.Nfreqs > 1:
            prihdr["CHWIDTH"] = np.median(self.channel_width)
        else:
            prihdr["CHWIDTH"] = delta_freq_array

        if self.freq_range is not None:
            freq_range_use = self.freq_range[0, :]
        else:
            freq_range_use = self.freq_range
        if self.cal_type == "delay":
            prihdr["FRQRANGE"] = ",".join(map(str, freq_range_use))

        if self.time_range is not None:
            prihdr["TMERANGE"] = ",".join(map(str, self.time_range[0, :]))

        if self.observer:
            prihdr["OBSERVER"] = self.observer

        if self.telescope.instrument:
            prihdr["INSTRUME"] = self.telescope.instrument

        if self.git_origin_cal:
            prihdr["ORIGCAL"] = self.git_origin_cal
        if self.git_hash_cal:
            prihdr["HASHCAL"] = self.git_hash_cal

        if self.quality_array is not None:
            prihdr["HASQLTY"] = True
        else:
            prihdr["HASQLTY"] = False

        # Define primary header values
        # Arrays have (column-major) dimensions of
        # [Nimages, Njones, Ntimes, Nfreqs, 1, Nantennas]
        # For a "delay"-type calibration, Nfreqs is a shallow axis

        # set the axis for number of arrays
        prihdr["CTYPE1"] = ("Narrays", "Number of image arrays.")
        prihdr["CUNIT1"] = "Integer"
        prihdr["CDELT1"] = 1
        prihdr["CRPIX1"] = 1
        prihdr["CRVAL1"] = 1

        # Jones axis
        prihdr["CTYPE2"] = ("JONES", "Jones matrix array")
        prihdr["CUNIT2"] = ("Integer", "representative integer for polarization.")
        prihdr["CRPIX2"] = 1
        prihdr["CRVAL2"] = self.jones_array[0]  # always start with first jones.
        prihdr["CDELT2"] = jones_spacing

        # time axis
        prihdr["CTYPE3"] = ("TIME", "Time axis.")
        prihdr["CUNIT3"] = ("JD", "Time in julian date format")
        prihdr["CRPIX3"] = 1
        prihdr["CRVAL3"] = time_zero
        prihdr["CDELT3"] = time_spacing

        # freq axis
        prihdr["CTYPE4"] = ("FREQS", "Frequency.")
        prihdr["CUNIT4"] = "Hz"
        prihdr["CRPIX4"] = 1
        prihdr["CRVAL4"] = ref_freq
        prihdr["CDELT4"] = delta_freq_array

        # spw axis: number of spectral windows
        prihdr["CTYPE5"] = ("IF", "Spectral window number.")
        prihdr["CUNIT5"] = "Integer"
        prihdr["CRPIX5"] = 1
        prihdr["CRVAL5"] = 1
        prihdr["CDELT5"] = 1

        # antenna axis
        prihdr["CTYPE6"] = ("ANTAXIS", "See ANTARR in ANTENNA extension for values.")
        prihdr["CUNIT6"] = "Integer"
        prihdr["CRPIX6"] = 1
        prihdr["CRVAL6"] = 1
        prihdr["CDELT6"] = -1

        # end standard keywords; begin user-defined keywords
        for key, value in self.extra_keywords.items():
            # header keywords have to be 8 characters or less
            if len(str(key)) > 8:
                warnings.warn(
                    f"key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 as required "
                    "by the calfits file format."
                )
            keyword = key[:8].upper()
            if isinstance(value, dict | list | np.ndarray):
                raise TypeError(
                    f"Extra keyword {key} is of {type(value)}. "
                    "Only strings and numbers are "
                    "supported in calfits."
                )

            if keyword == "COMMENT":
                for line in value.splitlines():
                    prihdr.add_comment(line)
            else:
                prihdr[keyword] = value

        for line in self.history.splitlines():
            prihdr.add_history(line)

        # define data section based on calibration type
        if self.cal_type == "gain":
            calfits_data_shape = (
                self.Nants_data,
                1,
                self.Nfreqs,
                self.Ntimes,
                self.Njones,
                1,
            )
            pridata = np.concatenate(
                [
                    np.reshape(self.gain_array.real, calfits_data_shape),
                    np.reshape(self.gain_array.imag, calfits_data_shape),
                    np.reshape(self.flag_array, calfits_data_shape),
                ],
                axis=-1,
            )

            if self.quality_array is not None:
                pridata = np.concatenate(
                    [pridata, np.reshape(self.quality_array, calfits_data_shape)],
                    axis=-1,
                )

        elif self.cal_type == "delay":
            calfits_data_shape = (self.Nants_data, 1, 1, self.Ntimes, self.Njones, 1)
            pridata = np.reshape(self.delay_array, calfits_data_shape)
            if self.quality_array is not None:
                pridata = np.concatenate(
                    [pridata, np.reshape(self.quality_array, calfits_data_shape)],
                    axis=-1,
                )

            # Set headers for the second hdu containing the flags. Only in
            # cal_type=delay
            # Can't put in primary header because frequency axis is shallow there,
            # but not here
            # Header values are the same as the primary header
            sechdr["CTYPE1"] = ("Narrays", "Number of image arrays.")
            sechdr["CUNIT1"] = "Integer"
            sechdr["CRPIX1"] = 1
            sechdr["CRVAL1"] = 1
            sechdr["CDELT1"] = 1

            sechdr["CTYPE2"] = ("JONES", "Jones matrix array")
            sechdr["CUNIT2"] = ("Integer", "representative integer for polarization.")
            sechdr["CRPIX2"] = 1
            sechdr["CRVAL2"] = self.jones_array[0]  # always start with first jones.
            sechdr["CDELT2"] = jones_spacing

            sechdr["CTYPE3"] = ("TIME", "Time axis.")
            sechdr["CUNIT3"] = ("JD", "Time in julian date format")
            sechdr["CRPIX3"] = 1
            sechdr["CRVAL3"] = time_zero
            sechdr["CDELT3"] = time_spacing

            sechdr["CTYPE4"] = ("FREQS", "Valid frequencies to apply delay.")
            sechdr["CUNIT4"] = "Hz"
            sechdr["CRPIX4"] = 1
            sechdr["CRVAL4"] = ref_freq
            sechdr["CDELT4"] = delta_freq_array

            sechdr["CTYPE5"] = ("IF", "Spectral window number.")
            sechdr["CUNIT5"] = "Integer"
            sechdr["CRPIX5"] = 1
            sechdr["CRVAL5"] = 1
            sechdr["CDELT5"] = 1

            sechdr["CTYPE6"] = (
                "ANTAXIS",
                "See ANTARR in ANTENNA extension for values.",
            )

            # convert from bool to int64; undone on read
            calfits_data_shape = (
                self.Nants_data,
                1,
                self.Nfreqs,
                self.Ntimes,
                self.Njones,
                1,
            )
            flag_array_use = np.repeat(self.flag_array, self.Nfreqs, axis=2)

            secdata = np.reshape(flag_array_use.astype(np.int64), calfits_data_shape)

        if self.total_quality_array is not None:
            # Set headers for the hdu containing the total_quality_array
            # No antenna axis, so we have [Njones, Ntime, Nfreq, 1]
            totqualhdr["CTYPE1"] = ("JONES", "Jones matrix array")
            totqualhdr["CUNIT1"] = (
                "Integer",
                "representative integer for polarization.",
            )
            totqualhdr["CRPIX1"] = 1
            totqualhdr["CRVAL1"] = self.jones_array[0]  # always start with first jones.
            totqualhdr["CDELT1"] = jones_spacing

            totqualhdr["CTYPE2"] = ("TIME", "Time axis.")
            totqualhdr["CUNIT2"] = ("JD", "Time in julian date format")
            totqualhdr["CRPIX2"] = 1
            totqualhdr["CRVAL2"] = time_zero
            totqualhdr["CDELT2"] = time_spacing

            totqualhdr["CTYPE3"] = ("FREQS", "Valid frequencies to apply delay.")
            totqualhdr["CUNIT3"] = "Hz"
            totqualhdr["CRPIX3"] = 1
            totqualhdr["CRVAL3"] = ref_freq
            totqualhdr["CDELT3"] = delta_freq_array

            # spws axis: number of spectral windows
            totqualhdr["CTYPE4"] = ("IF", "Spectral window number.")
            totqualhdr["CUNIT4"] = "Integer"
            totqualhdr["CRPIX4"] = 1
            totqualhdr["CRVAL4"] = 1
            totqualhdr["CDELT4"] = 1

            if self.cal_type == "gain":
                calfits_tqa_shape = (1, self.Nfreqs, self.Ntimes, self.Njones)
            else:
                calfits_tqa_shape = (1, 1, self.Ntimes, self.Njones)
            totqualdata = np.reshape(self.total_quality_array, calfits_tqa_shape)

        # make HDUs
        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)

        polaa = np.zeros(self.telescope.Nants, dtype=float)
        polab = np.zeros(self.telescope.Nants, dtype=float)
        poltya = np.full(self.telescope.Nants, "", dtype="<U1")
        poltyb = np.full(self.telescope.Nants, "", dtype="<U1")
        mntsta = np.zeros(self.telescope.Nants, dtype=int)

        # Should be no "if None" statement here b/c feed_array should be required
        for idx, feeds in enumerate(self.telescope.feed_array):
            # Format everything uppercase as expected by
            feeds = [feed.upper() for feed in feeds]

            # See if we need to flip to UVFITS convention, which wants XY/RL
            feed_a, feed_b = (1, 0) if feeds in [["Y", "X"], ["L", "R"]] else (0, 1)
            poltya[idx] = feeds[feed_a]
            polaa[idx] = np.degrees(self.telescope.feed_angle[idx, feed_a])
            if len(feeds) == 2:
                poltyb[idx] = feeds[feed_b]
                polab[idx] = np.degrees(self.telescope.feed_angle[idx, feed_b])

        # ant HDU
        col1 = fits.Column(
            name="ANTNAME", format="8A", array=self.telescope.antenna_names
        )
        col2 = fits.Column(
            name="ANTINDEX", format="D", array=self.telescope.antenna_numbers
        )
        if self.Nants_data == self.telescope.Nants:
            col3 = fits.Column(name="ANTARR", format="D", array=self.ant_array)
        else:
            # ant_array is shorter than the other columns.
            # Pad the extra rows with -1s. Need to undo on read.
            nants_add = self.telescope.Nants - self.Nants_data
            ant_array_use = np.append(
                self.ant_array, np.zeros(nants_add, dtype=np.int64) - 1
            )
            col3 = fits.Column(name="ANTARR", format="D", array=ant_array_use)
        col4 = fits.Column(
            name="ANTXYZ", format="3D", array=self.telescope.antenna_positions
        )
        col5 = fits.Column(name="POLTYA", format="1A", array=poltya)
        col6 = fits.Column(name="POLAA", format="1E", array=polaa)
        col7 = fits.Column(name="POLTYB", format="1A", array=poltyb)
        col8 = fits.Column(name="POLAB", format="1E", array=polab)
        collist = [col1, col2, col3, col4, col5, col6, col7, col8]

        if self.telescope.antenna_diameters is not None:
            collist.append(
                fits.Column(
                    name="ANTDIAM", format="D", array=self.telescope.antenna_diameters
                )
            )

        if self.telescope.mount_type is not None:
            mntsta = np.array(
                [
                    utils.antenna.MOUNT_STR2NUM_DICT[mount]
                    for mount in self.telescope.mount_type
                ]
            )
            collist.append(fits.Column(name="MNTSTA", format="1J", array=mntsta))

        cols = fits.ColDefs(collist)
        ant_hdu = fits.BinTableHDU.from_columns(cols)
        ant_hdu.header["EXTNAME"] = "ANTENNAS"

        hdulist = fits.HDUList([prihdu, ant_hdu])

        if self.cal_type != "gain":
            sechdu = fits.ImageHDU(data=secdata, header=sechdr)
            hdulist.append(sechdu)

        if self.total_quality_array is not None:
            totqualhdu = fits.ImageHDU(data=totqualdata, header=totqualhdr)
            hdulist.append(totqualhdu)

        hdulist.writeto(filename, overwrite=clobber)
        hdulist.close()

    @copy_replace_short_description(UVCal.read_calfits, style=DocstringStyle.NUMPYDOC)
    def read_calfits(
        self,
        filename,
        *,
        read_data=True,
        default_mount_type="other",
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        astrometry_library=None,
    ):
        """Read data from a calfits file."""
        # update filename attribute
        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

        with fits.open(filename) as fname:
            hdr = fname[0].header.copy()
            hdunames = fits_utils._indexhdus(fname)

            anthdu = fname[hdunames["ANTENNAS"]]
            self.telescope.Nants = anthdu.header["NAXIS2"]
            antdata = anthdu.data
            self.telescope.antenna_names = np.array(list(map(str, antdata["ANTNAME"])))
            self.telescope.antenna_numbers = np.array(
                list(map(int, antdata["ANTINDEX"]))
            )
            self.ant_array = np.array(list(map(int, antdata["ANTARR"])))
            if np.min(self.ant_array) < 0:
                # ant_array was shorter than the other columns, so it was
                # padded with -1s.
                # Remove the padded entries.
                self.ant_array = self.ant_array[np.where(self.ant_array >= 0)[0]]

            if "ANTXYZ" in antdata.names:
                self.telescope.antenna_positions = antdata["ANTXYZ"]

            if "ANTDIAM" in antdata.names:
                self.telescope.antenna_diameters = antdata["ANTDIAM"]

            if "MNTSTA" in antdata.names:
                self.telescope.mount_type = [
                    utils.antenna.MOUNT_NUM2STR_DICT[mount]
                    for mount in antdata["MNTSTA"]
                ]

            self.Njones = hdr.pop("NAXIS2")
            self.jones_array = fits_utils._gethduaxis(fname[0], 2)

            self.telescope.name = hdr.pop("TELESCOP")
            x_orientation = hdr.pop("XORIENT", None)

            if x_orientation is None:
                # This is for the "newer" version of calfits, since XORIENT is no longer
                # written to the header (but we can use it to set the new parameters).
                # Tranpose here so that the shape is (Nants, Nfeeds)
                # NB: Older versions of the fits reader produce " " here, so we use
                # strip so that the output across versions is a uniform ""
                self.telescope.feed_array = np.array(
                    [
                        [item.strip().lower() for item in antdata["POLTYA"]],
                        [item.strip().lower() for item in antdata["POLTYB"]],
                    ]
                ).T
                self.telescope.Nfeeds = 2
                self.telescope.feed_angle = np.radians(
                    [antdata["POLAA"], antdata["POLAB"]]
                ).T

                # If POLYTB is all missing, assuming this is a single-feed setup
                if all(self.telescope.feed_array[:, 1] == ""):
                    self.telescope.feed_array = self.telescope.feed_array[:, :1]
                    self.telescope.feed_angle = self.telescope.feed_angle[:, :1]
                    self.telescope.Nfeeds = 1

            x_telescope = hdr.pop("ARRAYX", None)
            y_telescope = hdr.pop("ARRAYY", None)
            z_telescope = hdr.pop("ARRAYZ", None)
            lat = hdr.pop("LAT", None)
            lon = hdr.pop("LON", None)
            alt = hdr.pop("ALT", None)

            telescope_frame = hdr.pop("FRAME", "itrs")
            ellipsoid = None
            if telescope_frame == "mcmf":
                try:
                    from lunarsky import MoonLocation
                except ImportError as ie:
                    raise ValueError(
                        "Need to install `lunarsky` package to work with MCMF frames."
                    ) from ie
                ellipsoid = hdr.pop("ELLIPSOI", "SPHERE")

            if (
                x_telescope is not None
                and y_telescope is not None
                and z_telescope is not None
            ):
                if telescope_frame == "itrs":
                    self.telescope.location = EarthLocation.from_geocentric(
                        x_telescope, y_telescope, z_telescope, unit="m"
                    )
                else:
                    self.telescope.location = MoonLocation.from_selenocentric(
                        x_telescope, y_telescope, z_telescope, unit="m"
                    )
                    self.telescope.location.ellipsoid = ellipsoid

            elif lat is not None and lon is not None and alt is not None:
                if telescope_frame == "itrs":
                    self.telescope.location = EarthLocation.from_geodetic(
                        lat=lat * units.rad, lon=lon * units.rad, height=alt * units.m
                    )
                else:
                    self.telescope.location = MoonLocation.from_selenodetic(
                        lat=lat * units.rad,
                        lon=lon * units.rad,
                        height=alt * units.m,
                        ellipsoid=ellipsoid,
                    )

            try:
                self.set_telescope_params(
                    x_orientation=x_orientation, mount_type=default_mount_type
                )
            except ValueError as ve:
                warnings.warn(str(ve))

            self.history = str(hdr.get("HISTORY", ""))

            if not utils.history._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                if not self.history.endswith("\n"):
                    self.history += "\n"

                self.history += self.pyuvdata_version_str

            self.gain_convention = hdr.pop("GNCONVEN")
            self.gain_scale = hdr.pop("GNSCALE", None)
            self.pol_convention = hdr.pop("POLCONV", None)
            self.cal_type = hdr.pop("CALTYPE")

            # old files might have a freq range for gain types but we don't want them
            if self.cal_type == "delay":
                self.freq_range = np.array(
                    [list(map(float, hdr.pop("FRQRANGE").split(",")))]
                )
            else:
                hdr.pop("FRQRANGE", None)

            self.cal_style = hdr.pop("CALSTYLE")
            if self.cal_style == "sky":
                self._set_sky()
            elif self.cal_style == "redundant":
                self._set_redundant()

            self.sky_catalog = hdr.pop("CATALOG", None)
            self.ref_antenna_name = hdr.pop("REFANT", None)
            self.Nsources = hdr.pop("NSOURCES", None)
            bl_range_string = hdr.pop("BL_RANGE", None)
            if bl_range_string is not None:
                self.baseline_range = np.asarray(
                    [float(b) for b in bl_range_string.strip("[").strip("]").split(",")]
                )
            self.diffuse_model = hdr.pop("DIFFUSE", None)

            self.observer = hdr.pop("OBSERVER", None)
            self.telescope.instrument = hdr.pop("INSTRUME", None)

            self.git_origin_cal = hdr.pop("ORIGCAL", None)
            self.git_hash_cal = hdr.pop("HASHCAL", None)

            # generate polarization and time array for either cal_type.
            self.Ntimes = hdr.pop("NAXIS3")
            main_hdr_time_array = fits_utils._gethduaxis(fname[0], 3)
            self.integration_time = np.full(self.Ntimes, hdr.pop("INTTIME"))

            # needs to come after Ntimes is defined.
            time_range = hdr.pop("TMERANGE", None)
            if time_range is not None and self.Ntimes == 1:
                self.time_range = np.asarray(
                    list(map(np.float64, time_range.split(",")))
                )
                self.time_range = self.time_range[np.newaxis, :]
            else:
                self.time_array = main_hdr_time_array

            if self.telescope.location is not None:
                proc = self.set_lsts_from_time_array(
                    background=background_lsts, astrometry_library=astrometry_library
                )
            else:
                proc = None

            self.Nspws = hdr.pop("NAXIS5")
            assert self.Nspws == 1, (
                "This file appears to have multiple spectral windows, which is not "
                "supported by the calfits format."
            )
            # subtract 1 to be zero-indexed
            self.spw_array = fits_utils._gethduaxis(fname[0], 5) - 1

            self.Nants_data = hdr.pop("NAXIS6")
            if self.cal_type == "gain":
                self._set_gain()
                self.Nfreqs = hdr.pop("NAXIS4")
                self.freq_array = fits_utils._gethduaxis(fname[0], 4)
                self.channel_width = np.full(self.Nfreqs, hdr.pop("CHWIDTH"))

                self.flex_spw_id_array = np.full(
                    self.Nfreqs, self.spw_array[0], dtype=int
                )
            if self.cal_type == "delay":
                self._set_delay()
                self.Nfreqs = 1

                sechdu = fname[hdunames["FLAGS"]]
                assert self.Nspws == 1, (
                    "This file appears to have multiple spectral windows, which is not "
                    "supported by the calfits format."
                )
                spw_array = fits_utils._gethduaxis(sechdu, 5) - 1

                if not np.allclose(spw_array, self.spw_array):
                    raise ValueError(
                        "Spectral window values are different in FLAGS HDU than"
                        " in primary HDU"
                    )

                time_array = fits_utils._gethduaxis(sechdu, 3)
                if not np.allclose(
                    time_array,
                    main_hdr_time_array,
                    rtol=self._time_array.tols[0],
                    atol=self._time_array.tols[0],
                ):
                    raise ValueError(
                        "Time values are different in FLAGS HDU than in primary HDU"
                    )

                jones_array = fits_utils._gethduaxis(sechdu, 2)
                if not np.allclose(
                    jones_array,
                    self.jones_array,
                    rtol=self._jones_array.tols[0],
                    atol=self._jones_array.tols[0],
                ):
                    raise ValueError(
                        "Jones values are different in FLAGS HDU than in primary HDU"
                    )

            # get data.
            has_quality = hdr.pop("HASQLTY", True)
            if read_data:
                data = fname[0].data
                if self.cal_type == "gain":
                    self.gain_array = (
                        data[:, 0, :, :, :, 0] + 1j * data[:, 0, :, :, :, 1]
                    )
                    self.flag_array = data[:, 0, :, :, :, 2].astype("bool")
                    if has_quality:
                        self.quality_array = data[:, 0, :, :, :, -1]
                if self.cal_type == "delay":
                    self.delay_array = data[:, 0, :, :, :, 0]
                    if has_quality:
                        self.quality_array = data[:, 0, :, :, :, 1]

                    flag_data = sechdu.data
                    self.flag_array = flag_data[:, 0, :, :, :, 0].astype("bool")

                    # Combine the flags down to one windows worth
                    self.flag_array = np.all(self.flag_array, axis=1, keepdims=True)

                # get total quality array if present
                if "TOTQLTY" in hdunames:
                    totqualhdu = fname[hdunames["TOTQLTY"]]
                    self.total_quality_array = totqualhdu.data[0]
                    spw_array = fits_utils._gethduaxis(totqualhdu, 4) - 1
                    if not np.allclose(spw_array, self.spw_array):
                        raise ValueError(
                            "Spectral window values are different in "
                            "TOTQLTY HDU than in primary HDU. primary HDU "
                            f"has {self.spw_array}, TOTQLTY has {spw_array}"
                        )

                    if self.cal_type != "delay":
                        # delay-type files won't have a freq_array
                        freq_array = fits_utils._gethduaxis(totqualhdu, 3)
                        if not np.allclose(
                            freq_array,
                            self.freq_array,
                            rtol=self._freq_array.tols[0],
                            atol=self._freq_array.tols[0],
                        ):
                            raise ValueError(
                                "Frequency values are different in TOTQLTY HDU than"
                                " in primary HDU"
                            )

                    time_array = fits_utils._gethduaxis(totqualhdu, 2)
                    if not np.allclose(
                        time_array,
                        main_hdr_time_array,
                        rtol=self._time_array.tols[0],
                        atol=self._time_array.tols[0],
                    ):
                        raise ValueError(
                            "Time values are different in TOTQLTY HDU than in "
                            "primary HDU"
                        )

                    jones_array = fits_utils._gethduaxis(totqualhdu, 1)
                    if not np.allclose(
                        jones_array,
                        self.jones_array,
                        rtol=self._jones_array.tols[0],
                        atol=self._jones_array.tols[0],
                    ):
                        raise ValueError(
                            "Jones values are different in TOTQLTY HDU than in "
                            "primary HDU"
                        )

                else:
                    self.total_quality_array = None

            self.extra_keywords = fits_utils._get_extra_keywords(hdr)

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
