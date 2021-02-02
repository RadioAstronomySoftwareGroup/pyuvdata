# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Class for reading and writing calibration FITS files."""
import warnings

import numpy as np
from astropy.io import fits

from .uvcal import UVCal
from .. import utils as uvutils

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
        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )

        if self.Nfreqs > 1:
            freq_spacing = self.freq_array[0, 1:] - self.freq_array[0, :-1]
            if not np.isclose(
                np.min(freq_spacing),
                np.max(freq_spacing),
                rtol=self._freq_array.tols[0],
                atol=self._freq_array.tols[1],
            ):
                raise ValueError(
                    "The frequencies are not evenly spaced (probably "
                    "because of a select operation). The calfits format "
                    "does not support unevenly spaced frequencies."
                )
            if np.isclose(freq_spacing[0], self.channel_width):
                freq_spacing = self.channel_width
            else:
                rounded_spacing = np.around(
                    freq_spacing, int(np.ceil(np.log10(self._freq_array.tols[1]) * -1))
                )
                freq_spacing = rounded_spacing[0]
        else:
            freq_spacing = self.channel_width

        if self.Ntimes > 1:
            time_spacing = np.diff(self.time_array)
            if not np.isclose(
                np.min(time_spacing),
                np.max(time_spacing),
                rtol=self._time_array.tols[0],
                atol=self._time_array.tols[1],
            ):
                raise ValueError(
                    "The times are not evenly spaced (probably "
                    "because of a select operation). The calfits format "
                    "does not support unevenly spaced times."
                )
            if np.isclose(time_spacing[0], self.integration_time / (24.0 * 60.0 ** 2)):
                time_spacing = self.integration_time / (24.0 * 60.0 ** 2)
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
            time_spacing = self.integration_time / (24.0 * 60.0 ** 2)

        if self.Njones > 1:
            jones_spacing = np.diff(self.jones_array)
            if np.min(jones_spacing) < np.max(jones_spacing):
                raise ValueError(
                    "The jones values are not evenly spaced."
                    "The calibration fits file format does not"
                    " support unevenly spaced polarizations."
                )
            jones_spacing = jones_spacing[0]
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
        prihdr["TELESCOP"] = self.telescope_name
        if self.telescope_location is not None:
            prihdr["ARRAYX"] = self.telescope_location[0]
            prihdr["ARRAYY"] = self.telescope_location[1]
            prihdr["ARRAYZ"] = self.telescope_location[2]
            prihdr["LAT"] = self.telescope_location_lat_lon_alt_degrees[0]
            prihdr["LON"] = self.telescope_location_lat_lon_alt_degrees[1]
            prihdr["ALT"] = self.telescope_location_lat_lon_alt[2]
        prihdr["GNCONVEN"] = self.gain_convention
        prihdr["CALTYPE"] = self.cal_type
        prihdr["CALSTYLE"] = self.cal_style
        if self.sky_field is not None:
            prihdr["FIELD"] = self.sky_field
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
        prihdr["INTTIME"] = self.integration_time
        prihdr["CHWIDTH"] = self.channel_width
        prihdr["XORIENT"] = self.x_orientation
        if self.cal_type == "delay":
            prihdr["FRQRANGE"] = ",".join(map(str, self.freq_range))
        elif self.freq_range is not None:
            prihdr["FRQRANGE"] = ",".join(map(str, self.freq_range))
        prihdr["TMERANGE"] = ",".join(map(str, self.time_range))

        if self.observer:
            prihdr["OBSERVER"] = self.observer
        if self.git_origin_cal:
            prihdr["ORIGCAL"] = self.git_origin_cal
        if self.git_hash_cal:
            prihdr["HASHCAL"] = self.git_hash_cal

        if self.cal_type == "unknown":
            raise ValueError(
                "unknown calibration type. Do not know how to " "store parameters"
            )

        # Define primary header values
        # Arrays have (column-major) dimensions of
        # [Nimages, Njones, Ntimes, Nfreqs, Nspw, Nantennas]
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
        prihdr["CRVAL3"] = self.time_array[0]
        prihdr["CDELT3"] = time_spacing

        # freq axis
        prihdr["CTYPE4"] = ("FREQS", "Frequency.")
        prihdr["CUNIT4"] = "Hz"
        prihdr["CRPIX4"] = 1
        prihdr["CRVAL4"] = self.freq_array[0][0]
        prihdr["CDELT4"] = freq_spacing

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
                    "key {key} in extra_keywords is longer than 8 "
                    "characters. It will be truncated to 8 as required "
                    "by the calfits file format.".format(key=key)
                )
            keyword = key[:8].upper()
            if isinstance(value, (dict, list, np.ndarray)):
                raise TypeError(
                    "Extra keyword {keyword} is of {keytype}. "
                    "Only strings and numbers are "
                    "supported in calfits.".format(keyword=key, keytype=type(value))
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
            if self.input_flag_array is not None:
                pridata = np.concatenate(
                    [
                        self.gain_array.real[:, :, :, :, :, np.newaxis],
                        self.gain_array.imag[:, :, :, :, :, np.newaxis],
                        self.flag_array[:, :, :, :, :, np.newaxis],
                        self.input_flag_array[:, :, :, :, :, np.newaxis],
                        self.quality_array[:, :, :, :, :, np.newaxis],
                    ],
                    axis=-1,
                )
            else:
                pridata = np.concatenate(
                    [
                        self.gain_array.real[:, :, :, :, :, np.newaxis],
                        self.gain_array.imag[:, :, :, :, :, np.newaxis],
                        self.flag_array[:, :, :, :, :, np.newaxis],
                        self.quality_array[:, :, :, :, :, np.newaxis],
                    ],
                    axis=-1,
                )

        elif self.cal_type == "delay":
            pridata = np.concatenate(
                [
                    self.delay_array[:, :, :, :, :, np.newaxis],
                    self.quality_array[:, :, :, :, :, np.newaxis],
                ],
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
            sechdr["CRVAL3"] = self.time_array[0]
            sechdr["CDELT3"] = time_spacing

            sechdr["CTYPE4"] = ("FREQS", "Valid frequencies to apply delay.")
            sechdr["CUNIT4"] = "Hz"
            sechdr["CRPIX4"] = 1
            sechdr["CRVAL4"] = self.freq_array[0][0]
            sechdr["CDELT4"] = freq_spacing

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
            if self.input_flag_array is not None:
                secdata = np.concatenate(
                    [
                        self.flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis],
                        self.input_flag_array.astype(np.int64)[
                            :, :, :, :, :, np.newaxis
                        ],
                    ],
                    axis=-1,
                )
            else:
                secdata = self.flag_array.astype(np.int64)[:, :, :, :, :, np.newaxis]

        if self.total_quality_array is not None:
            # Set headers for the hdu containing the total_quality_array
            # No antenna axis, so we have [Njones, Ntime, Nfreq, Nspws]
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
            totqualhdr["CRVAL2"] = self.time_array[0]
            totqualhdr["CDELT2"] = time_spacing

            totqualhdr["CTYPE3"] = ("FREQS", "Valid frequencies to apply delay.")
            totqualhdr["CUNIT3"] = "Hz"
            totqualhdr["CRPIX3"] = 1
            totqualhdr["CRVAL3"] = self.freq_array[0][0]
            totqualhdr["CDELT3"] = freq_spacing

            # spws axis: number of spectral windows
            totqualhdr["CTYPE4"] = ("IF", "Spectral window number.")
            totqualhdr["CUNIT4"] = "Integer"
            totqualhdr["CRPIX4"] = 1
            totqualhdr["CRVAL4"] = 1
            totqualhdr["CDELT4"] = 1
            totqualdata = self.total_quality_array

        # make HDUs
        prihdu = fits.PrimaryHDU(data=pridata, header=prihdr)

        # ant HDU
        col1 = fits.Column(name="ANTNAME", format="8A", array=self.antenna_names)
        col2 = fits.Column(name="ANTINDEX", format="D", array=self.antenna_numbers)
        if self.Nants_data == self.Nants_telescope:
            col3 = fits.Column(name="ANTARR", format="D", array=self.ant_array)
        else:
            # ant_array is shorter than the other columns.
            # Pad the extra rows with -1s. Need to undo on read.
            nants_add = self.Nants_telescope - self.Nants_data
            ant_array_use = np.append(
                self.ant_array, np.zeros(nants_add, dtype=np.int64) - 1
            )
            col3 = fits.Column(name="ANTARR", format="D", array=ant_array_use)
        if self.antenna_positions is not None:
            col4 = fits.Column(name="ANTXYZ", format="3D", array=self.antenna_positions)
            cols = fits.ColDefs([col1, col2, col3, col4])
        else:
            cols = fits.ColDefs([col1, col2, col3])
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

    def read_calfits(
        self,
        filename,
        read_data=True,
        background_lsts=True,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
    ):
        """
        Read data from a calfits file.

        Parameters
        ----------
        filename : str
            The calfits file to read from.
        read_data : bool
            Read in the gains or delays, quality arrays and flag arrays.
            If set to False, only the metadata will be read in. Setting read_data to
            False results in a metadata only object.
        background_lsts : bool
            When set to True, the lst_array is calculated in a background thread.
        run_check : bool
            Option to check for the existence and proper shapes of
            parameters after reading in the file.
        check_extra : bool
            Option to check optional parameters as well as required ones.
        run_check_acceptability : bool
            Option to check acceptable range of the values of
            parameters after reading in the file.

        """
        with fits.open(filename) as fname:
            hdr = fname[0].header.copy()
            hdunames = uvutils._fits_indexhdus(fname)

            anthdu = fname[hdunames["ANTENNAS"]]
            self.Nants_telescope = anthdu.header["NAXIS2"]
            antdata = anthdu.data
            self.antenna_names = np.array(list(map(str, antdata["ANTNAME"])))
            self.antenna_numbers = np.array(list(map(int, antdata["ANTINDEX"])))
            self.ant_array = np.array(list(map(int, antdata["ANTARR"])))
            if np.min(self.ant_array) < 0:
                # ant_array was shorter than the other columns, so it was
                # padded with -1s.
                # Remove the padded entries.
                self.ant_array = self.ant_array[np.where(self.ant_array >= 0)[0]]

            if anthdu.header["TFIELDS"] > 3:
                self.antenna_positions = antdata["ANTXYZ"]

            self.channel_width = hdr.pop("CHWIDTH")
            self.integration_time = hdr.pop("INTTIME")
            self.telescope_name = hdr.pop("TELESCOP")

            x_telescope = hdr.pop("ARRAYX", None)
            y_telescope = hdr.pop("ARRAYY", None)
            z_telescope = hdr.pop("ARRAYZ", None)
            lat = hdr.pop("LAT", None)
            lon = hdr.pop("LON", None)
            alt = hdr.pop("ALT", None)
            if (
                x_telescope is not None
                and y_telescope is not None
                and z_telescope is not None
            ):
                self.telescope_location = np.array(
                    [x_telescope, y_telescope, z_telescope]
                )
            elif lat is not None and lon is not None and alt is not None:
                self.telescope_location_lat_lon_alt_degrees = (lat, lon, alt)
            if self.telescope_location is None or self.antenna_positions is None:
                try:
                    self.set_telescope_params()
                except ValueError as ve:
                    warnings.warn(str(ve))

            self.history = str(hdr.get("HISTORY", ""))

            if not uvutils._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                if not self.history.endswith("\n"):
                    self.history += "\n"

                self.history += self.pyuvdata_version_str

            self.time_range = list(map(float, hdr.pop("TMERANGE").split(",")))
            self.gain_convention = hdr.pop("GNCONVEN")
            self.gain_scale = hdr.pop("GNSCALE", None)
            self.x_orientation = hdr.pop("XORIENT")
            self.cal_type = hdr.pop("CALTYPE")
            if self.cal_type == "delay":
                self.freq_range = list(map(float, hdr.pop("FRQRANGE").split(",")))
            else:
                if "FRQRANGE" in hdr:
                    self.freq_range = list(map(float, hdr.pop("FRQRANGE").split(",")))

            self.cal_style = hdr.pop("CALSTYLE")
            if self.cal_style == "sky":
                self._set_sky()
            elif self.cal_style == "redundant":
                self._set_redundant()

            self.sky_field = hdr.pop("FIELD", None)
            self.sky_catalog = hdr.pop("CATALOG", None)
            self.ref_antenna_name = hdr.pop("REFANT", None)
            self.Nsources = hdr.pop("NSOURCES", None)
            bl_range_string = hdr.pop("BL_RANGE", None)
            if bl_range_string is not None:
                self.baseline_range = [
                    float(b) for b in bl_range_string.strip("[").strip("]").split(",")
                ]
            self.diffuse_model = hdr.pop("DIFFUSE", None)

            self.observer = hdr.pop("OBSERVER", None)
            self.git_origin_cal = hdr.pop("ORIGCAL", None)
            self.git_hash_cal = hdr.pop("HASHCAL", None)

            # generate polarization and time array for either cal_type.
            self.Njones = hdr.pop("NAXIS2")
            self.jones_array = uvutils._fits_gethduaxis(fname[0], 2)
            self.Ntimes = hdr.pop("NAXIS3")
            self.time_array = uvutils._fits_gethduaxis(fname[0], 3)

            if self.telescope_location is not None:
                proc = self.set_lsts_from_time_array(background=background_lsts)
            else:
                proc = None

            self.Nspws = hdr.pop("NAXIS5")
            # subtract 1 to be zero-indexed
            self.spw_array = uvutils._fits_gethduaxis(fname[0], 5) - 1

            self.Nants_data = hdr.pop("NAXIS6")
            if self.cal_type == "gain":
                self._set_gain()

                # generate frequency array from primary data unit.
                self.Nfreqs = hdr.pop("NAXIS4")
                self.freq_array = uvutils._fits_gethduaxis(fname[0], 4)
                self.freq_array.shape = (self.Nspws,) + self.freq_array.shape
            if self.cal_type == "delay":
                self._set_delay()

                sechdu = fname[hdunames["FLAGS"]]
                # generate frequency array from flag data unit
                # (no freq axis in primary).
                self.Nfreqs = sechdu.header["NAXIS4"]
                self.freq_array = uvutils._fits_gethduaxis(sechdu, 4)
                self.freq_array.shape = (self.Nspws,) + self.freq_array.shape

                spw_array = uvutils._fits_gethduaxis(sechdu, 5) - 1

                if not np.allclose(spw_array, self.spw_array):
                    raise ValueError(
                        "Spectral window values are different in FLAGS HDU than"
                        " in primary HDU"
                    )

                time_array = uvutils._fits_gethduaxis(sechdu, 3)
                if not np.allclose(
                    time_array,
                    self.time_array,
                    rtol=self._time_array.tols[0],
                    atol=self._time_array.tols[0],
                ):
                    raise ValueError(
                        "Time values are different in FLAGS HDU than in primary HDU"
                    )

                jones_array = uvutils._fits_gethduaxis(sechdu, 2)
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
            if read_data:
                data = fname[0].data
                if self.cal_type == "gain":
                    self.gain_array = (
                        data[:, :, :, :, :, 0] + 1j * data[:, :, :, :, :, 1]
                    )
                    self.flag_array = data[:, :, :, :, :, 2].astype("bool")
                    if hdr.pop("NAXIS1") == 5:
                        self.input_flag_array = data[:, :, :, :, :, 3].astype("bool")
                        self.quality_array = data[:, :, :, :, :, 4]
                    else:
                        self.quality_array = data[:, :, :, :, :, 3]

                if self.cal_type == "delay":

                    self.delay_array = data[:, :, :, :, :, 0]
                    self.quality_array = data[:, :, :, :, :, 1]

                    flag_data = sechdu.data
                    if sechdu.header["NAXIS1"] == 2:
                        self.flag_array = flag_data[:, :, :, :, :, 0].astype("bool")
                        self.input_flag_array = flag_data[:, :, :, :, :, 1].astype(
                            "bool"
                        )
                    else:
                        self.flag_array = flag_data[:, :, :, :, :, 0].astype("bool")

                # get total quality array if present
                if "TOTQLTY" in hdunames:
                    totqualhdu = fname[hdunames["TOTQLTY"]]
                    self.total_quality_array = totqualhdu.data
                    spw_array = uvutils._fits_gethduaxis(totqualhdu, 4) - 1
                    if not np.allclose(spw_array, self.spw_array):
                        raise ValueError(
                            "Spectral window values are different in "
                            "TOTQLTY HDU than in primary HDU. primary HDU "
                            "has {pspw}, TOTQLTY has {tspw}".format(
                                pspw=self.spw_array, tspw=spw_array
                            )
                        )

                    if self.cal_type != "delay":
                        # delay-type files won't have a freq_array
                        freq_array = uvutils._fits_gethduaxis(totqualhdu, 3)
                        freq_array.shape = (self.Nspws,) + freq_array.shape
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

                    time_array = uvutils._fits_gethduaxis(totqualhdu, 2)
                    if not np.allclose(
                        time_array,
                        self.time_array,
                        rtol=self._time_array.tols[0],
                        atol=self._time_array.tols[0],
                    ):
                        raise ValueError(
                            "Time values are different in TOTQLTY HDU than in "
                            "primary HDU"
                        )

                    jones_array = uvutils._fits_gethduaxis(totqualhdu, 1)
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

            self.extra_keywords = uvutils._get_fits_extra_keywords(hdr)

        # wait for LSTs if set in background
        if proc is not None:
            proc.join()

        if run_check:
            self.check(
                check_extra=check_extra, run_check_acceptability=run_check_acceptability
            )
