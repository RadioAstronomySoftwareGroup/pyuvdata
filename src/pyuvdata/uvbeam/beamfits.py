# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Class for reading and writing beamfits files."""

import os
import warnings

import numpy as np
from astropy.io import fits
from docstring_parser import DocstringStyle

from .. import utils
from ..docstrings import copy_replace_short_description
from ..utils.io import fits as fits_utils
from . import UVBeam

__all__ = ["BeamFITS"]

hpx_primary_ax_nums = {
    "pixel": 1,
    "freq": 2,
    "feed_pol": 3,
    "spw": 4,
    "basisvec": 5,
    "complex": 6,
}
reg_primary_ax_nums = {
    "img_ax1": 1,
    "img_ax2": 2,
    "freq": 3,
    "feed_pol": 4,
    "spw": 5,
    "basisvec": 6,
    "complex": 7,
}

hxp_basisvec_ax_nums = {"pixel": 1, "ncomp": 2, "basisvec": 3}
reg_basisvec_ax_nums = {"img_ax1": 1, "img_ax2": 2, "ncomp": 3, "basisvec": 4}

fits_axisname_dict = {
    "hpx_inds": "PIX_IND",
    "azimuth": "AZIMUTH",
    "zen_angle": "ZENANGLE",
    "zenorth_x": "ZENX-SIN",
    "zenorth_y": "ZENY-SIN",
}


class BeamFITS(UVBeam):
    """
    Defines a fits-specific subclass of UVBeam for reading and writing beamfits files.

    This class should not be interacted with directly, instead use the
    read_beamfits and write_beamfits methods on the UVBeam class.

    The beamfits format supports regularly gridded or healpix beam files.
    The format defined here for healpix beams is not compatible with true healpix
    formats because it needs to support multiple dimensions (e.g. polarization,
    frequency, efield vectors).
    """

    # @profile
    @copy_replace_short_description(UVBeam.read_beamfits, style=DocstringStyle.NUMPYDOC)
    def read_beamfits(
        self,
        filename,
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        check_auto_power=True,
        fix_auto_power=True,
        freq_range=None,
        az_range=None,
        za_range=None,
        mount_type=None,
    ):
        """Read the data from a beamfits file."""
        # update filename attribute
        basename = os.path.basename(filename)
        self.filename = [basename]
        self._filename.form = (1,)

        with fits.open(filename) as fname:
            primary_hdu = fname[0]
            primary_header = primary_hdu.header.copy()
            hdunames = fits_utils._indexhdus(fname)  # find the rest of the tables
            data = primary_hdu.data
            # only support simple antenna_types for now.
            # support for phased arrays should be added
            self._set_simple()
            self.beam_type = primary_header.pop("BTYPE", None)
            if self.beam_type is not None:
                self.beam_type = self.beam_type.lower()
            else:
                bunit = primary_header.pop("BUNIT", None)
                if bunit is not None and bunit.lower().strip() == "jy/beam":
                    self.beam_type = "power"

            if self.beam_type == "intensity":
                self.beam_type = "power"

            n_dimensions = primary_header.pop("NAXIS")
            ctypes = [
                primary_header[ctype].lower()
                for ctype in (key for key in primary_header if "ctype" in key.lower())
            ]

            self.pixel_coordinate_system = primary_header.pop("COORDSYS", None)
            if self.pixel_coordinate_system is None:
                for cs, cs_dict in self.coordinate_system_dict.items():
                    ax_names = [
                        fits_axisname_dict[ax].lower() for ax in cs_dict["axes"]
                    ]
                    if ax_names == ctypes[0 : len(ax_names)]:
                        coord_list = ctypes[0 : len(ax_names)]
                        self.pixel_coordinate_system = cs
            else:
                ax_names = [
                    fits_axisname_dict[ax].lower()
                    for ax in self.coordinate_system_dict[self.pixel_coordinate_system][
                        "axes"
                    ]
                ]
                coord_list = ctypes[0 : len(ax_names)]
                if coord_list != ax_names:
                    raise ValueError(
                        "Coordinate axis list does not match coordinate system"
                    )

            if self.pixel_coordinate_system == "healpix":
                if az_range is not None or za_range is not None:
                    raise ValueError(
                        "az_range and za_range are not supported for healpix beams!"
                    )

                # get pixel values out of HPX_IND extension
                hpx_hdu = fname[hdunames["HPX_INDS"]]
                self.Npixels = hpx_hdu.header["NAXIS2"]
                hpx_data = hpx_hdu.data
                self.pixel_array = hpx_data["hpx_inds"]

                ax_nums = hpx_primary_ax_nums
                self.nside = primary_header.pop("NSIDE", None)
                self.ordering = primary_header.pop("ORDERING", None)
                data_Npixels = primary_header.pop("NAXIS" + str(ax_nums["pixel"]))
                if data_Npixels != self.Npixels:
                    raise ValueError(
                        "Number of pixels in HPX_IND extension does "
                        "not match number of pixels in data array"
                    )
            else:
                ax_nums = reg_primary_ax_nums
                self.Naxes1 = primary_header.pop("NAXIS" + str(ax_nums["img_ax1"]))
                self.Naxes2 = primary_header.pop("NAXIS" + str(ax_nums["img_ax2"]))

                self.axis1_array = fits_utils._gethduaxis(
                    primary_hdu, ax_nums["img_ax1"]
                )
                self.axis2_array = fits_utils._gethduaxis(
                    primary_hdu, ax_nums["img_ax2"]
                )

                # if units aren't defined they are degrees by FITS convention
                # convert degrees to radians because UVBeam uses radians
                axis1_units = primary_header.pop(
                    "CUNIT" + str(ax_nums["img_ax1"]), "deg"
                )
                if axis1_units == "deg":
                    self.axis1_array = np.deg2rad(self.axis1_array)
                elif axis1_units != "rad":
                    raise ValueError(
                        'Units of first axis array are not "deg" or "rad".'
                    )

                if az_range is not None:
                    azmin = np.where(self.axis1_array >= np.deg2rad(az_range[0]))[0][0]
                    azmax = (
                        np.where(self.axis1_array <= np.deg2rad(az_range[1]))[0][-1] + 1
                    )
                    self.Naxes1 = azmax - azmin
                    az_mask = slice(azmin, azmax)
                    self.axis1_array = self.axis1_array[az_mask]

                axis2_units = primary_header.pop(
                    "CUNIT" + str(ax_nums["img_ax2"]), "deg"
                )
                if axis2_units == "deg":
                    self.axis2_array = np.deg2rad(self.axis2_array)
                elif axis2_units != "rad":
                    raise ValueError(
                        'Units of second axis array are not "deg" or "rad".'
                    )

                if za_range is not None:
                    zamin = np.where(self.axis2_array >= np.deg2rad(za_range[0]))[0][0]
                    zamax = (
                        np.where(self.axis2_array <= np.deg2rad(za_range[1]))[0][-1] + 1
                    )
                    self.Naxes2 = zamax - zamin
                    za_mask = slice(zamin, zamax)
                    self.axis2_array = self.axis2_array[za_mask]

            n_efield_dims = max(ax_nums[key] for key in ax_nums)

            # shapes
            if (
                primary_header.pop("CTYPE" + str(ax_nums["freq"])).lower().strip()
                == "freq"
            ):
                self.Nfreqs = primary_header.pop("NAXIS" + str(ax_nums["freq"]))

            if n_dimensions > ax_nums["spw"] - 1 and (
                primary_header.pop("CTYPE" + str(ax_nums["spw"])).lower().strip()
                == "if"
            ):
                file_nspws = primary_header.pop("NAXIS" + str(ax_nums["spw"]), None)
                if file_nspws > 1:
                    raise NotImplementedError(
                        "UVBeam does not support having a spectral window axis "
                        "larger than one."
                    )

            if n_dimensions > ax_nums["basisvec"] - 1 and (
                primary_header.pop("CTYPE" + str(ax_nums["basisvec"])).lower().strip()
                == "vecind"
            ):
                self.Naxes_vec = primary_header.pop(
                    "NAXIS" + str(ax_nums["basisvec"]), None
                )

            if self.Naxes_vec is None and self.beam_type == "power":
                if self.Naxes_vec is None:
                    self.Naxes_vec = 1

                # add extra empty dimensions to data_array as appropriate
                while len(data.shape) < n_efield_dims - 1:
                    data = np.expand_dims(data, axis=0)

            self.freq_array = fits_utils._gethduaxis(primary_hdu, ax_nums["freq"])
            # default frequency axis is Hz, but check for corresonding CUNIT
            freq_units = primary_header.pop("CUNIT" + str(ax_nums["freq"]), "Hz")
            if freq_units != "Hz":
                freq_factor = {"kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
                if freq_units in freq_factor:
                    self.freq_array = self.freq_array * freq_factor[freq_units]
                else:
                    raise ValueError("Frequency units not recognized.")

            if freq_range is not None:
                fmin = np.where(self.freq_array >= freq_range[0])[0][0]
                fmax = np.where(self.freq_array <= freq_range[1])[0][-1] + 1
                freq_mask = slice(fmin, fmax)
                self.Nfreqs = fmax - fmin
                self.freq_array = self.freq_array[freq_mask]

            if az_range is not None:
                data = data[..., az_mask]
            if za_range is not None:
                data = data[..., za_mask, :]
            if freq_range is not None:
                if self.pixel_coordinate_system == "healpix":
                    data = data[..., freq_mask, :]
                else:
                    data = data[..., freq_mask, :, :]
            if self.beam_type == "power":
                # check for case where the data is complex (e.g. for xy beams)
                if n_dimensions > ax_nums["complex"] - 1:
                    complex_arrs = np.split(data, 2, axis=0)
                    self.data_array = np.squeeze(
                        complex_arrs[0] + 1j * complex_arrs[1], axis=0
                    )
                else:
                    self.data_array = data

                # Note: This axis is called STOKES by analogy with the equivalent
                # uvfits axis
                # However, this is confusing because it is NOT a true Stokes axis,
                #   it is really the polarization axis.
                if (
                    primary_header.pop("CTYPE" + str(ax_nums["feed_pol"]))
                    .lower()
                    .strip()
                    == "stokes"
                ):
                    self.Npols = primary_header.pop("NAXIS" + str(ax_nums["feed_pol"]))

                self.polarization_array = np.int32(
                    fits_utils._gethduaxis(primary_hdu, ax_nums["feed_pol"])
                )
                self._set_power()
            elif self.beam_type == "efield":
                self._set_efield()
                if n_dimensions < n_efield_dims:
                    raise ValueError(
                        "beam_type is efield and data dimensionality is too low"
                    )
                self.data_array = data[0] + 1j * data[1]
            else:
                raise ValueError(
                    f"Unknown beam_type: {self.beam_type}, beam_type should be "
                    '"efield" or "power".'
                )

            self.data_array = np.squeeze(self.data_array, axis=1)
            self.data_normalization = primary_header.pop("NORMSTD", None)

            self.telescope_name = primary_header.pop("TELESCOP")
            self.feed_name = primary_header.pop("FEED", None)
            self.feed_version = primary_header.pop("FEEDVER", None)
            self.model_name = primary_header.pop("MODEL", None)
            self.model_version = primary_header.pop("MODELVER", None)
            self.mount_type = primary_header.pop("MNTSTA", mount_type)
            x_orientation = primary_header.pop("XORIENT", "east")
            feedlist = primary_header.pop("FEEDLIST", None)
            if feedlist is not None:
                self.feed_array = np.array(feedlist[1:-1].split(", "))
                self.Nfeeds = len(self.feed_array)
            feedang = primary_header.pop("FEEDANG", None)
            if feedang is not None:
                self.feed_angle = np.array(
                    [float(item) for item in feedang[1:-1].split(", ")]
                )

            self.history = str(primary_header.get("HISTORY", ""))
            if not utils.history._check_history_version(
                self.history, self.pyuvdata_version_str
            ):
                self.history += self.pyuvdata_version_str

            self.extra_keywords = fits_utils._get_extra_keywords(primary_header)

            # read BASISVEC HDU if present
            if "BASISVEC" in hdunames:
                basisvec_hdu = fname[hdunames["BASISVEC"]]
                basisvec_header = basisvec_hdu.header
                self.basis_vector_array = basisvec_hdu.data

                if self.pixel_coordinate_system == "healpix":
                    basisvec_ax_nums = hxp_basisvec_ax_nums
                    if (
                        basisvec_header["CTYPE" + str(basisvec_ax_nums["pixel"])]
                        .lower()
                        .strip()
                        != "pix_ind"
                    ):
                        raise ValueError(
                            "First axis in BASISVEC HDU must be 'Pix_Ind' for "
                            "healpix beams"
                        )

                    basisvec_Npixels = basisvec_header.pop(
                        "NAXIS" + str(basisvec_ax_nums["pixel"])
                    )

                    if basisvec_Npixels != self.Npixels:
                        raise ValueError(
                            "Number of pixels in BASISVEC HDU does not match "
                            "primary HDU"
                        )
                else:
                    if az_range is not None:
                        self.basis_vector_array = self.basis_vector_array[..., az_mask]
                    if za_range is not None:
                        self.basis_vector_array = self.basis_vector_array[
                            ..., za_mask, :
                        ]

                    basisvec_ax_nums = reg_basisvec_ax_nums
                    basisvec_coord_list = [
                        basisvec_header[
                            "CTYPE" + str(basisvec_ax_nums["img_ax1"])
                        ].lower(),
                        basisvec_header[
                            "CTYPE" + str(basisvec_ax_nums["img_ax2"])
                        ].lower(),
                    ]
                    basisvec_axis1_array = fits_utils._gethduaxis(
                        basisvec_hdu, basisvec_ax_nums["img_ax1"]
                    )
                    basisvec_axis2_array = fits_utils._gethduaxis(
                        basisvec_hdu, basisvec_ax_nums["img_ax2"]
                    )

                    # if units aren't defined they are degrees by FITS convention
                    # convert degrees to radians because UVBeam uses radians
                    axis1_units = basisvec_header.pop(
                        "CUNIT" + str(basisvec_ax_nums["img_ax1"]), "deg"
                    )
                    if axis1_units == "deg":
                        basisvec_axis1_array = np.deg2rad(basisvec_axis1_array)
                    elif axis1_units != "rad":
                        raise ValueError(
                            "Units of first axis array in BASISVEC HDU are not"
                            " 'deg' or 'rad'."
                        )
                    axis2_units = basisvec_header.pop(
                        "CUNIT" + str(basisvec_ax_nums["img_ax2"]), "deg"
                    )
                    if axis2_units == "deg":
                        basisvec_axis2_array = np.deg2rad(basisvec_axis2_array)
                    elif axis2_units != "rad":
                        raise ValueError(
                            "Units of second axis array in BASISVEC HDU are not"
                            " 'deg' or 'rad'."
                        )

                    if az_range is not None:
                        basisvec_axis1_array = basisvec_axis1_array[az_mask]
                    if not np.all(basisvec_axis1_array == self.axis1_array):
                        raise ValueError(
                            "First image axis in BASISVEC HDU does not match "
                            "primary HDU"
                        )
                    if za_range is not None:
                        basisvec_axis2_array = basisvec_axis2_array[za_mask]
                    if not np.all(basisvec_axis2_array == self.axis2_array):
                        raise ValueError(
                            "Second image axis in BASISVEC HDU does not "
                            "match primary HDU"
                        )
                    if basisvec_coord_list != coord_list:
                        raise ValueError(
                            "Pixel coordinate list in BASISVEC HDU does not "
                            "match primary HDU"
                        )

                basisvec_Naxes_vec = basisvec_header[
                    "NAXIS" + str(basisvec_ax_nums["basisvec"])
                ]
                self.Ncomponents_vec = basisvec_header[
                    "NAXIS" + str(basisvec_ax_nums["ncomp"])
                ]

                basisvec_cs = basisvec_header["COORDSYS"]
                if basisvec_cs != self.pixel_coordinate_system:
                    raise ValueError(
                        "Pixel coordinate system in BASISVEC HDU does "
                        "not match primary HDU"
                    )

                if basisvec_Naxes_vec != self.Naxes_vec:
                    raise ValueError(
                        "Number of vector coordinate axes in BASISVEC "
                        "HDU does not match primary HDU"
                    )

            # check to see if BANDPARM HDU exists and read it out if it does
            if "BANDPARM" in hdunames:
                bandpass_hdu = fname[hdunames["BANDPARM"]]
                bandpass_header = bandpass_hdu.header.copy()
                self.reference_impedance = bandpass_header.pop("ref_imp", None)

                freq_data = bandpass_hdu.data
                columns = [c.name for c in freq_data.columns]
                self.bandpass_array = freq_data["bandpass"]
                if freq_range is not None:
                    self.bandpass_array = self.bandpass_array[freq_mask]

                if "rx_temp" in columns:
                    self.receiver_temperature_array = freq_data["rx_temp"]
                    if freq_range is not None:
                        self.receiver_temperature_array = (
                            self.receiver_temperature_array[freq_mask]
                        )
                if "loss" in columns:
                    self.loss_array = freq_data["loss"]
                    if freq_range is not None:
                        self.loss_array = self.loss_array[freq_mask]
                if "mismatch" in columns:
                    self.mismatch_array = freq_data["mismatch"]
                    if freq_range is not None:
                        self.mismatch_array = self.mismatch_array[freq_mask]
                if "s11" in columns:
                    s11 = freq_data["s11"]
                    s12 = freq_data["s12"]
                    s21 = freq_data["s21"]
                    s22 = freq_data["s22"]
                    self.s_parameters = np.zeros((4, len(s11)))
                    self.s_parameters[0, :] = s11
                    self.s_parameters[1, :] = s12
                    self.s_parameters[2, :] = s21
                    self.s_parameters[3, :] = s22
                    if freq_range is not None:
                        self.s_parameters = self.s_parameters[:, freq_mask]
            else:
                # no bandpass information, set it to an array of ones
                self.bandpass_array = np.ones(self.Nfreqs)

        # Handle x-orientation keyword here
        if self.feed_angle is None or self.feed_array is None:
            self.set_feeds_from_x_orientation(x_orientation)

        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )

    @copy_replace_short_description(
        UVBeam.write_beamfits, style=DocstringStyle.NUMPYDOC
    )
    def write_beamfits(
        self,
        filename,
        *,
        run_check=True,
        check_extra=True,
        run_check_acceptability=True,
        check_auto_power=True,
        fix_auto_power=False,
        clobber=False,
    ):
        """Write the data to a beamfits file."""
        if run_check:
            self.check(
                check_extra=check_extra,
                run_check_acceptability=run_check_acceptability,
                check_auto_power=check_auto_power,
                fix_auto_power=fix_auto_power,
            )

        if self.antenna_type != "simple":
            raise ValueError(
                "This beam fits writer currently only supports "
                "simple (rather than phased array) antenna beams"
            )

        if self.Nfreqs > 1:
            freq_spacing = self.freq_array[1:] - self.freq_array[:-1]
            if not utils.tools._test_array_constant(
                freq_spacing, tols=self._freq_array.tols
            ):
                raise ValueError(
                    "The frequencies are not evenly spaced (probably "
                    "because of a select operation). The beamfits format "
                    "does not support unevenly spaced frequencies."
                )
            freq_spacing = freq_spacing[0]
        else:
            freq_spacing = 1

        if self.pixel_coordinate_system == "healpix":
            ax_nums = hpx_primary_ax_nums
        else:
            ax_nums = reg_primary_ax_nums
            if self.Naxes1 > 1:
                axis1_spacing = self.axis1_array[1] - self.axis1_array[0]
            else:
                axis1_spacing = 1

            if self.Naxes2 > 1:
                axis2_spacing = self.axis2_array[1] - self.axis2_array[0]
            else:
                axis2_spacing = 1

        primary_header = fits.Header()

        # Conforming to fits format
        primary_header["SIMPLE"] = True
        primary_header["BITPIX"] = 32

        primary_header["BTYPE"] = self.beam_type
        primary_header["NORMSTD"] = self.data_normalization
        primary_header["COORDSYS"] = self.pixel_coordinate_system

        # metadata
        primary_header["TELESCOP"] = self.telescope_name
        primary_header["FEED"] = self.feed_name
        primary_header["FEEDVER"] = self.feed_version
        primary_header["MODEL"] = self.model_name
        primary_header["MODELVER"] = self.model_version

        if self.feed_array is not None:
            primary_header["FEEDLIST"] = "[" + ", ".join(self.feed_array) + "]"
        if self.feed_angle is not None:
            primary_header["FEEDANG"] = (
                "[" + ", ".join(str(item) for item in self.feed_angle) + "]"
            )
        if self.mount_type is not None:
            primary_header["MNTSTA"] = self.mount_type

        if self.pixel_coordinate_system == "healpix":
            primary_header["NSIDE"] = self.nside
            primary_header["ORDERING"] = self.ordering

            # set up pixel axis
            primary_header["CTYPE" + str(ax_nums["pixel"])] = (
                "Pix_Ind",
                "Index into pixel array in HPX_INDS extension.",
            )
            primary_header["CRVAL" + str(ax_nums["pixel"])] = 1
            primary_header["CRPIX" + str(ax_nums["pixel"])] = 1
            primary_header["CDELT" + str(ax_nums["pixel"])] = 1

        else:
            # set up first image axis
            # FITs standard is to use degrees (not radians as in the UVBeam object)
            deg_axis1_array = np.rad2deg(self.axis1_array)
            deg_axis1_spacing = np.rad2deg(axis1_spacing)
            primary_header["CTYPE" + str(ax_nums["img_ax1"])] = fits_axisname_dict[
                self.coordinate_system_dict[self.pixel_coordinate_system]["axes"][0]
            ]
            primary_header["CRVAL" + str(ax_nums["img_ax1"])] = deg_axis1_array[0]
            primary_header["CRPIX" + str(ax_nums["img_ax1"])] = 1
            primary_header["CDELT" + str(ax_nums["img_ax1"])] = deg_axis1_spacing
            primary_header["CUNIT" + str(ax_nums["img_ax1"])] = "deg"

            # set up second image axis
            deg_axis2_array = np.rad2deg(self.axis2_array)
            deg_axis2_spacing = np.rad2deg(axis2_spacing)
            primary_header["CTYPE" + str(ax_nums["img_ax2"])] = fits_axisname_dict[
                self.coordinate_system_dict[self.pixel_coordinate_system]["axes"][1]
            ]
            primary_header["CRVAL" + str(ax_nums["img_ax2"])] = deg_axis2_array[0]
            primary_header["CRPIX" + str(ax_nums["img_ax2"])] = 1
            primary_header["CDELT" + str(ax_nums["img_ax2"])] = deg_axis2_spacing
            primary_header["CUNIT" + str(ax_nums["img_ax2"])] = "deg"

        # set up frequency axis
        primary_header["CTYPE" + str(ax_nums["freq"])] = "FREQ"
        primary_header["CUNIT" + str(ax_nums["freq"])] = "Hz"
        primary_header["CRVAL" + str(ax_nums["freq"])] = self.freq_array[0]
        primary_header["CRPIX" + str(ax_nums["freq"])] = 1
        primary_header["CDELT" + str(ax_nums["freq"])] = freq_spacing

        # set up feed or pol axis
        if self.beam_type == "power":
            if self.Npols > 1:
                if not utils.tools._test_array_constant_spacing(
                    self._polarization_array
                ):
                    raise ValueError(
                        "The polarization values are not evenly "
                        "spaced (probably because of a select operation). "
                        "The uvfits format does not support unevenly "
                        "spaced polarizations."
                    )
                pol_spacing = self.polarization_array[1] - self.polarization_array[0]
            else:
                pol_spacing = 1

            # Note: This axis is called STOKES by analogy with the equivalent
            # uvfits axis
            # However, this is confusing because it is NOT a true Stokes axis,
            #   it is really the polarization axis.
            primary_header["CTYPE" + str(ax_nums["feed_pol"])] = (
                "STOKES",
                "Polarization integers, see uvbeam memo",
            )
            primary_header["CRVAL" + str(ax_nums["feed_pol"])] = (
                self.polarization_array[0]
            )
            primary_header["CDELT" + str(ax_nums["feed_pol"])] = pol_spacing

            # handle case where data_array is complex (e.g. for xy beams)
            if np.iscomplexobj(self.data_array):
                primary_data = np.concatenate(
                    [
                        np.expand_dims(self.data_array.real, axis=0),
                        np.expand_dims(self.data_array.imag, axis=0),
                    ],
                    axis=0,
                )
            else:
                primary_data = self.data_array
        elif self.beam_type == "efield":
            primary_header["CTYPE" + str(ax_nums["feed_pol"])] = (
                "FEEDIND",
                'Feed: index into "FEEDLIST".',
            )
            primary_header["CRVAL" + str(ax_nums["feed_pol"])] = 1
            primary_header["CDELT" + str(ax_nums["feed_pol"])] = 1

            primary_data = np.concatenate(
                [
                    np.expand_dims(self.data_array.real, axis=0),
                    np.expand_dims(self.data_array.imag, axis=0),
                ],
                axis=0,
            )
        else:
            raise ValueError(
                f"Unknown beam_type: {self.beam_type}, beam_type should be "
                '"efield" or "power".'
            )

        # Add a third axis to the data for consistency
        primary_data = primary_data[:, :, np.newaxis]

        primary_header["CRPIX" + str(ax_nums["feed_pol"])] = 1

        # set up spw axis
        primary_header["CTYPE" + str(ax_nums["spw"])] = (
            "IF",
            "Spectral window number.",
        )
        primary_header["CUNIT" + str(ax_nums["spw"])] = "Integer"
        primary_header["CRVAL" + str(ax_nums["spw"])] = 1
        primary_header["CRPIX" + str(ax_nums["spw"])] = 1
        primary_header["CDELT" + str(ax_nums["spw"])] = 1

        # set up basis vector axis
        primary_header["CTYPE" + str(ax_nums["basisvec"])] = (
            "VECIND",
            "Basis vector index.",
        )
        primary_header["CUNIT" + str(ax_nums["basisvec"])] = "Integer"
        primary_header["CRVAL" + str(ax_nums["basisvec"])] = 1
        primary_header["CRPIX" + str(ax_nums["basisvec"])] = 1
        primary_header["CDELT" + str(ax_nums["basisvec"])] = 1

        if np.iscomplexobj(self.data_array):
            # set up complex axis
            primary_header["CTYPE" + str(ax_nums["complex"])] = (
                "COMPLEX",
                "real, imaginary",
            )
            primary_header["CRVAL" + str(ax_nums["complex"])] = 1
            primary_header["CRPIX" + str(ax_nums["complex"])] = 1
            primary_header["CDELT" + str(ax_nums["complex"])] = 1

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
                    primary_header.add_comment(line)
            else:
                primary_header[keyword] = value

        for line in self.history.splitlines():
            primary_header.add_history(line)

        primary_hdu = fits.PrimaryHDU(data=primary_data, header=primary_header)
        hdulist = fits.HDUList([primary_hdu])

        if self.basis_vector_array is not None:
            basisvec_header = fits.Header()
            basisvec_header["EXTNAME"] = "BASISVEC"
            basisvec_header["COORDSYS"] = self.pixel_coordinate_system

            if self.pixel_coordinate_system == "healpix":
                basisvec_ax_nums = hxp_basisvec_ax_nums

                # set up pixel axis
                basisvec_header["CTYPE" + str(basisvec_ax_nums["pixel"])] = (
                    "Pix_Ind",
                    "Index into pixel array in HPX_INDS extension.",
                )
                basisvec_header["CRVAL" + str(basisvec_ax_nums["pixel"])] = 1
                basisvec_header["CRPIX" + str(basisvec_ax_nums["pixel"])] = 1
                basisvec_header["CDELT" + str(basisvec_ax_nums["pixel"])] = 1

            else:
                basisvec_ax_nums = reg_basisvec_ax_nums

                # set up first image axis
                basisvec_header["CTYPE" + str(basisvec_ax_nums["img_ax1"])] = (
                    fits_axisname_dict[
                        self.coordinate_system_dict[self.pixel_coordinate_system][
                            "axes"
                        ][0]
                    ]
                )
                basisvec_header["CRVAL" + str(basisvec_ax_nums["img_ax1"])] = (
                    deg_axis1_array[0]
                )
                basisvec_header["CRPIX" + str(basisvec_ax_nums["img_ax1"])] = 1
                basisvec_header["CDELT" + str(basisvec_ax_nums["img_ax1"])] = (
                    deg_axis1_spacing
                )
                basisvec_header["CUNIT" + str(basisvec_ax_nums["img_ax1"])] = "deg"

                # set up second image axis
                basisvec_header["CTYPE" + str(basisvec_ax_nums["img_ax2"])] = (
                    fits_axisname_dict[
                        self.coordinate_system_dict[self.pixel_coordinate_system][
                            "axes"
                        ][1]
                    ]
                )
                basisvec_header["CRVAL" + str(basisvec_ax_nums["img_ax2"])] = (
                    deg_axis2_array[0]
                )
                basisvec_header["CRPIX" + str(basisvec_ax_nums["img_ax2"])] = 1
                basisvec_header["CDELT" + str(basisvec_ax_nums["img_ax2"])] = (
                    deg_axis2_spacing
                )
                basisvec_header["CUNIT" + str(basisvec_ax_nums["img_ax2"])] = "deg"

            # set up vector component axis (length Ncomponents_vec)
            basisvec_header["CTYPE" + str(basisvec_ax_nums["ncomp"])] = (
                "COMPIND",
                "Vector component index",
            )
            basisvec_header["CUNIT" + str(basisvec_ax_nums["ncomp"])] = "Integer"
            basisvec_header["CRVAL" + str(basisvec_ax_nums["ncomp"])] = 1
            basisvec_header["CRPIX" + str(basisvec_ax_nums["ncomp"])] = 1
            basisvec_header["CDELT" + str(basisvec_ax_nums["ncomp"])] = 1

            # set up vector coordinate system axis (length Naxis_vec)
            basisvec_header["CTYPE" + str(basisvec_ax_nums["basisvec"])] = (
                "VECCOORD",
                "Basis vector index",
            )
            basisvec_header["CUNIT" + str(basisvec_ax_nums["basisvec"])] = "Integer"
            basisvec_header["CRVAL" + str(basisvec_ax_nums["basisvec"])] = 1
            basisvec_header["CRPIX" + str(basisvec_ax_nums["basisvec"])] = 1
            basisvec_header["CDELT" + str(basisvec_ax_nums["basisvec"])] = 1

            basisvec_data = self.basis_vector_array
            basisvec_hdu = fits.ImageHDU(data=basisvec_data, header=basisvec_header)
            hdulist.append(basisvec_hdu)

        if self.pixel_coordinate_system == "healpix":
            # make healpix pixel number column. 'K' format indicates 64-bit integer
            c1 = fits.Column(name="hpx_inds", format="K", array=self.pixel_array)
            coldefs = fits.ColDefs([c1])
            hpx_hdu = fits.BinTableHDU.from_columns(coldefs)
            hpx_hdu.header["EXTNAME"] = "HPX_INDS"
            hdulist.append(hpx_hdu)

        # check for frequency-specific optional arrays. If they're not None,
        # add them to the BANDPARM binary table HDU along with the bandpass_array
        bandpass_col = fits.Column(
            name="bandpass", format="D", array=self.bandpass_array
        )
        col_list = [bandpass_col]
        if self.receiver_temperature_array is not None:
            rx_temp_col = fits.Column(
                name="rx_temp", format="D", array=self.receiver_temperature_array
            )
            col_list.append(rx_temp_col)
        if self.loss_array is not None:
            loss_col = fits.Column(name="loss", format="D", array=self.loss_array)
            col_list.append(loss_col)
        if self.mismatch_array is not None:
            mismatch_col = fits.Column(
                name="mismatch", format="D", array=self.mismatch_array
            )
            col_list.append(mismatch_col)
        if self.s_parameters is not None:
            s11_col = fits.Column(name="s11", format="D", array=self.s_parameters[0, :])
            s12_col = fits.Column(name="s12", format="D", array=self.s_parameters[1, :])
            s21_col = fits.Column(name="s21", format="D", array=self.s_parameters[2, :])
            s22_col = fits.Column(name="s22", format="D", array=self.s_parameters[3, :])
            col_list += [s11_col, s12_col, s21_col, s22_col]

        coldefs = fits.ColDefs(col_list)
        bandpass_hdu = fits.BinTableHDU.from_columns(coldefs)
        bandpass_hdu.header["EXTNAME"] = "BANDPARM"
        if self.reference_impedance is not None:
            bandpass_hdu.header["ref_imp"] = self.reference_impedance
        hdulist.append(bandpass_hdu)

        hdulist.writeto(filename, overwrite=clobber)
        hdulist.close()
