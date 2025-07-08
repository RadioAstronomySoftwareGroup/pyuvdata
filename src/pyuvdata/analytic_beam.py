# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Analytic beam class definitions."""

from __future__ import annotations

import dataclasses
import importlib
import warnings
from dataclasses import InitVar, dataclass, field, replace
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import yaml
from astropy.constants import c as speed_of_light
from scipy.special import j1

from . import utils
from .docstrings import combine_docstrings
from .uvbeam.uvbeam import UVBeam

__all__ = ["AnalyticBeam", "AiryBeam", "GaussianBeam", "ShortDipoleBeam", "UniformBeam"]


def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    try:
        return a == b
    except TypeError:  # pragma: no cover
        return NotImplemented


def dc_eq(dc1, dc2) -> bool:
    """Check if two dataclasses which hold numpy arrays are equal."""
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented  # better than False
    fields1 = dataclasses.fields(dc1)
    dict1 = {}
    for fld in fields1:
        if fld.compare:
            dict1[fld.name] = getattr(dc1, fld.name)
    fields2 = dataclasses.fields(dc2)
    dict2 = {}
    for fld in fields2:
        if fld.compare:
            dict2[fld.name] = getattr(dc2, fld.name)

    # I can't figure out how to cause this to happen, but I still think it
    # should be checked.
    if dict1.keys() != dict2.keys():  # pragma: no cover
        return False
    return all(
        array_safe_eq(a1, a2)
        for a1, a2 in zip(dict1.values(), dict2.values(), strict=True)
    )


@dataclass(kw_only=True)
class AnalyticBeam:
    """
    Analytic beam base class.

    Attributes
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l.
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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

    Parameters
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l.
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
        "orbiting" (antenna is in motion, and its orientation depends on orbital
        parameters), "x-y" (primary rotates first in the plane connecting east,
        west, and zenith, and then perpendicular to that plane),
        "alt-az+nasmyth-r" ("alt-az" mount with a right-handed 90-degree tertiary
        mirror), "alt-az+nasmyth-l" ("alt-az" mount with a left-handed 90-degree
        tertiary mirror), "phased" (antenna is "electronically steered" by
        summing the voltages of multiple elements, e.g. MWA), "fixed" (antenna
        beam pattern is fixed in azimuth and elevation, e.g., HERA), and "other"
        (also referred to in some formats as "bizarre"). See the "Conventions"
        page of the documentation for further details. Default is "fixed".
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.
    x_orientation : str
        Physical orientation of the feed for the x feed. Not meaningful for
        unpolarized analytic beams, but clarifies the orientation of the dipole
        for for polarized beams like the ShortDipoleBeam and matches with the
        meaning on UVBeam objects. Used to set values of the feed_angle attribute,
        ignored in feed_angle is provided. Default is "east".

    """

    feed_array: npt.ArrayLike[str] | None = None
    feed_angle: npt.ArrayLike[float] | None = None
    mount_type: (
        Literal[
            "alt-az",
            "equatorial",
            "orbiting",
            "x-y",
            "alt-az+nasmyth-r",
            "alt-az+nasmyth-l",
            "phased",
            "fixed",
            "other",
        ]
        | None
    ) = "fixed"

    include_cross_pols: InitVar[bool] = True
    x_orientation: InitVar[Literal["east", "north"] | None] = "east"

    basis_vector_type = None

    # In the future, this might allow for cartesian basis vectors in some orientation.
    # In that case, the Naxes_vec would be 3 rather than 2
    _basis_vec_dict = {"az_za": 2}

    __types__: ClassVar[dict] = {}

    def __init_subclass__(cls):
        """Initialize any imported subclass."""
        if (
            cls.__name__ != "UnpolarizedAnalyticBeam"
            and not hasattr(cls, "_efield_eval")
            and not hasattr(cls, "_power_eval")
        ):
            raise TypeError(
                "Either _efield_eval or _power_eval method must be defined on "
                f"{cls.__name__}. Defining _efield_eval is the most general "
                "approach because it can represent polarized and negative going "
                "beams. If only _power_eval is defined, the E-field beam is "
                "defined as the square root of the auto pol power beam."
            )

        if cls.basis_vector_type is None:
            warnings.warn(
                "basis_vector_type was not defined, defaulting to azimuth and "
                "zenith_angle."
            )
            cls.basis_vector_type = "az_za"

        if cls.basis_vector_type not in cls._basis_vec_dict:
            raise ValueError(
                f"basis_vector_type for {cls.__name__} is {cls.basis_vector_type}, "
                f"must be one of {list(cls._basis_vec_dict.keys())}"
            )

        cls.__types__[cls.__name__] = cls

    def get_x_orientation_from_feeds(self) -> Literal["east", "north", None]:
        """
        Get x-orientation equivalent value based on feed information.

        Returns
        -------
        x_orientation : str
            One of "east", "north", or None, based on values present in
            AnalyticBeam.feed_array and AnalyticBeam.feed_angle.
        """
        return utils.pol.get_x_orientation_from_feeds(
            feed_array=self.feed_array, feed_angle=self.feed_angle, tols=[0, 1e-6]
        )

    def set_feeds_from_x_orientation(self, x_orientation):
        """
        Set feed information based on x-orientation value.

        Populates newer parameters describing feed-orientation
        (`AnalyticBeam.feed_array` and `AnalyticBeam.feed_angle`) based on the "older"
        x-orientation string. Note that this method will overwrite any previously
        populated values.

        Parameters
        ----------
        x_orientation : str
            String describing how the x-orientation is oriented. Must be either "north"/
            "n"/"ns" (x-polarization of antenna has a position angle of 0 degrees with
            respect to zenith/north) or "east"/"e"/"ew" (x-polarization of antenna has a
            position angle of 90 degrees with respect to zenith/north).
        """
        pol = self.polarization_array if hasattr(self, "polarization_array") else None
        _, self.feed_array, self.feed_angle = utils.pol.get_feeds_from_x_orientation(
            x_orientation=x_orientation,
            feeds=self.feed_array,
            polarization_array=pol,
            nants=0,
        )

    def __getattribute__(self, __name):
        """Handle getting old attributes."""
        if __name == "x_orientation":
            warnings.warn(
                "The AnalyticBeam.x_orientation attribute is deprecated, and has "
                "been superseded by AnalyticBeam.feed_angle and "
                "AnalyticBeam.feed_array. This will become an error in version 3.4. "
                "To set the equivalent value in the future, you can substitute "
                "accessing this parameter with a call to "
                "AnalyticBeam.get_x_orientation_from_feeds().",
                DeprecationWarning,
            )
            return self.get_x_orientation_from_feeds()

        return super().__getattribute__(__name)

    def __setattr__(self, __name, __value):
        """Handle setting old attributes."""
        if __name == "x_orientation":
            warnings.warn(
                "The AnalyticBeam.x_orientation attribute is deprecated, and has "
                "been superseded by UVBeam.feed_angle and AnalyticBeam.feed_array. "
                "This will become an error in version 3.4. To get the equivalent "
                "value in the future, you can substitute accessing this parameter "
                "with a call to AnalyticBeam.set_feeds_from_x_orientation().",
                DeprecationWarning,
            )
            if __value is not None:
                return self.set_feeds_from_x_orientation(__value)

        return super().__setattr__(__name, __value)

    def validate(self):
        """Validate inputs, placeholder for subclasses."""
        pass

    def __post_init__(self, include_cross_pols, x_orientation):
        """
        Post-initialization validation and conversions.

        Parameters
        ----------
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne)
            for the power beam.

        """
        self.validate()

        if self.feed_array is not None:
            allowed_feeds = ["n", "e", "x", "y", "r", "l"]
            for feed in self.feed_array:
                if feed not in allowed_feeds:
                    raise ValueError(
                        f"Feeds must be one of: {allowed_feeds}, "
                        f"got feeds: {self.feed_array}"
                    )
            self.feed_array = np.asarray(self.feed_array)
            if any(np.isin(self.feed_array, ["e", "n"])):
                warnings.warn(
                    'Support for physically oriented feeds (e.g., "n", and "e") has '
                    "been deprecated, with the only allowed options now available "
                    'being "x", "y", "l", or "r". The physical orientation of the feed '
                    "is now recorded in the AnalyticBeam.feed_angle parameter. This "
                    "will become an error in version 3.4.",
                    DeprecationWarning,
                )
                self.feed_angle = np.where(self.feed_array == "e", np.pi / 2, 0.0)
                self.feed_array[self.feed_array == "e"] = "x"
                self.feed_array[self.feed_array == "n"] = "y"
        else:
            self.feed_array = np.asarray(["x", "y"])

        if self.feed_angle is not None:
            self.feed_angle = np.asarray(self.feed_angle)
        elif x_orientation is not None:
            self.set_feeds_from_x_orientation(x_orientation)

        linear_pol = False
        for feed_name in ["x", "y"]:
            if feed_name in self.feed_array:
                linear_pol = True

        if self.feed_angle is None and linear_pol:
            raise ValueError(
                "feed_angle or x_orientation must be specified "
                "for linear polarization feeds"
            )

        if len(self.feed_array) == 1:
            include_cross_pols = False

        self.polarization_array = utils.pol.convert_feeds_to_pols(
            self.feed_array,
            include_cross_pols,
            x_orientation=self.get_x_orientation_from_feeds(),
        )

    def __eq__(self, other):
        """Define equality."""
        # have to define this because feed_array is a numpy array, so equality
        # needs to be checked with all_close not `==`
        return dc_eq(self, other)

    @property
    def Naxes_vec(self):  # noqa N802
        """The number of vector axes."""
        return self._basis_vec_dict[self.basis_vector_type]

    @property
    def Nfeeds(self):  # noqa N802
        """The number of feeds."""
        return len(self.feed_array)

    @property
    def Npols(self):  # noqa N802
        """The number of polarizations."""
        return self.polarization_array.size

    def clone(self, **kw):
        """Create a new instance of the object with updated parameters."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "The AnalyticBeam.x_orientation attribute is deprecated"
            )
            return replace(self, **kw)

    @property
    def east_ind(self):
        """The index of the east feed in the feed array."""
        if self.get_x_orientation_from_feeds() == "east" and "x" in self.feed_array:
            east_name = "x"
        elif self.get_x_orientation_from_feeds() == "north" and "y" in self.feed_array:
            east_name = "y"
        else:
            # this is not a linearly polarized feed
            return None
        return np.nonzero(np.asarray(self.feed_array) == east_name)[0][0]

    @property
    def north_ind(self):
        """The index of the north feed in the feed array."""
        if self.get_x_orientation_from_feeds() == "north" and "x" in self.feed_array:
            north_name = "x"
        elif self.get_x_orientation_from_feeds() == "east" and "y" in self.feed_array:
            north_name = "y"
        else:
            # this is not a linearly polarized feed
            return None
        return np.nonzero(np.asarray(self.feed_array) == north_name)[0][0]

    def _check_eval_inputs(
        self,
        *,
        az_array: npt.NDArray[float],
        za_array: npt.NDArray[float],
        freq_array: npt.NDArray[float],
    ):
        """Check the inputs for the eval methods."""
        if az_array.ndim > 1 or za_array.ndim > 1 or freq_array.ndim > 1:
            raise ValueError(
                "az_array, za_array and freq_array must all be one dimensional."
            )

        if az_array.shape != za_array.shape:
            raise ValueError("az_array and za_array must have the same shape.")

    def _get_empty_data_array(
        self, grid_shape: tuple[int, int], beam_type: str = "efield"
    ) -> npt.NDArray[float]:
        """Get the empty data to fill in the eval methods."""
        if beam_type == "efield":
            return np.zeros(
                (self.Naxes_vec, self.Nfeeds, *grid_shape), dtype=np.complex128
            )
        else:
            return np.zeros(
                (1, self.Npols, *grid_shape),
                dtype=np.complex128 if self.Npols > self.Nfeeds else np.float64,
            )

    def efield_eval(
        self,
        *,
        az_array: npt.NDArray[float],
        za_array: npt.NDArray[float],
        freq_array: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """
        Evaluate the efield at the given coordinates.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to evaluate the beam at in radians. Must be the same shape
            as za_array.
        za_array : array_like of floats, optional
            Zenith values to evaluate the beam at in radians. Must be the same shape
            as az_array.
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

        Returns
        -------
        array_like of complex
            An array of beam values. The shape of the evaluated data will be:
            (Naxes_vec, Nfeeds, freq_array.size, az_array.size)

        """
        self._check_eval_inputs(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        za_grid, _ = np.meshgrid(za_array, freq_array)
        az_grid, f_grid = np.meshgrid(az_array, freq_array)

        if hasattr(self, "_efield_eval"):
            return self._efield_eval(
                az_grid=az_grid, za_grid=za_grid, f_grid=f_grid
            ).astype(complex)
        else:
            # the polarization array always has the auto pols first, so we can just
            # use the first Nfeed elements.
            power_vals = self._power_eval(
                az_grid=az_grid, za_grid=za_grid, f_grid=f_grid
            )[0, 0 : self.Nfeeds].real

            data_array = self._get_empty_data_array(az_grid.shape)

            for fn in np.arange(self.Nfeeds):
                data_array[0, fn, :, :] = np.sqrt(power_vals[fn] / 2.0)
                data_array[1, fn, :, :] = np.sqrt(power_vals[fn] / 2.0)

            return data_array

    def power_eval(
        self,
        *,
        az_array: npt.NDArray[float],
        za_array: npt.NDArray[float],
        freq_array: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """
        Evaluate the power at the given coordinates.

        Parameters
        ----------
        az_array : array_like of floats, optional
            Azimuth values to evaluate the beam at in radians. Must be the same shape
            as za_array.
        za_array : array_like of floats, optional
            Zenith values to evaluate the beam at in radians. Must be the same shape
            as az_array.
        freq_array : array_like of floats, optional
            Frequency values to evaluate the beam at in Hertz.

        Returns
        -------
        array_like of float or complex
            An array of beam values. The shape of the evaluated data will be:
            (1, Npols, freq_array.size, az_array.size). The dtype will be
            a complex type if cross-pols are included, otherwise it will be a
            float type.

        """
        self._check_eval_inputs(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        za_grid, _ = np.meshgrid(za_array, freq_array)
        az_grid, f_grid = np.meshgrid(az_array, freq_array)

        if hasattr(self, "_power_eval"):
            return self._power_eval(
                az_grid=az_grid, za_grid=za_grid, f_grid=f_grid
            ).astype(complex if self.Npols > self.Nfeeds else float)
        else:
            efield_vals = self._efield_eval(
                az_grid=az_grid, za_grid=za_grid, f_grid=f_grid
            ).astype(complex)

            data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")

            for feed_i in np.arange(self.Nfeeds):
                data_array[0, feed_i] = (
                    np.abs(efield_vals[0, feed_i]) ** 2
                    + np.abs(efield_vals[1, feed_i]) ** 2
                )

            if self.Npols > self.Nfeeds:
                # do cross pols
                data_array[0, 2] = efield_vals[0, 0] * np.conj(
                    efield_vals[0, 1]
                ) + efield_vals[1, 0] * np.conj(efield_vals[1, 1])
                data_array[0, 3] = np.conj(data_array[0, 2])

            return data_array

    @combine_docstrings(UVBeam.new)
    def to_uvbeam(
        self,
        freq_array: npt.NDArray[float],
        beam_type: Literal["efield", "power"] = "efield",
        pixel_coordinate_system: (
            Literal["az_za", "orthoslant_zenith", "healpix"] | None
        ) = None,
        axis1_array: npt.NDArray[float] | None = None,
        axis2_array: npt.NDArray[float] | None = None,
        nside: int | None = None,
        healpix_pixel_array: npt.NDArray[int] | None = None,
        ordering: Literal["ring", "nested"] | None = None,
    ):
        """Generate a UVBeam object from an AnalyticBeam object.

        This method evaluates the analytic beam at a set of locations and
        frequencies to create a UVBeam object. This can be useful for testing
        and some other operations, but it is of course an approximation.

        Parameters
        ----------
        freq_array : ndarray of float
            Array of frequencies in Hz to evaluate the beam at.
        beam_type : str
            Beam type, either "efield" or "power".
        pixel_coordinate_system : str
            Pixel coordinate system, options are "az_za", "orthoslant_zenith" and
            "healpix". Forced to be "healpix" if ``nside`` is given and by
            *default* set to "az_za" if not. Currently, only "az_za" and "healpix"
            are implemented.
        axis1_array : ndarray of float
            Coordinates along first pixel axis (e.g. azimuth for an azimuth/zenith
            angle coordinate system) to evaluate the beam at. Must be regularly
            spaced. Should not provided for healpix coordinates.
        axis2_array : ndarray of float
            Coordinates along second pixel axis (e.g. zenith angle for an
            azimuth/zenith angle coordinate system) to evaluate the beam at. Must
            be regularly spaced. Should not provided for healpix coordinates.
        nside : int
            Healpix nside parameter, should only be provided for healpix coordinates.
        healpix_pixel_array : ndarray of int
            Healpix pixels to include. If nside is provided, defaults to all the
            pixels in the Healpix map.
        ordering : str
            Healpix ordering parameter, defaults to "ring" if nside is provided.

        """
        if beam_type not in ["efield", "power"]:
            raise ValueError("Beam type must be 'efield' or 'power'")

        if beam_type == "efield":
            polarization_array = None
        else:
            polarization_array = self.polarization_array

        mount_type = self.mount_type if hasattr(self, "mount_type") else None

        if pixel_coordinate_system is not None:
            allowed_coord_sys = list(UVBeam().coordinate_system_dict.keys())
            if pixel_coordinate_system not in allowed_coord_sys:
                raise ValueError(
                    f"Unknown coordinate system {pixel_coordinate_system}. UVBeam "
                    f"supported coordinate systems are: {allowed_coord_sys}."
                )

            if pixel_coordinate_system not in ["az_za", "healpix"]:
                raise NotImplementedError(
                    "Currently this method only supports 'az_za' and 'healpix' "
                    "pixel_coordinate_systems."
                )

        uvb = UVBeam.new(
            telescope_name="Analytic Beam",
            data_normalization="physical",
            feed_name=self.__repr__(),
            feed_version="1.0",
            model_name=self.__repr__(),
            model_version="1.0",
            freq_array=freq_array,
            feed_array=self.feed_array,
            feed_angle=self.feed_angle,
            mount_type=mount_type,
            polarization_array=polarization_array,
            pixel_coordinate_system=pixel_coordinate_system,
            axis1_array=axis1_array,
            axis2_array=axis2_array,
            nside=nside,
            healpix_pixel_array=healpix_pixel_array,
            ordering=ordering,
            history=f"created from a pyuvdata analytic beam: {self.__repr__()}",
        )

        if uvb.pixel_coordinate_system == "healpix":
            try:
                from astropy_healpix import HEALPix
            except ImportError as e:
                raise ImportError(
                    "astropy_healpix is not installed but is "
                    "required for healpix functionality. "
                    "Install 'astropy-healpix' using conda or pip."
                ) from e
            hp_obj = HEALPix(nside=uvb.nside, order=uvb.ordering)
            hpx_lon, hpx_lat = hp_obj.healpix_to_lonlat(uvb.pixel_array)
            za_array, az_array = utils.coordinates.hpx_latlon_to_zenithangle_azimuth(
                hpx_lat.radian, hpx_lon.radian
            )

        else:
            az_array, za_array = np.meshgrid(uvb.axis1_array, uvb.axis2_array)
            az_array = az_array.flatten()
            za_array = za_array.flatten()

        if beam_type == "efield":
            eval_function = "efield_eval"
        else:
            eval_function = "power_eval"

        data_array = getattr(self, eval_function)(
            az_array=az_array, za_array=za_array, freq_array=freq_array
        )

        if uvb.pixel_coordinate_system == "az_za":
            data_array = data_array.reshape(uvb.data_array.shape)

        uvb.data_array = data_array

        uvb.check()
        return uvb


def _analytic_beam_constructor(loader, node):
    """
    Define a yaml constructor for analytic beams.

    The yaml must specify a "class" field with an importable class and any
    required inputs to that class's constructor.

    Example yamls (note that the node key can be anything, it does not need to
    be 'beam'):

    * beam: !AnalyticBeam
       class: AiryBeam
       diameter: 4.0

    * beam: !AnalyticBeam
        class: GaussianBeam
        reference_frequency: 120000000.
        spectral_index: -1.5
        sigma: 0.26

    * beam: !AnalyticBeam
       class: ShortDipoleBeam


    Parameters
    ----------
    loader: yaml.Loader
        An instance of a yaml Loader object.
    node: yaml.Node
        A yaml node object.

    Returns
    -------
    beam
        An instance of an AnalyticBeam subclass.

    """
    values = loader.construct_mapping(node, deep=True)
    if "class" not in values:
        raise ValueError("yaml entries for AnalyticBeam must specify a class")
    class_parts = (values.pop("class")).split(".")
    class_name = class_parts[-1]

    if class_name not in AnalyticBeam.__types__ and len(class_parts) > 1:
        module = (".").join(class_parts[:-1])
        module = importlib.import_module(module)

    if class_name not in AnalyticBeam.__types__:
        raise NameError(
            f"{class_name} is not a known AnalyticBeam. Available options are: "
            f"{list(AnalyticBeam.__types__.keys())}. If it is a custom beam, "
            "either ensure the module is imported, or specify the beam with "
            "dot-pathed modules included (i.e. `my_module.MyAnalyticBeam`)"
        )

    beam_class = AnalyticBeam.__types__[class_name]

    beam = beam_class(**values)

    return beam


yaml.add_constructor(
    "!AnalyticBeam", _analytic_beam_constructor, Loader=yaml.SafeLoader
)
yaml.add_constructor(
    "!AnalyticBeam", _analytic_beam_constructor, Loader=yaml.FullLoader
)


def _analytic_beam_representer(dumper, beam):
    """
    Define a yaml representer for analytic beams.

    Parameters
    ----------
    dumper: yaml.Dumper
        An instance of a yaml Loader object.
    beam: AnalyticBeam subclass
        An analytic beam object.

    Returns
    -------
    str
        The yaml representation of the analytic beam.

    """
    mapping = {
        "class": beam.__module__ + "." + beam.__class__.__name__,
        **dataclasses.asdict(beam),
    }
    if "feed_array" in mapping:
        mapping["feed_array"] = mapping["feed_array"].tolist()
    if "feed_angle" in mapping:
        mapping["feed_angle"] = mapping["feed_angle"].tolist()

    return dumper.represent_mapping("!AnalyticBeam", mapping)


yaml.add_multi_representer(
    AnalyticBeam, _analytic_beam_representer, Dumper=yaml.SafeDumper
)
yaml.add_multi_representer(AnalyticBeam, _analytic_beam_representer, Dumper=yaml.Dumper)


@dataclass(kw_only=True, eq=False)
class UnpolarizedAnalyticBeam(AnalyticBeam):
    """
    Unpolarized analytic beam base class.

    Attributes
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l.
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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

    Parameters
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l. Default is ["x", "y"].
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.
    x_orientation : str
        Physical orientation of the feed for the x feed. Not meaningful for
        unpolarized analytic beams, but clarifies the orientation of the dipole
        for for polarized beams like the ShortDipoleBeam and matches with the
        meaning on UVBeam objects. Used to set values of the feed_angle attribute,
        ignored in feed_angle is provided. Default is "east".

    """

    feed_array: npt.npt.ArrayLike[str] | None = field(
        default=None, repr=False, compare=False
    )
    feed_angle: npt.npt.ArrayLike[float] | None = field(
        default=None, repr=False, compare=False
    )
    mount_type: Literal[
        "alt-az",
        "equatorial",
        "orbiting",
        "x-y",
        "alt-az+nasmyth-r",
        "alt-az+nasmyth-l",
        "phased",
        "fixed",
        "other",
    ] = field(default="fixed", repr=False, compare=True)

    # the basis vector type doesn't matter for unpolarized beams, just hardcode
    # it here so subclasses don't have to deal with it.
    basis_vector_type = "az_za"


@dataclass(kw_only=True, eq=False)
class AiryBeam(UnpolarizedAnalyticBeam):
    """
    A zenith pointed Airy beam.

    Airy beams are the diffraction pattern of a circular aperture, so represent
    an idealized dish. Requires a dish diameter in meters and is inherently
    chromatic and unpolarized.

    The unpolarized nature leads to some results that may be surprising to radio
    astronomers: if two feeds are specified they will have identical responses
    and the cross power beam between the two feeds will be identical to the
    power beam for a single feed.

    Attributes
    ----------
    diameter : float
        Dish diameter in meters.

    Parameters
    ----------
    diameter : float
        Dish diameter in meters.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.

    """

    diameter: float

    # Have to define this because an Airy E-field response can go negative,
    # so it cannot just be calculated from the sqrt of a power beam.
    def _efield_eval(
        self,
        *,
        az_grid: npt.NDArray[float],
        za_grid: npt.NDArray[float],
        f_grid: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_grid.shape)

        kvals = (2.0 * np.pi) * f_grid / speed_of_light.to("m/s").value
        xvals = (self.diameter / 2.0) * np.sin(za_grid) * kvals
        values = np.zeros_like(xvals)
        nz = xvals != 0.0
        ze = xvals == 0.0
        values[nz] = 2.0 * j1(xvals[nz]) / xvals[nz]
        values[ze] = 1.0

        for fn in np.arange(self.Nfeeds):
            data_array[0, fn, :, :] = values / np.sqrt(2.0)
            data_array[1, fn, :, :] = values / np.sqrt(2.0)

        return data_array


def diameter_to_sigma(diameter: float, freq_array: npt.NDArray[float]) -> float:
    """
    Find the sigma that gives a beam width similar to an Airy disk.

    Find the stddev of a gaussian with fwhm equal to that of
    an Airy disk's main lobe for a given diameter.

    Parameters
    ----------
    diameter : float
        Antenna diameter in meters
    freq_array : array of float
        Frequencies in Hz

    Returns
    -------
    sigma : float
        The standard deviation in zenith angle radians for a Gaussian beam
        with FWHM equal to that of an Airy disk's main lobe for an aperture
        with the given diameter.

    """
    wavelengths = speed_of_light.to("m/s").value / freq_array

    scalar = 2.2150894  # Found by fitting a Gaussian to an Airy disk function

    sigma = np.arcsin(scalar * wavelengths / (np.pi * diameter)) * 2 / 2.355

    return sigma


@dataclass(kw_only=True, eq=False)
class GaussianBeam(UnpolarizedAnalyticBeam):
    """
    A circular, zenith pointed Gaussian beam.

    Requires either a dish diameter in meters or a standard deviation sigma in
    radians. Gaussian beams specified by a diameter will have their width
    matched to an Airy beam at each simulated frequency, so are inherently
    chromatic. For Gaussian beams specified with sigma, the sigma_type defines
    whether the width specified by sigma specifies the width of the E-Field beam
    (default) or power beam in zenith angle. If only sigma is specified, the
    beam is achromatic, optionally both the spectral_index and reference_frequency
    parameters can be set to generate a chromatic beam with standard deviation
    defined by a power law:

    stddev(f) = sigma * (f/ref_freq)**(spectral_index)

    The unpolarized nature leads to some results that may be
    surprising to radio astronomers: if two feeds are specified they will have
    identical responses and the cross power beam between the two feeds will be
    identical to the power beam for a single feed.

    Attributes
    ----------
    sigma : float
        Standard deviation in radians for the gaussian beam. Only one of sigma
        and diameter should be set.
    sigma_type : str
        Either "efield" or "power" to indicate whether the sigma specifies the size of
        the efield or power beam. Ignored if `sigma` is None.
    diameter : float
        Dish diameter in meters to use to define the size of the gaussian beam, by
        matching the FWHM of the gaussian to the FWHM of an Airy disk. This will result
        in a frequency dependent beam.  Only one of sigma and diameter should be set.
    spectral_index : float
        Option to scale the gaussian beam width as a power law with frequency. If set
        to anything other than zero, the beam will be frequency dependent and the
        `reference_frequency` must be set. Ignored if `sigma` is None.
    reference_frequency : float
        The reference frequency for the beam width power law, required if `sigma` is not
        None and `spectral_index` is not zero. Ignored if `sigma` is None.

    Parameters
    ----------
    sigma : float
        Standard deviation in radians for the gaussian beam. Only one of sigma
        and diameter should be set.
    sigma_type : str
        Either "efield" or "power" to indicate whether the sigma specifies the size of
        the efield or power beam. Ignored if `sigma` is None.
    diameter : float
        Dish diameter in meters to use to define the size of the gaussian beam, by
        matching the FWHM of the gaussian to the FWHM of an Airy disk. This will result
        in a frequency dependent beam.  Only one of sigma and diameter should be set.
    spectral_index : float
        Option to scale the gaussian beam width as a power law with frequency. If set
        to anything other than zero, the beam will be frequency dependent and the
        `reference_frequency` must be set. Ignored if `sigma` is None.
    reference_frequency : float
        The reference frequency for the beam width power law, required if `sigma` is not
        None and `spectral_index` is not zero. Ignored if `sigma` is None.
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.

    """

    sigma: float | None = None
    sigma_type: Literal["efield", "power"] = "efield"
    diameter: float | None = None
    spectral_index: float = 0.0
    reference_frequency: float = None

    def validate(self):
        """Post-initialization validation and conversions."""
        if (self.diameter is None and self.sigma is None) or (
            self.diameter is not None and self.sigma is not None
        ):
            if self.diameter is None:
                raise ValueError("Either diameter or sigma must be set.")
            else:
                raise ValueError("Only one of diameter or sigma can be set.")

        if self.sigma is not None:
            if self.sigma_type not in ["efield", "power"]:
                raise ValueError("sigma_type must be 'efield' or 'power'.")

            if self.sigma_type == "power":
                self.sigma = np.sqrt(2) * self.sigma

            if self.spectral_index != 0.0 and self.reference_frequency is None:
                raise ValueError(
                    "reference_frequency must be set if `spectral_index` is not zero."
                )
            if self.reference_frequency is None:
                self.reference_frequency = 1.0

    def get_sigmas(self, freq_array: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Get the sigmas for the gaussian beam using the diameter (if defined).

        Parameters
        ----------
        freq_array : array of floats
            Frequency values to get the sigmas for in Hertz.

        Returns
        -------
        sigmas : array_like of float
            Beam sigma values as a function of frequency. Size will match the
            freq_array size.

        """
        if self.diameter is not None:
            sigmas = diameter_to_sigma(self.diameter, freq_array)
        elif self.sigma is not None:
            sigmas = (
                self.sigma
                * (freq_array / self.reference_frequency) ** self.spectral_index
            )
        return sigmas

    def _power_eval(
        self,
        *,
        az_grid: npt.NDArray[float],
        za_grid: npt.NDArray[float],
        f_grid: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the power at the given coordinates."""
        sigmas = self.get_sigmas(f_grid)

        values = np.exp(-(za_grid**2) / (sigmas**2))
        data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")
        for fn in np.arange(self.Npols):
            # For power beams the first axis is shallow because we don't have to worry
            # about polarization.
            data_array[0, fn, :, :] = values

        return data_array


class ShortDipoleBeam(AnalyticBeam):
    """
    A zenith pointed analytic short dipole beam with two crossed feeds.

    A classical short (Hertzian) dipole beam with two crossed feeds aligned east
    and north. Short dipole beams are intrinsically polarized but achromatic.
    Does not require any parameters, but the orientation of the dipole labelled
    as "x" can be specified to align "north" or "east" via the x_orientation
    parameter (matching the parameter of the same name on UVBeam and UVData
    objects).

    Attributes
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y.
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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

    Parameters
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l. Default is ["x", "y"]
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.
    x_orientation : str
        Physical orientation of the feed for the x feed. Not meaningful for
        unpolarized analytic beams, but clarifies the orientation of the dipole
        for for polarized beams like the ShortDipoleBeam and matches with the
        meaning on UVBeam objects. Used to set values of the feed_angle attribute,
        ignored in feed_angle is provided. Default is "east".

    """

    basis_vector_type = "az_za"

    def validate(self):
        """Post-initialization validation and conversions."""
        if self.feed_array is None:
            self.feed_array = ["x", "y"]
        self.feed_array = np.asarray(self.feed_array)

        allowed_feeds = ["n", "e", "x", "y"]
        for feed in self.feed_array:
            if feed not in allowed_feeds:
                raise ValueError(
                    f"Feeds must be one of: {allowed_feeds}, "
                    f"got feeds: {self.feed_array}"
                )

        if self.feed_angle is not None:
            x_orient = utils.pol.get_x_orientation_from_feeds(
                feed_array=self.feed_array, feed_angle=self.feed_angle, tols=[0, 1e-6]
            )
            if x_orient is None:
                raise NotImplementedError(
                    "ShortDipoleBeams currently only support dipoles aligned to East "
                    "and North, it does not yet have support for arbitrary feed angles."
                )
            else:
                # use the standard feed angles for consistency
                _, _, feed_angle = utils.pol.get_feeds_from_x_orientation(
                    x_orientation=x_orient,
                    nants=1,
                    feed_array=self.feed_array[np.newaxis, :],
                )
                self.feed_angle = np.squeeze(feed_angle)

    def _efield_eval(
        self,
        *,
        az_grid: npt.NDArray[float],
        za_grid: npt.NDArray[float],
        f_grid: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the efield at the given coordinates."""
        data_array = self._get_empty_data_array(az_grid.shape)

        # The first dimension is for [azimuth, zenith angle] in that order
        # the second dimension is for feed [e, n] in that order
        data_array[0, self.east_ind] = -np.sin(az_grid)
        data_array[0, self.north_ind] = np.cos(az_grid)
        data_array[1, self.east_ind] = np.cos(za_grid) * np.cos(az_grid)
        data_array[1, self.north_ind] = np.cos(za_grid) * np.sin(az_grid)

        return data_array

    def _power_eval(
        self,
        *,
        az_grid: npt.NDArray[float],
        za_grid: npt.NDArray[float],
        f_grid: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")

        # these are just the sum in quadrature of the efield components.
        # some trig work is done to reduce the number of cos/sin evaluations
        data_array[0, 0] = 1 - (np.sin(za_grid) * np.cos(az_grid)) ** 2
        data_array[0, 1] = 1 - (np.sin(za_grid) * np.sin(az_grid)) ** 2

        if self.Npols > self.Nfeeds:
            # cross pols are included
            data_array[0, 2] = -(np.sin(za_grid) ** 2) * np.sin(2.0 * az_grid) / 2.0
            data_array[0, 3] = data_array[0, 2]

        return data_array


@dataclass(kw_only=True, eq=False)
class UniformBeam(UnpolarizedAnalyticBeam):
    """
    A uniform beam.

    Uniform beams have identical responses in all directions, so are quite
    unphysical but can be useful for testing other aspects of simulators. They
    are unpolarized and achromatic and do not take any parameters.

    The unpolarized nature leads to some results that may be surprising to radio
    astronomers: if two feeds are specified they will have identical responses
    and the cross power beam between the two feeds will be identical to the
    power beam for a single feed.

    Attributes
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l.
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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

    Parameters
    ----------
    feed_array : array-like of str
        Feeds to define this beam for, e.g. x & y or r & l. Default is ["x", "y"].
    feed_angle : array-like of float
        Position angle of a given feed, units of radians. A feed angle of 0 is
        typically oriented toward zenith for steerable antennas, otherwise toward
        north for fixed antennas (e.g., HERA, LWA). More details on this can be found
        on the "Conventions" page of the docs.
    mount_type : str
        Antenna mount type, which describes the optics of the antenna in question.
        Supported options include: "alt-az" (primary rotates in azimuth and
        elevation), "equatorial" (primary rotates in hour angle and declination)
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
    include_cross_pols : bool
        Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
        the power beam.
    x_orientation : str
        Physical orientation of the feed for the x feed. Not meaningful for
        unpolarized analytic beams, but clarifies the orientation of the dipole
        for for polarized beams like the ShortDipoleBeam and matches with the
        meaning on UVBeam objects. Used to set values of the feed_angle attribute,
        ignored in feed_angle is provided. Default is "east".

    """

    def _power_eval(
        self,
        *,
        az_grid: npt.NDArray[float],
        za_grid: npt.NDArray[float],
        f_grid: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Evaluate the power at the given coordinates."""
        data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")

        data_array = data_array + 1.0

        return data_array
