# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for phasing."""

from copy import deepcopy

import erfa
import numpy as np
from astropy import units
from astropy.coordinates import AltAz, Distance, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

from . import _phasing
from .coordinates import get_loc_obj
from .times import get_lst_for_time
from .tools import _nants_to_nblts, _ntimes_to_nblts


def old_uvw_calc(ra, dec, initial_uvw):
    """
    Calculate old uvws from unphased ones in an icrs or gcrs frame.

    This method should not be used and is only retained for testing the
    undo_old_uvw_calc method, which is needed for fixing phases.

    This code expects input uvws relative to the telescope location in the same frame
    that ra/dec are in (e.g. icrs or gcrs) and returns phased ones in the same frame.

    Parameters
    ----------
    ra : float
        Right ascension of phase center.
    dec : float
        Declination of phase center.
    initial_uvw : ndarray of float
        Unphased uvws or positions relative to the array center,
        shape (Nlocs, 3).

    Returns
    -------
    uvw : ndarray of float
        uvw array in the same frame as initial_uvws, ra and dec.

    """
    if initial_uvw.ndim == 1:
        initial_uvw = initial_uvw[np.newaxis, :]

    return _phasing._old_uvw_calc(
        np.float64(ra),
        np.float64(dec),
        np.ascontiguousarray(initial_uvw.T, dtype=np.float64),
    ).T


def undo_old_uvw_calc(ra, dec, uvw):
    """
    Undo the old phasing calculation on uvws in an icrs or gcrs frame.

    This code expects phased uvws or positions in the same frame that ra/dec
    are in (e.g. icrs or gcrs) and returns unphased ones in the same frame.

    Parameters
    ----------
    ra : float
        Right ascension of phase center.
    dec : float
        Declination of phase center.
    uvw : ndarray of float
        Phased uvws or positions relative to the array center,
        shape (Nlocs, 3).

    Returns
    -------
    unphased_uvws : ndarray of float
        Unphased uvws or positions relative to the array center,
        shape (Nlocs, 3).

    """
    if uvw.ndim == 1:
        uvw = uvw[np.newaxis, :]

    return _phasing._undo_old_uvw_calc(
        np.float64(ra), np.float64(dec), np.ascontiguousarray(uvw.T, dtype=np.float64)
    ).T


def polar2_to_cart3(*, lon_array, lat_array):
    """
    Convert 2D polar coordinates into 3D cartesian coordinates.

    This is a simple routine for converting a set of spherical angular coordinates
    into a 3D cartesian vectors, where the x-direction is set by the position (0, 0).

    Parameters
    ----------
    lon_array : float or ndarray
        Longitude coordinates, which increases in the counter-clockwise direction.
        Units of radians. Can either be a float or ndarray -- if the latter, must have
        the same shape as lat_array.
    lat_array : float or ndarray
        Latitude coordinates, where 0 falls on the equator of the sphere.  Units of
        radians. Can either be a float or ndarray -- if the latter, must have the same
        shape as lat_array.

    Returns
    -------
    xyz_array : ndarray of float
        Cartesian coordinates of the given longitude and latitude on a unit sphere.
        Shape is (3, coord_shape), where coord_shape is the shape of lon_array and
        lat_array if they were provided as type ndarray, otherwise (3,).
    """
    # Check to make sure that we are not playing with mixed types
    if type(lon_array) is not type(lat_array):
        raise ValueError(
            "lon_array and lat_array must either both be floats or ndarrays."
        )
    if isinstance(lon_array, np.ndarray) and lon_array.shape != lat_array.shape:
        raise ValueError("lon_array and lat_array must have the same shape.")

    # Once we know that lon_array and lat_array are of the same shape,
    # time to create our 3D set of vectors!
    xyz_array = np.array(
        [
            np.cos(lon_array) * np.cos(lat_array),
            np.sin(lon_array) * np.cos(lat_array),
            np.sin(lat_array),
        ],
        dtype=float,
    )

    return xyz_array


def cart3_to_polar2(xyz_array):
    """
    Convert 3D cartesian coordinates into 2D polar coordinates.

    This is a simple routine for converting a set of 3D cartesian vectors into
    spherical coordinates, where the position (0, 0) lies along the x-direction.

    Parameters
    ----------
    xyz_array : ndarray of float
        Cartesian coordinates, need not be of unit vector length. Shape is
        (3, coord_shape).

    Returns
    -------
    lon_array : ndarray of float
        Longitude coordinates, which increases in the counter-clockwise direction.
        Units of radians, shape is (coord_shape,).
    lat_array : ndarray of float
        Latitude coordinates, where 0 falls on the equator of the sphere.  Units of
        radians, shape is (coord_shape,).
    """
    if not isinstance(xyz_array, np.ndarray):
        raise ValueError("xyz_array must be an ndarray.")
    if xyz_array.ndim == 0:
        raise ValueError("xyz_array must have ndim > 0")
    if xyz_array.shape[0] != 3:
        raise ValueError("xyz_array must be length 3 across the zeroth axis.")

    # The longitude coord is relatively easy to calculate, just take the X and Y
    # components and find the arctac of the pair.
    lon_array = np.mod(np.arctan2(xyz_array[1], xyz_array[0]), 2.0 * np.pi, dtype=float)

    # If we _knew_ that xyz_array was always of length 1, then this call could be a much
    # simpler one to arcsin. But to make this generic, we'll use the length of the XY
    # component along with arctan2.
    lat_array = np.arctan2(
        xyz_array[2], np.sqrt((xyz_array[0:2] ** 2.0).sum(axis=0)), dtype=float
    )

    # Return the two arrays
    return lon_array, lat_array


def _rotate_matmul_wrapper(*, xyz_array, rot_matrix, n_rot):
    """
    Apply a rotation matrix to a series of vectors.

    This is a simple convenience function which wraps numpy's matmul function for use
    with various vector rotation functions in this module. This code could, in
    principle, be replaced by a cythonized piece of code, although the matmul function
    is _pretty_ well optimized already. This function is not meant to be called by
    users, but is instead used by multiple higher-level utility functions (namely those
    that perform rotations).

    Parameters
    ----------
    xyz_array : ndarray of floats
        Array of vectors to be rotated. When nrot > 1, shape may be (n_rot, 3, n_vec)
        or (1, 3, n_vec), the latter is useful for when performing multiple rotations
        on a fixed set of vectors. If nrot = 1, shape may be (1, 3, n_vec), (3, n_vec),
        or (3,).
    rot_matrix : ndarray of floats
        Series of rotation matricies to be applied to the stack of vectors. Must be
        of shape (n_rot, 3, 3)
    n_rot : int
        Number of individual rotation matricies to be applied.

    Returns
    -------
    rotated_xyz : ndarray of floats
        Array of vectors that have been rotated, of shape (n_rot, 3, n_vectors,).
    """
    # Do a quick check to make sure that things look sensible
    if rot_matrix.shape != (n_rot, 3, 3):
        raise ValueError(
            f"rot_matrix must be of shape (n_rot, 3, 3), where n_rot={n_rot}."
        )
    if (xyz_array.ndim == 3) and (
        (xyz_array.shape[0] not in [1, n_rot]) or (xyz_array.shape[-2] != 3)
    ):
        raise ValueError("Misshaped xyz_array - expected shape (n_rot, 3, n_vectors).")
    if (xyz_array.ndim < 3) and (xyz_array.shape[0] != 3):
        raise ValueError("Misshaped xyz_array - expected shape (3, n_vectors) or (3,).")
    rotated_xyz = np.matmul(rot_matrix, xyz_array)

    return rotated_xyz


def _rotate_one_axis(xyz_array, *, rot_amount, rot_axis):
    """
    Rotate an array of 3D positions around the a single axis (x, y, or z).

    This function performs a basic rotation of 3D vectors about one of the priciple
    axes -- the x-axis, the y-axis, or the z-axis.

    Note that the rotations here obey the right-hand rule -- that is to say, from the
    perspective of the positive side of the axis of rotation, a positive rotation will
    cause points on the plane intersecting this axis to move in a counter-clockwise
    fashion.

    Parameters
    ----------
    xyz_array : ndarray of float
        Set of 3-dimensional vectors be rotated, in typical right-handed cartesian
        order, e.g. (x, y, z). Shape is (Nrot, 3, Nvectors).
    rot_amount : float or ndarray of float
        Amount (in radians) to rotate the given set of coordinates. Can either be a
        single float (or ndarray of shape (1,)) if rotating all vectors by the same
        amount, otherwise expected to be shape (Nrot,).
    rot_axis : int
        Axis around which the rotation is applied. 0 is the x-axis, 1 is the y-axis,
        and 2 is the z-axis.

    Returns
    -------
    rotated_xyz : ndarray of float
        Set of rotated 3-dimensional vectors, shape (Nrot, 3, Nvector).
    """
    # If rot_amount is None or all zeros, then this is just one big old no-op.
    if (rot_amount is None) or np.all(rot_amount == 0.0):
        if np.ndim(xyz_array) == 1:
            return deepcopy(xyz_array[np.newaxis, :, np.newaxis])
        elif np.ndim(xyz_array) == 2:
            return deepcopy(xyz_array[np.newaxis, :, :])
        else:
            return deepcopy(xyz_array)

    # Check and see how big of a rotation matrix we need
    n_rot = 1 if (not isinstance(rot_amount, np.ndarray)) else (rot_amount.shape[0])
    n_vec = xyz_array.shape[-1]

    # The promotion of values to float64 is to suppress numerical precision issues,
    # since the matrix math can - in limited circumstances - introduce precision errors
    # of order 10x the limiting numerical precision of the float. For a float32/single,
    # thats a part in 1e6 (~arcsec-level errors), but for a float64 it translates to
    # a part in 1e15.
    rot_matrix = np.zeros((3, 3, n_rot), dtype=np.float64)

    # Figure out which pieces of the matrix we need to update
    temp_jdx = (rot_axis + 1) % 3
    temp_idx = (rot_axis + 2) % 3

    # Fill in the rotation matricies accordingly
    rot_matrix[rot_axis, rot_axis] = 1
    rot_matrix[temp_idx, temp_idx] = np.cos(rot_amount, dtype=np.float64)
    rot_matrix[temp_jdx, temp_jdx] = rot_matrix[temp_idx, temp_idx]
    rot_matrix[temp_idx, temp_jdx] = np.sin(rot_amount, dtype=np.float64)
    rot_matrix[temp_jdx, temp_idx] = -rot_matrix[temp_idx, temp_jdx]

    # The rot matrix was shape (3, 3, n_rot) to help speed up filling in the elements
    # of each matrix, but now we want to flip it into its proper shape of (n_rot, 3, 3)
    rot_matrix = np.transpose(rot_matrix, axes=[2, 0, 1])

    if (n_rot == 1) and (n_vec == 1) and (xyz_array.ndim == 3):
        # This is a special case where we allow the rotation axis to "expand" along
        # the 0th axis of the rot_amount arrays. For xyz_array, if n_vectors = 1
        # but n_rot !=1, then it's a lot faster (by about 10x) to "switch it up" and
        # swap the n_vector and  n_rot axes, and then swap them back once everything
        # else is done.
        return np.transpose(
            _rotate_matmul_wrapper(
                xyz_array=np.transpose(xyz_array, axes=[2, 1, 0]),
                rot_matrix=rot_matrix,
                n_rot=n_rot,
            ),
            axes=[2, 1, 0],
        )
    else:
        return _rotate_matmul_wrapper(
            xyz_array=xyz_array, rot_matrix=rot_matrix, n_rot=n_rot
        )


def _rotate_two_axis(xyz_array, *, rot_amount1, rot_amount2, rot_axis1, rot_axis2):
    """
    Rotate an array of 3D positions sequentially around a pair of axes (x, y, or z).

    This function performs a sequential pair of basic rotations of 3D vectors about
    the priciple axes -- the x-axis, the y-axis, or the z-axis.

    Note that the rotations here obey the right-hand rule -- that is to say, from the
    perspective of the positive side of the axis of rotation, a positive rotation will
    cause points on the plane intersecting this axis to move in a counter-clockwise
    fashion.

    Parameters
    ----------
    xyz_array : ndarray of float
        Set of 3-dimensional vectors be rotated, in typical right-handed cartesian
        order, e.g. (x, y, z). Shape is (Nrot, 3, Nvectors).
    rot_amount1 : float or ndarray of float
        Amount (in radians) of rotatation to apply during the first rotation of the
        sequence, to the given set of coordinates. Can either be a single float (or
        ndarray of shape (1,)) if rotating all vectors by the same amount, otherwise
        expected to be shape (Nrot,).
    rot_amount2 : float or ndarray of float
        Amount (in radians) of rotatation to apply during the second rotation of the
        sequence, to the given set of coordinates. Can either be a single float (or
        ndarray of shape (1,)) if rotating all vectors by the same amount, otherwise
        expected to be shape (Nrot,).
    rot_axis1 : int
        Axis around which the first rotation is applied. 0 is the x-axis, 1 is the
        y-axis, and 2 is the z-axis.
    rot_axis2 : int
        Axis around which the second rotation is applied. 0 is the x-axis, 1 is the
        y-axis, and 2 is the z-axis.

    Returns
    -------
    rotated_xyz : ndarray of float
        Set of rotated 3-dimensional vectors, shape (Nrot, 3, Nvector).

    """
    # Capture some special cases upfront, where we can save ourselves a bit of work
    no_rot1 = (rot_amount1 is None) or np.all(rot_amount1 == 0.0)
    no_rot2 = (rot_amount2 is None) or np.all(rot_amount2 == 0.0)
    if no_rot1 and no_rot2:
        # If rot_amount is None, then this is just one big old no-op.
        return deepcopy(xyz_array)
    elif no_rot1:
        # If rot_amount1 is None, then ignore it and just work w/ the 2nd rotation
        return _rotate_one_axis(xyz_array, rot_amount=rot_amount2, rot_axis=rot_axis2)
    elif no_rot2:
        # If rot_amount2 is None, then ignore it and just work w/ the 1st rotation
        return _rotate_one_axis(xyz_array, rot_amount=rot_amount1, rot_axis=rot_axis1)
    elif rot_axis1 == rot_axis2:
        # Capture the case where someone wants to do a sequence of rotations on the same
        # axis. Also known as just rotating a single axis.
        return _rotate_one_axis(
            xyz_array, rot_amount=rot_amount1 + rot_amount2, rot_axis=rot_axis1
        )

    # Figure out how many individual rotation matricies we need, accounting for the
    # fact that these can either be floats or ndarrays.
    n_rot = max(
        rot_amount1.shape[0] if isinstance(rot_amount1, np.ndarray) else 1,
        rot_amount2.shape[0] if isinstance(rot_amount2, np.ndarray) else 1,
    )
    n_vec = xyz_array.shape[-1]

    # The promotion of values to float64 is to suppress numerical precision issues,
    # since the matrix math can - in limited circumstances - introduce precision errors
    # of order 10x the limiting numerical precision of the float. For a float32/single,
    # thats a part in 1e6 (~arcsec-level errors), but for a float64 it translates to
    # a part in 1e15.
    rot_matrix = np.empty((3, 3, n_rot), dtype=np.float64)

    # There are two permulations per pair of axes -- when the pair is right-hand
    # oriented vs left-hand oriented. Check here which one it is. For example,
    # rotating first on the x-axis, second on the y-axis is considered a
    # "right-handed" pair, whereas z-axis first, then y-axis would be considered
    # a "left-handed" pair.
    lhd_order = np.mod(rot_axis2 - rot_axis1, 3) != 1

    temp_idx = [
        np.mod(rot_axis1 - lhd_order, 3),
        np.mod(rot_axis1 + 1 - lhd_order, 3),
        np.mod(rot_axis1 + 2 - lhd_order, 3),
    ]

    # We're using lots of sin and cos calculations -- doing them once upfront saves
    # quite a bit of time by eliminating redundant calculations
    sin_lo = np.sin(rot_amount2 if lhd_order else rot_amount1, dtype=np.float64)
    cos_lo = np.cos(rot_amount2 if lhd_order else rot_amount1, dtype=np.float64)
    sin_hi = np.sin(rot_amount1 if lhd_order else rot_amount2, dtype=np.float64)
    cos_hi = np.cos(rot_amount1 if lhd_order else rot_amount2, dtype=np.float64)

    # Take care of the diagonal terms first, since they aren't actually affected by the
    # order of rotational opertations
    rot_matrix[temp_idx[0], temp_idx[0]] = cos_hi
    rot_matrix[temp_idx[1], temp_idx[1]] = cos_lo
    rot_matrix[temp_idx[2], temp_idx[2]] = cos_lo * cos_hi

    # Now time for the off-diagonal terms, as a set of 3 pairs. The rotation matrix
    # for a left-hand oriented pair of rotation axes (e.g., x-rot, then y-rot) is just
    # a transpose of the right-hand orientation of the same pair (e.g., y-rot, then
    # x-rot).
    rot_matrix[temp_idx[0 + lhd_order], temp_idx[1 - lhd_order]] = sin_lo * sin_hi
    rot_matrix[temp_idx[0 - lhd_order], temp_idx[lhd_order - 1]] = (
        cos_lo * sin_hi * ((-1.0) ** lhd_order)
    )

    rot_matrix[temp_idx[1 - lhd_order], temp_idx[0 + lhd_order]] = 0.0
    rot_matrix[temp_idx[1 + lhd_order], temp_idx[2 - lhd_order]] = sin_lo * (
        (-1.0) ** (1 + lhd_order)
    )

    rot_matrix[temp_idx[lhd_order - 1], temp_idx[0 - lhd_order]] = sin_hi * (
        (-1.0) ** (1 + lhd_order)
    )
    rot_matrix[temp_idx[2 - lhd_order], temp_idx[1 + lhd_order]] = (
        sin_lo * cos_hi * ((-1.0) ** (lhd_order))
    )

    # The rot matrix was shape (3, 3, n_rot) to help speed up filling in the elements
    # of each matrix, but now we want to flip it into its proper shape of (n_rot, 3, 3)
    rot_matrix = np.transpose(rot_matrix, axes=[2, 0, 1])

    if (n_rot == 1) and (n_vec == 1) and (xyz_array.ndim == 3):
        # This is a special case where we allow the rotation axis to "expand" along
        # the 0th axis of the rot_amount arrays. For xyz_array, if n_vectors = 1
        # but n_rot !=1, then it's a lot faster (by about 10x) to "switch it up" and
        # swap the n_vector and  n_rot axes, and then swap them back once everything
        # else is done.
        return np.transpose(
            _rotate_matmul_wrapper(  # xyz_array, rot_matrix, n_rot
                xyz_array=np.transpose(xyz_array, axes=[2, 1, 0]),
                rot_matrix=rot_matrix,
                n_rot=n_rot,
            ),
            axes=[2, 1, 0],
        )
    else:
        return _rotate_matmul_wrapper(
            xyz_array=xyz_array, rot_matrix=rot_matrix, n_rot=n_rot
        )


def calc_uvw(
    *,
    app_ra=None,
    app_dec=None,
    frame_pa=None,
    lst_array=None,
    use_ant_pos=True,
    uvw_array=None,
    antenna_positions=None,
    antenna_numbers=None,
    ant_1_array=None,
    ant_2_array=None,
    old_app_ra=None,
    old_app_dec=None,
    old_frame_pa=None,
    telescope_lat=None,
    telescope_lon=None,
    from_enu=False,
    to_enu=False,
):
    """
    Calculate an array of baseline coordinates, in either uvw or ENU.

    This routine is meant as a convenience function for producing baseline coordinates
    based under a few different circumstances:

    1) Calculating ENU coordinates using antenna positions
    2) Calculating uvw coordinates at a given sky position using antenna positions
    3) Converting from ENU coordinates to uvw coordinates
    4) Converting from uvw coordinate to ENU coordinates
    5) Converting from uvw coordinates at one sky position to another sky position

    Different conversion pathways have different parameters that are required.

    Parameters
    ----------
    app_ra : ndarray of float
        Apparent RA of the target phase center, required if calculating baseline
        coordinates in uvw-space (vs ENU-space). Shape is (Nblts,), units are
        radians.
    app_dec : ndarray of float
        Apparent declination of the target phase center, required if calculating
        baseline coordinates in uvw-space (vs ENU-space). Shape is (Nblts,),
        units are radians.
    frame_pa : ndarray of float
        Position angle between the great circle of declination in the apparent frame
        versus that of the reference frame, used for making sure that "North" on
        the derived maps points towards a particular celestial pole (not just the
        topocentric one). Required if not deriving baseline coordinates from antenna
        positions, from_enu=False, and a value for old_frame_pa is given. Shape is
        (Nblts,), units are radians.
    old_app_ra : ndarray of float
        Apparent RA of the previous phase center, required if not deriving baseline
        coordinates from antenna positions and from_enu=False. Shape is (Nblts,),
        units are radians.
    old_app_dec : ndarray of float
        Apparent declination of the previous phase center, required if not deriving
        baseline coordinates from antenna positions and from_enu=False. Shape is
        (Nblts,), units are radians.
    old_frame_pa : ndarray of float
        Frame position angle of the previous phase center, required if not deriving
        baseline coordinates from antenna positions, from_enu=False, and a value
        for frame_pa is supplied. Shape is (Nblts,), units are radians.
    lst_array : ndarray of float
        Local apparent sidereal time, required if deriving baseline coordinates from
        antenna positions, or converting to/from ENU coordinates. Shape is (Nblts,).
    use_ant_pos : bool
        Switch to determine whether to derive uvw values from the antenna positions
        (if set to True), or to use the previously calculated uvw coordinates to derive
        new the new baseline vectors (if set to False). Default is True.
    uvw_array : ndarray of float
        Array of previous baseline coordinates (in either uvw or ENU), required if
        not deriving new coordinates from antenna positions.  Shape is (Nblts, 3).
    antenna_positions : ndarray of float
        List of antenna positions relative to array center in ECEF coordinates,
        required if not providing `uvw_array`. Shape is (Nants, 3).
    antenna_numbers: ndarray of int
        List of antenna numbers, ordered in the same way as `antenna_positions` (e.g.,
        `antenna_numbers[0]` should given the number of antenna that resides at ECEF
        position given by `antenna_positions[0]`). Shape is (Nants,), requred if not
        providing `uvw_array`. Contains all unique entires of the joint set of
        `ant_1_array` and `ant_2_array`.
    ant_1_array : ndarray of int
        Antenna number of the first antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,).
    ant_2_array : ndarray of int
        Antenna number of the second antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,).
    telescope_lat : float
        Latitude of the phase center, units radians, required if deriving baseline
        coordinates from antenna positions, or converting to/from ENU coordinates.
    telescope_lon : float
        Longitude of the phase center, units radians, required if deriving baseline
        coordinates from antenna positions, or converting to/from ENU coordinates.
    from_enu : boolean
        Set to True if uvw_array is expressed in ENU coordinates. Default is False.
    to_enu : boolean
        Set to True if you would like the output expressed in ENU coordinates. Default
        is False.

    Returns
    -------
    new_coords : ndarray of float64
        Set of baseline coordinates, shape (Nblts, 3).
    """
    if to_enu:
        if lst_array is None and not use_ant_pos:
            raise ValueError(
                "Must include lst_array to calculate baselines in ENU coordinates!"
            )
        if telescope_lat is None:
            raise ValueError(
                "Must include telescope_lat to calculate baselines in ENU coordinates!"
            )
    else:
        if ((app_ra is None) or (app_dec is None)) and frame_pa is None:
            raise ValueError(
                "Must include both app_ra and app_dec, or frame_pa to calculate "
                "baselines in uvw coordinates!"
            )

    if use_ant_pos:
        # Assume at this point we are dealing w/ antenna positions
        if antenna_positions is None:
            raise ValueError("Must include antenna_positions if use_ant_pos=True.")
        if (ant_1_array is None) or (ant_2_array is None) or (antenna_numbers is None):
            raise ValueError(
                "Must include ant_1_array, ant_2_array, and antenna_numbers "
                "setting use_ant_pos=True."
            )
        if lst_array is None and not to_enu:
            raise ValueError(
                "Must include lst_array if use_ant_pos=True and not calculating "
                "baselines in ENU coordinates."
            )
        if telescope_lon is None:
            raise ValueError("Must include telescope_lon if use_ant_pos=True.")

        ant_dict = {ant_num: idx for idx, ant_num in enumerate(antenna_numbers)}
        ant_1_index = np.array(
            [ant_dict[ant_num] for ant_num in ant_1_array], dtype=int
        )
        ant_2_index = np.array(
            [ant_dict[ant_num] for ant_num in ant_2_array], dtype=int
        )

        N_ants = antenna_positions.shape[0]
        # Use the app_ra, app_dec, and lst_array arrays to figure out how many unique
        # rotations are actually needed. If the ratio of Nblts to number of unique
        # entries is favorable, we can just rotate the antenna positions and save
        # outselves a bit of work.
        if to_enu:
            # If to_enu, skip all this -- there's only one unique ha + dec combo
            unique_mask = np.zeros(len(ant_1_index), dtype=np.bool_)
            unique_mask[0] = True
        else:
            unique_mask = np.append(
                True,
                (
                    ((lst_array[:-1] - app_ra[:-1]) != (lst_array[1:] - app_ra[1:]))
                    | (app_dec[:-1] != app_dec[1:])
                ),
            )

        # GHA -> Hour Angle as measured at Greenwich (because antenna coords are
        # centered such that x-plane intersects the meridian at longitude 0).
        if to_enu:
            # Unprojected coordinates are given in the ENU convention -- that's
            # equivalent to calculating uvw's based on zenith. We can use that to our
            # advantage and spoof the gha and dec based on telescope lon and lat
            unique_gha = np.zeros(1) - telescope_lon
            unique_dec = np.zeros(1) + telescope_lat
            unique_pa = None
        else:
            unique_gha = (lst_array[unique_mask] - app_ra[unique_mask]) - telescope_lon
            unique_dec = app_dec[unique_mask]
            unique_pa = 0.0 if frame_pa is None else frame_pa[unique_mask]

        # Tranpose the ant vectors so that they are in the proper shape
        ant_vectors = np.transpose(antenna_positions)[np.newaxis, :, :]
        # Apply rotations, and then reorganize the ndarray so that you can access
        # individual antenna vectors quickly.
        ant_rot_vectors = np.reshape(
            np.transpose(
                _rotate_one_axis(
                    _rotate_two_axis(
                        ant_vectors,
                        rot_amount1=unique_gha,
                        rot_amount2=unique_dec,
                        rot_axis1=2,
                        rot_axis2=1,
                    ),
                    rot_amount=unique_pa,
                    rot_axis=0,
                ),
                axes=[0, 2, 1],
            ),
            (-1, 3),
        )

        unique_mask[0] = False
        unique_map = np.cumsum(unique_mask) * N_ants
        new_coords = (
            ant_rot_vectors[unique_map + ant_2_index]
            - ant_rot_vectors[unique_map + ant_1_index]
        )
    else:
        if uvw_array is None:
            raise ValueError("Must include uvw_array if use_ant_pos=False.")
        if from_enu:
            if to_enu:
                # Well this was pointless... returning your uvws unharmed
                return uvw_array
            # Unprojected coordinates appear to be stored in ENU coordinates -- that's
            # equivalent to calculating uvw's based on zenith. We can use that to our
            # advantage and spoof old_app_ra and old_app_dec based on lst_array and
            # telescope_lat
            if telescope_lat is None:
                raise ValueError(
                    "Must include telescope_lat if moving between "
                    "ENU (i.e., 'unprojected') and uvw coordinates!"
                )
            if lst_array is None:
                raise ValueError(
                    "Must include lst_array if moving between ENU "
                    "(i.e., 'unprojected') and uvw coordinates!"
                )
        else:
            if (old_frame_pa is None) and not (frame_pa is None or to_enu):
                raise ValueError(
                    "Must include old_frame_pa values if data are phased and "
                    "applying new position angle values (frame_pa)."
                )
            if ((old_app_ra is None) and not (app_ra is None or to_enu)) or (
                (old_app_dec is None) and not (app_dec is None or to_enu)
            ):
                raise ValueError(
                    "Must include old_app_ra and old_app_dec values when data are "
                    "already phased and phasing to a new position."
                )
        # For this operation, all we need is the delta-ha coverage, which _should_ be
        # entirely encapsulated by the change in RA.
        if (app_ra is None) and (old_app_ra is None):
            gha_delta_array = 0.0
        else:
            gha_delta_array = (lst_array if from_enu else old_app_ra) - (
                lst_array if to_enu else app_ra
            )

        # Notice below there's an axis re-orientation here, to go from uvw -> XYZ,
        # where X is pointing in the direction of the source. This is mostly here
        # for convenience and code legibility -- a slightly different pair of
        # rotations would give you the same result w/o needing to cycle the axes.

        # Up front, we want to trap the corner-case where the sky position you are
        # phasing up to hasn't changed, just the position angle (i.e., which way is
        # up on the map). This is a much easier transform to handle.
        if np.all(gha_delta_array == 0.0) and np.all(old_app_dec == app_dec):
            new_coords = _rotate_one_axis(
                uvw_array[:, [2, 0, 1], np.newaxis],
                rot_amount=frame_pa - (0.0 if old_frame_pa is None else old_frame_pa),
                rot_axis=0,
            )[:, :, 0]
        else:
            new_coords = _rotate_two_axis(
                _rotate_two_axis(
                    uvw_array[:, [2, 0, 1], np.newaxis],
                    rot_amount1=(
                        0.0 if (from_enu or old_frame_pa is None) else (-old_frame_pa)
                    ),
                    rot_amount2=(-telescope_lat) if from_enu else (-old_app_dec),
                    rot_axis1=0,
                    rot_axis2=1,
                ),
                rot_amount1=gha_delta_array,
                rot_amount2=telescope_lat if to_enu else app_dec,
                rot_axis1=2,
                rot_axis2=1,
            )

            # One final rotation applied here, to compensate for the fact that we want
            # the Dec-axis of our image (Fourier dual to the v-axis) to be aligned with
            # the chosen frame, if we not in ENU coordinates
            if not to_enu:
                new_coords = _rotate_one_axis(
                    new_coords, rot_amount=frame_pa, rot_axis=0
                )

            # Finally drop the now-vestigal last axis of the array
            new_coords = new_coords[:, :, 0]

    # There's one last task to do, which is to re-align the axes from projected
    # XYZ -> uvw, where X (which points towards the source) falls on the w axis,
    # and Y and Z fall on the u and v axes, respectively.
    return new_coords[:, [1, 2, 0]]


def transform_sidereal_coords(
    *,
    longitude,
    latitude,
    in_coord_frame,
    out_coord_frame,
    in_coord_epoch=None,
    out_coord_epoch=None,
    time_array=None,
):
    """
    Transform a given set of coordinates from one sidereal coordinate frame to another.

    Uses astropy to convert from a coordinates from sidereal frame into another.
    This function will support transforms from several frames, including GCRS,
    FK5 (i.e., J2000), FK4 (i.e., B1950), Galactic, Supergalactic, CIRS, HCRS, and
    a few others (basically anything that doesn't require knowing the observers
    location on Earth/other celestial body).

    Parameters
    ----------
    lon_coord : float or ndarray of floats
        Logitudinal coordinate to be transformed, typically expressed as the right
        ascension, in units of radians. Can either be a float, or an ndarray of
        floats with shape (Ncoords,). Must agree with lat_coord.
    lat_coord : float or ndarray of floats
        Latitudinal coordinate to be transformed, typically expressed as the
        declination, in units of radians. Can either be a float, or an ndarray of
        floats with shape (Ncoords,). Must agree with lon_coord.
    in_coord_frame : string
        Reference frame for the provided coordinates.  Expected to match a list of
        those supported within the astropy SkyCoord object. An incomplete list includes
        'gcrs', 'fk4', 'fk5', 'galactic', 'supergalactic', 'cirs', and 'hcrs'.
    out_coord_frame : string
        Reference frame to output coordinates in. Expected to match a list of
        those supported within the astropy SkyCoord object. An incomplete list includes
        'gcrs', 'fk4', 'fk5', 'galactic', 'supergalactic', 'cirs', and 'hcrs'.
    in_coord_epoch : float
        Epoch for the input coordinate frame. Optional parameter, only required
        when using either the FK4 (B1950) or FK5 (J2000) coordinate systems. Units are
        in fractional years.
    out_coord_epoch : float
        Epoch for the output coordinate frame. Optional parameter, only required
        when using either the FK4 (B1950) or FK5 (J2000) coordinate systems. Units are
        in fractional years.
    time_array : float or ndarray of floats
        Julian date(s) to which the coordinates correspond to, only used in frames
        with annular motion terms (e.g., abberation in GCRS). Can either be a float,
        or an ndarray of floats with shape (Ntimes,), assuming that either lat_coord
        and lon_coord are floats, or that Ntimes == Ncoords.

    Returns
    -------
    new_lat : float or ndarray of floats
        Longitudinal coordinates, in units of radians. Output will be an ndarray
        if any inputs were, with shape (Ncoords,) or (Ntimes,), depending on inputs.
    new_lon : float or ndarray of floats
        Latidudinal coordinates, in units of radians. Output will be an ndarray
        if any inputs were, with shape (Ncoords,) or (Ntimes,), depending on inputs.
    """
    lon_coord = longitude * units.rad
    lat_coord = latitude * units.rad

    # Check here to make sure that lat_coord and lon_coord are the same length,
    # either 1 or len(time_array)
    if lat_coord.shape != lon_coord.shape:
        raise ValueError("lon and lat must be the same shape.")

    if lon_coord.ndim == 0:
        lon_coord.shape += (1,)
        lat_coord.shape += (1,)

    # Check to make sure that we have a properly formatted epoch for our in-bound
    # coordinate frame
    in_epoch = None
    if isinstance(in_coord_epoch, str | Time):
        # If its a string or a Time object, we don't need to do anything more
        in_epoch = Time(in_coord_epoch)
    elif in_coord_epoch is not None:
        if in_coord_frame.lower() in ["fk4", "fk4noeterms"]:
            in_epoch = Time(in_coord_epoch, format="byear")
        else:
            in_epoch = Time(in_coord_epoch, format="jyear")

    # Now do the same for the outbound frame
    out_epoch = None
    if isinstance(out_coord_epoch, str | Time):
        # If its a string or a Time object, we don't need to do anything more
        out_epoch = Time(out_coord_epoch)
    elif out_coord_epoch is not None:
        if out_coord_frame.lower() in ["fk4", "fk4noeterms"]:
            out_epoch = Time(out_coord_epoch, format="byear")
        else:
            out_epoch = Time(out_coord_epoch, format="jyear")

    # Make sure that time array matched up with what we expect. Thanks to astropy
    # weirdness, time_array has to be the same length as lat/lon coords
    rep_time = False
    rep_crds = False
    if time_array is None:
        time_obj_array = None
    else:
        if isinstance(time_array, Time):
            time_obj_array = time_array
        else:
            time_obj_array = Time(time_array, format="jd", scale="utc")
        if (time_obj_array.size != 1) and (lon_coord.size != 1):
            if time_obj_array.shape != lon_coord.shape:
                raise ValueError(
                    "Shape of time_array must be either that of "
                    " lat_coord/lon_coord if len(time_array) > 1."
                )
        else:
            rep_crds = (time_obj_array.size != 1) and (lon_coord.size == 1)
            rep_time = (time_obj_array.size == 1) and (lon_coord.size != 1)
    if rep_crds:
        lon_coord = np.repeat(lon_coord, len(time_array))
        lat_coord = np.repeat(lat_coord, len(time_array))
    if rep_time:
        time_obj_array = Time(
            np.repeat(time_obj_array.jd, len(lon_coord)), format="jd", scale="utc"
        )
    coord_object = SkyCoord(
        lon_coord,
        lat_coord,
        frame=in_coord_frame,
        equinox=in_epoch,
        obstime=time_obj_array,
    )

    # Easiest, most general way to transform to the new frame is to create a dummy
    # SkyCoord with all the attributes needed -- note that we particularly need this
    # in order to use a non-standard equinox/epoch
    new_coord = coord_object.transform_to(
        SkyCoord(0, 0, unit="rad", frame=out_coord_frame, equinox=out_epoch)
    )

    return new_coord.spherical.lon.rad, new_coord.spherical.lat.rad


def transform_icrs_to_app(
    *,
    time_array,
    ra,
    dec,
    telescope_loc,
    telescope_frame="itrs",
    ellipsoid=None,
    epoch=2000.0,
    pm_ra=None,
    pm_dec=None,
    vrad=None,
    dist=None,
    astrometry_library=None,
):
    """
    Transform a set of coordinates in ICRS to topocentric/apparent coordinates.

    This utility uses one of three libraries (astropy, NOVAS, or ERFA) to calculate
    the apparent (i.e., topocentric) coordinates of a source at a given time and
    location, given a set of coordinates expressed in the ICRS frame. These coordinates
    are most typically used for defining the phase center of the array (i.e, calculating
    baseline vectors).

    As of astropy v4.2, the agreement between the three libraries is consistent down to
    the level of better than 1 mas, with the values produced by astropy and pyERFA
    consistent to bettter than 10 µas (this is not surprising, given that astropy uses
    pyERFA under the hood for astrometry). ERFA is the default as it outputs
    coordinates natively in the apparent frame (whereas NOVAS and astropy do not), as
    well as the fact that of the three libraries, it produces results the fastest.

    Parameters
    ----------
    time_array : float or array-like of float
        Julian dates to calculate coordinate positions for. Can either be a single
        float, or an array-like of shape (Ntimes,).
    ra : float or array-like of float
        ICRS RA of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (with the exception of telescope location parameters).
    dec : float or array-like of float
        ICRS Dec of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (with the exception of telescope location parameters).
    telescope_loc : array-like of floats or EarthLocation or MoonLocation
        ITRS latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    epoch : int or float or str or Time object
        Epoch of the coordinate data supplied, only used when supplying proper motion
        values. If supplying a number, it will assumed to be in Julian years. Default
        is J2000.0.
    pm_ra : float or array-like of float
        Proper motion in RA of the source, expressed in units of milliarcsec / year.
        Proper motion values are applied relative to the J2000 (i.e., RA/Dec ICRS
        values should be set to their expected values when the epoch is 2000.0).
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Note that
        units are in dRA/dt, not cos(Dec)*dRA/dt. Not required.
    pm_dec : float or array-like of float
        Proper motion in Dec of the source, expressed in units of milliarcsec / year.
        Proper motion values are applied relative to the J2000 (i.e., RA/Dec ICRS
        values should be set to their expected values when the epoch is 2000.0).
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Not
        required.
    vrad : float or array-like of float
        Radial velocity of the source, expressed in units of km / sec. Can either be
        a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (namely ra_coord and dec_coord). Not required.
    dist : float or array-like of float
        Distance of the source, expressed in milliarcseconds. Can either be a single
        float or array of shape (Ntimes,), although this must be consistent with other
        parameters (namely ra_coord and dec_coord). Not required.
    astrometry_library : str
        Library used for running the coordinate conversions. Allowed options are
        'erfa' (which uses the pyERFA), 'novas' (which uses the python-novas library),
        and 'astropy' (which uses the astropy utilities). Default is erfa unless
        the telescope_location is a MoonLocation object, in which case the default is
        astropy.

    Returns
    -------
    app_ra : ndarray of floats
        Apparent right ascension coordinates, in units of radians, of shape (Ntimes,).
    app_dec : ndarray of floats
        Apparent declination coordinates, in units of radians, of shape (Ntimes,).
    """
    site_loc, on_moon = get_loc_obj(
        telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
        angle_units=units.rad,
        return_moon=True,
    )

    # Make sure that the library requested is actually permitted
    if astrometry_library is None:
        if on_moon:
            astrometry_library = "astropy"
        else:
            astrometry_library = "erfa"

    if astrometry_library not in ["erfa", "novas", "astropy"]:
        raise ValueError(
            "Requested coordinate transformation library is not supported, please "
            "select either 'erfa', 'novas', or 'astropy' for astrometry_library."
        )
    ra_coord = ra * units.rad
    dec_coord = dec * units.rad

    # Check here to make sure that ra_coord and dec_coord are the same length,
    # either 1 or len(time_array)
    multi_coord = ra_coord.size != 1
    if ra_coord.shape != dec_coord.shape:
        raise ValueError("ra and dec must be the same shape.")

    pm_ra_coord = None if pm_ra is None else pm_ra * (units.mas / units.yr)
    pm_dec_coord = None if pm_dec is None else pm_dec * (units.mas / units.yr)
    d_coord = (
        None if (dist is None or np.all(dist == 0.0)) else Distance(dist * units.pc)
    )
    v_coord = None if vrad is None else vrad * (units.km / units.s)

    opt_list = [pm_ra_coord, pm_dec_coord, d_coord, v_coord]
    opt_names = ["pm_ra", "pm_dec", "dist", "vrad"]
    # Check the optional inputs, make sure that they're sensible
    for item, name in zip(opt_list, opt_names, strict=True):
        if item is not None and ra_coord.shape != item.shape:
            raise ValueError(f"{name} must be the same shape as ra and dec.")

    if on_moon and astrometry_library != "astropy":
        raise NotImplementedError(
            "MoonLocation telescopes are only supported with the 'astropy' astrometry "
            "library"
        )

    # Useful for both astropy and novas methods, the latter of which gives easy
    # access to the IERS data that we want.
    if isinstance(time_array, Time):
        time_obj_array = time_array
    else:
        time_obj_array = Time(time_array, format="jd", scale="utc")

    if time_obj_array.size != 1:
        if (time_obj_array.shape != ra_coord.shape) and multi_coord:
            raise ValueError(
                "time_array must be of either of length 1 (single "
                "float) or same length as ra and dec."
            )
    elif time_obj_array.ndim == 0:
        # Make the array at least 1-dimensional so we don't run into indexing
        # issues later.
        time_obj_array = Time([time_obj_array])

    # Check to make sure that we have a properly formatted epoch for our in-bound
    # coordinate frame
    coord_epoch = None
    if isinstance(epoch, str | Time):
        # If its a string or a Time object, we don't need to do anything more
        coord_epoch = Time(epoch)
    elif epoch is not None:
        coord_epoch = Time(epoch, format="jyear")

    # Note if time_array is a single element
    multi_time = time_obj_array.size != 1

    # Get IERS data, which is needed for NOVAS and ERFA
    polar_motion_data = iers.earth_orientation_table.get()

    pm_x_array, pm_y_array = polar_motion_data.pm_xy(time_obj_array)
    delta_x_array, delta_y_array = polar_motion_data.dcip_xy(time_obj_array)

    pm_x_array = pm_x_array.to_value("arcsec")
    pm_y_array = pm_y_array.to_value("arcsec")
    delta_x_array = delta_x_array.to_value("marcsec")
    delta_y_array = delta_y_array.to_value("marcsec")
    # Catch the case where we don't have CIP delta values yet (they don't typically have
    # predictive values like the polar motion does)
    delta_x_array[np.isnan(delta_x_array)] = 0.0
    delta_y_array[np.isnan(delta_y_array)] = 0.0

    # If the source was instantiated w/ floats, it'll be a 0-dim object, which will
    # throw errors if we try to treat it as an array. Reshape to a 1D array of len 1
    # so that all the calls can be uniform
    if ra_coord.ndim == 0:
        ra_coord.shape += (1,)
        dec_coord.shape += (1,)
        if pm_ra_coord is not None:
            pm_ra_coord.shape += (1,)
        if pm_dec_coord is not None:
            pm_dec_coord.shape += (1,)
        if d_coord is not None:
            d_coord.shape += (1,)
        if v_coord is not None:
            v_coord.shape += (1,)

    if astrometry_library == "astropy":
        # Astropy doesn't have (oddly enough) a way of getting at the apparent RA/Dec
        # directly, but we can cheat this by going to AltAz, and then coverting back
        # to apparent RA/Dec using the telescope lat and LAST.
        if (epoch is not None) and (pm_ra is not None) and (pm_dec is not None):
            # astropy is a bit weird in how it handles proper motion, so rather than
            # fight with it to do it all in one step, we separate it into two: first
            # apply proper motion to ICRS, then transform to topocentric.
            sky_coord = SkyCoord(
                ra=ra_coord,
                dec=dec_coord,
                pm_ra_cosdec=pm_ra_coord * np.cos(dec_coord),
                pm_dec=pm_dec_coord,
                frame="icrs",
            )

            sky_coord = sky_coord.apply_space_motion(dt=(time_obj_array - coord_epoch))
            ra_coord = sky_coord.ra
            dec_coord = sky_coord.dec
            if d_coord is not None:
                d_coord = d_coord.repeat(ra_coord.size)
            if v_coord is not None:
                v_coord = v_coord.repeat(ra_coord.size)

        if not on_moon:
            time_obj_array = Time(time_obj_array, location=site_loc)

            sky_coord = SkyCoord(
                ra=ra_coord,
                dec=dec_coord,
                distance=d_coord,
                radial_velocity=v_coord,
                frame="icrs",
            )

            azel_data = sky_coord.transform_to(
                SkyCoord(
                    np.zeros_like(time_obj_array) * units.rad,
                    np.zeros_like(time_obj_array) * units.rad,
                    location=site_loc,
                    obstime=time_obj_array,
                    frame="altaz",
                )
            )
        else:
            from lunarsky import SkyCoord as LunarSkyCoord, Time as LTime

            sky_coord = LunarSkyCoord(
                ra=ra_coord,
                dec=dec_coord,
                distance=d_coord,
                radial_velocity=v_coord,
                frame="icrs",
            )

            azel_data = sky_coord.transform_to(
                LunarSkyCoord(
                    np.zeros_like(time_obj_array) * units.rad,
                    np.zeros_like(time_obj_array) * units.rad,
                    location=site_loc,
                    obstime=time_obj_array,
                    frame="lunartopo",
                )
            )
            time_obj_array = LTime(time_obj_array, location=site_loc)

        app_ha, app_dec = erfa.ae2hd(
            azel_data.az.rad, azel_data.alt.rad, site_loc.lat.rad
        )
        app_ra = np.mod(
            time_obj_array.sidereal_time("apparent").rad - app_ha, 2 * np.pi
        )

    elif astrometry_library == "novas":
        # Import the NOVAS library only if it's needed/available.
        try:
            import novas_de405  # noqa
            from novas import compat as novas
            from novas.compat import eph_manager
        except ImportError as e:
            raise ImportError(
                "novas and/or novas_de405 are not installed but is required for "
                "NOVAS functionality"
            ) from e

        # Call is needed to load high-precision ephem data in NOVAS
        jd_start, jd_end, number = eph_manager.ephem_open()

        # Define the obs location, which is needed to calculate diurnal abb term
        # and polar wobble corrections
        site_loc = novas.make_on_surface(
            site_loc.lat.deg,  # latitude in deg
            site_loc.lon.deg,  # Longitude in deg
            site_loc.height.to_value("m"),  # Height in meters
            0.0,  # Temperature, set to 0 for now (no atm refrac)
            0.0,  # Pressure, set to 0 for now (no atm refrac)
        )

        # NOVAS wants things in terrestial time and UT1
        tt_time_array = time_obj_array.tt.jd
        ut1_time_array = time_obj_array.ut1.jd
        gast_array = time_obj_array.sidereal_time("apparent", "greenwich").rad

        if np.any(tt_time_array < jd_start) or np.any(tt_time_array > jd_end):
            raise ValueError(
                "No current support for JPL ephems outside of 1700 - 2300 AD. "
                "Check back later (or possibly earlier)..."
            )

        app_ra = np.zeros(tt_time_array.shape) + np.zeros(ra_coord.shape)
        app_dec = np.zeros(tt_time_array.shape) + np.zeros(ra_coord.shape)

        for idx in range(len(app_ra)):
            if multi_coord or (idx == 0):
                # Create a catalog entry for the source in question
                if pm_ra is None:
                    pm_ra_use = 0.0
                else:
                    pm_ra_use = pm_ra_coord[idx].to_value("mas/yr") * np.cos(
                        dec_coord[idx].to_value("rad")
                    )

                if pm_dec is None:
                    pm_dec_use = 0.0
                else:
                    pm_dec_use = pm_dec_coord[idx].to_value("mas/yr")

                if dist is None or np.any(dist == 0.0):
                    parallax = 0.0
                else:
                    parallax = d_coord[idx].kiloparsec ** -1.0

                if vrad is None:
                    vrad_use = 0.0
                else:
                    vrad_use = v_coord[idx].to_value("km/s")

                cat_entry = novas.make_cat_entry(
                    "dummy_name",  # Dummy source name
                    "GKK",  # Catalog ID, fixed for now
                    156,  # Star ID number, fixed for now
                    ra_coord[idx].to_value("hourangle"),
                    dec_coord[idx].to_value("deg"),
                    pm_ra_use,
                    pm_dec_use,
                    parallax,
                    vrad_use,
                )

            # Update polar wobble parameters for a given timestamp
            if multi_time or (idx == 0):
                gast = gast_array[idx]
                pm_x = pm_x_array[idx] * np.cos(gast) + pm_y_array[idx] * np.sin(gast)
                pm_y = pm_y_array[idx] * np.cos(gast) - pm_x_array[idx] * np.sin(gast)
                tt_time = tt_time_array[idx]
                ut1_time = ut1_time_array[idx]
                novas.cel_pole(tt_time, 2, delta_x_array[idx], delta_y_array[idx])

            # Calculate topocentric RA/Dec values
            [temp_ra, temp_dec] = novas.topo_star(
                tt_time, (tt_time - ut1_time) * 86400.0, cat_entry, site_loc, accuracy=0
            )
            xyz_array = polar2_to_cart3(
                lon_array=temp_ra * (np.pi / 12.0), lat_array=temp_dec * (np.pi / 180.0)
            )
            xyz_array = novas.wobble(tt_time, pm_x, pm_y, xyz_array, 1)

            app_ra[idx], app_dec[idx] = cart3_to_polar2(np.array(xyz_array))
    elif astrometry_library == "erfa":
        # liberfa wants things in radians
        pm_x_array *= np.pi / (3600.0 * 180.0)
        pm_y_array *= np.pi / (3600.0 * 180.0)

        if pm_ra is None:
            pm_ra_use = 0.0
        else:
            pm_ra_use = pm_ra_coord.to_value("rad/yr")

        if pm_dec is None:
            pm_dec_use = 0.0
        else:
            pm_dec_use = pm_dec_coord.to_value("rad/yr")

        if dist is None or np.any(dist == 0.0):
            parallax = 0.0
        else:
            parallax = d_coord.pc**-1.0

        if vrad is None:
            vrad_use = 0
        else:
            vrad_use = v_coord.to_value("km/s")

        [_, _, _, app_dec, app_ra, eqn_org] = erfa.atco13(
            ra_coord.to_value("rad"),
            dec_coord.to_value("rad"),
            pm_ra_use,
            pm_dec_use,
            parallax,
            vrad_use,
            time_obj_array.utc.jd1,
            time_obj_array.utc.jd2,
            time_obj_array.delta_ut1_utc,
            site_loc.lon.rad,
            site_loc.lat.rad,
            site_loc.height.to_value("m"),
            pm_x_array,
            pm_y_array,
            0,  # ait pressure, used for refraction (ignored)
            0,  # amb temperature, used for refraction (ignored)
            0,  # rel humidity, used for refraction (ignored)
            0,  # wavelength, used for refraction (ignored)
        )

        app_ra = np.mod(app_ra - eqn_org, 2 * np.pi)

    return app_ra, app_dec


def transform_app_to_icrs(
    *,
    time_array,
    app_ra,
    app_dec,
    telescope_loc,
    telescope_frame="itrs",
    ellipsoid="SPHERE",
    astrometry_library=None,
):
    """
    Transform a set of coordinates in topocentric/apparent to ICRS coordinates.

    This utility uses either astropy or erfa to calculate the ICRS  coordinates of
    a given set of apparent source coordinates. These coordinates are most typically
    used for defining the celestial/catalog position of a source. Note that at present,
    this is only implemented in astropy and pyERFA, although it could hypothetically
    be extended to NOVAS at some point.

    Parameters
    ----------
    time_array : float or ndarray of float
        Julian dates to calculate coordinate positions for. Can either be a single
        float, or an ndarray of shape (Ntimes,).
    app_ra : float or ndarray of float
        ICRS RA of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ncoord,). Note that if time_array is
        not a singleton value, then Ncoord must be equal to Ntimes.
    app_dec : float or ndarray of float
        ICRS Dec of the celestial target, expressed in units of radians. Can either
        be a single float or array of shape (Ncoord,). Note that if time_array is
        not a singleton value, then Ncoord must be equal to Ntimes.
    telescope_loc : tuple of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    astrometry_library : str
        Library used for running the coordinate conversions. Allowed options are
        'erfa' (which uses the pyERFA), and 'astropy' (which uses the astropy
        utilities). Default is erfa unless the telescope_location is a MoonLocation
        object, in which case the default is astropy.

    Returns
    -------
    icrs_ra : ndarray of floats
        ICRS right ascension coordinates, in units of radians, of either shape
        (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    icrs_dec : ndarray of floats
        ICRS declination coordinates, in units of radians, of either shape
        (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    """
    site_loc, on_moon = get_loc_obj(
        telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
        angle_units=units.rad,
        return_moon=True,
    )

    # Make sure that the library requested is actually permitted
    if astrometry_library is None:
        if on_moon:
            astrometry_library = "astropy"
        else:
            astrometry_library = "erfa"

    if astrometry_library not in ["erfa", "astropy"]:
        raise ValueError(
            "Requested coordinate transformation library is not supported, please "
            "select either 'erfa' or 'astropy' for astrometry_library."
        )

    ra_coord = app_ra * units.rad
    dec_coord = app_dec * units.rad

    # Check here to make sure that ra_coord and dec_coord are the same length,
    # either 1 or len(time_array)
    multi_coord = ra_coord.size != 1
    if ra_coord.shape != dec_coord.shape:
        raise ValueError("app_ra and app_dec must be the same shape.")

    if on_moon and astrometry_library != "astropy":
        raise NotImplementedError(
            "MoonLocation telescopes are only supported with the 'astropy' astrometry "
            "library"
        )

    assert time_array.size > 0
    if isinstance(time_array, Time):
        time_obj_array = time_array
    else:
        time_obj_array = Time(time_array, format="jd", scale="utc")

    if time_obj_array.size != 1:
        if (time_obj_array.shape != ra_coord.shape) and multi_coord:
            raise ValueError(
                "time_array must be of either of length 1 (single "
                "float) or same length as ra and dec."
            )
    elif time_obj_array.ndim == 0:
        # Make the array at least 1-dimensional so we don't run into indexing
        # issues later.
        time_obj_array = Time([time_obj_array])

    if astrometry_library == "astropy":
        if on_moon:
            from lunarsky import Time as LTime

            time_obj_array = LTime(time_obj_array, location=site_loc)
        else:
            time_obj_array = Time(time_obj_array, location=site_loc)

        az_coord, el_coord = erfa.hd2ae(
            np.mod(
                time_obj_array.sidereal_time("apparent").rad - ra_coord.to_value("rad"),
                2 * np.pi,
            ),
            dec_coord.to_value("rad"),
            site_loc.lat.rad,
        )

        if not on_moon:
            sky_coord = SkyCoord(
                az_coord * units.rad,
                el_coord * units.rad,
                frame="altaz",
                location=site_loc,
                obstime=time_obj_array,
            )
        else:
            from lunarsky import SkyCoord as LunarSkyCoord

            sky_coord = LunarSkyCoord(
                az_coord * units.rad,
                el_coord * units.rad,
                frame="lunartopo",
                location=site_loc,
                obstime=time_obj_array,
            )

        coord_data = sky_coord.transform_to("icrs")
        icrs_ra = coord_data.ra.rad
        icrs_dec = coord_data.dec.rad
    elif astrometry_library == "erfa":
        # Get IERS data, which is needed for highest precision
        polar_motion_data = iers.earth_orientation_table.get()

        pm_x_array, pm_y_array = polar_motion_data.pm_xy(time_obj_array)
        pm_x_array = pm_x_array.to_value("rad")
        pm_y_array = pm_y_array.to_value("rad")

        bpn_matrix = erfa.pnm06a(time_obj_array.tt.jd1, time_obj_array.tt.jd2)
        cip_x, cip_y = erfa.bpn2xy(bpn_matrix)
        cio_s = erfa.s06(time_obj_array.tt.jd1, time_obj_array.tt.jd2, cip_x, cip_y)
        eqn_org = erfa.eors(bpn_matrix, cio_s)

        # Observed to ICRS via ERFA
        icrs_ra, icrs_dec = erfa.atoc13(
            "r",
            ra_coord.to_value("rad") + eqn_org,
            dec_coord.to_value("rad"),
            time_obj_array.utc.jd1,
            time_obj_array.utc.jd2,
            time_obj_array.delta_ut1_utc,
            site_loc.lon.rad,
            site_loc.lat.rad,
            site_loc.height.value,
            pm_x_array,
            pm_y_array,
            0,  # atm pressure, used for refraction (ignored)
            0,  # amb temperature, used for refraction (ignored)
            0,  # rel humidity, used for refraction (ignored)
            0,  # wavelength, used for refraction (ignored)
        )

    # Return back the two RA/Dec arrays
    return icrs_ra, icrs_dec


def calc_parallactic_angle(*, app_ra, app_dec, lst_array, telescope_lat):
    """
    Calculate the parallactic angle between RA/Dec and the AltAz frame.

    Parameters
    ----------
    app_ra : ndarray of floats
        Array of apparent RA values in units of radians, shape (Ntimes,).
    app_dec : ndarray of floats
        Array of apparent dec values in units of radians, shape (Ntimes,).
    telescope_lat : float
        Latitude of the observatory, in units of radians.
    lst_array : float or ndarray of float
        Array of local apparent sidereal timesto calculate position angle values
        for, in units of radians. Can either be a single float or an array of shape
        (Ntimes,).
    """
    # This is just a simple wrapped around the pas function in ERFA
    return erfa.pas(app_ra, app_dec, lst_array, telescope_lat)


def calc_frame_pos_angle(
    *,
    time_array,
    app_ra,
    app_dec,
    telescope_loc,
    ref_frame,
    ref_epoch=None,
    telescope_frame="itrs",
    ellipsoid="SPHERE",
    offset_pos=(np.pi / 360.0),
):
    """
    Calculate an position angle given apparent position and reference frame.

    This function is used to determine the position angle between the great
    circle of declination in apparent coordinates, versus that in a given
    reference frame. Note that this is slightly different than parallactic
    angle, which is the difference between apparent declination and elevation.

    Paramters
    ---------
    time_array : ndarray of floats
        Array of julian dates to calculate position angle values for, of shape
        (Ntimes,).
    app_ra : ndarray of floats
        Array of apparent RA values in units of radians, shape (Ntimes,).
    app_dec : ndarray of floats
        Array of apparent dec values in units of radians, shape (Ntimes,).
    telescope_loc : tuple of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the observer.
        Can either be provided as an astropy EarthLocation, or an array-like of shape
        (3,) containing the latitude, longitude, and altitude, in that order, with units
        of radians, radians, and meters, respectively.
    ref_frame : str
        Coordinate frame to calculate position angles for. Can be any of the
        several supported frames in astropy (a limited list: fk4, fk5, icrs,
        gcrs, cirs, galactic).
    ref_epoch : str or flt
        Epoch of the coordinates, only used when ref_frame = fk4 or fk5. Given
        in unites of fractional years, either as a float or as a string with
        the epoch abbreviation (e.g, Julian epoch 2000.0 would be J2000.0).
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    offset_pos : float
        Distance of the offset position used to calculate the frame PA. Default
        is 0.5 degrees, which should be sufficent for most applications.


    Returns
    -------
    frame_pa : ndarray of floats
        Array of position angles, in units of radians.
    """
    # Check to see if the position angles should default to zero
    if (ref_frame is None) or (ref_frame == "topo"):
        # No-op detected, ENGAGE MAXIMUM SNARK!
        return np.zeros_like(time_array)

    assert offset_pos > 0, "offset_pos must be greater than 0."

    # This creates an array of unique entries of ra + dec + time, since the processing
    # time for each element can be non-negligible, and entries along the Nblt axis can
    # be highly redundant.
    unique_mask = np.union1d(
        np.union1d(
            np.unique(app_ra, return_index=True)[1],
            np.unique(app_dec, return_index=True)[1],
        ),
        np.unique(time_array, return_index=True)[1],
    )

    # Pluck out the unique entries for each
    unique_ra = app_ra[unique_mask]
    unique_dec = app_dec[unique_mask]
    unique_time = time_array[unique_mask]

    # Figure out how many elements we need to transform
    n_coord = len(unique_mask)

    # Offset north/south positions by 0.5 deg, such that the PA is determined over a
    # 1 deg arc.
    up_dec = unique_dec + offset_pos
    dn_dec = unique_dec - offset_pos
    up_ra = np.array(unique_ra)
    dn_ra = np.array(unique_ra)

    # Wrap the positions if they happen to go over the poles
    select_mask = up_dec > (np.pi / 2.0)
    up_ra[select_mask] = np.mod(up_ra[select_mask] + np.pi, 2.0 * np.pi)
    up_dec[select_mask] = np.pi - up_dec[select_mask]

    select_mask = dn_dec < (-np.pi / 2.0)
    dn_ra[select_mask] = np.mod(dn_ra[select_mask] + np.pi, 2.0 * np.pi)
    dn_dec[select_mask] = (-np.pi) - dn_dec[select_mask]

    # Run the set of offset coordinates through the "reverse" transform. The two offset
    # positions are concat'd together to help reduce overheads
    ref_ra, ref_dec = calc_sidereal_coords(
        time_array=np.tile(unique_time, 2),
        app_ra=np.concatenate((dn_ra, up_ra)),
        app_dec=np.concatenate((dn_dec, up_dec)),
        telescope_loc=telescope_loc,
        coord_frame=ref_frame,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
        coord_epoch=ref_epoch,
    )

    # Use the pas function from ERFA to calculate the position angle. The negative sign
    # is here because we're measuring PA of app -> frame, but we want frame -> app.
    unique_pa = -erfa.pas(
        ref_ra[:n_coord], ref_dec[:n_coord], ref_ra[n_coord:], ref_dec[n_coord:]
    )

    # Finally, we have to go back through and "fill in" the redundant entries
    frame_pa = np.zeros_like(app_ra)
    for idx in range(n_coord):
        select_mask = np.logical_and(
            np.logical_and(unique_ra[idx] == app_ra, unique_dec[idx] == app_dec),
            unique_time[idx] == time_array,
        )
        frame_pa[select_mask] = unique_pa[idx]

    return frame_pa


def lookup_jplhorizons(
    target_name,
    time_array,
    *,
    telescope_loc=None,
    high_cadence=False,
    force_indv_lookup=None,
):
    """
    Lookup solar system body coordinates via the JPL-Horizons service.

    This utility is useful for generating ephemerides, which can then be interpolated in
    order to provide positional data for a target which is moving, such as planetary
    bodies and other solar system objects. Use of this function requires the
    installation of the `astroquery` module.


    Parameters
    ----------
    target_name : str
        Name of the target to gather an ephemeris for. Must match the name
        in the JPL-Horizons database.
    time_array : array-like of float
        Times in UTC Julian days to gather an ephemeris for.
    telescope_loc : tuple of floats or EarthLocation
        ITRS latitude, longitude, and altitude (rel to sea-level) of the observer.
        Can either be provided as an EarthLocation object, or an
        array-like of shape (3,) containing the latitude, longitude, and altitude,
        in that order, with units of radians, radians, and meters, respectively.
    high_cadence : bool
        If set to True, will calculate ephemeris points every 3 minutes in time, as
        opposed to the default of every 3 hours.
    force_indv_lookup : bool
        If set to True, will calculate coordinate values for each value found within
        `time_array`. If False, a regularized time grid is sampled that encloses the
        values contained within `time_array`. Default is False, unless `time_array` is
        of length 1, in which the default is set to True.


    Returns
    -------
    ephem_times : ndarray of float
        Times for which the ephemeris values were calculated, in UTC Julian days.
    ephem_ra : ndarray of float
        ICRS Right ascension of the target at the values within `ephem_times`, in
        units of radians.
    ephem_dec : ndarray of float
        ICRS Declination of the target at the values within `ephem_times`, in units
        of radians.
    ephem_dist : ndarray of float
        Distance of the target relative to the observer, at the values within
        `ephem_times`, in units of parsecs.
    ephem_vel : ndarray of float
        Velocity of the targets relative to the observer, at the values within
        `ephem_times`, in units of km/sec.
    """
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError as err:
        raise ImportError(
            "astroquery is not installed but is required for "
            "planet ephemeris functionality"
        ) from err
    from json import load as json_load
    from os.path import join as path_join

    from pyuvdata.data import DATA_PATH

    # Get the telescope location into a format that JPL-Horizons can understand,
    # which is nominally a dict w/ entries for lon (units of deg), lat (units of
    # deg), and elevation (units of km).
    if isinstance(telescope_loc, EarthLocation):
        site_loc = {
            "lon": telescope_loc.lon.deg,
            "lat": telescope_loc.lat.deg,
            "elevation": telescope_loc.height.to_value(unit=units.km),
        }
    elif telescope_loc is None:
        # Setting to None will report the geocentric position
        site_loc = None
    elif isinstance(telescope_loc, tuple | list) or (
        isinstance(telescope_loc, np.ndarray) and telescope_loc.size > 1
    ):
        # MoonLocations are instances of np.ndarray but have size 1
        site_loc = {
            "lon": telescope_loc[1] * (180.0 / np.pi),
            "lat": telescope_loc[0] * (180.0 / np.pi),
            "elevation": telescope_loc[2] * (0.001),  # m -> km
        }
    else:
        bad_type = False
        try:
            from lunarsky import MoonLocation

            if isinstance(telescope_loc, MoonLocation):
                raise NotImplementedError(
                    "Cannot lookup JPL positions for telescopes with a MoonLocation"
                )
            else:
                bad_type = True
        except ImportError:  # pragma: no cover
            # getting here requires having astroquery but not lunarsky. We don't
            # have a CI like that.
            bad_type = True
        if bad_type:
            raise ValueError(
                f"telescope_loc is not a valid type: {type(telescope_loc)}"
            )

    # If force_indv_lookup is True, or unset but only providing a single value, then
    # just calculate the RA/Dec for the times requested rather than creating a table
    # to interpolate from.
    if force_indv_lookup or (
        (np.array(time_array).size == 1) and (force_indv_lookup is None)
    ):
        epoch_list = np.unique(time_array)
        if len(epoch_list) > 50:
            raise ValueError(
                "Requesting too many individual ephem points from JPL-Horizons. This "
                "can be remedied by setting force_indv_lookup=False or limiting the "
                "number of values in time_array."
            )
    else:
        # When querying for multiple times, its faster (and kinder to the
        # good folks at JPL) to create a range to query, and then interpolate
        # between values. The extra buffer of 0.001 or 0.25 days for high and
        # low cadence is to give enough data points to allow for spline
        # interpolation of the data.
        if high_cadence:
            start_time = np.min(time_array) - 0.001
            stop_time = np.max(time_array) + 0.001
            step_time = "3m"
            n_entries = (stop_time - start_time) * (1440.0 / 3.0)
        else:
            # The start/stop time here are setup to maximize reusability of the
            # data, since astroquery appears to cache the results from previous
            # queries.
            start_time = (0.25 * np.floor(4.0 * np.min(time_array))) - 0.25
            stop_time = (0.25 * np.ceil(4.0 * np.max(time_array))) + 0.25
            step_time = "3h"
            n_entries = (stop_time - start_time) * (24.0 / 3.0)
        # We don't want to overtax the JPL service, so limit ourselves to 1000
        # individual queries at a time. Note that this is likely a conservative
        # cap for JPL-Horizons, but there should be exceptionally few applications
        # that actually require more than this.
        if n_entries > 1000:
            if (len(np.unique(time_array)) <= 50) and (force_indv_lookup is None):
                # If we have a _very_ sparse set of epochs, pass that along instead
                epoch_list = np.unique(time_array)
            else:
                # Otherwise, time to raise an error
                raise ValueError(
                    "Too many ephem points requested from JPL-Horizons. This "
                    "can be remedied by setting high_cadance=False or limiting "
                    "the number of values in time_array."
                )
        else:
            epoch_list = {
                "start": Time(start_time, format="jd").isot,
                "stop": Time(stop_time, format="jd").isot,
                "step": step_time,
            }
    # Check to make sure dates are within the 1700-2200 time range,
    # since not all targets are supported outside of this range
    if (np.min(time_array) < 2341973.0) or (np.max(time_array) > 2524593.0):
        raise ValueError(
            "No current support for JPL ephems outside of 1700 - 2300 AD. "
            "Check back later (or possibly earlier)..."
        )

    # JPL-Horizons has a separate catalog with what it calls 'major bodies',
    # and will throw an error if you use the wrong catalog when calling for
    # astrometry. We'll use the dict below to capture this behavior.
    with open(path_join(DATA_PATH, "jpl_major_bodies.json")) as fhandle:
        major_body_dict = json_load(fhandle)

    target_id = target_name
    id_type = "smallbody"
    # If we find the target in the major body database, then we can extract the
    # target ID to make the query a bit more robust (otherwise JPL-Horizons will fail
    # on account that id will find multiple partial matches: e.g., "Mars" will be
    # matched with "Mars", "Mars Explorer", "Mars Barycenter"..., and JPL-Horizons will
    # not know which to choose).
    if target_name in major_body_dict:
        target_id = major_body_dict[target_name]
        id_type = None

    query_obj = Horizons(
        id=target_id, location=site_loc, epochs=epoch_list, id_type=id_type
    )
    # If not in the major bodies catalog, try the minor bodies list, and if
    # still not found, throw an error.
    try:
        ephem_data = query_obj.ephemerides(extra_precision=True)
    except KeyError:
        # This is a fix for an astroquery + JPL-Horizons bug, that's related to
        # API change on JPL's side. In this case, the source is identified, but
        # astroquery can't correctly parse the return message from JPL-Horizons.
        # See astroquery issue #2169.
        ephem_data = query_obj.ephemerides(extra_precision=False)  # pragma: no cover
    except ValueError as err:
        query_obj._session.close()
        if "Unknown target" in str(err):
            raise ValueError(
                "Target ID is not recognized in either the small or major bodies "
                "catalogs, please consult the JPL-Horizons database for supported "
                "targets (https://ssd.jpl.nasa.gov/?horizons)."
            ) from err
        else:
            raise  # pragma: no cover
    # This is explicitly closed here to trap a bug that occassionally throws an
    # unexpected warning, see astroquery issue #1807
    query_obj._session.close()

    # Now that we have the ephem data, extract out the relevant data
    ephem_times = np.array(ephem_data["datetime_jd"])
    ephem_ra = np.array(ephem_data["RA"]) * (np.pi / 180.0)
    ephem_dec = np.array(ephem_data["DEC"]) * (np.pi / 180.0)
    ephem_dist = np.array(ephem_data["delta"])  # AU
    ephem_vel = np.array(ephem_data["delta_rate"])  # km/s

    return ephem_times, ephem_ra, ephem_dec, ephem_dist, ephem_vel


def interpolate_ephem(
    *, time_array, ephem_times, ephem_ra, ephem_dec, ephem_dist=None, ephem_vel=None
):
    """
    Interpolates ephemerides to give positions for requested times.

    This is a simple tool for calculated interpolated RA and Dec positions, as well
    as distances and velocities, for a given ephemeris. Under the hood, the method
    uses as cubic spline interpolation to calculate values at the requested times,
    provided that there are enough values to interpolate over to do so (requires
    >= 4 points), otherwise a linear interpolation is used.

    Parameters
    ----------
    time_array : array-like of floats
        Times to interpolate positions for, in UTC Julian days.
    ephem_times : array-like of floats
        Times in UTC Julian days which describe that match to the recorded postions
        of the target. Must be array-like, of shape (Npts,), where Npts is the number
        of ephemeris points.
    ephem_ra : array-like of floats
        Right ascencion of the target, at the times given in `ephem_times`. Units are
        in radians, must have the same shape as `ephem_times`.
    ephem_dec : array-like of floats
        Declination of the target, at the times given in `ephem_times`. Units are
        in radians, must have the same shape as `ephem_times`.
    ephem_dist : array-like of floats
        Distance of the target from the observer, at the times given in `ephem_times`.
        Optional argument, in units of parsecs. Must have the same shape as
        `ephem_times`.
    ephem_vel : array-like of floats
        Velocities of the target, at the times given in `ephem_times`. Optional
        argument, in units of km/sec. Must have the same shape as `ephem_times`.

    Returns
    -------
    ra_vals : ndarray of float
        Interpolated RA values, returned as an ndarray of floats with
        units of radians, and the same shape as `time_array`.
    dec_vals : ndarray of float
        Interpolated declination values, returned as an ndarray of floats with
        units of radians, and the same shape as `time_array`.
    dist_vals : None or ndarray of float
        If `ephem_dist` was provided, an ndarray of floats (with same shape as
        `time_array`) with the interpolated target distances, in units of parsecs.
        If `ephem_dist` was not provided, this returns as None.
    vel_vals : None or ndarray of float
        If `ephem_vals` was provided, an ndarray of floats (with same shape as
        `time_array`) with the interpolated target velocities, in units of km/sec.
        If `ephem_vals` was not provided, this returns as None.

    """
    # We're importing this here since it's only used for this one function
    from scipy.interpolate import interp1d

    ephem_shape = np.array(ephem_times).shape

    # Make sure that things look reasonable
    if np.array(ephem_ra).shape != ephem_shape:
        raise ValueError("ephem_ra must have the same shape as ephem_times.")

    if np.array(ephem_dec).shape != ephem_shape:
        raise ValueError("ephem_dec must have the same shape as ephem_times.")

    if (np.array(ephem_dist).shape != ephem_shape) and (ephem_dist is not None):
        raise ValueError("ephem_dist must have the same shape as ephem_times.")

    if (np.array(ephem_vel).shape != ephem_shape) and (ephem_vel is not None):
        raise ValueError("ephem_vel must have the same shape as ephem_times.")

    ra_vals = np.zeros_like(time_array, dtype=float)
    dec_vals = np.zeros_like(time_array, dtype=float)
    dist_vals = None if ephem_dist is None else np.zeros_like(time_array, dtype=float)
    vel_vals = None if ephem_vel is None else np.zeros_like(time_array, dtype=float)

    if len(ephem_times) == 1:
        ra_vals += ephem_ra
        dec_vals += ephem_dec
        if ephem_dist is not None:
            dist_vals += ephem_dist
        if ephem_vel is not None:
            vel_vals += ephem_vel
    else:
        if len(ephem_times) > 3:
            interp_kind = "cubic"
        else:
            interp_kind = "linear"

        # If we have values that line up perfectly, just use those directly
        select_mask = np.isin(time_array, ephem_times)
        if np.any(select_mask):
            time_select = time_array[select_mask]
            ra_vals[select_mask] = interp1d(ephem_times, ephem_ra, kind="nearest")(
                time_select
            )
            dec_vals[select_mask] = interp1d(ephem_times, ephem_dec, kind="nearest")(
                time_select
            )
            if ephem_dist is not None:
                dist_vals[select_mask] = interp1d(
                    ephem_times, ephem_dist, kind="nearest"
                )(time_select)
            if ephem_vel is not None:
                vel_vals[select_mask] = interp1d(
                    ephem_times, ephem_vel, kind="nearest"
                )(time_select)

        # If we have values lining up between grid points, use spline interpolation
        # to calculate their values
        select_mask = ~select_mask
        if np.any(select_mask):
            time_select = time_array[select_mask]
            ra_vals[select_mask] = interp1d(ephem_times, ephem_ra, kind=interp_kind)(
                time_select
            )
            dec_vals[select_mask] = interp1d(ephem_times, ephem_dec, kind=interp_kind)(
                time_select
            )
            if ephem_dist is not None:
                dist_vals[select_mask] = interp1d(
                    ephem_times, ephem_dist, kind=interp_kind
                )(time_select)
            if ephem_vel is not None:
                vel_vals[select_mask] = interp1d(
                    ephem_times, ephem_vel, kind=interp_kind
                )(time_select)

    return (ra_vals, dec_vals, dist_vals, vel_vals)


def calc_app_coords(
    *,
    lon_coord,
    lat_coord,
    coord_frame="icrs",
    coord_epoch=None,
    coord_times=None,
    coord_type="sidereal",
    time_array=None,
    lst_array=None,
    telescope_loc=None,
    telescope_frame="itrs",
    ellipsoid=None,
    pm_ra=None,
    pm_dec=None,
    vrad=None,
    dist=None,
    all_times_unique: bool | None = None,
):
    """
    Calculate apparent coordinates for several different coordinate types.

    This function calculates apparent positions at the current epoch.

    Parameters
    ----------
    lon_coord : float or ndarray of float
        Longitudinal (e.g., RA) coordinates, units of radians. Must match the same
        shape as lat_coord.
    lat_coord : float or ndarray of float
        Latitudinal (e.g., Dec) coordinates, units of radians. Must match the same
        shape as lon_coord.
    coord_frame : string
        The requested reference frame for the output coordinates, can be any frame
        that is presently supported by astropy.
    coord_epoch : float or str or Time object
        Epoch for ref_frame, nominally only used if converting to either the FK4 or
        FK5 frames, in units of fractional years. If provided as a float and the
        coord_frame is an FK4-variant, value will assumed to be given in Besselian
        years (i.e., 1950 would be 'B1950'), otherwise the year is assumed to be
        in Julian years.
    coord_times : float or ndarray of float
        Only used when `coord_type="ephem"`, the JD UTC time for each value of
        `lon_coord` and `lat_coord`. These values are used to interpolate `lon_coord`
        and `lat_coord` values to those times listed in `time_array`.
    coord_type : str
        Type of source to calculate coordinates for. Must be one of:

            - "sidereal" (fixed RA/Dec),
            - "ephem" (RA/Dec that moves with time),
            - "driftscan" (fixed az/el position),
            - "unprojected" (alias for "driftscan" with (Az, Alt) = (0 deg, 90 deg)).
            - "near_field" (equivalent to sidereal, with the addition of
              near-field corrections)
    time_array : float or ndarray of float or Time object
        Times for which the apparent coordinates are to be calculated, in UTC JD.
        If more than a single element, must be the same shape as lon_coord and
        lat_coord if both of those are arrays (instead of single floats).
    telescope_loc : array-like of floats or EarthLocation or MoonLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, a lunarsky
        Moonlocation, or a tuple of shape (3,) containing (in order) the latitude,
        longitude, and altitude for a position on Earth in units of radians, radians,
        and meters, respectively.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    pm_ra : float or ndarray of float
        Proper motion in RA of the source, expressed in units of milliarcsec / year.
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Not
        required, motion is calculated relative to the value of `coord_epoch`.
    pm_dec : float or ndarray of float
        Proper motion in Dec of the source, expressed in units of milliarcsec / year.
        Can either be a single float or array of shape (Ntimes,), although this must
        be consistent with other parameters (namely ra_coord and dec_coord). Not
        required, motion is calculated relative to the value of `coord_epoch`.
    vrad : float or ndarray of float
        Radial velocity of the source, expressed in units of km / sec. Can either be
        a single float or array of shape (Ntimes,), although this must be consistent
        with other parameters (namely ra_coord and dec_coord). Not required.
    dist : float or ndarray of float
        Distance of the source, expressed in milliarcseconds. Can either be a single
        float or array of shape (Ntimes,), although this must be consistent with other
        parameters (namely ra_coord and dec_coord). Not required.
    all_times_unique
        Boolean specifying whether all the times that were passed are unique. Default
        is to determine this within the function, but specifying it here can improve
        performance.

    Returns
    -------
    app_ra : ndarray of floats
        Apparent right ascension coordinates, in units of radians.
    app_dec : ndarray of floats
        Apparent declination coordinates, in units of radians.
    """
    site_loc = get_loc_obj(
        telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
        angle_units=units.rad,
    )

    # Time objects and unique don't seem to play well together, so we break apart
    # their handling here
    if isinstance(time_array, Time):
        time_array = time_array.utc.jd

    if not all_times_unique:
        unique_time_array, unique_mask = np.unique(time_array, return_index=True)
    else:
        unique_time_array = time_array
        unique_mask = slice(None)

    if coord_type in ["driftscan", "unprojected"]:
        if lst_array is None:
            unique_lst = get_lst_for_time(unique_time_array, telescope_loc=site_loc)
        else:
            unique_lst = lst_array[unique_mask]

    if coord_type == "sidereal" or coord_type == "near_field":
        # If the coordinates are not in the ICRS frame, go ahead and transform them now
        if coord_frame != "icrs":
            icrs_ra, icrs_dec = transform_sidereal_coords(
                longitude=lon_coord,
                latitude=lat_coord,
                in_coord_frame=coord_frame,
                out_coord_frame="icrs",
                in_coord_epoch=coord_epoch,
                time_array=unique_time_array,
            )
        else:
            icrs_ra = lon_coord
            icrs_dec = lat_coord
        unique_app_ra, unique_app_dec = transform_icrs_to_app(
            time_array=unique_time_array,
            ra=icrs_ra,
            dec=icrs_dec,
            telescope_loc=site_loc,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
            vrad=vrad,
            dist=dist,
        )

    elif coord_type == "driftscan":
        # Use the ERFA function ae2hd, which will do all the heavy
        # lifting for us
        unique_app_ha, unique_app_dec = erfa.ae2hd(
            lon_coord, lat_coord, site_loc.lat.rad
        )
        # The above returns HA/Dec, so we just need to rotate by
        # the LST to get back app RA and Dec
        unique_app_ra = np.mod(unique_app_ha + unique_lst, 2 * np.pi)
        unique_app_dec = unique_app_dec + np.zeros_like(unique_app_ra)
    elif coord_type == "ephem":
        interp_ra, interp_dec, _, _ = interpolate_ephem(
            time_array=unique_time_array,
            ephem_times=coord_times,
            ephem_ra=lon_coord,
            ephem_dec=lat_coord,
        )
        if coord_frame != "icrs":
            icrs_ra, icrs_dec = transform_sidereal_coords(
                longitude=interp_ra,
                latitude=interp_dec,
                in_coord_frame=coord_frame,
                out_coord_frame="icrs",
                in_coord_epoch=coord_epoch,
                time_array=unique_time_array,
            )
        else:
            icrs_ra = interp_ra
            icrs_dec = interp_dec
        # TODO: Vel and distance handling to be integrated here, once they are are
        # needed for velocity frame tracking
        unique_app_ra, unique_app_dec = transform_icrs_to_app(
            time_array=unique_time_array,
            ra=icrs_ra,
            dec=icrs_dec,
            telescope_loc=site_loc,
            pm_ra=pm_ra,
            pm_dec=pm_dec,
        )
    elif coord_type == "unprojected":
        # This is the easiest one - this is just supposed to be ENU, so set the
        # apparent coords to the current lst and telescope_lat.
        unique_app_ra = unique_lst.copy()
        unique_app_dec = np.zeros_like(unique_app_ra) + site_loc.lat.rad
    else:
        raise ValueError(f"Object type {coord_type} is not recognized.")

    # Now that we've calculated all the unique values, time to backfill through the
    # "redundant" entries in the Nblt axis.
    app_ra = np.zeros(np.array(time_array).shape)
    app_dec = np.zeros(np.array(time_array).shape)

    for idx, unique_time in enumerate(unique_time_array):
        if not all_times_unique:
            select_mask = time_array == unique_time
        else:
            select_mask = idx
        app_ra[select_mask] = unique_app_ra[idx]
        app_dec[select_mask] = unique_app_dec[idx]

    return app_ra, app_dec


def calc_sidereal_coords(
    *,
    time_array,
    app_ra,
    app_dec,
    telescope_loc,
    coord_frame,
    telescope_frame="itrs",
    ellipsoid=None,
    coord_epoch=None,
):
    """
    Calculate sidereal coordinates given apparent coordinates.

    This function calculates coordinates in the requested frame (at a given epoch)
    from a set of apparent coordinates.

    Parameters
    ----------
    time_array : float or ndarray of float or Time object
        Times for which the apparent coordinates were calculated, in UTC JD. Must
        match the shape of app_ra and app_dec.
    app_ra : float or ndarray of float
        Array of apparent right ascension coordinates, units of radians. Must match
        the shape of time_array and app_dec.
    app_ra : float or ndarray of float
        Array of apparent right declination coordinates, units of radians. Must match
        the shape of time_array and app_dec.
    telescope_loc : tuple of floats or EarthLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, or a tuple
        of shape (3,) containing (in order) the latitude, longitude, and altitude,
        in units of radians, radians, and meters, respectively.
    coord_frame : string
        The requested reference frame for the output coordinates, can be any frame
        that is presently supported by astropy. Default is ICRS.
    telescope_frame: str, optional
        Reference frame for telescope location. Options are itrs (default) or mcmf.
        Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    coord_epoch : float or str or Time object
        Epoch for ref_frame, nominally only used if converting to either the FK4 or
        FK5 frames, in units of fractional years. If provided as a float and the
        ref_frame is an FK4-variant, value will assumed to be given in Besselian
        years (i.e., 1950 would be 'B1950'), otherwise the year is assumed to be
        in Julian years.

    Returns
    -------
    ref_ra : ndarray of floats
        Right ascension coordinates in the requested frame, in units of radians.
        Either shape (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    ref_dec : ndarray of floats
        Declination coordinates in the requested frame, in units of radians.
        Either shape (Ntimes,) if Ntimes >1, otherwise (Ncoord,).
    """
    # Check to make sure that we have a properly formatted epoch for our in-bound
    # coordinate frame
    epoch = None
    if isinstance(coord_epoch, str | Time):
        # If its a string or a Time object, we don't need to do anything more
        epoch = Time(coord_epoch)
    elif coord_epoch is not None:
        if coord_frame.lower() in ["fk4", "fk4noeterms"]:
            epoch = Time(coord_epoch, format="byear")
        else:
            epoch = Time(coord_epoch, format="jyear")

    if telescope_frame == "mcmf" and ellipsoid is None:
        ellipsoid = "SPHERE"

    icrs_ra, icrs_dec = transform_app_to_icrs(
        time_array=time_array,
        app_ra=app_ra,
        app_dec=app_dec,
        telescope_loc=telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
    )

    if coord_frame == "icrs":
        ref_ra, ref_dec = (icrs_ra, icrs_dec)
    else:
        ref_ra, ref_dec = transform_sidereal_coords(
            longitude=icrs_ra,
            latitude=icrs_dec,
            in_coord_frame="icrs",
            out_coord_frame=coord_frame,
            out_coord_epoch=epoch,
            time_array=time_array,
        )

    return ref_ra, ref_dec


def uvw_track_generator(
    *,
    lon_coord=None,
    lat_coord=None,
    coord_frame="icrs",
    coord_epoch=None,
    coord_type="sidereal",
    time_array=None,
    telescope_loc=None,
    telescope_frame="itrs",
    ellipsoid=None,
    antenna_positions=None,
    antenna_numbers=None,
    ant_1_array=None,
    ant_2_array=None,
    uvw_array=None,
    force_postive_u=False,
):
    """
    Calculate uvw coordinates (among other values) for a given position on the sky.

    This function is meant to be a user-friendly wrapper around several pieces of code
    for effectively simulating a track.

    Parameters
    ----------
    lon_coord : float or ndarray of float
        Longitudinal (e.g., RA) coordinates, units of radians. Must match the same
        shape as lat_coord.
    lat_coord : float or ndarray of float
        Latitudinal (e.g., Dec) coordinates, units of radians. Must match the same
        shape as lon_coord.
    coord_frame : string
        The requested reference frame for the output coordinates, can be any frame
        that is presently supported by astropy.
    coord_epoch : float or str or Time object, optional
        Epoch for ref_frame, nominally only used if converting to either the FK4 or
        FK5 frames, in units of fractional years. If provided as a float and the
        ref_frame is an FK4-variant, value will assumed to be given in Besselian
        years (i.e., 1950 would be 'B1950'), otherwise the year is assumed to be
        in Julian years.
    coord_type : str
        Type of source to calculate coordinates for. Must be one of:
            "sidereal" (fixed RA/Dec),
            "ephem" (RA/Dec that moves with time),
            "driftscan" (fixed az/el position),
            "unprojected" (alias for "driftscan" with (Az, Alt) = (0 deg, 90 deg)).
    time_array : ndarray of float or Time object
        Times for which the apparent coordinates were calculated, in UTC JD. Must
        match the shape of lon_coord and lat_coord.
    telescope_loc : array-like of floats or EarthLocation or MoonLocation
        ITRF latitude, longitude, and altitude (rel to sea-level) of the phase center
        of the array. Can either be provided as an astropy EarthLocation, a lunarsky
        Moonlocation, or a tuple of shape (3,) containing (in order) the latitude,
        longitude, and altitude for a position on Earth in units of degrees, degrees,
        and meters, respectively.
    telescope_frame : str, optional
        Reference frame for latitude/longitude/altitude. Options are itrs (default) or
        mcmf. Only used if telescope_loc is not an EarthLocation or MoonLocation.
    ellipsoid : str
        Ellipsoid to use for lunar coordinates. Must be one of "SPHERE",
        "GSFC", "GRAIL23", "CE-1-LAM-GEO" (see lunarsky package for details). Default
        is "SPHERE". Only used if frame is mcmf.
    antenna_positions : ndarray of float
        List of antenna positions relative to array center in ECEF coordinates,
        required if not providing `uvw_array`. Shape is (Nants, 3).
    antenna_numbers: ndarray of int, optional
        List of antenna numbers, ordered in the same way as `antenna_positions` (e.g.,
        `antenna_numbers[0]` should given the number of antenna that resides at ECEF
        position given by `antenna_positions[0]`). Shape is (Nants,), requred if
        supplying ant_1_array and ant_2_array.
    ant_1_array : ndarray of int, optional
        Antenna number of the first antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,). If not supplied, then
        the method will automatically fill in ant_1_array with all unique antenna
        pairings for each time/position.
    ant_2_array : ndarray of int, optional
        Antenna number of the second antenna in the baseline pair, for all baselines
        Required if not providing `uvw_array`, shape is (Nblts,). If not supplied, then
        the method will automatically fill in ant_2_array with all unique antenna
        pairings for each time/position.
    uvw_array : ndarray of float, optional
        Array of baseline coordinates (in ENU), required if not deriving new coordinates
        from antenna positions. Setting this value will will cause antenna positions to
        be ignored. Shape is (Nblts, 3).
    force_positive_u : bool, optional
        If set to true, then forces the conjugation of each individual baseline to be
        set such that the uvw coordinates land on the positive-u side of the uv-plane.
        Default is False.

    Returns
    -------
    obs_dict : dict
        Dictionary containing the results of the simulation, which includes:
            "uvw" the uvw-coordinates (meters),
            "app_ra" apparent RA of the sources (radians),
            "app_dec"  apparent Dec of the sources (radians),
            "frame_pa"  ngle between apparent north and `coord_frame` north (radians),
            "lst" local apparent sidereal time (radians),
            "site_loc" EarthLocation or MoonLocation for the telescope site.
    """
    site_loc = get_loc_obj(
        telescope_loc,
        telescope_frame=telescope_frame,
        ellipsoid=ellipsoid,
        angle_units=units.deg,
    )

    if not isinstance(lon_coord, np.ndarray):
        lon_coord = np.array(lon_coord)
    if not isinstance(lat_coord, np.ndarray):
        lat_coord = np.array(lat_coord)
    if not isinstance(time_array, np.ndarray):
        time_array = np.array(time_array)

    if lon_coord.ndim == 0:
        lon_coord = lon_coord.reshape(1)
    if lat_coord.ndim == 0:
        lat_coord = lat_coord.reshape(1)
    if time_array.ndim == 0:
        time_array = time_array.reshape(1)

    Ntimes = len(time_array)
    if uvw_array is None and all(
        item is None for item in [antenna_numbers, ant_1_array, ant_2_array]
    ):
        antenna_numbers = np.arange(1, 1 + len(antenna_positions))
        ant_1_array = []
        ant_2_array = []
        for idx in range(len(antenna_positions)):
            for jdx in range(idx + 1, len(antenna_positions)):
                ant_1_array.append(idx + 1)
                ant_2_array.append(jdx + 1)

        Nbase = len(ant_1_array)

        ant_1_array = np.tile(ant_1_array, Ntimes)
        ant_2_array = np.tile(ant_2_array, Ntimes)
        if len(lon_coord) == len(time_array):
            lon_coord = np.repeat(lon_coord, Nbase)
            lat_coord = np.repeat(lat_coord, Nbase)

        time_array = np.repeat(time_array, Nbase)

    lst_array = get_lst_for_time(jd_array=time_array, telescope_loc=site_loc)
    app_ra, app_dec = calc_app_coords(
        lon_coord=lon_coord,
        lat_coord=lat_coord,
        coord_frame=coord_frame,
        coord_type=coord_type,
        time_array=time_array,
        lst_array=lst_array,
        telescope_loc=site_loc,
    )

    frame_pa = calc_frame_pos_angle(
        time_array=time_array,
        app_ra=app_ra,
        app_dec=app_dec,
        telescope_loc=site_loc,
        ref_frame=coord_frame,
        ref_epoch=coord_epoch,
    )

    uvws = calc_uvw(
        app_ra=app_ra,
        app_dec=app_dec,
        frame_pa=frame_pa,
        lst_array=lst_array,
        antenna_positions=antenna_positions,
        antenna_numbers=antenna_numbers,
        ant_1_array=ant_1_array,
        ant_2_array=ant_2_array,
        telescope_lon=site_loc.lon.rad,
        telescope_lat=site_loc.lat.rad,
        uvw_array=uvw_array,
        use_ant_pos=(uvw_array is None),
        from_enu=(uvw_array is not None),
    )

    if force_postive_u:
        mask = (uvws[:, 0] < 0.0) | ((uvws[:, 0] == 0.0) & (uvws[:, 1] < 0.0))
        uvws[mask, :] *= -1.0

    return {
        "uvw": uvws,
        "app_ra": app_ra,
        "app_dec": app_dec,
        "frame_pa": frame_pa,
        "lst": lst_array,
        "site_loc": site_loc,
    }


def _get_focus_xyz(uvd, focus, ra, dec):
    """
    Return the x,y,z coordinates of the focal point.

    The focal point corresponds to the location of
    the near-field object of interest in the ENU
    frame centered on the median position of the
    antennas.

    Parameters
    ----------
    uvd : UVData object
        UVData object
    focus : float
        Focal distance of the array (km)
    ra : float
        Right ascension of the focal point ie phase center (deg; shape (Ntimes,))
    dec : float
        Declination of the focal point ie phase center (deg; shape (Ntimes,))

    Returns
    -------
    x, y, z: ndarray, ndarray, ndarray
        ENU-frame coordinates of the focal point (meters) (shape (Ntimes,))
    """
    # Obtain timesteps
    timesteps = Time(np.unique(uvd.time_array), format="jd")

    # Initialize sky-based coordinates using right ascension and declination
    obj = SkyCoord(ra * units.deg, dec * units.deg)

    # Initialize EarthLocation object centred on the telescope
    loc = uvd.telescope.location.itrs.cartesian.xyz.value
    antpos = uvd.telescope.antenna_positions + loc
    x, y, z = np.median(antpos, axis=0)

    telescope = EarthLocation(x, y, z, unit=units.m)

    # Convert sky object to an AltAz frame centered on the telescope
    obj = obj.transform_to(AltAz(obstime=timesteps, location=telescope))

    # Obtain altitude and azimuth
    theta, phi = obj.alt.to(units.rad), obj.az.to(units.rad)

    # Obtain x,y,z ENU coordinates
    x = focus * 1e3 * np.cos(theta) * np.sin(phi)
    y = focus * 1e3 * np.cos(theta) * np.cos(phi)
    z = focus * 1e3 * np.sin(theta)

    return x, y, z


def _get_nearfield_delay(uvd, focus_x, focus_y, focus_z):
    """
    Calculate near-field phase/delay along the Nblts axis.

    Parameters
    ----------
    uvd : UVData object
        UVData object
    focus_x, focus_y, focus_z : ndarray, ndarray, ndarray
        ENU-frame coordinates of focal point (Each of shape (Ntimes,))

    Returns
    -------
    new_w : ndarray
        The calculated near-field delay (or w-term) for each visibility along
        the Nblts axis
    """
    # Get indices to convert between Nants and Nblts
    ind1, ind2 = _nants_to_nblts(uvd)

    # The center of the ENU frame should be located at the median position of the array
    antpos = uvd.telescope.get_enu_antpos() - np.median(
        uvd.telescope.get_enu_antpos(), axis=0
    )

    # Get tile positions for each baseline
    tile1 = antpos[ind1]  # Shape (Nblts, 3)
    tile2 = antpos[ind2]  # Shape (Nblts, 3)

    # Focus points have shape (Ntimes,); convert to shape (Nblts,)
    t_inds = _ntimes_to_nblts(uvd)
    (focus_x, focus_y, focus_z) = (focus_x[t_inds], focus_y[t_inds], focus_z[t_inds])

    # Calculate distance from antennas to focal point
    # for each visibility along the Nblts axis
    r1 = np.sqrt(
        (tile1[:, 0] - focus_x) ** 2
        + (tile1[:, 1] - focus_y) ** 2
        + (tile1[:, 2] - focus_z) ** 2
    )
    r2 = np.sqrt(
        (tile2[:, 0] - focus_x) ** 2
        + (tile2[:, 1] - focus_y) ** 2
        + (tile2[:, 2] - focus_z) ** 2
    )

    # Get the uvw array along the Nblts axis; select only the w's
    old_w = uvd.uvw_array[:, -1]

    # Calculate near-field delay
    new_w = r1 - r2

    # Mask autocorrelations
    mask = np.not_equal(uvd.ant_1_array, uvd.ant_2_array)
    new_w = np.where(mask, new_w, old_w)

    return new_w  # Shape (Nblts,)
