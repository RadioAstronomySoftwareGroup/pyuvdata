# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for baseline numbers."""
import re
import warnings

import numpy as np

from . import _bls
from .pol import polnum2str, polstr2num

__all__ = ["baseline_to_antnums", "antnums_to_baseline"]


def baseline_to_antnums(baseline, *, Nants_telescope):  # noqa: N803
    """
    Get the antenna numbers corresponding to a given baseline number.

    Parameters
    ----------
    baseline : int or array_like of ints
        baseline number
    Nants_telescope : int
        number of antennas

    Returns
    -------
    int or array_like of int
        first antenna number(s)
    int or array_like of int
        second antenna number(s)

    """
    if Nants_telescope > 2147483648:
        raise ValueError(f"error Nants={Nants_telescope}>2147483648 not supported")
    if np.any(np.asarray(baseline) < 0):
        raise ValueError("negative baseline numbers are not supported")
    if np.any(np.asarray(baseline) > 4611686018498691072):
        raise ValueError("baseline numbers > 4611686018498691072 are not supported")

    return_array = isinstance(baseline, (np.ndarray, list, tuple))
    ant1, ant2 = _bls.baseline_to_antnums(
        np.ascontiguousarray(baseline, dtype=np.uint64)
    )
    if return_array:
        return ant1.astype(int), ant2.astype(int)
    else:
        return int(ant1.item(0)), int(ant2.item(0))


def antnums_to_baseline(
    ant1,
    ant2,
    *,
    Nants_telescope,  # noqa: N803
    attempt256=False,
    use_miriad_convention=False,
):
    """
    Get the baseline number corresponding to two given antenna numbers.

    Parameters
    ----------
    ant1 : int or array_like of int
        first antenna number
    ant2 : int or array_like of int
        second antenna number
    Nants_telescope : int
        number of antennas
    attempt256 : bool
        Option to try to use the older 256 standard used in
        many uvfits files. If there are antenna numbers >= 256, the 2048
        standard will be used unless there are antenna numbers >= 2048
        or Nants_telescope > 2048. In that case, the 2147483648 standard
        will be used. Default is False.
    use_miriad_convention : bool
        Option to use the MIRIAD convention where BASELINE id is
        `bl = 256 * ant1 + ant2` if `ant2 < 256`, otherwise
        `bl = 2048 * ant1 + ant2 + 2**16`.
        Note antennas should be 1-indexed (start at 1, not 0)

    Returns
    -------
    int or array of int
        baseline number corresponding to the two antenna numbers.

    """
    if Nants_telescope is not None and Nants_telescope > 2147483648:
        raise ValueError(
            "cannot convert ant1, ant2 to a baseline index "
            f"with Nants={Nants_telescope}>2147483648."
        )
    if np.any(np.concatenate((np.unique(ant1), np.unique(ant2))) >= 2147483648):
        raise ValueError(
            "cannot convert ant1, ant2 to a baseline index "
            "with antenna numbers greater than 2147483647."
        )
    if np.any(np.concatenate((np.unique(ant1), np.unique(ant2))) < 0):
        raise ValueError(
            "cannot convert ant1, ant2 to a baseline index "
            "with antenna numbers less than zero."
        )

    nants_less2048 = True
    if Nants_telescope is not None and Nants_telescope > 2048:
        nants_less2048 = False

    return_array = isinstance(ant1, (np.ndarray, list, tuple))
    baseline = _bls.antnums_to_baseline(
        np.ascontiguousarray(ant1, dtype=np.uint64),
        np.ascontiguousarray(ant2, dtype=np.uint64),
        attempt256=attempt256,
        nants_less2048=nants_less2048,
        use_miriad_convention=use_miriad_convention,
    )
    if return_array:
        return baseline
    else:
        return baseline.item(0)


def baseline_index_flip(baseline, *, Nants_telescope):  # noqa: N803
    """Change baseline number to reverse antenna order."""
    ant1, ant2 = baseline_to_antnums(baseline, Nants_telescope=Nants_telescope)
    return antnums_to_baseline(ant2, ant1, Nants_telescope=Nants_telescope)


def parse_ants(uv, ant_str, *, print_toggle=False, x_orientation=None):
    """
    Get antpair and polarization from parsing an aipy-style ant string.

    Used to support the select function. Generates two lists of antenna pair
    tuples and polarization indices based on parsing of the string ant_str.
    If no valid polarizations (pseudo-Stokes params, or combinations of [lr]
    or [xy]) or antenna numbers are found in ant_str, ant_pairs_nums and
    polarizations are returned as None.

    Parameters
    ----------
    uv : UVBase Object
        A UVBased object that supports the following functions and parameters:
        - get_ants
        - get_antpairs
        - get_pols
        These are used to construct the baseline ant_pair_nums
        and polarizations returned.
    ant_str : str
        String containing antenna information to parse. Can be 'all',
        'auto', 'cross', or combinations of antenna numbers and polarization
        indicators 'l' and 'r' or 'x' and 'y'.  Minus signs can also be used
        in front of an antenna number or baseline to exclude it from being
        output in ant_pairs_nums. If ant_str has a minus sign as the first
        character, 'all,' will be appended to the beginning of the string.
        See the tutorial for examples of valid strings and their behavior.
    print_toggle : bool
        Boolean for printing parsed baselines for a visual user check.
    x_orientation : str, optional
        Orientation of the physical dipole corresponding to what is
        labelled as the x polarization ("east" or "north") to allow for
        converting from E/N strings. If input uv object has an `x_orientation`
        parameter and the input to this function is `None`, the value from the
        object will be used. Any input given to this function will override the
        value on the uv object. See corresonding parameter on UVData
        for more details.

    Returns
    -------
    ant_pairs_nums : list of tuples of int or None
        List of tuples containing the parsed pairs of antenna numbers, or
        None if ant_str is 'all' or a pseudo-Stokes polarizations.
    polarizations : list of int or None
        List of desired polarizations or None if ant_str does not contain a
        polarization specification.

    """
    required_attrs = ["get_ants", "get_antpairs", "get_pols"]
    if not all(hasattr(uv, attr) for attr in required_attrs):
        raise ValueError(
            "UVBased objects must have all the following attributes in order "
            f"to call 'parse_ants': {required_attrs}."
        )

    if x_orientation is None and (
        hasattr(uv.telescope, "x_orientation")
        and uv.telescope.x_orientation is not None
    ):
        x_orientation = uv.telescope.x_orientation

    ant_re = r"(\(((-?\d+[lrxy]?,?)+)\)|-?\d+[lrxy]?)"
    bl_re = "(^(%s_%s|%s),?)" % (ant_re, ant_re, ant_re)
    str_pos = 0
    ant_pairs_nums = []
    polarizations = []
    ants_data = uv.get_ants()
    ant_pairs_data = uv.get_antpairs()
    pols_data = uv.get_pols()
    warned_ants = []
    warned_pols = []

    if ant_str.startswith("-"):
        ant_str = "all," + ant_str

    while str_pos < len(ant_str):
        m = re.search(bl_re, ant_str[str_pos:])
        if m is None:
            if ant_str[str_pos:].upper().startswith("ALL"):
                if len(ant_str[str_pos:].split(",")) > 1:
                    ant_pairs_nums = uv.get_antpairs()
            elif ant_str[str_pos:].upper().startswith("AUTO"):
                for pair in ant_pairs_data:
                    if pair[0] == pair[1] and pair not in ant_pairs_nums:
                        ant_pairs_nums.append(pair)
            elif ant_str[str_pos:].upper().startswith("CROSS"):
                for pair in ant_pairs_data:
                    if not (pair[0] == pair[1] or pair in ant_pairs_nums):
                        ant_pairs_nums.append(pair)
            elif ant_str[str_pos:].upper().startswith("PI"):
                polarizations.append(polstr2num("pI"))
            elif ant_str[str_pos:].upper().startswith("PQ"):
                polarizations.append(polstr2num("pQ"))
            elif ant_str[str_pos:].upper().startswith("PU"):
                polarizations.append(polstr2num("pU"))
            elif ant_str[str_pos:].upper().startswith("PV"):
                polarizations.append(polstr2num("pV"))
            else:
                raise ValueError(f"Unparsable argument {ant_str}")

            comma_cnt = ant_str[str_pos:].find(",")
            if comma_cnt >= 0:
                str_pos += comma_cnt + 1
            else:
                str_pos = len(ant_str)
        else:
            m = m.groups()
            str_pos += len(m[0])
            if m[2] is None:
                ant_i_list = [m[8]]
                ant_j_list = list(uv.get_ants())
            else:
                if m[3] is None:
                    ant_i_list = [m[2]]
                else:
                    ant_i_list = m[3].split(",")

                if m[6] is None:
                    ant_j_list = [m[5]]
                else:
                    ant_j_list = m[6].split(",")

            for ant_i in ant_i_list:
                include_i = True
                if isinstance(ant_i, str) and ant_i.startswith("-"):
                    ant_i = ant_i[1:]  # nibble the - off the string
                    include_i = False

                for ant_j in ant_j_list:
                    include_j = True
                    if isinstance(ant_j, str) and ant_j.startswith("-"):
                        ant_j = ant_j[1:]
                        include_j = False

                    pols = None
                    ant_i, ant_j = str(ant_i), str(ant_j)
                    if not ant_i.isdigit():
                        ai = re.search(r"(\d+)([x,y,l,r])", ant_i).groups()

                    if not ant_j.isdigit():
                        aj = re.search(r"(\d+)([x,y,l,r])", ant_j).groups()

                    if ant_i.isdigit() and ant_j.isdigit():
                        ai = [ant_i, ""]
                        aj = [ant_j, ""]
                    elif ant_i.isdigit() and not ant_j.isdigit():
                        if "x" in ant_j or "y" in ant_j:
                            pols = ["x" + aj[1], "y" + aj[1]]
                        else:
                            pols = ["l" + aj[1], "r" + aj[1]]
                        ai = [ant_i, ""]
                    elif not ant_i.isdigit() and ant_j.isdigit():
                        if "x" in ant_i or "y" in ant_i:
                            pols = [ai[1] + "x", ai[1] + "y"]
                        else:
                            pols = [ai[1] + "l", ai[1] + "r"]
                        aj = [ant_j, ""]
                    elif not ant_i.isdigit() and not ant_j.isdigit():
                        pols = [ai[1] + aj[1]]

                    ant_tuple = (abs(int(ai[0])), abs(int(aj[0])))

                    # Order tuple according to order in object
                    if ant_tuple in ant_pairs_data:
                        pass
                    elif ant_tuple[::-1] in ant_pairs_data:
                        ant_tuple = ant_tuple[::-1]
                    else:
                        if not (
                            ant_tuple[0] in ants_data or ant_tuple[0] in warned_ants
                        ):
                            warned_ants.append(ant_tuple[0])
                        if not (
                            ant_tuple[1] in ants_data or ant_tuple[1] in warned_ants
                        ):
                            warned_ants.append(ant_tuple[1])
                        if pols is not None:
                            for pol in pols:
                                if not (pol.lower() in pols_data or pol in warned_pols):
                                    warned_pols.append(pol)
                        continue

                    if include_i and include_j:
                        if ant_tuple not in ant_pairs_nums:
                            ant_pairs_nums.append(ant_tuple)
                        if pols is not None:
                            for pol in pols:
                                if (
                                    pol.lower() in pols_data
                                    and polstr2num(pol, x_orientation=x_orientation)
                                    not in polarizations
                                ):
                                    polarizations.append(
                                        polstr2num(pol, x_orientation=x_orientation)
                                    )
                                elif not (
                                    pol.lower() in pols_data or pol in warned_pols
                                ):
                                    warned_pols.append(pol)
                    else:
                        if pols is not None:
                            for pol in pols:
                                if pol.lower() in pols_data:
                                    if uv.Npols == 1 and [pol.lower()] == pols_data:
                                        ant_pairs_nums.remove(ant_tuple)
                                    if (
                                        polstr2num(pol, x_orientation=x_orientation)
                                        in polarizations
                                    ):
                                        polarizations.remove(
                                            polstr2num(pol, x_orientation=x_orientation)
                                        )
                                elif not (
                                    pol.lower() in pols_data or pol in warned_pols
                                ):
                                    warned_pols.append(pol)
                        elif ant_tuple in ant_pairs_nums:
                            ant_pairs_nums.remove(ant_tuple)

    if ant_str.upper() == "ALL":
        ant_pairs_nums = None
    elif len(ant_pairs_nums) == 0:
        if not ant_str.upper() in ["AUTO", "CROSS"]:
            ant_pairs_nums = None

    if len(polarizations) == 0:
        polarizations = None
    else:
        polarizations.sort(reverse=True)

    if print_toggle:
        print("\nParsed antenna pairs:")
        if ant_pairs_nums is not None:
            for pair in ant_pairs_nums:
                print(pair)

        print("\nParsed polarizations:")
        if polarizations is not None:
            for pol in polarizations:
                print(polnum2str(pol, x_orientation=x_orientation))

    if len(warned_ants) > 0:
        warnings.warn(
            "Warning: Antenna number {a} passed, but not present "
            "in the ant_1_array or ant_2_array".format(
                a=(",").join(map(str, warned_ants))
            )
        )

    if len(warned_pols) > 0:
        warnings.warn(
            "Warning: Polarization {p} is not present in the polarization_array".format(
                p=(",").join(warned_pols).upper()
            )
        )

    return ant_pairs_nums, polarizations
