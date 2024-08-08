# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Read antenna position files."""

import numpy as np


def read_antpos_csv(antenna_positions_file):
    """
    Interpret an antenna positions file.

    Parameters
    ----------
    antenna_positions_file : str
        Name of the antenna_positions_file, which is assumed to be in DATA_PATH.
        Should be a csv file with the following columns:

            - "name": antenna names
            - "number": antenna numbers
            - "x": x ECEF coordinate relative to the telescope location.
            - "y": y ECEF coordinate relative to the telescope location.
            - "z": z ECEF coordinate relative to the telescope location.

    Returns
    -------
    antenna_names : array of str
        Antenna names.
    antenna_names : array of int
        Antenna numbers.
    antenna_positions : array of float
        Antenna positions in ECEF relative to the telescope location.

    """
    columns = ["name", "number", "x", "y", "z"]
    formats = ["U10", "i8", np.longdouble, np.longdouble, np.longdouble]

    dt = np.rec.format_parser(formats, columns, [])
    ant_array = np.genfromtxt(
        antenna_positions_file,
        delimiter=",",
        autostrip=True,
        skip_header=1,
        dtype=dt.dtype,
    )
    antenna_names = ant_array["name"]
    antenna_numbers = ant_array["number"]
    antenna_positions = np.stack((ant_array["x"], ant_array["y"], ant_array["z"])).T

    return antenna_names, antenna_numbers, antenna_positions.astype("float")
