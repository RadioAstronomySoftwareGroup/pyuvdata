# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""pytest fixtures for utils tests."""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import pyuvdata.utils.phasing as phs_utils
from pyuvdata import utils

from .test_coordinates import frame_selenoid, selenoids


@pytest.fixture(scope="session")
def astrometry_args():
    default_args = {
        "time_array": 2456789.0 + np.array([0.0, 1.25, 10.5, 100.75]),
        "icrs_ra": 2.468,
        "icrs_dec": 1.234,
        "epoch": 2000.0,
        "telescope_loc": (0.123, -0.456, 4321.0),
        "telescope_frame": "itrs",
        "pm_ra": 12.3,
        "pm_dec": 45.6,
        "vrad": 31.4,
        "dist": 73.31,
        "library": "erfa",
    }
    default_args["lst_array"] = utils.get_lst_for_time(
        jd_array=default_args["time_array"],
        latitude=default_args["telescope_loc"][0] * (180.0 / np.pi),
        longitude=default_args["telescope_loc"][1] * (180.0 / np.pi),
        altitude=default_args["telescope_loc"][2],
        frame="itrs",
    )

    default_args["drift_coord"] = SkyCoord(
        default_args["lst_array"],
        [default_args["telescope_loc"][0]] * len(default_args["lst_array"]),
        unit="rad",
    )

    if len(frame_selenoid) > 1:
        default_args["moon_telescope_loc"] = (
            0.6875 * np.pi / 180.0,
            24.433 * np.pi / 180.0,
            0.3,
        )
        default_args["moon_lst_array"] = {}
        default_args["moon_drift_coord"] = {}
        for selenoid in selenoids:
            default_args["moon_lst_array"][selenoid] = utils.get_lst_for_time(
                jd_array=default_args["time_array"],
                latitude=default_args["moon_telescope_loc"][0] * (180.0 / np.pi),
                longitude=default_args["moon_telescope_loc"][1] * (180.0 / np.pi),
                altitude=default_args["moon_telescope_loc"][2],
                frame="mcmf",
                ellipsoid=selenoid,
            )
            default_args["moon_drift_coord"][selenoid] = SkyCoord(
                default_args["moon_lst_array"][selenoid],
                [default_args["moon_telescope_loc"][0]]
                * len(default_args["moon_lst_array"][selenoid]),
                unit="rad",
            )

    default_args["icrs_coord"] = SkyCoord(
        default_args["icrs_ra"], default_args["icrs_dec"], unit="rad"
    )

    default_args["fk5_ra"], default_args["fk5_dec"] = (
        phs_utils.transform_sidereal_coords(
            longitude=default_args["icrs_ra"],
            latitude=default_args["icrs_dec"],
            in_coord_frame="icrs",
            out_coord_frame="fk5",
            in_coord_epoch="J2000.0",
            out_coord_epoch="J2000.0",
        )
    )

    # These are values calculated w/o the optional arguments, e.g. pm, vrad, dist
    default_args["app_ra"], default_args["app_dec"] = phs_utils.transform_icrs_to_app(
        time_array=default_args["time_array"],
        ra=default_args["icrs_ra"],
        dec=default_args["icrs_dec"],
        telescope_loc=default_args["telescope_loc"],
    )

    default_args["app_coord"] = SkyCoord(
        default_args["app_ra"], default_args["app_dec"], unit="rad"
    )

    if len(frame_selenoid) > 1:
        default_args["moon_app_ra"] = {}
        default_args["moon_app_dec"] = {}
        default_args["moon_app_coord"] = {}
        for selenoid in selenoids:
            (
                default_args["moon_app_ra"][selenoid],
                default_args["moon_app_dec"][selenoid],
            ) = phs_utils.transform_icrs_to_app(
                time_array=default_args["time_array"],
                ra=default_args["icrs_ra"],
                dec=default_args["icrs_dec"],
                telescope_loc=default_args["moon_telescope_loc"],
                telescope_frame="mcmf",
                ellipsoid=selenoid,
            )

            default_args["moon_app_coord"][selenoid] = SkyCoord(
                default_args["moon_app_ra"][selenoid],
                default_args["moon_app_dec"][selenoid],
                unit="rad",
            )

    yield default_args
