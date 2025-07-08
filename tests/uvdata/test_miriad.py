# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Miriad object.

Note that because of the way that file handling is done in the C++ API, a miriad
file is not closed until the destructor function (tp_dealloc) is called.

The following lines were made before a major rewrite of the miriad interface.
As of April 2020, they should no longer necessarily be true,
but are preserved here in case segfaults arise again.

Due to implementation details of CPython, it is sometimes not enough to `del` the
object--a manual garbage collection may be required. When adding new tests,
proper cleanup consists of: (1) deleting any Miriad objects or UVData objects,
(2) performing garbage collection (with `gc.collect()`), and (3) removing the
directory corresponding to the miriad file. Each test should do this, to avoid
open file handles interfering with other tests. The exception to this is if a
test uses the `uv_in_paper` or `uv_in_uvfits` fixture, as these handle cleanup
on their own.
"""

import os
import shutil

import numpy as np
import pytest
from astropy import constants as const, units
from astropy.coordinates import Angle
from astropy.time import Time, TimeDelta

from pyuvdata import UVData, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.telescopes import known_telescope_location
from pyuvdata.testing import check_warnings
from pyuvdata.uvdata.miriad import Miriad

aipy_extracts = pytest.importorskip("pyuvdata.uvdata.aipy_extracts")

# always ignore the Altitude not present warning
pytestmark = pytest.mark.filterwarnings("ignore:Altitude is not present in Miriad")

paper_miriad_file = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")

# This is a dictionary of warning strings to aid warning checks
warn_dict = {
    "default_vals": (
        "writing default values for restfreq, vsource, veldop, jyperk, and systemp"
    ),
    "uvw_mismatch": (
        "The uvw_array does not match the expected values given the antenna positions"
    ),
    "driftscan": "This object has a driftscan phase center. Miriad does not really ",
    "long_key": "key test_long_key in extra_keywords is longer than 8 characters.",
    "ant_diameters": (
        "Antenna diameters are not uniform, but miriad only supports a single diameter."
    ),
    "time_mismatch": (
        "Some visibility times did not match ephem times so the ra and dec "
        "values for those visibilities were interpolated or set to "
        "the closest time if they would have required extrapolation."
    ),
    "altitude_missing_paper": (
        "Altitude is not present in Miriad file, using "
        "known location altitude value for PAPER and lat/lon from file."
    ),
    "altitude_missing_lat_long": (
        "Altitude is not present in file and latitude and longitude values do not match"
    ),
    "altitude_missing_long": (
        "Altitude is not present in file and longitude value does not match"
    ),
    "altitude_missing_lat": (
        "Altitude is not present in file and latitude value does not match"
    ),
    "altitude_missing_foo": (
        "Altitude is not present in Miriad file, and "
        "telescope foo is not in known_telescopes. "
        "Telescope location will be set using antenna positions."
    ),
    "no_telescope_loc": (
        "Telescope location is not set, but antenna positions are "
        "present. Mean antenna latitude and longitude values match file "
        "values, so telescope_position will be set using the mean of the "
        "antenna altitudes"
    ),
    "unclear_projection": (
        "It is not clear from the file if the data are projected or not."
    ),
    "telescope_at_sealevel": (
        "Telescope location is set at sealevel at the file lat/lon "
        "coordinates. Antenna positions are present, but the mean antenna "
        "position does not give a telescope location on the surface of the "
        "earth. Antenna positions do not appear to be on the surface of the "
        "earth and will be treated as relative."
    ),
    "telescope_at_sealevel_lat": (
        "Telescope location is set at sealevel at the file lat/lon coordinates. "
        "Antenna positions are present, but the mean antenna latitude value does "
        "not match file values so they are not used for altitude."
    ),
    "telescope_at_sealevel_lat_long": (
        "Telescope location is set at sealevel at the file lat/lon coordinates. Antenna"
        " positions are present, but the mean antenna latitude and longitude values do"
        " not match file values so they are not used for altitude."
    ),
    "telescope_at_sealevel_foo": (
        "Altitude is not present in Miriad file, and telescope foo is not in"
        " known_telescopes."
    ),
    "projection_false_offset": (
        "projected is False, but RA, Dec is off from lst, latitude by more than 1.0 deg"
    ),
}


def _write_miriad(uv: UVData, filename: str, warn: str = None, **kwargs):
    """Write miriad file, capturing warnings for pytest

    Parameters:
    -----------
    uv: UVData
        uvdata object to call write_miriad on
    filename: str
        Name of file to write to
    warn: str or None
        warnings to catch. Defaults to catch 'writing default values ...'
    """
    if warn is None:
        warn = [warn_dict["default_vals"], warn_dict["uvw_mismatch"]]
    with check_warnings(UserWarning, match=warn):
        uv.write_miriad(filename, **kwargs)


@pytest.fixture(scope="function")
def uv_in_paper(paper_miriad, tmp_path):
    uv_in = paper_miriad
    write_file = os.path.join(tmp_path, "outtest_miriad.uv")
    _write_miriad(uv_in, write_file, clobber=True)
    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.fixture(scope="function")
def uv_in_uvfits(paper_miriad, tmp_path):
    uv_in = paper_miriad
    write_file = os.path.join(tmp_path, "outtest_miriad.uvfits")

    uv_out = UVData()

    yield uv_in, uv_out, write_file

    # cleanup
    del uv_in, uv_out


@pytest.mark.filterwarnings("ignore:Telescope ATCA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_atca(tmp_path):
    uv_in = UVData()
    uv_out = UVData()
    atca_file = os.path.join(DATA_PATH, "atca_miriad/")
    testfile = os.path.join(tmp_path, "outtest_atca_miriad.uv")
    with check_warnings(
        UserWarning,
        [
            (
                "Altitude is not present in Miriad file, and "
                "telescope ATCA is not in known_telescopes. "
            ),
            "Altitude is not present",
            warn_dict["telescope_at_sealevel"],
            warn_dict["uvw_mismatch"],
            warn_dict["unclear_projection"],
        ],
    ):
        uv_in.read(atca_file)

    _write_miriad(uv_in, testfile, clobber=True)
    uv_out.read(testfile)

    # make sure filename is what we expect
    assert uv_in.filename[0] == "atca_miriad"
    assert uv_out.filename[0] == "outtest_atca_miriad.uv"
    assert len(uv_in.filename) == len(uv_out.filename)
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_nrao_write_miriad_read_miriad(casa_uvfits, tmp_path):
    """Test reading in a CASA tutorial uvfits file, writing and reading as miriad"""
    uvfits_uv = casa_uvfits
    miriad_uv = UVData()
    writefile = os.path.join(tmp_path, "outtest_miriad.uv")
    _write_miriad(uvfits_uv, writefile, clobber=True)
    miriad_uv.read(writefile)

    # check that setting projected also works
    miriad_uv2 = UVData.from_file(writefile, projected=True)
    assert miriad_uv2 == miriad_uv

    # check that setting projected also works
    with check_warnings(UserWarning, match=warn_dict["uvw_mismatch"]):
        miriad_uv2 = UVData.from_file(writefile, projected=True)
    assert miriad_uv2 == miriad_uv

    # make sure filename is what we expect
    assert uvfits_uv.filename[0] == "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
    assert miriad_uv.filename[0] == "outtest_miriad.uv"
    assert len(uvfits_uv.filename) == len(miriad_uv.filename)
    uvfits_uv.filename = miriad_uv.filename

    miriad_uv._consolidate_phase_center_catalogs(
        reference_catalog=uvfits_uv.phase_center_catalog
    )
    assert uvfits_uv == miriad_uv


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_carma(tmp_path):
    uv_in = UVData()
    uv_out = UVData()
    carma_file = os.path.join(DATA_PATH, "carma_miriad")
    testfile = os.path.join(tmp_path, "outtest_carma_miriad.uv")

    with check_warnings(
        UserWarning,
        [
            (
                "Altitude is not present in Miriad file, "
                "using known location values for SZA."
            ),
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_in.read(carma_file)

    # Extra keywords cannnot handle lists, dicts, or arrays, so drop them from the
    # dataset, so that the writer doesn't run into issues.
    # TODO: Capture these extra keywords
    for item in list(uv_in.extra_keywords.keys()):
        if isinstance(uv_in.extra_keywords[item], dict | list | np.ndarray):
            uv_in.extra_keywords.pop(item)

    _write_miriad(uv_in, testfile, clobber=True)
    uv_out.read(testfile)

    # make sure filename is what we expect
    assert uv_in.filename == ["carma_miriad"]
    assert uv_out.filename == ["outtest_carma_miriad.uv"]
    uv_in.filename = uv_out.filename

    assert uv_in == uv_out

    # We should get the same result if we feed in these parameters, since the original
    # file had the LST calculated on read, and its def a phased dataset
    uv_out.read(testfile, calc_lst=False)

    assert uv_in == uv_out

    _write_miriad(uv_in, testfile, clobber=True, calc_lst=True)
    uv_out.read(testfile, calc_lst=False)

    # Finally, make sure that if we calc LSTs on write, but not read, that we still
    # get the same answer.
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:pamatten in extra_keywords is a list, array")
@pytest.mark.filterwarnings("ignore:psys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:psysattn in extra_keywords is a list, array or")
@pytest.mark.filterwarnings("ignore:ambpsys in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:bfmask in extra_keywords is a list, array or dict")
@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:Writing in the MS file that the units of the data")
def test_read_carma_miriad_write_ms(tmp_path):
    """
    Check a roundtrip between CARMA-MIRIAD and MS formats.

    Note that this check does a few different operations -- these are consolidated into
    one test because several of them rely upon manipulating the same file object.
    """
    pytest.importorskip("casacore")
    from casacore import tables

    uv_in = UVData()
    uv_out = UVData()
    carma_file = os.path.join(DATA_PATH, "carma_miriad")
    testfile = os.path.join(tmp_path, "outtest_carma_miriad.ms")

    with check_warnings(
        UserWarning,
        [
            (
                "Altitude is not present in Miriad file, "
                "using known location values for SZA."
            ),
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_in.read(carma_file)

    # MIRIAD is missing these in the file, so we'll fill it in here.
    uv_in.telescope.antenna_diameters = np.zeros(uv_in.telescope.Nants)
    uv_in.telescope.antenna_diameters[:6] = 10.0
    uv_in.telescope.antenna_diameters[15:] = 3.5

    # We need to recalculate app coords here for one source ("NOISE"), which was
    # not actually correctly calculated in the online CARMA system (long story). Since
    # the MS format requires recalculating apparent coords after read in, we'll
    # calculate them here just to verify that everything matches.
    uv_in._set_app_coords_helper()
    with check_warnings(
        UserWarning,
        [
            warn_dict["uvw_mismatch"],
            "Writing in the MS file that the units of the data are",
        ],
    ):
        uv_in.write_ms(testfile, clobber=True)

    uv_out.read(testfile)

    # Make sure the MS extra keywords are as expected
    assert uv_out.extra_keywords["DATA_COL"] == "DATA"
    assert uv_out.extra_keywords["observer"] == "SZA"
    uv_in.extra_keywords["DATA_COL"] = "DATA"
    uv_in.extra_keywords["observer"] = "SZA"

    # make sure filename is what we expect
    assert uv_in.filename == ["carma_miriad"]
    assert uv_out.filename == ["outtest_carma_miriad.ms"]
    uv_in.filename = uv_out.filename

    # Do a quick check on the history to verify they're different.
    assert uv_in.history != uv_out.history
    uv_in.history = uv_out.history

    # Final equality check
    assert uv_in.__eq__(uv_out, allowed_failures=["filename"])

    # Manipulate the table so that all fields have the same name (as is permitted
    # for MS files), and wipe out the source_ID information
    tb_field = tables.table(os.path.join(testfile, "FIELD"), ack=False, readonly=False)
    tb_field.putcol("NAME", ["TEST"] * 3)
    tb_field.removecols("SOURCE_ID")
    tb_field.close()

    # Check and see that the naming convention lines up as expected -- only the internal
    # catalog entries (specifically the names/keys and catalog IDs) should have changed.
    uv_out.read(testfile)

    uv_out.phase_center_catalog = uv_in.phase_center_catalog
    uv_out._set_app_coords_helper()

    # Final equality check
    assert uv_in.__eq__(uv_out, allowed_failures=["filename"])

    # Check to make sure we raise an error if overwritting a file w/o clobber enabled
    with pytest.raises(OSError, match="File exists; skipping"):
        uv_in.write_ms(testfile, clobber=False)


@pytest.mark.filterwarnings("ignore:LST values stored in this file are not ")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_miriad_write_uvfits(uv_in_uvfits):
    """
    Miriad to uvfits loopback test.

    Read in Miriad files, write out as uvfits, read back in and check for
    object equality.
    """
    miriad_uv, uvfits_uv, testfile = uv_in_uvfits

    miriad_uv.write_uvfits(testfile, force_phase=True)
    uvfits_uv.read_uvfits(testfile)

    # make sure filename is what we expect
    assert miriad_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uvfits_uv.filename == ["outtest_miriad.uvfits"]
    miriad_uv.filename = uvfits_uv.filename

    for item in ["dut1", "earth_omega", "gst0", "rdate", "timesys"]:
        # Check to make sure that the UVFITS-specific paramters are set on the
        # UVFITS-based obj, and not on our original object. Then set it to None for the
        # UVFITS-based obj.
        assert getattr(miriad_uv, item) is None
        assert getattr(uvfits_uv, item) is not None
        setattr(uvfits_uv, item, None)

    uvfits_uv._consolidate_phase_center_catalogs(
        reference_catalog=miriad_uv.phase_center_catalog, ignore_name=True
    )
    assert miriad_uv == uvfits_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_read_warning_lat_lon_corrected():
    miriad_uv = UVData()
    # check warning when correct_lat_lon is set to False
    with check_warnings(
        UserWarning,
        [
            (
                "Altitude is not present in Miriad file, using known location "
                "altitude value for PAPER and lat/lon from file."
            ),
            warn_dict["uvw_mismatch"],
        ],
    ):
        miriad_uv.read(paper_miriad_file, correct_lat_lon=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    ["override_dict", "correct_lat_lon"],
    [
        [
            {"latitud": known_telescope_location("paper").lat.rad + 10 * np.pi / 180.0},
            True,
        ],
        [
            {"longitu": known_telescope_location("paper").lon.rad + 10 * np.pi / 180.0},
            True,
        ],
        [
            {
                "latitud": known_telescope_location("paper").lat.rad
                + 10 * np.pi / 180.0,
                "longitu": known_telescope_location("paper").lon.rad
                + 10 * np.pi / 180.0,
            },
            False,
        ],
        [{"telescop": "foo"}, True],
        [{}, False],
    ],
)
def test_wronglatlon(tmp_path, override_dict, correct_lat_lon):
    """
    Check for appropriate warnings with incorrect lat/lon values or missing telescope
    """
    testfile = os.path.join(tmp_path, "paper_tweaked_loc.uv")

    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)

    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file, make changes
    aipy_uv2.init_from_uv(aipy_uv, override=override_dict, exclude=["altitude"])
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    warn_list = [warn_dict["uvw_mismatch"]]

    if (
        "latitud" in override_dict or "longitu" in override_dict
    ) and not correct_lat_lon:
        warn_list.append(warn_dict["projection_false_offset"])

    if "telescop" in override_dict:
        warn_list.extend(
            [
                warn_dict["altitude_missing_foo"],
                warn_dict["altitude_missing_foo"],
                warn_dict["telescope_at_sealevel"],
            ]
        )
    elif "latitud" in override_dict and "longitu" in override_dict:
        warn_list.append(warn_dict["altitude_missing_lat_long"])
    elif "latitud" in override_dict:
        warn_list.append(warn_dict["altitude_missing_lat"])
    elif "longitu" in override_dict:
        warn_list.append(warn_dict["altitude_missing_long"])
    else:
        warn_list.append(warn_dict["altitude_missing_paper"])

    with check_warnings(UserWarning, match=warn_list):
        UVData.from_file(testfile, correct_lat_lon=correct_lat_lon)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_location_handling(paper_miriad_main, tmp_path):
    uv_in = paper_miriad_main
    uv_out = UVData()

    testfile = os.path.join(tmp_path, "outtest_miriad.uv")
    aipy_uv = aipy_extracts.UV(paper_miriad_file)

    if os.path.exists(testfile):
        shutil.rmtree(testfile)

    # Test for using antenna positions to get telescope position
    # extract antenna positions and rotate them for miriad
    nants = aipy_uv["nants"]
    rel_ecef_antpos = np.zeros(
        (nants, 3), dtype=uv_in.telescope.antenna_positions.dtype
    )
    for ai, num in enumerate(uv_in.telescope.antenna_numbers):
        rel_ecef_antpos[num, :] = uv_in.telescope.antenna_positions[ai, :]

    # find zeros so antpos can be zeroed there too
    antpos_length = np.sqrt(np.sum(np.abs(rel_ecef_antpos) ** 2, axis=1))

    ecef_antpos = rel_ecef_antpos + uv_in.telescope._location.xyz()
    antpos = utils.rotECEF_from_ECEF(ecef_antpos, uv_in.telescope.location.lon.rad)

    # zero out bad locations (these are checked on read)
    antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    antpos = antpos.T.flatten() / const.c.to_value("m/ns")

    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={"telescop": "foo", "antpos": antpos})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with check_warnings(
        UserWarning,
        [
            warn_dict["altitude_missing_foo"],
            warn_dict["altitude_missing_foo"],  # raised twice
            warn_dict["no_telescope_loc"],
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_out.read(testfile)

    # test for handling no altitude, unknown telescope, no antenna positions
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={"telescop": "foo"}, exclude=["antpos"])
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()
    with check_warnings(
        UserWarning,
        match=[
            warn_dict["altitude_missing_foo"],
            warn_dict["altitude_missing_foo"],  # raised twice
            "Antenna positions are not present in the file.",
            "Antenna positions are not present in the file.",  # raised twice
            "Telescope location is set at sealevel at the file lat/lon "
            "coordinates because neither altitude nor antenna positions are "
            "present in the file",
        ],
    ):
        uv_out.read(testfile, run_check=False)

    # Test for handling when antenna positions have a different mean latitude
    # than the file latitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude
    new_lat = aipy_uv["latitud"] * 1.5
    aipy_uv2.init_from_uv(
        aipy_uv, override={"telescop": "foo", "antpos": antpos, "latitud": new_lat}
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with check_warnings(
        UserWarning,
        [
            warn_dict["altitude_missing_foo"],
            warn_dict["altitude_missing_foo"],
            warn_dict["telescope_at_sealevel_foo"],
            warn_dict["projection_false_offset"],
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions have a different mean longitude
    # than the file longitude
    # this is harder because of the rotation that's done on the antenna positions
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file longitude
    new_lon = aipy_uv["longitu"] + np.pi
    aipy_uv2.init_from_uv(
        aipy_uv, override={"telescop": "foo", "antpos": antpos, "longitu": new_lon}
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with check_warnings(
        UserWarning,
        [
            warn_dict["altitude_missing_foo"],
            warn_dict["altitude_missing_foo"],
            (
                "Telescope location is set at sealevel at the "
                "file lat/lon coordinates. Antenna positions "
                "are present, but the mean antenna longitude "
                "value does not match"
            ),
            warn_dict["projection_false_offset"],
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions have a different mean longitude &
    # latitude than the file longitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude and longitude
    aipy_uv2.init_from_uv(
        aipy_uv,
        override={
            "telescop": "foo",
            "antpos": antpos,
            "latitud": new_lat,
            "longitu": new_lon,
        },
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with check_warnings(
        UserWarning,
        [
            warn_dict["altitude_missing_foo"],
            warn_dict["altitude_missing_foo"],
            warn_dict["telescope_at_sealevel_lat_long"],
            warn_dict["projection_false_offset"],
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_out.read(testfile)

    # Test for handling when antenna positions are far enough apart to make the
    # mean position inside the earth

    good_antpos = np.where(antpos_length > 0)[0]
    rot_ants = good_antpos[: len(good_antpos) // 2]
    rot_antpos = utils.rotECEF_from_ECEF(
        ecef_antpos[rot_ants, :], uv_in.telescope.location.lon.rad + np.pi
    )
    modified_antpos = utils.rotECEF_from_ECEF(
        ecef_antpos, uv_in.telescope.location.lon.rad
    )
    modified_antpos[rot_ants, :] = rot_antpos

    # zero out bad locations (these are checked on read)
    modified_antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    modified_antpos = modified_antpos.T.flatten() / const.c.to_value("m/ns")

    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(paper_miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use modified absolute antenna positions
    aipy_uv2.init_from_uv(
        aipy_uv, override={"telescop": "foo", "antpos": modified_antpos}
    )
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with check_warnings(
        UserWarning,
        [
            warn_dict["altitude_missing_foo"],
            warn_dict["altitude_missing_foo"],
            (
                "Telescope location is set at sealevel at the "
                "file lat/lon coordinates. Antenna positions "
                "are present, but the mean antenna position "
                "does not give a telescope location on the "
                "surface of the earth."
            ),
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_out.read(testfile)

    # cleanup
    aipy_uv.close()


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_singletimeselect_unprojected(tmp_path):
    """
    Check behavior with writing & reading after selecting a single time from
    an unprojected file.
    """
    uv_in = UVData.from_file(paper_miriad_file)

    uv_in_copy = uv_in.copy()
    uv_in.select(times=uv_in.time_array[0])
    testfile = os.path.join(tmp_path, "single_time_unprojected")
    _write_miriad(uv_in, testfile, clobber=True)
    uv_out = UVData.from_file(testfile)

    # remove phsframe to test detecting projection from single time properly
    testfile2 = os.path.join(tmp_path, "single_time_unprojected_noframe")
    aipy_uv = aipy_extracts.UV(testfile)
    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile2, status="new")
    # initialize headers from old file
    aipy_uv2.init_from_uv(aipy_uv, exclude=["phsframe"])
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    with check_warnings(
        UserWarning,
        match=[
            "It is not clear from the file if the data are projected or not. "
            "Since the 'epoch' variable is not present it will be labeled as "
            "unprojected. If that is incorrect you can use the 'projected' parameter "
            "on this method to set it properly.",
            "The uvw_array does not match the expected values",
        ],
    ):
        uv_out2 = UVData.from_file(testfile2)

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    uv_out2._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_out2 == uv_out
    assert uv_in == uv_out

    # check that setting projected works
    uv_out.read(testfile, projected=False)
    assert uv_in == uv_out

    # also check that setting projected works
    with check_warnings(UserWarning, match=warn_dict["uvw_mismatch"]):
        uv_out.read(testfile, projected=False)
        assert uv_in == uv_out

    # check again with more than one time but only 1 unflagged time
    time_gt0_array = np.where(uv_in_copy.time_array > uv_in_copy.time_array[0])[0]
    uv_in_copy.flag_array[time_gt0_array, :, :] = True

    # get unflagged blts
    blt_good = np.where(~np.all(uv_in_copy.flag_array, axis=(1, 2)))
    assert np.isclose(
        np.mean(np.diff(uv_in_copy.time_array[blt_good])),
        0.0,
        rtol=uv_in._time_array.tols[0],
        atol=uv_in._time_array.tols[1],
    )

    _write_miriad(uv_in_copy, testfile, clobber=True)
    uv_out.read(testfile)

    uv_out._consolidate_phase_center_catalogs(other=uv_in_copy)
    assert uv_in_copy == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:The provided name zenith is already used")
@pytest.mark.parametrize("frame", ["fk5", "fk4"])
def test_loop_multi_phase(tmp_path, paper_miriad, frame):
    uv_in = paper_miriad
    testfile = os.path.join(tmp_path, "outtest_miriad.uv")
    testfile2 = os.path.join(tmp_path, "outtest_miriad2.uv")

    mask = np.full(uv_in.Nblts, False)
    mask[: uv_in.Nblts // 2] = True
    if frame == "fk5":
        epoch = 2000
    elif frame == "fk4":
        epoch = 1950
    uv_in.phase(
        ra=0, dec=0, phase_frame=frame, select_mask=mask, cat_name="foo", epoch=epoch
    )

    _write_miriad(uv_in, testfile, clobber=True)
    uv2 = UVData.from_file(testfile)

    uv2._consolidate_phase_center_catalogs(other=uv_in)
    assert uv2 == uv_in

    # remove phsframe to test frame setting based on epoch
    aipy_uv = aipy_extracts.UV(testfile)
    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile2, status="new")
    # initialize headers from old file
    aipy_uv2.init_from_uv(aipy_uv, exclude=["phsframe"])
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    uv3 = UVData.from_file(testfile2)

    # without the "phsframe" variable, the unprojected phase center gets interpreted as
    # an ephem type phase center.
    zen_id, _ = utils.phase_center_catalog.look_in_catalog(
        uv3.phase_center_catalog, cat_name="zenith"
    )
    new_id = uv3._add_phase_center(cat_name="zenith", cat_type="unprojected")
    uv3.phase_center_id_array[np.nonzero(uv3.phase_center_id_array == zen_id)] = new_id
    uv3._clear_unused_phase_centers()
    uv3._consolidate_phase_center_catalogs(other=uv_in)

    assert uv3 == uv_in


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_multi_phase_error(tmp_path, paper_miriad):
    uv_in = paper_miriad
    testfile = os.path.join(tmp_path, "outtest_miriad.uv")

    mask = np.full(uv_in.Nblts, False)
    mask[: uv_in.Nblts // 2] = True
    uv_in.phase(
        ra=0, dec=0, phase_frame="fk5", select_mask=mask, cat_name="foo", epoch=200
    )

    _write_miriad(uv_in, testfile, clobber=True)
    with pytest.raises(
        ValueError, match="projected is False but there are multiple sources."
    ):
        UVData.from_file(testfile, projected=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_only_itrs(tmp_path, paper_miriad):
    pytest.importorskip("lunarsky")
    from lunarsky import MoonLocation

    uv_in = paper_miriad
    testfile = os.path.join(tmp_path, "outtest_miriad.uv")

    enu_antpos = uv_in.telescope.get_enu_antpos()
    latitude, longitude, altitude = uv_in.telescope.location_lat_lon_alt
    uv_in.telescope.location = MoonLocation.from_selenodetic(
        lat=latitude * units.rad, lon=longitude * units.rad, height=altitude * units.m
    )
    new_full_antpos = utils.ECEF_from_ENU(
        enu=enu_antpos, center_loc=uv_in.telescope.location
    )

    uv_in.telescope.antenna_positions = (
        new_full_antpos - uv_in.telescope._location.xyz()
    )
    uv_in.set_lsts_from_time_array()
    uv_in.check()

    with pytest.raises(
        ValueError, match="Only ITRS telescope locations are supported in Miriad files."
    ):
        _write_miriad(uv_in, testfile, clobber=True)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("cut_ephem_pts", [True, False])
@pytest.mark.parametrize("extrapolate", [True, False])
def test_miriad_ephem(tmp_path, casa_uvfits, cut_ephem_pts, extrapolate):
    pytest.importorskip("astroquery")

    from ssl import SSLError

    from requests import RequestException

    uv_in = casa_uvfits
    # Need to spread out the times to get ra/dec changes that are different at
    # miriad's precision
    unique_times = np.unique(uv_in.time_array)
    for t_ind, ut in enumerate(unique_times):
        time_mask = np.nonzero(uv_in.time_array == ut)[0]
        uv_in.time_array[time_mask] += t_ind * 0.01
    uv_in.set_lsts_from_time_array()

    testfile = os.path.join(tmp_path, "outtest_miriad.uv")

    # Handle this part with care, since we don't want the test to fail if we are unable
    # to reach the JPL-Horizons service.
    try:
        uv_in.phase(ra=0, dec=0, epoch="J2000", lookup_name="Mars", cat_name="Mars")
    except (SSLError, RequestException) as err:
        pytest.skip("SSL/Connection error w/ JPL Horizons: " + str(err))

    if extrapolate:
        # change cat_times to force extrapolation
        uv_in.phase_center_catalog[1]["cat_times"] += 0.5

    if cut_ephem_pts:
        uv_in.phase_center_catalog[1]["cat_times"] = uv_in.phase_center_catalog[1][
            "cat_times"
        ][0]
        uv_in.phase_center_catalog[1]["cat_lon"] = uv_in.phase_center_catalog[1][
            "cat_lon"
        ][0]
        uv_in.phase_center_catalog[1]["cat_lat"] = uv_in.phase_center_catalog[1][
            "cat_lat"
        ][0]
        uv_in.phase_center_catalog[1]["cat_vrad"] = uv_in.phase_center_catalog[1][
            "cat_vrad"
        ][0]
        uv_in.phase_center_catalog[1]["cat_dist"] = uv_in.phase_center_catalog[1][
            "cat_dist"
        ][0]

    _write_miriad(
        uv_in,
        testfile,
        clobber=True,
        warn=[warn_dict["default_vals"], warn_dict["time_mismatch"]],
    )
    uv2 = UVData.from_file(testfile)

    uv2._update_phase_center_id(0, new_id=1)
    uv2.phase_center_catalog[1]["info_source"] = uv_in.phase_center_catalog[1][
        "info_source"
    ]

    if cut_ephem_pts:
        # Only one ephem points results in only one ra/dec, so it is interpretted as
        # a sidereal rather than ephem phase center
        assert uv2.phase_center_catalog[1]["cat_type"] == "sidereal"
        assert (
            uv2.phase_center_catalog[1]["cat_lon"]
            == uv_in.phase_center_catalog[1]["cat_lon"]
        )
        assert (
            uv2.phase_center_catalog[1]["cat_lat"]
            == uv_in.phase_center_catalog[1]["cat_lat"]
        )

        # just replace the phase_center_catalog before testing for equality
        uv2.phase_center_catalog = uv_in.phase_center_catalog
    else:
        # the number of catalog times is changed
        # adjust phase center catalogs to make them equal
        assert uv2._phase_center_catalog != uv_in._phase_center_catalog
        uv2.phase_center_catalog[1]["cat_times"] = uv_in.phase_center_catalog[1][
            "cat_times"
        ]
        uv2.phase_center_catalog[1]["cat_lon"] = uv_in.phase_center_catalog[1][
            "cat_lon"
        ]
        uv2.phase_center_catalog[1]["cat_lat"] = uv_in.phase_center_catalog[1][
            "cat_lat"
        ]
        uv2.phase_center_catalog[1]["cat_vrad"] = uv_in.phase_center_catalog[1][
            "cat_vrad"
        ]
        uv2.phase_center_catalog[1]["cat_dist"] = uv_in.phase_center_catalog[1][
            "cat_dist"
        ]
        assert uv2._phase_center_catalog == uv_in._phase_center_catalog

    assert uv2 == uv_in


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_driftscan(tmp_path, paper_miriad):
    uv_in = paper_miriad
    testfile = os.path.join(tmp_path, "outtest_miriad.uv")

    uv2 = uv_in.copy()
    uv2.phase(
        lat=Angle("80d").rad,
        lon=0,
        phase_frame="altaz",
        cat_type="driftscan",
        cat_name="drift_alt80",
    )
    warn_list = [warn_dict["driftscan"], warn_dict["default_vals"]]
    _write_miriad(uv2, testfile, clobber=True, warn=warn_list)

    uv3 = UVData.from_file(testfile)
    assert np.all(uv3._check_for_cat_type("ephem"))

    # put it back to a driftscan
    uv3._update_phase_center_id(0, new_id=1)
    uv3.phase_center_catalog = uv2.phase_center_catalog

    assert uv2 == uv3


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_poltoind(uv_in_paper):
    miriad_uv, _, _ = uv_in_paper

    pol_arr = miriad_uv.polarization_array

    miriad = miriad_uv._convert_to_filetype("miriad")
    miriad.polarization_array = None
    with pytest.raises(ValueError) as cm:
        miriad._pol_to_ind(pol_arr[0])
    assert str(cm.value).startswith(
        "Can't index polarization -7 because polarization_array is not set"
    )

    miriad.polarization_array = [pol_arr[0], pol_arr[0]]
    with pytest.raises(ValueError) as cm:
        miriad._pol_to_ind(pol_arr[0])
    assert str(cm.value).startswith("multiple matches for pol=-7 in polarization_array")


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwd_name,kwd_value,errstr",
    (
        ["testdict", {"testkey": 23}, "Extra keyword testdict is of <class 'dict'>"],
        ["testlist", [12, 14, 90], "Extra keyword testlist is of <class 'list'>"],
        [
            "testarr",
            np.array([12, 14, 90]),
            "Extra keyword testarr is of <class 'numpy.ndarray'>",
        ],
        ["test_long_key", True, None],
        [
            "complex1",
            np.complex64(5.3 + 1.2j),
            "Extra keyword complex1 is of <class 'numpy.complex64'>",
        ],
        ["complex2", 6.9 + 4.6j, "Extra keyword complex2 is of <class 'complex'>"],
    ),
)
def test_miriad_extra_keywords_errors(uv_in_paper, kwd_name, kwd_value, errstr):
    uv_in, _, testfile = uv_in_paper

    uvw_warn_str = "The uvw_array does not match the expected values"

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords[kwd_name] = kwd_value
    with check_warnings(UserWarning, uvw_warn_str):
        uv_in.check()

    if errstr is not None:
        with pytest.raises(TypeError, match=errstr):
            _write_miriad(
                uv_in,
                testfile,
                clobber=True,
                run_check=False,
                warn=[errstr, warn_dict["default_vals"]],
            )
    else:
        _write_miriad(
            uv_in,
            testfile,
            clobber=True,
            run_check=False,
            warn=[warn_dict["default_vals"], warn_dict["long_key"]],
        )


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "kwd_names,kwd_values",
    (
        [["bool", "bool2"], [True, False]],
        [["int1", "int2"], [np.int64(5), 7]],
        [["float1", "float2"], [np.float64(5.3), 6.9]],
        [["str", "longstr"], ["hello", "this is a very long string " * 1000]],
    ),
)
def test_miriad_extra_keywords(uv_in_paper, tmp_path, kwd_names, kwd_values):
    uv_in, uv_out, testfile = uv_in_paper

    for name, value in zip(kwd_names, kwd_values, strict=True):
        uv_in.extra_keywords[name] = value
    _write_miriad(uv_in, testfile, clobber=True)
    uv_out.read(testfile)

    # make sure filename is what we expect
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_roundtrip_optional_params(uv_in_paper):
    uv_in, uv_out, testfile = uv_in_paper

    uv_in.telescope.set_feeds_from_x_orientation(
        "east", polarization_array=uv_in.polarization_array
    )
    uv_in.pol_convention = "sum"
    uv_in.vis_units = "Jy"
    uv_in.reorder_blts()

    _write_miriad(uv_in, testfile, clobber=True)
    uv_out.read(testfile)

    # make sure filename is what we expect
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out

    # test with bda as well (single entry in tuple)
    uv_in.reorder_blts("bda")

    _write_miriad(uv_in, testfile, clobber=True)
    uv_out.read(testfile)

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_breakread_miriad(uv_in_paper, tmp_path):
    """Test Miriad file checking."""
    uv_in, uv_out, testfile = uv_in_paper

    with pytest.raises(IOError) as cm:
        uv_in.read("foo", file_type="miriad")
    assert str(cm.value).startswith("foo not found")

    uv_in_copy = uv_in.copy()
    uv_in_copy.Nblts += 10
    _write_miriad(
        uv_in_copy,
        testfile,
        clobber=True,
        run_check=False,
        warn=warn_dict["default_vals"],
    )
    with check_warnings(
        UserWarning, "Nblts does not match the number of unique blts in the data"
    ):
        uv_out.read(testfile, run_check=False)

    uv_in_copy = uv_in.copy()
    uv_in_copy.Nbls += 10
    _write_miriad(
        uv_in_copy,
        testfile,
        clobber=True,
        run_check=False,
        warn=warn_dict["default_vals"],
    )
    with check_warnings(
        UserWarning, "Nbls does not match the number of unique baselines in the data"
    ):
        uv_out.read(testfile, run_check=False)

    uv_in_copy = uv_in.copy()
    uv_in_copy.Ntimes += 10
    _write_miriad(
        uv_in_copy,
        testfile,
        clobber=True,
        run_check=False,
        warn=warn_dict["default_vals"],
    )
    with check_warnings(
        UserWarning, "Ntimes does not match the number of unique times in the data"
    ):
        uv_out.read(testfile, run_check=False)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad(uv_in_paper):
    """
    PAPER file Miriad loopback test.

    Read in Miriad PAPER file, write out as new Miriad file, read back in and
    check for object equality.
    """
    uv_in, uv_out, write_file = uv_in_paper
    uv_out.read(write_file)

    # make sure filename is what we expect
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out

    # check that we can read & write phased data
    uv_in2 = uv_in.copy()
    uv_in2.phase_to_time(Time(np.mean(uv_in2.time_array), format="jd"))
    _write_miriad(uv_in2, write_file, clobber=True, warn=warn_dict["default_vals"])
    uv_out.read(write_file)

    uv_out._consolidate_phase_center_catalogs(other=uv_in2)
    assert uv_in2 == uv_out
    del uv_in2

    # check that trying to overwrite without clobber raises an error
    with pytest.raises(OSError) as cm:
        _write_miriad(uv_in, write_file, clobber=False)
    assert str(cm.value).startswith("File exists: skipping")

    # check that if x_orientation is set, it's read back out properly
    uv_in.telescope.set_feeds_from_x_orientation(
        "east", polarization_array=uv_in.polarization_array
    )
    _write_miriad(uv_in, write_file, clobber=True)
    uv_out.read(write_file)
    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_antenna_diameters(uv_in_paper):
    # check that if antenna_diameters is set, it's read back out properly
    uv_in, uv_out, write_file = uv_in_paper
    uv_in.telescope.antenna_diameters = (
        np.zeros((uv_in.telescope.Nants,), dtype=np.float32) + 14.0
    )
    _write_miriad(uv_in, write_file, clobber=True)
    uv_out.read(write_file)

    # make sure that filenames make sense
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out

    # check that antenna diameters get written if not exactly float
    uv_in.telescope.antenna_diameters = (
        np.zeros((uv_in.telescope.Nants,), dtype=np.float32) + 14.0
    )
    _write_miriad(uv_in, write_file, clobber=True)
    uv_out.read(write_file)
    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out

    # check warning when antenna diameters vary
    uv_in.telescope.antenna_diameters = (
        np.zeros((uv_in.telescope.Nants,), dtype=np.float32) + 14.0
    )
    uv_in.telescope.antenna_diameters[1] = 15.0
    _write_miriad(
        uv_in,
        write_file,
        clobber=True,
        warn=[
            warn_dict["uvw_mismatch"],
            warn_dict["default_vals"],
            warn_dict["ant_diameters"],
        ],
    )
    uv_out.read(write_file)
    assert uv_out.telescope.antenna_diameters is None
    uv_out.telescope.antenna_diameters = uv_in.telescope.antenna_diameters
    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:It is not clear from the file if the data are")
def test_miriad_write_read_diameters(tmp_path):
    # check for backwards compatibility with old keyword 'diameter' for
    # antenna diameters

    orig_file = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA")
    testfile = os.path.join(tmp_path, "diameter_miriad")

    aipy_uv = aipy_extracts.UV(orig_file)

    if os.path.exists(testfile):
        shutil.rmtree(testfile)

    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    # remove "antdiam" and put in "diameter"
    aipy_uv2.init_from_uv(aipy_uv, exclude=["antdiam"])
    aipy_uv2.add_var("diameter", "d")
    aipy_uv2["diameter"] = "14."
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    uv_in = UVData.from_file(orig_file)
    uv_out = UVData.from_file(testfile)

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_and_aipy_reads(uv_in_paper):
    uv_in, _, write_file = uv_in_paper
    # check that variables 'ischan' and 'nschan' were written to new file
    # need to use aipy, since pyuvdata is not currently capturing these variables
    uv_aipy = aipy_extracts.UV(write_file)
    nfreqs = uv_in.Nfreqs
    nschan = uv_aipy["nschan"]
    ischan = uv_aipy["ischan"]
    assert nschan == nfreqs
    assert ischan == 1

    # cleanup
    uv_aipy.close()


def test_miriad_telescope_locations():
    # test load_telescope_coords w/ blank Miriad
    uv_in = Miriad()
    uv = aipy_extracts.UV(paper_miriad_file)
    uv_in._load_telescope_coords(uv)
    assert uv_in.telescope.location is not None
    uv.close()
    # test load_antpos w/ blank Miriad
    uv_in = Miriad()
    uv = aipy_extracts.UV(paper_miriad_file)
    uv_in._load_antpos(uv)
    assert uv_in.telescope.antenna_positions is not None


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_miriad_integration_time_precision(uv_in_paper):
    uv_in, uv_out, write_file = uv_in_paper

    # test that changing precision of integraiton_time is okay
    # tolerance of integration_time (1e-3) is larger than floating point type
    # conversions
    uv_in.integration_time = uv_in.integration_time.astype(np.float32)
    _write_miriad(uv_in, write_file, clobber=True)
    new_uv = UVData()
    new_uv.read(write_file)

    # make sure filenames are what we expect
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert new_uv.filename == ["outtest_miriad.uv"]
    uv_in.filename = new_uv.filename

    new_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == new_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"bls": [(0, 0), (0, 1), (4, 2)]},
        {"bls": [(0, 0), (2, 4)], "antenna_nums": [0, 2, 4]},
        {"bls": (2, 4, "xy")},
        {"bls": [(4, 2, "yx")]},
        {"polarizations": [-7], "bls": [(4, 4)]},
        {"bls": [(4, 4, "xy")]},
    ],
)
def test_read_write_read_miriad_partial_bls(uv_in_paper, select_kwargs, tmp_path):
    # check partial read selections
    full, uv_out, write_file = uv_in_paper

    _write_miriad(full, write_file, clobber=True)
    uv_in = UVData()

    # test only specified bls were read, and that flipped antpair is loaded too
    uv_in.read(write_file, **select_kwargs)
    antpairs = uv_in.get_antpairs()
    # indexing here is to ignore polarization if present, maybe there is a better way
    bls = select_kwargs["bls"]
    if isinstance(bls, tuple):
        bls = [bls]
    assert np.all([bl[:2] in antpairs or bl[:2][::-1] in antpairs for bl in bls])
    exp_uv = full.select(inplace=False, **select_kwargs)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_antenna_nums(uv_in_paper, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    _write_miriad(full, write_file, clobber=True)
    uv_in = UVData()
    # test all bls w/ 0 are loaded
    uv_in.read(write_file, antenna_nums=[0])
    diff = set(full.get_antpairs()) - set(uv_in.get_antpairs())
    assert 0 not in np.unique(diff)
    exp_uv = full.select(antenna_nums=[0], inplace=False)
    assert np.max(exp_uv.ant_1_array) == 0
    assert np.max(exp_uv.ant_2_array) == 0

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "select_kwargs",
    [
        {"time_range": [2456865.607, 2456865.609]},
        {"time_range": [2456865.607, 2456865.609], "antenna_nums": [0]},
        {"time_range": [2456865.607, 2456865.609], "polarizations": [-7]},
    ],
)
def test_read_write_read_miriad_partial_times(uv_in_paper, select_kwargs, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    _write_miriad(full, write_file, clobber=True)
    # test time loading
    uv_in = UVData()
    uv_in.read(write_file, **select_kwargs)
    full_times = np.unique(
        full.time_array[
            (full.time_array > select_kwargs["time_range"][0])
            & (full.time_array < select_kwargs["time_range"][1])
        ]
    )
    np.testing.assert_allclose(
        np.unique(uv_in.time_array),
        full_times,
        rtol=uv_in._time_array.tols[0],
        atol=uv_in._time_array.tols[1],
    )
    # The exact time are calculated above, pop out the time range to compare
    # time range with selecting on exact times
    select_kwargs.pop("time_range", None)
    exp_uv = full.select(times=full_times, inplace=False, **select_kwargs)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize("pols", [["xy"], [-7]])
def test_read_write_read_miriad_partial_pols(uv_in_paper, pols, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    _write_miriad(full, write_file, clobber=True)

    # test polarization loading
    uv_in = UVData()
    uv_in.read(write_file, polarizations=pols)
    assert full.polarization_array == uv_in.polarization_array
    exp_uv = full.select(polarizations=pols, inplace=False)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_ant_str(uv_in_paper, tmp_path):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    _write_miriad(full, write_file, clobber=True)
    # test ant_str
    uv_in = UVData()
    uv_in.read(write_file, ant_str="auto")
    assert np.array([blp[0] == blp[1] for blp in uv_in.get_antpairs()]).all()
    exp_uv = full.select(ant_str="auto", inplace=False)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv

    uv_in.read(write_file, ant_str="cross")
    assert np.array([blp[0] != blp[1] for blp in uv_in.get_antpairs()]).all()
    exp_uv = full.select(ant_str="cross", inplace=False)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv

    uv_in.read(write_file, ant_str="all")

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    uv_in._consolidate_phase_center_catalogs(other=full)
    assert uv_in == full


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.parametrize(
    "err_type,select_kwargs,err_msg",
    [
        (
            ValueError,
            {"ant_str": "auto", "antenna_nums": [0, 1]},
            "Cannot provide ant_str with antenna_nums or bls",
        ),
        (ValueError, {"bls": "foo"}, "bls must be a list of tuples of antenna numbers"),
        (
            ValueError,
            {"bls": [[0, 1]]},
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            ValueError,
            {"bls": [("foo", "bar")]},
            "bls must be a list of tuples of antenna numbers",
        ),
        (
            ValueError,
            {"bls": [("foo",)]},
            "bls tuples must be all length-2 or all length-3",
        ),
        (
            ValueError,
            {"bls": [(1, 2), (2, 3, "xx")]},
            "bls tuples must be all length-2 or all length-3",
        ),
        (
            ValueError,
            {"bls": [(2, 4, 0)]},
            "The third element in each bl must be a polarization string",
        ),
        (
            ValueError,
            {"bls": [(2, 4, "xy")], "polarizations": ["xy"]},
            "Cannot provide length-3 tuples and also specify polarizations.",
        ),
        (
            ValueError,
            {"antenna_nums": np.array([(0, 10)])},
            "antenna_nums must be a list of antenna number integers",
        ),
        (
            ValueError,
            {"antenna_nums": 5},
            "antenna_nums must be a list of antenna number integers",
        ),
        (
            ValueError,
            {"antenna_nums": ["foo"]},
            "antenna_nums must be a list of antenna number integers",
        ),
        (
            ValueError,
            {"polarizations": "xx"},
            "pols must be a list of polarization strings or ints",
        ),
        (
            ValueError,
            {"polarizations": [5.3]},
            "pols must be a list of polarization strings or ints",
        ),
        (
            ValueError,
            {"polarizations": ["yy"]},
            "No data is present, probably as a result of select on read",
        ),
        (ValueError, {"polarizations": [-9]}, "No polarizations in data matched input"),
        (
            ValueError,
            {"time_range": "foo"},
            "time_range must be a len-2 list of Julian Date floats",
        ),
        (
            ValueError,
            {"time_range": [1, 2, 3]},
            "time_range must be a len-2 list of Julian Date floats",
        ),
        (
            ValueError,
            {"time_range": ["foo", "bar"]},
            "time_range must be a len-2 list of Julian Date floats",
        ),
        (
            ValueError,
            {"time_range": [10.1, 10.2]},
            "No data is present, probably as a result of select on read",
        ),
        (ValueError, {"ant_str": 0}, "ant_str must be a string"),
    ],
)
def test_read_write_read_miriad_partial_errors(
    uv_in_paper, err_type, select_kwargs, err_msg, tmp_path
):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    _write_miriad(full, write_file, clobber=True)
    uv_in = UVData()

    with pytest.raises(err_type, match=err_msg):
        uv_in.read(write_file, **select_kwargs)


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_with_warnings(uv_in_paper):
    full, uv_out, write_file = uv_in_paper

    # check partial read selections
    _write_miriad(full, write_file, clobber=True)

    uv_in = UVData()
    # check handling for generic read selections unsupported by read_miriad
    unique_times = np.unique(full.time_array)
    times_to_keep = unique_times[
        ((unique_times > 2456865.607) & (unique_times < 2456865.609))
    ]
    with check_warnings(
        UserWarning,
        [
            "Warning: a select on read keyword is set",
            warn_dict["uvw_mismatch"],
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_in.read(write_file, times=times_to_keep)

    exp_uv = full.select(times=times_to_keep, inplace=False)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == exp_uv

    uv_in = UVData()
    # check handling for generic read selections unsupported by read_miriad
    blts_select = np.where(full.time_array == unique_times[0])[0]
    ants_keep = [0, 2, 4]
    with check_warnings(
        UserWarning,
        [
            "Warning: blt_inds is set along with select on read",
            warn_dict["uvw_mismatch"],
            warn_dict["uvw_mismatch"],
        ],
    ):
        uv_in.read(write_file, blt_inds=blts_select, antenna_nums=ants_keep)
    exp_uv = full.select(blt_inds=blts_select, antenna_nums=ants_keep, inplace=False)

    # make sure filenames are what we expect
    assert uv_in.filename == ["outtest_miriad.uv"]
    assert exp_uv.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    uv_in.filename = exp_uv.filename

    exp_uv._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in != exp_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_write_read_miriad_partial_metadata_only(uv_in_paper, tmp_path):
    uv_in, uv_out, write_file = uv_in_paper
    write_file2 = os.path.join(tmp_path, "outtest_miriad2.uv")

    # try metadata only read
    uv_in_meta = UVData()
    uv_in_meta.read(paper_miriad_file, read_data=False)
    assert uv_in_meta.time_array is None
    assert uv_in_meta.data_array is None
    assert uv_in_meta.integration_time is None
    metadata = ["channel_width", "history", "vis_units"]
    for par in metadata:
        assert getattr(uv_in_meta, par) is not None

    telescope_metadata = [
        "antenna_positions",
        "antenna_names",
        "antenna_positions",
        "location",
    ]
    for par in telescope_metadata:
        assert getattr(uv_in_meta.telescope, par) is not None

    # metadata only multiple file read-in
    del uv_in_meta

    new_uv = uv_in.select(freq_chans=np.arange(5), inplace=False)
    _write_miriad(new_uv, write_file, clobber=True)
    new_uv = uv_in.select(freq_chans=np.arange(5) + 5, inplace=False)
    _write_miriad(new_uv, write_file2, clobber=True)

    uv_in.select(freq_chans=np.arange(10))

    uv_in2 = UVData()
    uv_in2.read(np.array([write_file, write_file2]))

    assert uv_in.history != uv_in2.history
    uv_in2.history = uv_in.history

    # make sure filenames are what we expect
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert set(uv_in2.filename) == {"outtest_miriad.uv", "outtest_miriad2.uv"}
    uv_in2.filename = uv_in.filename
    uv_in2._filename.form = (1,)

    uv_in2._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_in2


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_read_ms_write_miriad_casa_history(tmp_path):
    """
    Read in .ms file.
    Write to a miriad file, read back in and check for history parameter
    """
    pytest.importorskip("casacore")
    ms_uv = UVData()
    miriad_uv = UVData()
    ms_file = os.path.join(DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.ms")
    testfile = os.path.join(tmp_path, "outtest_miriad")
    ms_uv.read(ms_file)

    _write_miriad(ms_uv, testfile, clobber=True)
    miriad_uv.read(testfile)

    # make sure filenames are what we expect
    assert miriad_uv.filename == ["outtest_miriad"]
    assert ms_uv.filename == ["day2_TDEM0003_10s_norx_1src_1spw.ms"]
    miriad_uv.filename = ms_uv.filename

    # propagate scan numbers to the uvfits, ONLY for comparison
    miriad_uv.scan_number_array = ms_uv.scan_number_array

    assert miriad_uv == ms_uv


@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_rwr_miriad_antpos_issues(uv_in_paper, tmp_path):
    """
    test warnings and errors associated with antenna position issues in Miriad files

    Read in Miriad PAPER file, mess with various antpos issues and write out as
    a new Miriad file, read back in and check for appropriate behavior.
    """
    uv_in, uv_out, write_file = uv_in_paper

    uv_in_copy = uv_in.copy()
    uv_in_copy.telescope.antenna_positions = None
    _write_miriad(
        uv_in_copy,
        write_file,
        clobber=True,
        run_check=False,
        warn=warn_dict["default_vals"],
    )
    with check_warnings(
        UserWarning, match=["Antenna positions are not present in the file."] * 2
    ):
        uv_out.read(write_file, run_check=False)

    # make sure filenames are what we expect
    assert uv_in_copy.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in_copy.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in_copy)
    assert uv_in_copy == uv_out

    uv_in_copy = uv_in.copy()
    ants_with_data = list(set(uv_in_copy.ant_1_array).union(uv_in_copy.ant_2_array))
    ant_ind = np.where(uv_in_copy.telescope.antenna_numbers == ants_with_data[0])[0]
    uv_in_copy.telescope.antenna_positions[ant_ind, :] = [0, 0, 0]
    _write_miriad(
        uv_in_copy,
        write_file,
        clobber=True,
        no_antnums=True,
        warn=[warn_dict["default_vals"], warn_dict["uvw_mismatch"]],
    )
    with check_warnings(UserWarning, ["antenna number", warn_dict["uvw_mismatch"]]):
        uv_out.read(write_file)

    # make sure filenames are what we expect
    assert uv_in_copy.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in_copy.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in_copy)
    assert uv_in_copy == uv_out

    uv_in.telescope.antenna_positions = None
    ants_with_data = sorted(set(uv_in.ant_1_array).union(uv_in.ant_2_array))
    new_nums = []
    new_names = []
    for a in ants_with_data:
        new_nums.append(a)
        ind = np.where(uv_in.telescope.antenna_numbers == a)[0][0]
        new_names.append(uv_in.telescope.antenna_names[ind])
    uv_in.telescope.antenna_numbers = np.array(new_nums)
    uv_in.telescope.antenna_names = new_names
    uv_in.telescope.Nants = len(uv_in.telescope.antenna_numbers)
    _write_miriad(
        uv_in,
        write_file,
        clobber=True,
        no_antnums=True,
        run_check=False,
        warn=warn_dict["default_vals"],
    )
    with check_warnings(
        UserWarning, match=["Antenna positions are not present in the file."] * 2
    ):
        uv_out.read(write_file, run_check=False)

    # make sure filenames are what we expect
    assert uv_in.filename == ["zen.2456865.60537.xy.uvcRREAA"]
    assert uv_out.filename == ["outtest_miriad.uv"]
    uv_in.filename = uv_out.filename

    uv_out._consolidate_phase_center_catalogs(other=uv_in)
    assert uv_in == uv_out


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_multi_files(casa_uvfits, tmp_path):
    """
    Reading multiple files at once.
    """
    uv_full = casa_uvfits
    testfile1 = os.path.join(tmp_path, "uv1")
    testfile2 = os.path.join(tmp_path, "uv2")
    # rename telescope to avoid name warning
    uv_full.unproject_phase()
    uv_full.conjugate_bls("ant1<ant2")

    uv1 = uv_full.copy()
    uv2 = uv_full.copy()
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    _write_miriad(uv1, testfile1, clobber=True, warn=warn_dict["default_vals"])
    _write_miriad(uv2, testfile2, clobber=True, warn=warn_dict["default_vals"])
    del uv1
    uv1 = UVData()
    uv1.read([testfile1, testfile2], file_type="miriad")

    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis using"
        " pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1", "uv2"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    uv1._consolidate_phase_center_catalogs(
        reference_catalog=uv_full.phase_center_catalog
    )

    assert uv1 == uv_full

    # again, setting axis
    del uv1
    uv1 = UVData()
    uv1.read([testfile1, testfile2], axis="freq")
    # Check history is correct, before replacing and doing a full object check
    assert utils.history._check_histories(
        uv_full.history + "  Downselected to "
        "specific frequencies using pyuvdata. "
        "Combined data along frequency axis using"
        " pyuvdata.",
        uv1.history,
    )
    uv1.history = uv_full.history

    # make sure filenames are what we expect
    assert set(uv1.filename) == {"uv1", "uv2"}
    assert uv_full.filename == ["day2_TDEM0003_10s_norx_1src_1spw.uvfits"]
    uv1.filename = uv_full.filename
    uv1._filename.form = (1,)

    uv1._consolidate_phase_center_catalogs(
        reference_catalog=uv_full.phase_center_catalog
    )

    assert uv1 == uv_full


@pytest.mark.filterwarnings("ignore:Telescope EVLA is not")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
def test_antpos_units(casa_uvfits, tmp_path):
    """
    Read uvfits, write miriad. Check written antpos are in ns.
    """
    uv = casa_uvfits
    testfile = os.path.join(tmp_path, "uv_antpos_units")
    _write_miriad(uv, testfile, clobber=True)
    auv = aipy_extracts.UV(testfile)
    aantpos = auv["antpos"].reshape(3, -1).T * const.c.to_value("m/ns")
    aantpos = aantpos[uv.telescope.antenna_numbers, :]
    aantpos = (
        utils.ECEF_from_rotECEF(aantpos, uv.telescope.location.lon.rad)
        - uv.telescope._location.xyz()
    )
    np.testing.assert_allclose(
        aantpos, uv.telescope.antenna_positions, rtol=0, atol=1e-3
    )


@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only,")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match the expected values")
@pytest.mark.filterwarnings("ignore:It is not clear from the file if the data are")
def test_readmiriad_write_miriad_check_time_format(tmp_path):
    """
    test time_array is converted properly from Miriad format
    """
    # test read-in
    fname = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcAA")
    uvd = UVData()
    uvd.read(fname)
    uvd_t = uvd.time_array.min()
    uvd_l = uvd.lst_array.min()
    uv = aipy_extracts.UV(fname)
    uv_t = uv["time"] + uv["inttime"] / (24 * 3600.0) / 2

    t1 = Time(uv["time"], format="jd", location=uvd.telescope.location)
    dt = TimeDelta(uv["inttime"] / 2, format="sec")
    t2 = t1 + dt
    lsts = utils.get_lst_for_time(
        np.array([t1.jd, t2.jd]), telescope_loc=uvd.telescope.location
    )
    delta_lst = lsts[1] - lsts[0]
    uv_l = uv["lst"] + delta_lst

    # assert starting time array and lst array are shifted by half integration
    assert np.isclose(
        uvd_t, uv_t, rtol=uvd._time_array.tols[0], atol=uvd._time_array.tols[1]
    )

    # avoid errors if IERS table is too old (if the iers url is down)
    lst_tolerance = 1e-8
    np.testing.assert_allclose(uvd_l, uv_l, atol=lst_tolerance, rtol=0)
    # test write-out
    fout = os.path.join(tmp_path, "ex_miriad")
    _write_miriad(uvd, fout, clobber=True)
    # assert equal to original miriad time
    uv2 = aipy_extracts.UV(fout)
    assert np.isclose(
        uv["time"],
        uv2["time"],
        rtol=uvd._time_array.tols[0],
        atol=uvd._time_array.tols[1],
    )
    assert np.isclose(uv["lst"], uv2["lst"], atol=lst_tolerance, rtol=0)


def test_file_with_bad_extra_words():
    """Test file with bad extra words is iterated and popped correctly."""
    fname = os.path.join(DATA_PATH, "test_miriad_changing_extra.uv")
    uv = UVData()
    warn_message = [
        (
            "Altitude is not present in Miriad file, "
            "using known location values for PAPER."
        ),
        "Mean of empty slice.",
        "invalid value encountered",
        "npols=4 but found 1 pols in data file",
        "Mean of empty slice.",
        "invalid value encountered",
        (
            "antenna number 0 has visibilities associated with it, but it has a "
            "position of (0,0,0)"
        ),
        (
            "antenna number 26 has visibilities associated with it, "
            "but it has a position of (0,0,0)"
        ),
    ]
    warn_category = (
        [UserWarning]
        + [RuntimeWarning] * 2
        + [UserWarning]
        + [RuntimeWarning] * 2
        + [UserWarning] * 2
    )
    # This is an old PAPER file, run_check must be set to false
    # The antenna positions is (0, 0, 0) vector
    with check_warnings(warn_category, warn_message):
        uv.read_miriad(fname, run_check=False)


def test_miriad_read_xorient(tmp_path):
    """
    Read miriad w/ x_orientation keyword present, verify things make sense.
    """
    # set the xorient variable directly as we used to do to check for backwards
    # compatibility
    orig_file = os.path.join(DATA_PATH, "new.uvA")
    testfile = os.path.join(tmp_path, "xorient_miriad")

    aipy_uv = aipy_extracts.UV(orig_file)

    if os.path.exists(testfile):
        shutil.rmtree(testfile)

    # make new file
    aipy_uv2 = aipy_extracts.UV(testfile, status="new")
    # initialize headers from old file
    aipy_uv2.init_from_uv(aipy_uv)
    # add xorient into the file
    aipy_uv2.add_var("xorient", "a")
    aipy_uv2["xorient"] = "east"
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    aipy_uv2.close()

    uv = UVData.from_file(testfile)
    nants = uv.telescope.Nants

    assert uv.telescope.get_x_orientation_from_feeds() == "east"

    assert uv.telescope.Nfeeds == 1
    assert np.array_equal(uv.telescope.feed_array, [["x"]] * nants)
    assert np.array_equal(uv.telescope.feed_angle, [[np.pi / 2]] * nants)
