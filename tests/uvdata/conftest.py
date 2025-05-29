# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""pytest fixtures for UVData tests."""

import copy
import os
import warnings

import pytest

from pyuvdata import UVData
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings
from pyuvdata.uvdata.mir_parser import MirParser

casa_tutorial_uvfits = os.path.join(
    DATA_PATH, "day2_TDEM0003_10s_norx_1src_1spw.uvfits"
)
paper_miriad_file = os.path.join(DATA_PATH, "zen.2456865.60537.xy.uvcRREAA")


@pytest.fixture(scope="session")
def casa_uvfits_main():
    """Read in CASA tutorial uvfits file."""
    uv_in = UVData()
    with check_warnings(
        UserWarning, "The uvw_array does not match the expected values"
    ):
        uv_in.read(casa_tutorial_uvfits)

    yield uv_in

    # cleanup
    del uv_in


@pytest.fixture(scope="function")
def casa_uvfits(casa_uvfits_main):
    """Make function level CASA tutorial uvfits object."""
    casa_uvfits = casa_uvfits_main.copy()
    yield casa_uvfits

    # clean up when done
    del casa_uvfits

    return


@pytest.fixture(scope="session")
def hera_uvh5_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "zen.2458661.23480.HH.uvh5")
    uv_object.read(testfile)

    yield uv_object


@pytest.fixture(scope="function")
def hera_uvh5(hera_uvh5_main):
    # read in test file for the resampling in time functions
    uv_object = hera_uvh5_main.copy()

    yield uv_object


@pytest.fixture(scope="session")
def paper_miriad_main():
    # read in paper miriad file
    pytest.importorskip("pyuvdata.uvdata._miriad", exc_type=ImportError)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The uvw_array does not match")
        warnings.filterwarnings("ignore", "Altitude is not present in Miriad file")
        uv_object = UVData.from_file(paper_miriad_file)

    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="function")
def paper_miriad(paper_miriad_main):
    uv_object = paper_miriad_main.copy()
    uv_object.set_rectangularity()
    yield uv_object

    # cleanup
    del uv_object

    return


@pytest.fixture(scope="session")
def sma_mir_main():
    # read in test file for the resampling in time functions
    uv_object = UVData()
    testfile = os.path.join(DATA_PATH, "sma_test.mir")
    with check_warnings(
        UserWarning,
        match=[
            "> 25 ms errors detected reading in LST values from MIR data. ",
            "The lst_array is not self-consistent with the time_array and telescope ",
        ],
    ):
        uv_object.read(testfile)
    uv_object.set_lsts_from_time_array()

    yield uv_object


@pytest.fixture(scope="function")
def sma_mir(sma_mir_main):
    # read in test file for the resampling in time functions
    uv_object = sma_mir_main.copy()

    yield uv_object


@pytest.fixture(scope="session")
def mir_data_main():
    mir_data = MirParser()

    yield mir_data._load_test_data(load_cross=True, load_auto=True, has_auto=True)


@pytest.fixture(scope="function")
def mir_data(mir_data_main):
    mir_data = mir_data_main.copy()

    yield mir_data


@pytest.fixture(scope="function")
def fhd_test_files():
    # set up FHD files
    testdir = os.path.join(DATA_PATH, "fhd_vis_data/")

    tf_prefix = "1061316296_"
    tf_dict = {
        "data_files": [
            os.path.join(testdir, "vis_data", tf_prefix + fname)
            for fname in ["vis_XX.sav", "vis_YY.sav"]
        ],
        "model_files": [
            os.path.join(testdir, "vis_data", tf_prefix + fname)
            for fname in ["vis_model_XX.sav", "vis_model_YY.sav"]
        ],
        "flags_file": os.path.join(testdir, "vis_data", tf_prefix + "flags.sav"),
        "params_file": os.path.join(testdir, "metadata", tf_prefix + "params.sav"),
        "layout_file": os.path.join(testdir, "metadata", tf_prefix + "layout.sav"),
        "settings_file": os.path.join(testdir, "metadata", tf_prefix + "settings.txt"),
        "obs_file": os.path.join(testdir, "metadata", tf_prefix + "obs.sav"),
    }

    return tf_dict


@pytest.fixture(scope="function")
def fhd_data_files(fhd_test_files):
    file_dict = copy.deepcopy(fhd_test_files)
    file_dict["filename"] = file_dict["data_files"]
    del file_dict["data_files"]
    del file_dict["model_files"]

    return file_dict


@pytest.fixture(scope="function")
def fhd_model_files(fhd_test_files):
    file_dict = copy.deepcopy(fhd_test_files)
    file_dict["filename"] = file_dict["model_files"]
    del file_dict["data_files"]
    del file_dict["model_files"]

    return file_dict


@pytest.fixture(scope="session")
def uv_phase_comp_main():
    file1 = os.path.join(DATA_PATH, "1133866760.uvfits")
    file2 = os.path.join(DATA_PATH, "1133866760_rephase.uvfits")
    # These files came from an external source, don't want to rewrite them, so use
    # checkwarnings to capture the warning about non-real autos
    with check_warnings(
        UserWarning,
        match=[
            "Fixing auto-correlations to be be real-only, after some imaginary "
            "values were detected in data_array."
        ]
        * 2,
    ):
        uvd1 = UVData.from_file(file1)
        uvd2 = UVData.from_file(file2)

    yield uvd1, uvd2


@pytest.fixture(scope="function")
def uv_phase_comp(uv_phase_comp_main):
    uvd1, uvd2 = uv_phase_comp_main
    uvd1_copy = uvd1.copy()
    uvd2_copy = uvd2.copy()

    yield uvd1_copy, uvd2_copy
