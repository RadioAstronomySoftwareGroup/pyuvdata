# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
import os
import shutil

import numpy as np
import pytest
import yaml

from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings
from pyuvdata.uvbeam.cst_beam import CSTBeam

filenames = ["HERA_NicCST_150MHz.txt", "HERA_NicCST_123MHz.txt"]
cst_folder = "NicCSTbeams"
cst_files = [os.path.join(DATA_PATH, cst_folder, f) for f in filenames]
cst_yaml_file = os.path.join(DATA_PATH, cst_folder, "NicCSTbeams.yaml")
cst_yaml_vivaldi = os.path.join(DATA_PATH, cst_folder, "HERA_Vivaldi_CST_beams.yaml")


@pytest.fixture(scope="function")
def cst_efield_1freq_mod(cst_efield_1freq_main):
    """Make modified version of 1-freq efield beam."""
    beam = cst_efield_1freq_main.copy()

    beam.reference_impedance = 100
    beam.receiver_temperature_array = None
    beam.loss_array = None
    beam.mismatch_array = None
    beam.s_parameters = None

    yield beam
    del beam


@pytest.fixture(scope="function")
def cst_efield_2freq_mod(cst_efield_2freq_main):
    """Make modified version of 2-freq efield beam."""
    beam = cst_efield_2freq_main.copy()

    beam.reference_impedance = 100
    beam.receiver_temperature_array = None
    beam.loss_array = None
    beam.mismatch_array = None
    beam.s_parameters = None

    yield beam
    del beam


def test_basic_frequencyparse():
    beam1 = CSTBeam()

    parsed_freqs = [beam1.name2freq(f) for f in cst_files]
    assert parsed_freqs == [150e6, 123e6]


def test_frequencyparse_extra_numbers():
    beam1 = CSTBeam()
    test_path = os.path.join(
        "pyuvdata_1510194907049",
        "_t_env",
        "lib",
        "python2.7",
        "site-packages",
        "pyuvdata",
        "data",
    )
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    assert parsed_freqs == [150e6, 123e6]


def test_frequencyparse_nicf_path():
    beam1 = CSTBeam()
    test_path = os.path.join(
        "Simulations",
        "Radiation_patterns",
        "E-field pattern-Rigging height4.9m",
        "HERA_4.9m_E-pattern_100-200MHz",
    )
    test_files = [os.path.join(test_path, f) for f in filenames]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    assert parsed_freqs == [150e6, 123e6]


def test_frequencyparse_decimal_non_mhz():
    beam1 = CSTBeam()
    test_path = os.path.join(
        "Simulations",
        "Radiation_patterns",
        "E-field pattern-Rigging height4.9m",
        "HERA_4.9m_E-pattern_100-200MHz",
    )
    test_names = [
        "HERA_Sim_120.87kHz.txt",
        "HERA_Sim_120.87GHz.txt",
        "HERA_Sim_120.87Hz.txt",
    ]
    test_files = [os.path.join(test_path, f) for f in test_names]
    parsed_freqs = [beam1.name2freq(f) for f in test_files]
    assert parsed_freqs == [120.87e3, 120.87e9, 120.87]


def test_read_yaml(cst_efield_2freq_mod):
    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam1 = cst_efield_2freq_mod
    assert beam1.filename == ["HERA_NicCST_123MHz.txt", "HERA_NicCST_150MHz.txt"]

    beam2.read_cst_beam(cst_yaml_file, beam_type="efield")
    assert beam2.filename == sorted(
        [
            os.path.basename(cst_yaml_file),
            "HERA_NicCST_123MHz.txt",
            "HERA_NicCST_150MHz.txt",
        ]
    )

    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords


def test_read_yaml_onefile(cst_efield_1freq_mod, tmp_path):
    # copy the beam files to the tmp directory so that it can read them
    # when the yaml is stored there
    for fname in cst_files:
        shutil.copy2(src=fname, dst=tmp_path)

    test_yaml_file = os.path.join(tmp_path, "test_cst_settings.yaml")
    with open(cst_yaml_file) as file:
        settings_dict = yaml.safe_load(file)

    settings_dict["filenames"] = [settings_dict["filenames"][0]]
    settings_dict["frequencies"] = [settings_dict["frequencies"][0]]

    with open(test_yaml_file, "w") as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam1 = cst_efield_1freq_mod

    beam2.read_cst_beam(test_yaml_file, beam_type="efield")
    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords


def test_read_yaml_override(cst_efield_2freq_mod):
    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam1 = cst_efield_2freq_mod

    beam1.telescope_name = "test"

    with check_warnings(
        UserWarning,
        match=[
            "The telescope_name keyword is set, overriding "
            "the value in the settings yaml file.",
            "The mount_type keyword is set",
            "The feed_pol keyword is set",
        ],
    ):
        beam2.read_cst_beam(
            cst_yaml_file,
            beam_type="efield",
            telescope_name="test",
            mount_type="fixed",
            feed_pol="x",
            feed_array="x",
            feed_angle=[np.pi / 2],
        )

    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords


def test_read_yaml_freq_select(cst_efield_1freq_mod):
    # test frequency_select
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1 = cst_efield_1freq_mod

    beam2.read_cst_beam(cst_yaml_file, beam_type="efield", frequency_select=[150e6])

    assert beam1 == beam2

    # test error with using frequency_select where no such frequency
    freq = 180e6
    with pytest.raises(ValueError, match=f"frequency {freq} not in frequency list"):
        beam2.read_cst_beam(cst_yaml_file, beam_type="power", frequency_select=[freq])


def test_read_yaml_feed_pol_list(cst_efield_2freq_mod, cst_efield_1freq_mod):
    # make yaml with a list of (the same) feed_pols

    test_yaml_file = os.path.join(DATA_PATH, cst_folder, "test_cst_settings.yaml")
    with open(cst_yaml_file) as file:
        settings_dict = yaml.safe_load(file)

    settings_dict["feed_pol"] = ["x", "x"]

    with open(test_yaml_file, "w") as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam1 = cst_efield_2freq_mod

    beam2.read_cst_beam(test_yaml_file, beam_type="efield")
    assert beam1 == beam2

    assert beam2.reference_impedance == 100
    assert beam2.extra_keywords == extra_keywords

    # also test with frequency_select
    beam1 = cst_efield_1freq_mod

    beam2.read_cst_beam(test_yaml_file, beam_type="efield", frequency_select=[150e6])
    assert beam1 == beam2

    os.remove(test_yaml_file)


def test_read_yaml_multi_pol(tmp_path):
    # make yaml for one freq, 2 pols

    # copy the beam files to the tmp directory so that it can read them
    # when the yaml is stored there
    for fname in cst_files:
        shutil.copy2(src=fname, dst=tmp_path)
    test_yaml_file = os.path.join(tmp_path, "test_cst_settings.yaml")

    with open(cst_yaml_file) as file:
        settings_dict = yaml.safe_load(file)

    settings_dict["feed_pol"] = ["x", "y"]
    first_file = settings_dict["filenames"][0]
    settings_dict["filenames"] = [first_file, first_file]
    first_freq = settings_dict["frequencies"][0]
    settings_dict["frequencies"] = [first_freq, first_freq]

    with open(test_yaml_file, "w") as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    beam2 = UVBeam()

    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    with check_warnings(
        UserWarning, "No frequency provided. Detected frequency is", nwarnings=2
    ):
        beam1.read_cst_beam(
            [cst_files[0], cst_files[0]],
            beam_type="efield",
            feed_pol=["x", "y"],
            telescope_name="HERA",
            feed_name="Dipole",
            feed_version="1.0",
            mount_type="fixed",
            model_name="Dipole - Rigging height 4.9 m",
            model_version="1.0",
            x_orientation="east",
            reference_impedance=100,
            history="Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 2 files included to keep test data volume low.",
            extra_keywords=extra_keywords,
        )

    beam2.read_cst_beam(test_yaml_file, beam_type="efield")
    assert beam1 == beam2

    # also test with frequency_select
    beam2.read_cst_beam(test_yaml_file, beam_type="efield", frequency_select=[150e6])
    assert beam2.feed_array.tolist() == ["x", "y"]
    assert beam1 == beam2

    os.remove(test_yaml_file)


def test_read_yaml_errors(tmp_path):
    # test error if required key is not present in yaml file

    test_yaml_file = os.path.join(tmp_path, "test_cst_settings.yaml")
    with open(cst_yaml_file) as file:
        settings_dict = yaml.safe_load(file)

    settings_dict.pop("telescope_name")

    with open(test_yaml_file, "w") as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    with pytest.raises(
        ValueError,
        match=(
            "telescope_name is a required key in CST settings files but is not present."
        ),
    ):
        beam1.read_cst_beam(test_yaml_file, beam_type="power")

    with open(cst_yaml_file) as file:
        settings_dict = yaml.safe_load(file)
    settings_dict["filenames"] = settings_dict["filenames"][0]

    with open(test_yaml_file, "w") as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    with pytest.raises(ValueError, match=("filenames in yaml file must be a list.")):
        beam1.read_cst_beam(test_yaml_file, beam_type="power")

    with open(cst_yaml_file) as file:
        settings_dict = yaml.safe_load(file)
    settings_dict["frequencies"] = settings_dict["frequencies"][0]

    with open(test_yaml_file, "w") as outfile:
        yaml.dump(settings_dict, outfile, default_flow_style=False)

    beam1 = UVBeam()
    with pytest.raises(ValueError, match=("frequencies in yaml file must be a list.")):
        beam1.read_cst_beam(test_yaml_file, beam_type="power")

    os.remove(test_yaml_file)


def test_read_power(cst_power_2freq):
    beam2 = UVBeam()

    beam1 = cst_power_2freq

    assert beam1.pixel_coordinate_system == "az_za"
    assert beam1.beam_type == "power"
    assert beam1.data_array.shape == (1, 2, 2, 181, 360)
    assert np.max(beam1.data_array) == 8275.5409
    assert np.all(beam1.bandpass_array == 1)

    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, np.where(beam1.axis1_array == 0)[0]],
        beam1.data_array[:, 1, :, :, np.where(beam1.axis1_array == np.pi / 2.0)[0]],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )

    # test passing in other polarization
    beam2.read_cst_beam(
        np.array(cst_files),
        beam_type="power",
        frequency=np.array([150e6, 123e6]),
        feed_pol="y",
        mount_type="fixed",
        telescope_name="TEST",
        feed_name="bob",
        feed_version="0.1",
        model_name="E-field pattern - Rigging height 4.9m",
        model_version="1.0",
        x_orientation="east",
    )

    np.testing.assert_allclose(
        beam1.freq_array,
        beam2.freq_array,
        rtol=beam1._freq_array.tols[0],
        atol=beam1._freq_array.tols[1],
    )

    np.testing.assert_allclose(beam2.polarization_array, np.array([-6, -5]))
    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, :],
        beam2.data_array[:, 0, :, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )


def test_read_power_single_freq(cst_power_1freq):
    # test single frequency
    beam2 = UVBeam()

    beam1 = cst_power_1freq

    assert beam1.freq_array == [150e6]
    assert beam1.pixel_coordinate_system == "az_za"
    assert beam1.beam_type == "power"
    assert beam1.data_array.shape == (1, 2, 1, 181, 360)

    # test single frequency and not rotating the polarization
    with check_warnings(UserWarning, "No frequency provided. Detected frequency is"):
        beam2.read_cst_beam(
            cst_files[0],
            beam_type="power",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
            rotate_pol=False,
            x_orientation="east",
        )

    assert beam2.freq_array == [150e6]
    assert beam2.pixel_coordinate_system == "az_za"
    assert beam2.beam_type == "power"
    assert beam2.polarization_array == np.array([-5])
    assert beam2.data_array.shape == (1, 1, 1, 181, 360)
    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, :],
        beam2.data_array[:, 0, :, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )


def test_read_power_multi_pol():
    # test reading in multiple polarization files
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1.read_cst_beam(
        [cst_files[0], cst_files[0]],
        beam_type="power",
        frequency=[150e6],
        feed_pol=np.array(["xx", "yy"]),
        feed_array=np.array(["x", "y"]),
        feed_angle=np.array([0, np.pi / 2]),
        mount_type="fixed",
        telescope_name="TEST",
        feed_name="bob",
        feed_version="0.1",
        model_name="E-field pattern - Rigging height 4.9m",
        model_version="1.0",
    )
    assert beam1.data_array.shape == (1, 2, 1, 181, 360)
    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, :],
        beam1.data_array[:, 1, :, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )

    # test reading in cross polarization files
    beam2.read_cst_beam(
        [cst_files[0]],
        beam_type="power",
        frequency=[150e6],
        feed_pol=np.array(["xy"]),
        mount_type="fixed",
        telescope_name="TEST",
        feed_name="bob",
        feed_version="0.1",
        model_name="E-field pattern - Rigging height 4.9m",
        model_version="1.0",
        x_orientation="east",
    )
    np.testing.assert_allclose(beam2.polarization_array, np.array([-7, -8]))
    assert beam2.data_array.shape == (1, 2, 1, 181, 360)
    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, :],
        beam2.data_array[:, 0, :, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )


@pytest.mark.parametrize(
    ["files", "kwargs", "err_msg"],
    [
        [
            cst_files,
            {"beam_type": "power", "frequency": [150e6, 123e6, 100e6]},
            "If frequency and filename are both lists they need to be the same length",
        ],
        [
            cst_files[0],
            {"beam_type": "power", "frequency": [150e6, 123e6]},
            "Too many frequencies specified",
        ],
        [
            [cst_files[0], cst_files[0], cst_files[0]],
            {"beam_type": "power", "feed_pol": ["x", "y"]},
            "If feed_pol and filename are both lists they need to be the same length",
        ],
        [
            cst_files[0],
            {"beam_type": "power", "feed_pol": ["x", "y"]},
            "Too many feed_pols specified",
        ],
        [
            [[cst_files[0]], [cst_files[1]]],
            {
                "beam_type": "power",
                "frequency": [150e6, 123e6],
                "feed_pol": ["x"],
                "x_orientation": "east",
            },
            "filename can not be a nested list",
        ],
        [
            np.array([[cst_files[0]], [cst_files[1]]]),
            {"beam_type": "power", "frequency": [150e6, 123e6], "feed_pol": ["x"]},
            "filename can not be a multi-dimensional array",
        ],
        [
            cst_files,
            {"beam_type": "power", "frequency": [[150e6], [123e6]]},
            "frequency can not be a nested list",
        ],
        [
            cst_files,
            {"beam_type": "power", "frequency": np.array([[150e6], [123e6]])},
            "frequency can not be a multi-dimensional array",
        ],
        [
            cst_files,
            {"beam_type": "power", "frequency": 150e6, "feed_pol": [["x"], ["y"]]},
            "feed_pol can not be a nested list",
        ],
        [
            cst_files,
            {
                "beam_type": "power",
                "frequency": 150e6,
                "feed_pol": np.array([["x"], ["y"]]),
            },
            "feed_pol can not be a multi-dimensional array",
        ],
    ],
)
def test_read_errors(files, kwargs, err_msg):
    # test errors
    beam1 = UVBeam()

    kwargs.update(
        {
            "telescope_name": "TEST",
            "feed_name": "bob",
            "feed_version": "0.1",
            "model_name": "E-field pattern - Rigging height 4.9m",
            "model_version": "1.0",
            "mount_type": "fixed",
        }
    )

    with pytest.raises(ValueError, match=err_msg):
        beam1.read_cst_beam(files, **kwargs)


def test_read_efield(cst_efield_2freq):
    beam1 = cst_efield_2freq
    beam2 = UVBeam()

    assert beam1.pixel_coordinate_system == "az_za"
    assert beam1.beam_type == "efield"
    assert beam1.data_array.shape == (2, 2, 2, 181, 360)
    assert np.max(np.abs(beam1.data_array)) == 90.97

    # test passing in other polarization
    with check_warnings(UserWarning, ["Feed information not supplied"] * 2):
        beam2.read_cst_beam(
            cst_files,
            beam_type="efield",
            frequency=[150e6, 123e6],
            feed_pol="y",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )
    assert beam2.feed_array[0] == "y"
    assert beam2.feed_array[1] == "x"
    np.testing.assert_allclose(beam2.feed_angle, [0.0, np.pi / 2])
    assert beam1.data_array.shape == (2, 2, 2, 181, 360)
    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, :],
        beam2.data_array[:, 0, :, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )

    # test single frequency and not rotating the polarization
    with check_warnings(UserWarning, "No frequency provided. Detected frequency is"):
        beam2.read_cst_beam(
            cst_files[0],
            beam_type="efield",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
            rotate_pol=False,
            x_orientation="east",
        )

    assert beam2.pixel_coordinate_system == "az_za"
    assert beam2.beam_type == "efield"
    assert beam2.feed_array == np.array(["x"])
    assert beam2.data_array.shape == (2, 1, 1, 181, 360)

    np.testing.assert_allclose(
        beam1.data_array[:, 0, 1, :, :],
        beam2.data_array[:, 0, 0, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )

    # test reading in multiple polarization files
    beam1.read_cst_beam(
        [cst_files[0], cst_files[0]],
        beam_type="efield",
        frequency=[150e6],
        feed_pol=["x", "y"],
        mount_type="fixed",
        telescope_name="TEST",
        feed_name="bob",
        feed_version="0.1",
        model_name="E-field pattern - Rigging height 4.9m",
        model_version="1.0",
        feed_angle=[0, np.pi / 2],
    )
    assert beam1.data_array.shape == (2, 2, 1, 181, 360)
    np.testing.assert_allclose(
        beam1.data_array[:, 0, :, :, :],
        beam1.data_array[:, 1, :, :, :],
        rtol=beam1._data_array.tols[0],
        atol=beam1._data_array.tols[1],
    )


def test_no_deg_units(tmp_path):
    # need to write a modified file to test headers not in degrees
    testfile = os.path.join(tmp_path, "HERA_NicCST_150MHz_modified.txt")
    with open(cst_files[0]) as file:
        line1 = file.readline()
        line2 = file.readline()

    data = np.loadtxt(cst_files[0], skiprows=2)

    raw_names = line1.split("]")
    raw_names = [raw_name for raw_name in raw_names if "\n" not in raw_name]
    column_names = []
    column_names_simple = []
    units = []
    for raw_name in raw_names:
        column_name, unit = tuple(raw_name.split("["))
        column_names.append(column_name)
        column_names_simple.append("".join(column_name.lower().split(" ")))
        if unit != "deg.":
            units.append(unit)
        else:
            units.append("    ")

    new_column_headers = []
    for index, name in enumerate(column_names):
        new_column_headers.append(name + "[" + units[index] + "]")

    new_header = ""
    for col in new_column_headers:
        new_header += f"{col:12}"

    beam1 = UVBeam()
    beam2 = UVBeam()

    # format to match existing file
    existing_format = [
        "%8.3f",
        "%15.3f",
        "%20.3e",
        "%19.3e",
        "%19.3f",
        "%19.3e",
        "%19.3f",
        "%19.3e",
    ]
    np.savetxt(
        testfile,
        data,
        fmt=existing_format,
        header=new_header + "\n" + line2,
        comments="",
    )
    # this errors because the phi 2pi rotation doesn't work
    # (because they are degrees but the code thinks they're radians)
    with pytest.raises(ValueError, match="Rotating by pi/2 failed"):
        beam1.read_cst_beam(
            testfile,
            beam_type="efield",
            frequency=np.array([150e6]),
            feed_pol="y",
            feed_angle=0.0,
            mount_type="fixed",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )

    theta_col = np.where(np.array(column_names_simple) == "theta")[0][0]
    phi_col = np.where(np.array(column_names_simple) == "phi")[0][0]
    theta_phase_col = np.where(np.array(column_names_simple) == "phase(theta)")[0][0]
    phi_phase_col = np.where(np.array(column_names_simple) == "phase(phi)")[0][0]

    data[:, theta_col] = np.radians(data[:, theta_col])
    data[:, phi_col] = np.radians(data[:, phi_col])
    data[:, theta_phase_col] = np.radians(data[:, theta_phase_col])
    data[:, phi_phase_col] = np.radians(data[:, phi_phase_col])

    np.savetxt(
        testfile,
        data,
        fmt=existing_format,
        header=new_header + "\n" + line2,
        comments="",
    )
    # this errors because theta isn't regularly gridded (too few sig figs)
    with pytest.raises(
        ValueError, match="Data does not appear to be regularly gridded in zenith angle"
    ):
        beam1.read_cst_beam(
            testfile,
            beam_type="efield",
            frequency=np.array([150e6]),
            feed_pol="y",
            feed_angle=np.pi / 2,
            mount_type="fixed",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )

    # use more decimal places for theta so that it is regularly gridded
    new_format = [
        "%15.12e",
        "%15.3e",
        "%20.3e",
        "%19.3e",
        "%19.3f",
        "%19.3e",
        "%19.3f",
        "%19.3e",
    ]
    np.savetxt(
        testfile, data, fmt=new_format, header=new_header + "\n" + line2, comments=""
    )
    # this errors because phi isn't regularly gridded (too few sig figs)
    with pytest.raises(
        ValueError,
        match="Data does not appear to be regularly gridded in azimuth angle",
    ):
        beam1.read_cst_beam(
            testfile,
            beam_type="efield",
            frequency=np.array([150e6]),
            feed_pol="y",
            feed_angle=[0.0],
            mount_type="fixed",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )

    # use more decimal places so that it is regularly gridded and matches data
    new_format = [
        "%15.12e",
        "%15.12e",
        "%20.3e",
        "%19.3e",
        "%19.12f",
        "%19.3e",
        "%19.12f",
        "%19.3e",
    ]
    np.savetxt(
        testfile, data, fmt=new_format, header=new_header + "\n" + line2, comments=""
    )

    with check_warnings(
        UserWarning,
        [
            "Feed information not supplied and x-orientation not specified",
            "No frequency provided. Detected frequency is",
        ],
    ):
        beam1.read_cst_beam(
            cst_files[0],
            beam_type="efield",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )

    with check_warnings(
        UserWarning,
        [
            "Feed information not supplied and x-orientation not specified",
            "No frequency provided. Detected frequency is",
        ],
    ):
        beam2.read_cst_beam(
            testfile,
            beam_type="efield",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )

    assert beam1 == beam2

    # remove a row to make data not on a grid to catch that error
    data = data[1:, :]

    np.savetxt(
        testfile, data, fmt=new_format, header=new_header + "\n" + line2, comments=""
    )
    # this errors because theta & phi aren't on a strict grid
    with pytest.raises(ValueError, match="Data does not appear to be on a grid"):
        beam1.read_cst_beam(
            testfile,
            beam_type="efield",
            frequency=np.array([150e6]),
            feed_pol="y",
            feed_angle=0.0,
            mount_type="fixed",
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
        )


def test_wrong_column_names(tmp_path):
    # need to write modified files to test headers with wrong column names
    testfile = os.path.join(tmp_path, "HERA_NicCST_150MHz_modified.txt")
    with open(cst_files[0]) as file:
        line1 = file.readline()
        line2 = file.readline()

    data = np.loadtxt(cst_files[0], skiprows=2)

    raw_names = line1.split("]")
    raw_names = [raw_name for raw_name in raw_names if "\n" not in raw_name]
    column_names = []
    missing_power_column_names = []
    extra_power_column_names = []
    column_names_simple = []
    units = []
    for raw_name in raw_names:
        column_name, unit = tuple(raw_name.split("["))
        column_names.append(column_name)
        column_names_simple.append("".join(column_name.lower().split(" ")))
        units.append(unit)
        if column_name.strip() == "Abs(V   )":
            missing_power_column_names.append("Power")
        else:
            missing_power_column_names.append(column_name)
        if column_name.strip() == "Abs(Theta)":
            extra_power_column_names.append("Abs(E   )")
        else:
            extra_power_column_names.append(column_name)

    missing_power_column_headers = []
    for index, name in enumerate(missing_power_column_names):
        missing_power_column_headers.append(name + "[" + units[index] + "]")

    extra_power_column_headers = []
    for index, name in enumerate(extra_power_column_names):
        extra_power_column_headers.append(name + "[" + units[index] + "]")

    missing_power_header = ""
    for col in missing_power_column_headers:
        missing_power_header += f"{col:12}"

    beam1 = UVBeam()

    # format to match existing file
    existing_format = [
        "%8.3f",
        "%15.3f",
        "%20.3e",
        "%19.3e",
        "%19.3f",
        "%19.3e",
        "%19.3f",
        "%19.3e",
    ]
    np.savetxt(
        testfile,
        data,
        fmt=existing_format,
        header=missing_power_header + "\n" + line2,
        comments="",
    )
    # this errors because there's no recognized power column
    with pytest.raises(ValueError, match="No power column found in file:"):
        beam1.read_cst_beam(
            testfile,
            beam_type="power",
            frequency=np.array([150e6]),
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
            x_orientation="east",
        )

    extra_power_header = ""
    for col in extra_power_column_headers:
        extra_power_header += f"{col:12}"
    np.savetxt(
        testfile,
        data,
        fmt=existing_format,
        header=extra_power_header + "\n" + line2,
        comments="",
    )
    # this errors because there's multiple recognized power columns
    with pytest.raises(
        ValueError, match="Multiple possible power columns found in file:"
    ):
        beam1.read_cst_beam(
            testfile,
            beam_type="power",
            frequency=[150e6],
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            mount_type="fixed",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
            x_orientation="east",
        )


def test_hera_yaml():
    beam1 = UVBeam()
    beam2 = UVBeam()

    beam1.read_cst_beam(cst_yaml_vivaldi, beam_type="efield", frequency_select=[150e6])

    assert beam1.reference_impedance == 100
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }
    assert beam1.extra_keywords == extra_keywords

    beam2.read_cst_beam(cst_yaml_vivaldi, beam_type="power", frequency_select=[150e6])

    beam1.efield_to_power(calc_cross_pols=False)

    # The values in the beam file only have 4 sig figs, so they don't match precisely
    diff = np.abs(beam1.data_array - beam2.data_array)
    assert np.max(diff) < 2
    reldiff = diff / beam2.data_array
    assert np.max(reldiff) < 0.002

    # set data_array tolerances higher to test the rest of the object
    # tols are (relative, absolute)
    tols = [0.002, 0]
    beam1._data_array.tols = tols

    assert beam1.history != beam2.history
    beam1.history = beam2.history

    assert beam1 == beam2


@pytest.mark.parametrize(
    "kwargs,beam_type,err_msg",
    [
        [{"feed_array": ["y"]}, "efield", "Cannot set feed_array for efield beams"],
        [{"feed_array": np.ones((2, 2))}, "power", "feed_array cannot be a multi-dim"],
        [{"feed_angle": np.ones((2, 2))}, "power", "feed_angle cannot be a multi-dim"],
        [{"feed_angle": [0, 0]}, "efield", "feed_pol and feed_angle must contain"],
        [{"feed_angle": [0, 0]}, "power", "Must set either both or neither"],
        [
            {"feed_array": "y", "feed_angle": [0, 0]},
            "power",
            "feed_array and feed_angle must contain the same number",
        ],
    ],
)
def test_read_cst_feed_errors(kwargs, beam_type, err_msg):
    beam = UVBeam()
    with pytest.raises(ValueError, match=err_msg):
        beam.read_cst_beam(
            cst_files,
            beam_type=beam_type,
            frequency=[150e6, 123e6],
            feed_pol=np.array(["y"]),
            telescope_name="TEST",
            feed_name="bob",
            feed_version="0.1",
            model_name="E-field pattern - Rigging height 4.9m",
            model_version="1.0",
            **kwargs,
        )
