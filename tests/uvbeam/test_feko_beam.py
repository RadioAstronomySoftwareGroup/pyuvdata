import os

import numpy as np
import pytest

from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH

filename_x = "OVRO_LWA_x.ffe"
filename_y = "OVRO_LWA_y.ffe"
feko_folder = "OVRO_LWA_FEKOBeams"
feko_filename_x = os.path.join(DATA_PATH, feko_folder, filename_x)
feko_filename_y = os.path.join(DATA_PATH, feko_folder, filename_y)


@pytest.mark.parametrize(("btype"), [("power"), ("efield")])
def test_read_feko(btype):
    beam1 = UVBeam()
    beam2 = UVBeam()

    if btype == "efield":
        reference_impedance = 50
        extra_keywords = {"foo": "bar"}
    else:
        reference_impedance = None
        extra_keywords = None

    beam_feko1 = beam1.from_file(
        feko_filename_x,
        beam_type=btype,
        frequency=[10e6],
        feed_pol=None,
        telescope_name="LWA",
        feed_name="LWA",
        feed_version="1",
        model_name="FEKO_MROsoil",
        model_version="1.0",
        mount_type="fixed",
        feed_angle=np.pi / 2,  # E/W
        reference_impedance=reference_impedance,
        extra_keywords=extra_keywords,
    )

    beam_feko2 = beam2.from_file(
        feko_filename_y,
        beam_type=btype,
        frequency=np.array([10e6]),
        feed_pol=np.array(["y"]),
        telescope_name="LWA",
        feed_name="LWA",
        feed_version="1",
        model_name="FEKO_MROsoil",
        model_version="1.0",
        mount_type="fixed",
        feed_angle=0.0,  # N/S
        reference_impedance=reference_impedance,
        extra_keywords=extra_keywords,
    )
    if btype == "power":
        assert beam_feko1.beam_type == "power"
        assert beam_feko2.beam_type == "power"
        assert np.allclose(
            beam_feko1.data_array[0, :, :, 0, np.where(beam_feko1.axis1_array == 0)[0]],
            beam_feko1.data_array[
                0, :, :, 0, np.where(beam_feko1.axis1_array == np.pi / 2.0)[0]
            ],
        )
        assert beam_feko1.data_array.shape == (1, 1, 3, 181, 181)
    else:
        assert beam_feko1.beam_type == "efield"
        assert beam_feko2.beam_type == "efield"
        assert beam_feko1.data_array.shape == (2, 1, 3, 181, 181)
        assert beam_feko2.data_array.shape == beam_feko1.data_array.shape

        assert beam_feko1.reference_impedance == 50
        assert beam_feko1.extra_keywords == {"foo": "bar"}

    assert len(beam_feko1.freq_array) == 3
    assert len(beam_feko1.freq_array) == len(beam_feko2.freq_array)
    assert np.all(beam_feko1.bandpass_array) == 1

    feko_beam_multi = beam_feko1 + beam_feko2

    feko_beam_multi2 = UVBeam.from_file(
        [feko_filename_x, feko_filename_y],
        beam_type=btype,
        frequency=[10e6],
        feed_pol=["x", "y"],
        feed_angle=[np.pi / 2, 0.0],
        telescope_name="LWA",
        feed_name="LWA",
        feed_version="1",
        model_name="FEKO_MROsoil",
        model_version="1.0",
        mount_type="fixed",
        reference_impedance=reference_impedance,
        extra_keywords=extra_keywords,
    )

    assert feko_beam_multi == feko_beam_multi2


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        (
            {"feed_pol": np.array([["x"]])},
            "feed_pol can not be a multi-dimensional array",
        ),
        ({"feed_pol": ["x", "y"]}, "feed_pol must have exactly one element"),
        (
            {"filename": [filename_x, filename_y]},
            "If multiple FEKO files are passed, the feed_pol must be a list "
            "or array of the same length giving the feed_pol for each file.",
        ),
        (
            {
                "filename": [filename_x, filename_y],
                "feed_pol": ["x", "y"],
                "feed_angle": np.pi / 2,
            },
            "If multiple FEKO files are passed, the feed_angle must be a list or "
            "array of the same length giving the feed_angle for each file.",
        ),
    ],
)
def test_read_feko_input_errors(kwargs, msg):
    init_kwargs = {"filename": filename_x, "beam_type": "power", "feed_pol": "x"}
    init_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        UVBeam.from_file(**init_kwargs)


@pytest.mark.parametrize(
    ("error_type", "msg"),
    [
        ("grid", "Data does not appear to be on a grid"),
        ("zen_grid", "Data does not appear to be regularly gridded in zenith angle"),
        ("az_grid", "Data does not appear to be regularly gridded in azimuth angle"),
    ],
)
def test_read_feko_file_errors(tmp_path, error_type, msg):
    # read in file into list, modify it to trigger errors, write it back out.
    new_file = os.path.join(tmp_path, filename_x)

    with open(feko_filename_x) as file:
        # read a list of lines into data
        data = file.readlines()

    if error_type == "grid":
        data[10] = (
            "                    0.50000000E+0     0.00000000E+0    -1.51385203E-7     "
            "3.78444333E-8    -4.30077968E-3    -5.07565385E-3    -9.61862064E+1    "
            "-3.59112629E+0    -3.59112629E+0   \n"
        )
    elif error_type == "zen_grid":
        for li, line in enumerate(data):
            if line.startswith("                    0.0"):
                data[li] = line.replace(
                    "                    0.0", "                    0.5"
                )
    elif error_type == "az_grid":
        for li, line in enumerate(data):
            if line[38:].startswith("0.0"):
                data[li] = line[:38] + line[38:].replace("0.0", "0.5")

    with open(new_file, "w") as file:
        file.writelines(data)

    with pytest.raises(ValueError, match=msg):
        UVBeam.from_file(new_file, beam_type="efield")
