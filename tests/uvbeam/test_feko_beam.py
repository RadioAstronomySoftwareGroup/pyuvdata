import os

import numpy as np

from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH

filename = ["OVRO_LWA_x.ffe"]
feko_folder = "OVRO_LWA_FEKOBeams"
feko_filename = os.path.join(DATA_PATH, feko_folder, filename)


def test_read_power():
    beam1 = UVBeam()

    beam_feko1 = beam1.from_file(
        feko_filename,
        beam_type="power",
        frequency=None,
        feed_pol="x",
        telescope_name="LWA",
        feed_name="LWA",
        feed_version="1",
        model_name="FEKO_MROsoil",
        model_version="1.0",
    )

    assert beam_feko1.beam_type == "power"
    assert beam_feko1.data_array.shape == (2, 1, 2, 3, 181, 360)
    assert len(beam_feko1.freq_array[0]) == 3

    assert np.allclose(
        beam_feko1.data_array[:, 0, :, :, 0, np.where(beam_feko1.axis1_array == 0)[0]],
        beam_feko1.data_array[
            :, 0, :, :, 0, np.where(beam_feko1.axis1_array == np.pi / 2.0)[0]
        ],
    )
