# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import numpy as np
import pytest

from pyuvdata import ShortDipoleBeam
from pyuvdata.utils.plotting import get_az_za_grid, plot_beam_arrays


def test_plot_arrays(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!

    dipole_beam = ShortDipoleBeam()

    az_grid, za_grid = get_az_za_grid()
    az_array, za_array = np.meshgrid(az_grid, za_grid)

    beam_vals = dipole_beam.efield_eval(
        az_array=az_array.flatten(),
        za_array=za_array.flatten(),
        freq_array=np.asarray(np.asarray([100e6])),
    )
    beam_vals = beam_vals.reshape(2, 2, za_grid.size, az_grid.size)

    savefile = str(tmp_path / "test.png")

    plot_beam_arrays(
        beam_vals,
        az_array,
        za_array,
        complex_type="real",
        beam_type_label="E-field",
        beam_name="short dipole",
        savefile=savefile,
    )


def test_plot_arrays_errors():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!

    dipole_beam = ShortDipoleBeam()

    az_grid, za_grid = get_az_za_grid()
    az_array, za_array = np.meshgrid(az_grid, za_grid)

    beam_vals = dipole_beam.efield_eval(
        az_array=az_array.flatten(),
        za_array=za_array.flatten(),
        freq_array=np.asarray(np.asarray([100e6])),
    )

    with pytest.raises(
        ValueError,
        match="az_array and za_array must be shaped like the last dimension "
        "of beam_vals for irregular beam_vals",
    ):
        plot_beam_arrays(beam_vals[:, :, 0], az_array, za_array)

    beam_vals = beam_vals.reshape(2, 2, za_grid.size, az_grid.size)

    with pytest.raises(ValueError, match="beam_vals must be 3 or 4 dimensional."):
        plot_beam_arrays(beam_vals[0, 0], az_array, za_array)

    with pytest.raises(
        ValueError,
        match="feedpol_label must have the same number of elements as the 1st",
    ):
        plot_beam_arrays(beam_vals, az_array, za_array, feedpol_label=["foo"])

    with pytest.raises(
        ValueError,
        match="az_array and za_array must be shaped like the last 2 dimensions "
        "of beam_vals for regularly gridded beam_vals",
    ):
        plot_beam_arrays(beam_vals, az_array[0], za_array)
