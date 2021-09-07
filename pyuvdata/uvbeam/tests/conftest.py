# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2021 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""pytest fixtures for UVBeam tests."""
import os

import pytest
import numpy as np

from pyuvdata.data import DATA_PATH
from pyuvdata import UVBeam

filenames = ["HERA_NicCST_150MHz.txt", "HERA_NicCST_123MHz.txt"]
cst_folder = "NicCSTbeams"
cst_files = [os.path.join(DATA_PATH, cst_folder, f) for f in filenames]


def make_cst_beam(beam_type):
    """Make the default CST testing beam."""
    extra_keywords = {
        "software": "CST 2016",
        "sim_type": "E-farfield",
        "layout": "1 antenna",
        "port_num": 1,
    }

    beam = UVBeam()
    beam.read_cst_beam(
        cst_files,
        beam_type=beam_type,
        frequency=[150e6, 123e6],
        telescope_name="HERA",
        feed_name="Dipole",
        feed_version="1.0",
        feed_pol=["x"],
        model_name="Dipole - Rigging height 4.9 m",
        model_version="1.0",
        x_orientation="east",
        reference_impedance=100,
        history=(
            "Derived from https://github.com/Nicolas-Fagnoni/Simulations."
            "\nOnly 2 files included to keep test data volume low."
        ),
        extra_keywords=extra_keywords,
    )
    return beam


def cut_beam(beam):
    """Downselect a beam to a small sky area to speed tests up."""
    za_max = np.deg2rad(10.0)
    za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
    beam.select(axis2_inds=za_inds_use)
    return beam


def single_freq_version(beam):
    """Make a single freq version with expected history."""
    history_use = beam.history[: beam.history.find(" Combined data")]
    beam.select(freq_chans=1)
    beam.filename = [beam.filename[1]]
    beam._filename.form = (1,)
    beam.history = history_use
    return beam


@pytest.fixture(scope="session")
def cst_efield_2freq_main():
    """Make session level 2-freq efield beam."""
    return make_cst_beam("efield")


@pytest.fixture(scope="function")
def cst_efield_2freq(cst_efield_2freq_main):
    """Make function level 2-freq efield beam."""
    return cst_efield_2freq_main.copy()


@pytest.fixture(scope="session")
def cst_efield_2freq_cut_main(cst_efield_2freq_main):
    """Make session level cut down 2-freq efield beam."""
    return cut_beam(cst_efield_2freq_main.copy())


@pytest.fixture(scope="function")
def cst_efield_2freq_cut(cst_efield_2freq_cut_main):
    """Make function level cut down 2-freq efield beam."""
    return cst_efield_2freq_cut_main.copy()


@pytest.fixture(scope="session")
def cst_efield_2freq_cut_healpix_main(cst_efield_2freq_cut_main):
    """Make session level cut down HEALPix 2-freq efield beam."""
    pytest.importorskip("astropy_healpix")
    beam = cst_efield_2freq_cut_main.copy()
    beam.interpolation_function = "az_za_simple"
    beam.to_healpix()
    return beam


@pytest.fixture(scope="function")
def cst_efield_2freq_cut_healpix(cst_efield_2freq_cut_healpix_main):
    """Make function level cut down HEALPix 2-freq efield beam."""
    return cst_efield_2freq_cut_healpix_main.copy()


@pytest.fixture(scope="session")
def cst_efield_1freq_main(cst_efield_2freq_main):
    """Make session level single freq efield beam."""
    return single_freq_version(cst_efield_2freq_main.copy())


@pytest.fixture(scope="function")
def cst_efield_1freq(cst_efield_1freq_main):
    """Make function level single freq efield beam."""
    return cst_efield_1freq_main.copy()


@pytest.fixture(scope="session")
def cst_efield_1freq_cut_main(cst_efield_2freq_cut_main):
    """Make session level cut down single freq efield beam."""
    return single_freq_version(cst_efield_2freq_cut_main.copy())


@pytest.fixture(scope="function")
def cst_efield_1freq_cut(cst_efield_1freq_cut_main):
    """Make function level cut down single freq efield beam."""
    return cst_efield_1freq_cut_main.copy()


@pytest.fixture(scope="session")
def cst_efield_1freq_cut_healpix_main(cst_efield_2freq_cut_healpix_main):
    """Make session level HEALPix cut down single freq efield beam."""
    return single_freq_version(cst_efield_2freq_cut_healpix_main.copy())


@pytest.fixture(scope="function")
def cst_efield_1freq_cut_healpix(cst_efield_1freq_cut_healpix_main):
    """Make function level HEALPix cut down single freq efield beam."""
    return cst_efield_1freq_cut_healpix_main.copy()


@pytest.fixture(scope="session")
def cst_power_2freq_main():
    """Make session level 2-freq power beam."""
    return make_cst_beam("power")


@pytest.fixture(scope="function")
def cst_power_2freq(cst_power_2freq_main):
    """Make function level 2-freq efield beam."""
    return cst_power_2freq_main.copy()


@pytest.fixture(scope="session")
def cst_power_2freq_cut_main(cst_power_2freq_main):
    """Make session level cut down 2-freq power beam."""
    return cut_beam(cst_power_2freq_main.copy())


@pytest.fixture(scope="function")
def cst_power_2freq_cut(cst_power_2freq_cut_main):
    """Make function level cut down 2-freq power beam."""
    return cst_power_2freq_cut_main.copy()


@pytest.fixture(scope="session")
def cst_power_2freq_cut_healpix_main(cst_power_2freq_cut_main):
    """Make session level HEALPix cut down 2-freq power beam."""
    pytest.importorskip("astropy_healpix")
    beam = cst_power_2freq_cut_main.copy()
    beam.interpolation_function = "az_za_simple"
    beam.to_healpix()
    return beam


@pytest.fixture(scope="function")
def cst_power_2freq_cut_healpix(cst_power_2freq_cut_healpix_main):
    """Make function level HEALPix cut down 2-freq power beam."""
    return cst_power_2freq_cut_healpix_main.copy()


@pytest.fixture(scope="session")
def cst_power_1freq_main(cst_power_2freq_main):
    """Make session level single freq power beam."""
    return single_freq_version(cst_power_2freq_main.copy())


@pytest.fixture(scope="function")
def cst_power_1freq(cst_power_1freq_main):
    """Make function level single freq power beam."""
    return cst_power_1freq_main.copy()


@pytest.fixture(scope="session")
def cst_power_1freq_cut_main(cst_power_2freq_cut_main):
    """Make session level cut down single freq power beam."""
    return single_freq_version(cst_power_2freq_cut_main.copy())


@pytest.fixture(scope="function")
def cst_power_1freq_cut(cst_power_1freq_cut_main):
    """Make function level cut down single freq power beam."""
    return cst_power_1freq_cut_main.copy()


@pytest.fixture(scope="session")
def cst_power_1freq_cut_healpix_main(cst_power_2freq_cut_healpix_main):
    """Make session level HEALPix cut down single freq power beam."""
    return single_freq_version(cst_power_2freq_cut_healpix_main.copy())


@pytest.fixture(scope="function")
def cst_power_1freq_cut_healpix(cst_power_1freq_cut_healpix_main):
    """Make function level HEALPix cut down single freq power beam."""
    return cst_power_1freq_cut_healpix_main.copy()
