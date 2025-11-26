# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
import copy
import re

import numpy as np
import pytest
from astropy.io import fits

from pyuvdata import ShortDipoleBeam, UVBeam, utils
from pyuvdata.datasets import fetch_data
from pyuvdata.testing import check_warnings
from pyuvdata.uvbeam.mwa_beam import P1sin, P1sin_array


@pytest.fixture()
def mwabeam_kwargs(mwa_aee_files):
    return {
        "fee": {"pixels_per_deg": 1},
        "aee": {"zfile": mwa_aee_files["zfile"]},
        "aee_noxy": {
            "zfile": mwa_aee_files["zfile"],
            "include_cross_feed_coupling": False,
        },
    }


@pytest.fixture()
def uvbeam_kwargs(mwabeam_kwargs):
    uvbeam_kwargs = copy.deepcopy(mwabeam_kwargs)
    uvbeam_kwargs["aee"]["mwa_zfile"] = uvbeam_kwargs["aee"].pop("zfile")
    uvbeam_kwargs["aee_noxy"]["mwa_zfile"] = uvbeam_kwargs["aee_noxy"].pop("zfile")
    uvbeam_kwargs["aee_noxy"]["mwa_include_cross_feed_coupling"] = uvbeam_kwargs[
        "aee_noxy"
    ].pop("include_cross_feed_coupling")
    return uvbeam_kwargs


@pytest.fixture()
def beam_filenames(mwa_aee_files):
    return {
        "fee": fetch_data("mwa_full_EE"),
        "aee": mwa_aee_files["jfile"],
        "aee_noxy": mwa_aee_files["jfile"],
    }


@pytest.mark.parametrize("model", ["fee", "aee"])
def test_read_write_mwa(beam_filenames, mwabeam_kwargs, model, tmp_path):
    """Basic read/write test."""

    filename = beam_filenames[model]
    kwargs = mwabeam_kwargs[model]

    beam1 = UVBeam()
    beam2 = UVBeam()
    beam1.read_mwa_beam(filename, **kwargs)

    if model == "fee":
        assert beam1.filename == ["mwa_full_EE_test.h5"]
        assert beam1.data_array.shape == (2, 2, 3, 91, 360)
        # this is entirely empirical, just to prevent unexpected changes.
        # The actual values have been validated through external tests against
        # the mwa_pb repo.
        assert np.isclose(
            np.max(np.abs(beam1.data_array)),
            0.6823676193472403,
            rtol=beam1._data_array.tols[0],
            atol=beam1._data_array.tols[1],
        )
    else:
        assert beam1.filename == ["JMatrix_3freq.fits", "ZMatrix_3freq.fits"]
        assert beam1.data_array.shape == (2, 2, 3, 31, 121)

    assert beam1.pixel_coordinate_system == "az_za"
    assert beam1.beam_type == "efield"

    assert "x" in beam1.feed_array
    assert "y" in beam1.feed_array
    assert beam1.get_x_orientation_from_feeds() == "east"

    outfile_name = str(tmp_path / "mwa_beam_out.fits")
    beam1.write_beamfits(outfile_name, clobber=True)

    beam2.read_beamfits(outfile_name)

    assert beam1 == beam2


@pytest.mark.filterwarnings("ignore:There are some terminated dipoles")
@pytest.mark.parametrize("model", ["fee", "aee", "aee_noxy"])
def test_mwa_orientation(
    mwa_fee_1ppd, mwa_aee, mwa_aee_noxy, beam_filenames, uvbeam_kwargs, model
):
    if model == "fee":
        ebeam = mwa_fee_1ppd
        near_hor_za = 80
        near_zen_za = 2
    else:
        if model == "aee":
            ebeam = mwa_aee
        else:
            ebeam = mwa_aee_noxy
        near_hor_za = 81
        near_zen_za = 3
    power_beam = ebeam.efield_to_power(inplace=False)

    small_za_ind = np.nonzero(
        np.isclose(power_beam.axis2_array, near_zen_za * np.pi / 180)
    )

    east_az_ind = np.nonzero(np.isclose(power_beam.axis1_array, 0))
    north_az_ind = np.nonzero(np.isclose(power_beam.axis1_array, np.pi / 2))

    east_ind = np.nonzero(
        power_beam.polarization_array
        == utils.polstr2num(
            "ee", x_orientation=power_beam.get_x_orientation_from_feeds()
        )
    )[0]
    north_ind = np.nonzero(
        power_beam.polarization_array
        == utils.polstr2num(
            "nn", x_orientation=power_beam.get_x_orientation_from_feeds()
        )
    )[0]

    # check that the e/w dipole is more sensitive n/s
    assert (
        power_beam.data_array[0, east_ind, 0, small_za_ind, east_az_ind]
        < power_beam.data_array[0, east_ind, 0, small_za_ind, north_az_ind]
    )

    # check that the n/s dipole is more sensitive e/w
    assert (
        power_beam.data_array[0, north_ind, 0, small_za_ind, north_az_ind]
        < power_beam.data_array[0, north_ind, 0, small_za_ind, east_az_ind]
    )

    # check that for a single dipole (all others turned off) there is higher
    # azimuth-aligned response near the horizon than zenith angle-aligned response
    # for both feed orientations
    # this is true with all dipoles on too, but the difference is bigger for a
    # single dipole
    delays = np.full((2, 16), 32, dtype=int)
    delays[:, 5] = 0

    filename = beam_filenames[model]
    kwargs = uvbeam_kwargs[model]
    kwargs["delays"] = delays
    efield_beam = UVBeam.from_file(filename, **kwargs)

    large_za_val = np.nonzero(
        np.isclose(efield_beam.axis2_array, near_hor_za * np.pi / 180)
    )

    max_az_response = np.max(
        np.abs(efield_beam.data_array[0, east_ind, 0, large_za_val, :])
    )
    max_za_response = np.max(
        np.abs(efield_beam.data_array[1, east_ind, 0, large_za_val, :])
    )
    assert max_az_response > max_za_response

    max_az_response = np.max(
        np.abs(efield_beam.data_array[0, north_ind, 0, large_za_val, :])
    )
    max_za_response = np.max(
        np.abs(efield_beam.data_array[1, north_ind, 0, large_za_val, :])
    )
    assert max_az_response > max_za_response

    # go back to zenith pointed full tile beam
    # check the sign of the responses are as expected close to zenith
    efield_beam = ebeam
    small_za_ind = np.nonzero(
        np.isclose(power_beam.axis2_array, near_zen_za * np.pi / 180)
    )

    # first check zenith angle aligned response
    assert efield_beam.data_array[1, east_ind, 0, small_za_ind, east_az_ind] > 0
    assert efield_beam.data_array[1, north_ind, 0, small_za_ind, north_az_ind] > 0

    # then check azimuthal aligned response
    assert efield_beam.data_array[0, north_ind, 0, small_za_ind, east_az_ind] > 0
    assert efield_beam.data_array[0, east_ind, 0, small_za_ind, north_az_ind] < 0


@pytest.mark.parametrize(
    ("delay_set", "az_val", "za_range"),
    [
        (np.zeros(16), 0, [-0.1, 0.1]),
        (np.ones(16), 0, [-0.1, 0.1]),
        (np.tile(np.arange(0, 8, 2), 4), 0, [12, 15]),
        (np.tile((np.arange(0, 8, 2))[np.newaxis, :].T, 4).flatten(), 270, [12, 15]),
        (np.tile(np.flip(np.arange(0, 8, 2)), 4), 180, [12, 15]),
        (
            np.tile(np.flip(np.arange(0, 8, 2))[np.newaxis, :].T, 4).flatten(),
            90,
            [12, 15],
        ),
    ],
)
@pytest.mark.parametrize("model", ["fee", "aee", "aee_noxy"])
def test_mwa_pointing(
    delay_set, az_val, za_range, beam_filenames, uvbeam_kwargs, model
):
    # Test that pointing the beam moves the peak in the right direction.

    delays = np.empty((2, 16), dtype=int)

    for pol in range(2):
        delays[pol] = delay_set

    filename = beam_filenames[model]
    kwargs = uvbeam_kwargs[model]
    kwargs["delays"] = delays
    mwa_beam = UVBeam.from_file(filename, **kwargs)
    mwa_beam.efield_to_power(calc_cross_pols=False)

    # set up zenith angle, azimuth and frequency arrays to evaluate with
    # make a regular grid in direction cosines for nice plots
    n_vals = 100
    zmax = np.radians(90)  # Degrees
    axis_arr = np.arange(-n_vals / 2.0, n_vals / 2.0) / float(n_vals / 2.0)
    l_arr, m_arr = np.meshgrid(axis_arr, axis_arr)
    radius = np.sqrt(l_arr**2 + m_arr**2)
    za_array = radius * zmax
    az_array = np.arctan2(m_arr, l_arr)

    # Wrap the azimuth array to [0, 2pi] to match the extent of the UVBeam azimuth
    where_neg_az = np.nonzero(az_array < 0)
    az_array[where_neg_az] = az_array[where_neg_az] + np.pi * 2.0
    az_array = az_array.flatten()
    za_array = za_array.flatten()

    # find the values above the horizon so we don't try to interpolate the MWA beam
    # beyond the horizon
    above_hor = np.nonzero(za_array <= np.pi / 2.0)[0]
    az_array = az_array[above_hor]
    za_array = za_array[above_hor]

    # The MWA beam we have in our test data is small, it only has 3 frequencies,
    # so we will just get the value at one of those frequencies rather than
    # trying to interpolate to a new frequency.
    freqs = np.array([mwa_beam.freq_array[-1]])

    mwa_beam_vals, _ = mwa_beam.interp(
        az_array=az_array,
        za_array=za_array,
        freq_array=freqs,
        return_basis_vector=False,
    )
    mwa_beam_vals = np.squeeze(mwa_beam_vals)

    ee_power_vals = mwa_beam_vals[0]
    nn_power_vals = mwa_beam_vals[1]
    max_ee_loc = np.nonzero(ee_power_vals == np.max(ee_power_vals))
    max_nn_loc = np.nonzero(nn_power_vals == np.max(nn_power_vals))

    assert np.rad2deg(az_array[max_ee_loc]) == az_val
    assert np.rad2deg(az_array[max_nn_loc]) == az_val

    assert np.rad2deg(za_array[max_ee_loc]) > za_range[0]
    assert np.rad2deg(za_array[max_ee_loc]) < za_range[1]
    assert np.rad2deg(za_array[max_nn_loc]) > za_range[0]
    assert np.rad2deg(za_array[max_nn_loc]) < za_range[1]


def test_mwa_fhd_decompose(mwa_fee_1ppd):
    mwa_beam = mwa_fee_1ppd
    # select to above the horizon
    mwa_beam.select(axis2_inds=np.nonzero(mwa_beam.axis2_array <= np.pi / 2))

    small_za_ind = np.nonzero(np.isclose(mwa_beam.axis2_array, 2.0 * np.pi / 180))

    firesp, fproj = mwa_beam.decompose_feed_iresponse_projection()

    assert np.all(firesp.data_array[0, :, :, small_za_ind].real > 0)

    az_array, za_array = np.meshgrid(mwa_beam.axis1_array, mwa_beam.axis2_array)

    dipole_beam = ShortDipoleBeam()

    dipole_fproj = dipole_beam.feed_projection_eval(
        az_array=az_array.flatten(),
        za_array=za_array.flatten(),
        freq_array=np.asarray(np.asarray([mwa_beam.freq_array[-1]])),
    )
    dipole_fproj = dipole_fproj.reshape(2, 2, mwa_beam.Naxes2, mwa_beam.Naxes1)

    # MWA fproj is pretty similar to dipole_fproj up to mutual coupling effects
    # they are very similar near zenith
    fproj_diff = fproj.data_array[:, :, 0] - dipole_fproj

    np.testing.assert_allclose(
        fproj_diff[:, :, small_za_ind].real, 0, rtol=0, atol=2e-4
    )
    np.testing.assert_allclose(
        fproj_diff[:, :, small_za_ind].imag, 0, rtol=0, atol=2e-4
    )


@pytest.mark.parametrize("model", ["fee", "aee"])
def test_freq_range(mwa_fee_1ppd, mwa_aee, model, beam_filenames, mwabeam_kwargs):
    if model == "fee":
        beam1 = mwa_fee_1ppd
    else:
        beam1 = mwa_aee

    filename = beam_filenames[model]
    kwargs = mwabeam_kwargs[model]

    beam2 = UVBeam()

    # include all
    beam2.read_mwa_beam(filename, **kwargs, freq_range=[100e6, 200e6])
    assert beam1 == beam2

    beam2.read_mwa_beam(filename, **kwargs, freq_range=[100e6, 170e6])
    beam1.select(freq_chans=[0, 1])
    assert beam1.history != beam2.history
    beam1.history = beam2.history
    assert beam1 == beam2


def test_freq_range_errors():
    beam1 = UVBeam()

    with check_warnings(UserWarning, match="Only one available frequency"):
        beam1.read_mwa_beam(
            fetch_data("mwa_full_EE"), pixels_per_deg=1, freq_range=[100e6, 130e6]
        )

    with pytest.raises(ValueError, match="No frequencies available in freq_range"):
        beam1.read_mwa_beam(
            fetch_data("mwa_full_EE"), pixels_per_deg=1, freq_range=[100e6, 110e6]
        )

    with pytest.raises(ValueError, match="freq_range must have 2 elements."):
        beam1.read_mwa_beam(
            fetch_data("mwa_full_EE"), pixels_per_deg=1, freq_range=[100e6]
        )


def test_p1sin_array():
    pixels_per_deg = 5
    nmax = 10
    n_theta = np.floor(90 * pixels_per_deg) + 1
    theta_arr = np.deg2rad(np.arange(0, n_theta) / pixels_per_deg)
    (P_sin, P1) = P1sin_array(nmax, theta_arr)

    P_sin_orig = np.zeros((nmax**2 + 2 * nmax, np.size(theta_arr)))
    P1_orig = np.zeros((nmax**2 + 2 * nmax, np.size(theta_arr)))
    for theta_i, theta in enumerate(theta_arr):
        P_sin_temp, P1_temp = P1sin(nmax, theta)
        P_sin_orig[:, theta_i] = P_sin_temp
        P1_orig[:, theta_i] = P1_temp

    np.testing.assert_allclose(P1_orig, P1.T)
    np.testing.assert_allclose(P_sin_orig, P_sin.T)


def test_bad_amps():
    beam1 = UVBeam()

    amps = np.ones([2, 8])
    with pytest.raises(ValueError, match="amplitudes must be shape"):
        beam1.read_mwa_beam(
            fetch_data("mwa_full_EE"), pixels_per_deg=1, amplitudes=amps
        )


def test_bad_delays():
    beam1 = UVBeam()

    delays = np.zeros([2, 8], dtype="int")
    with pytest.raises(ValueError, match="delays must be shape"):
        beam1.read_mwa_beam(fetch_data("mwa_full_EE"), pixels_per_deg=1, delays=delays)

    delays = np.zeros((2, 16), dtype="int")
    delays = delays + 64
    with pytest.raises(ValueError, match="There are delays greater than 32"):
        beam1.read_mwa_beam(fetch_data("mwa_full_EE"), pixels_per_deg=1, delays=delays)

    delays = np.zeros((2, 16), dtype="float")
    with pytest.raises(ValueError, match="Delays must be integers."):
        beam1.read_mwa_beam(fetch_data("mwa_full_EE"), pixels_per_deg=1, delays=delays)


def test_dead_dipoles():
    beam1 = UVBeam()

    delays = np.zeros((2, 16), dtype="int")
    delays[:, 0] = 32

    with check_warnings(UserWarning, "There are some terminated dipoles"):
        beam1.read_mwa_beam(fetch_data("mwa_full_EE"), pixels_per_deg=1, delays=delays)

    delay_str = (
        "[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]"
    )
    gain_str = (
        "[[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "
        "1.0, 1.0, 1.0, 1.0, 1.0], "
        "[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "
        "1.0, 1.0, 1.0, 1.0]]"
    )
    history_str = (
        "Sujito et al. full embedded element beam, derived from "
        "https://github.com/MWATelescope/mwa_pb/"
        + "  delays set to "
        + delay_str
        + "  gains set to "
        + gain_str
        + beam1.pyuvdata_version_str
    )
    assert utils.history._check_histories(history_str, beam1.history)


@pytest.mark.parametrize(
    ("change", "errmsg"),
    [
        (
            "diff_exten",
            "model_type could not be determined for MWA beam file, use "
            "the model_type keyword to specify the type (one of 'fee' "
            "or 'aee'). Filename is",
        ),
        ("no_grid", "Data does not appear to be on a grid"),
        ("bad_th_reshape", "thetas do not appear to be on expected grid"),
        ("bad_ph_reshape", "phis do not appear to be on expected grid"),
        ("diff_th", "Inconsistent theta values across frequencies"),
        ("diff_ph", "Inconsistent phi values across frequencies"),
    ],
)
def test_aee_jfile_errors(tmp_path, mwa_aee_files, change, errmsg):

    if change == "diff_exten":
        testfile = str(tmp_path / "test_aee_j.foo")
    else:
        testfile = str(tmp_path / "test_aee_j.fits")

    with fits.open(mwa_aee_files["jfile"]) as jfile:
        first_hdu = jfile[0]
        data = first_hdu.data
        if change == "no_grid":
            # change the first theta value slightly
            data[0, 0] += 0.1
        elif change == "bad_th_reshape":
            # set the first theta to 2nd theta
            data[0, 0] = data[1, 0]
        elif change == "bad_ph_reshape":
            # set the first phi to 2nd phi
            data[0, 1] = data[100, 1]
        elif change == "diff_th":
            # set the first phi to 2nd phi
            data[:, 0] += 1
        elif change == "diff_ph":
            # set the first phi to 2nd phi
            data[:, 1] += 1

        first_hdu.data = data
        hdulist = [first_hdu]
        for nhdu in range(1, len(jfile)):
            hdulist.append(jfile[nhdu])

        hdulist = fits.HDUList(hdulist)

        hdulist.writeto(testfile, overwrite=True)
        hdulist.close()

    with pytest.raises(ValueError, match=re.escape(errmsg)):
        UVBeam.from_file(
            testfile, mwa_zfile=mwa_aee_files["zfile"], file_type="mwa_beam"
        )


@pytest.mark.parametrize(
    ("change", "errmsg"),
    [
        (
            "diff_nf",
            "Zmatrix file does not have as the same number of frequencies as "
            "Jmatrix file.",
        ),
        ("diff_f", "Zmatrix 0th freq does not match Jmatrix file."),
    ],
)
def test_aee_zfile_errors(tmp_path, mwa_aee_files, change, errmsg):

    testfile = str(tmp_path / "test_aee_z.fits")

    with fits.open(mwa_aee_files["zfile"]) as zfile:
        first_hdu = zfile[0]
        n_freqs = len(zfile)
        if change == "diff_nf":
            n_freqs = n_freqs - 1
        elif change == "diff_f":
            first_hdu.header["freq"] += 2

        hdulist = [first_hdu]
        for nhdu in range(1, n_freqs):
            hdulist.append(zfile[nhdu])

        hdulist = fits.HDUList(hdulist)

        hdulist.writeto(testfile, overwrite=True)
        hdulist.close()

    with pytest.raises(ValueError, match=errmsg):
        UVBeam.from_file(mwa_aee_files["jfile"], mwa_zfile=testfile)
