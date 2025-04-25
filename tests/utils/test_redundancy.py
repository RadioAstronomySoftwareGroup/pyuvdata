# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Tests for baseline redundancy utility functions."""

import copy
import os
import re

import numpy as np
import pytest

import pyuvdata.utils.redundancy as red_utils
from pyuvdata import UVData, utils
from pyuvdata.data import DATA_PATH
from pyuvdata.testing import check_warnings


@pytest.mark.parametrize("grid_alg", [True, False])
def test_redundancy_finder(grid_alg):
    """
    Check that get_baseline_redundancies and get_antenna_redundancies return consistent
    redundant groups for a test file with the HERA19 layout.
    """
    uvd = UVData()
    uvd.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    uvd.select(times=uvd.time_array[0])
    uvd.unproject_phase(use_ant_pos=True)
    # uvw_array is now equivalent to baseline positions
    uvd.conjugate_bls("ant1<ant2", use_enu=True)

    tol = 0.05  # meters

    bl_positions = uvd.uvw_array
    bl_pos_backup = copy.deepcopy(uvd.uvw_array)

    warn_str = (
        "The include_conjugates parameter is not set. The default is "
        "currently False, which produces different groups than the groups "
        "produced when using the `compress_by_redundancy` method. "
        "The default will change to True in version 3.4."
    )
    warn_type = DeprecationWarning

    with (
        pytest.raises(
            ValueError, match=re.escape("Baseline vectors must be shape (Nbls, 3)")
        ),
        check_warnings(warn_type, match=warn_str),
    ):
        red_utils.get_baseline_redundancies(
            uvd.baseline_array, bl_positions[0:2, 0:1], use_grid_alg=grid_alg
        )

    with check_warnings(warn_type, match=warn_str):
        baseline_groups, vec_bin_centers, lens = red_utils.get_baseline_redundancies(
            uvd.baseline_array, bl_positions, tol=tol, use_grid_alg=grid_alg
        )

    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            np.testing.assert_allclose(
                np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol, rtol=0
            )

    # Shift the baselines around in a circle. Check that the same baselines are
    # recovered to the corresponding tolerance increase.
    # This moves one baseline at a time by a fixed displacement and checks that
    # the redundant groups are the same.

    hightol = 0.25  # meters. Less than the smallest baseline in the file.
    tol_use = hightol
    if grid_alg in [None, True]:
        tol_use = hightol * 4
    Nbls = uvd.Nbls
    Nshifts = 5
    shift_angs = np.linspace(0, 2 * np.pi, Nshifts)
    base_shifts = np.stack(
        (
            (hightol - tol) * np.cos(shift_angs),
            (hightol - tol) * np.sin(shift_angs),
            np.zeros(Nshifts),
        )
    ).T
    for sh in base_shifts:
        for bi in range(Nbls):
            # Shift one baseline at a time.
            bl_positions_new = uvd.uvw_array
            bl_positions_new[bi] += sh

            with check_warnings(warn_type, match=warn_str):
                (baseline_groups_new, vec_bin_centers, lens) = (
                    red_utils.get_baseline_redundancies(
                        uvd.baseline_array,
                        bl_positions_new,
                        tol=tol_use,
                        use_grid_alg=grid_alg,
                    )
                )

            for gi, gp in enumerate(baseline_groups_new):
                for bl in gp:
                    bl_ind = np.where(uvd.baseline_array == bl)
                    bl_vec = bl_positions[bl_ind]
                    np.testing.assert_allclose(
                        np.sqrt(np.abs(np.dot(bl_vec, vec_bin_centers[gi]))),
                        lens[gi],
                        atol=tol_use,
                        rtol=0,
                    )

            # Compare baseline groups:
            a = [tuple(el) for el in baseline_groups]
            b = [tuple(el) for el in baseline_groups_new]
            assert set(a) == set(b)

    tol = 0.05

    antpos = uvd.telescope.get_enu_antpos()

    baseline_groups_ants, vec_bin_centers, lens = red_utils.get_antenna_redundancies(
        uvd.telescope.antenna_numbers,
        antpos,
        tol=tol,
        include_autos=False,
        use_grid_alg=grid_alg,
    )
    # Under these conditions, should see 19 redundant groups in the file.
    assert len(baseline_groups_ants) == 19

    # Check with conjugated baseline redundancies returned
    # Ensure at least one baseline has u==0 and v!=0 (for coverage of this case)
    bl_positions[16, 0] = 0
    (baseline_groups, vec_bin_centers, lens, conjugates) = (
        red_utils.get_baseline_redundancies(
            uvd.baseline_array,
            bl_positions,
            tol=tol,
            include_conjugates=True,
            use_grid_alg=grid_alg,
        )
    )

    # restore baseline (16,0) and repeat to get correct groups
    bl_positions = bl_pos_backup
    (baseline_groups, vec_bin_centers, lens, conjugates) = (
        red_utils.get_baseline_redundancies(
            uvd.baseline_array,
            bl_positions,
            tol=tol,
            include_conjugates=True,
            use_grid_alg=grid_alg,
        )
    )

    # Apply flips to compare with get_antenna_redundancies().
    bl_gps_unconj = copy.deepcopy(baseline_groups)
    for gi, gp in enumerate(bl_gps_unconj):
        for bi, bl in enumerate(gp):
            if bl in conjugates:
                bl_gps_unconj[gi][bi] = utils.redundancy.baseline_index_flip(
                    bl, Nants_telescope=uvd.telescope.Nants
                )
    bl_gps_unconj = [sorted(bgp) for bgp in bl_gps_unconj]
    bl_gps_ants = [sorted(bgp) for bgp in baseline_groups_ants]
    assert np.all(sorted(bl_gps_ants) == sorted(bl_gps_unconj))
    for gi, gp in enumerate(baseline_groups):
        for bl in gp:
            bl_ind = np.where(uvd.baseline_array == bl)
            bl_vec = bl_positions[bl_ind]
            if bl in conjugates:
                bl_vec *= -1
            assert np.isclose(
                np.sqrt(np.dot(bl_vec, vec_bin_centers[gi])), lens[gi], atol=tol
            )


def test_high_tolerance_redundancy_error():
    """
    Confirm that an error is raised if the redundancy tolerance is set too high,
    such that baselines end up in multiple groups
    """
    uvd = UVData()
    uvd.read_uvfits(
        os.path.join(DATA_PATH, "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits")
    )

    uvd.select(times=uvd.time_array[0])
    uvd.unproject_phase(use_ant_pos=True)
    # uvw_array is now equivalent to baseline positions
    uvd.conjugate_bls("ant1<ant2", use_enu=True)
    bl_positions = uvd.uvw_array

    tol = 20.05  # meters

    with pytest.raises(ValueError, match="Some baselines are falling into"):
        red_utils.get_baseline_redundancies(
            uvd.baseline_array,
            bl_positions,
            tol=tol,
            include_conjugates=True,
            use_grid_alg=False,
        )


@pytest.mark.parametrize("grid_alg", [True, False])
def test_redundancy_conjugates(grid_alg):
    """
    Check that redundancy finding with conjugation works.

    Check that the correct baselines are flipped.
    """
    Nants = 10
    tol = 0.5
    ant1_arr = np.tile(np.arange(Nants), Nants)
    ant2_arr = np.repeat(np.arange(Nants), Nants)
    Nbls = ant1_arr.size
    bl_inds = utils.antnums_to_baseline(ant1_arr, ant2_arr, Nants_telescope=Nants)

    maxbl = 100.0
    bl_vecs = np.random.uniform(-maxbl, maxbl, (Nbls, 3))
    bl_vecs[0, 0] = 0
    bl_vecs[1, 0:2] = 0

    expected_conjugates = []
    for i, (u, v, w) in enumerate(bl_vecs):
        uneg = u < -tol
        uzer = np.isclose(u, 0.0, atol=tol)
        vneg = v < -tol
        vzer = np.isclose(v, 0.0, atol=tol)
        wneg = w < -tol
        if uneg or (uzer and vneg) or (uzer and vzer and wneg):
            expected_conjugates.append(bl_inds[i])
    _, _, _, conjugates = red_utils.get_baseline_redundancies(
        bl_inds, bl_vecs, tol=tol, include_conjugates=True, use_grid_alg=grid_alg
    )

    assert sorted(conjugates) == sorted(expected_conjugates)


@pytest.mark.parametrize("grid_alg", [True, False])
def test_redundancy_finder_fully_redundant_array(grid_alg):
    """Test the redundancy finder for a fully redundant array."""
    uvd = UVData()
    uvd.read_uvfits(os.path.join(DATA_PATH, "test_redundant_array.uvfits"))
    uvd.select(times=uvd.time_array[0])

    tol = 1  # meters
    bl_positions = uvd.uvw_array

    baseline_groups, _, _, _ = red_utils.get_baseline_redundancies(
        uvd.baseline_array,
        bl_positions,
        tol=tol,
        include_conjugates=True,
        use_grid_alg=grid_alg,
    )

    # Only 1 set of redundant baselines
    assert len(baseline_groups) == 1
    #  Should return the input baselines
    assert baseline_groups[0].sort() == np.unique(uvd.baseline_array).sort()


@pytest.mark.parametrize("n_blocks", [1, 10])
def test_adjacency_lists(n_blocks):
    """Test the adjacency list method in utils."""
    # n_blocks: in _adj_list, loop over chunks of vectors when computing distances.

    # Make a grid.
    Nx = 5
    Lmax = 50

    xbase = np.linspace(0, Lmax, Nx)
    x, y, z = map(np.ndarray.flatten, np.meshgrid(xbase, xbase, xbase))

    # Make more vectors by shifting by Lmax/Nx/3 in x, y, and z:
    dx = (Lmax / Nx) / 3  # One third of cell size.
    x = np.append(x, x + dx)
    y = np.append(y, y + dx)
    z = np.append(z, z + dx)

    # Construct vectors
    vecs = np.vstack((x, y, z)).T
    Npts = x.size

    # Reorder randomly.
    np.random.shuffle(vecs)

    # Tolerance = half of cell diagonal.
    tol = Lmax / Nx * np.sqrt(2) / 2

    adj = red_utils._adj_list(vecs, tol, n_blocks=n_blocks)

    # Confirm that each adjacency set contains all of the vectors that
    # are within the tolerance distance.
    for vi in range(Npts):
        for vj in range(Npts):
            dist = np.linalg.norm(vecs[vi] - vecs[vj])
            if dist < tol:
                assert vj in adj[vi]
                assert vi in adj[vj]
            else:
                assert vj not in adj[vi]
                assert vi not in adj[vj]

    # The way the grid is set up, every clique should have two elements.
    assert all(len(vi) == 2 for vi in adj)


def test_strict_cliques():
    # Adjacency lists comprising only isolated cliques.
    adj_isol = [
        {0, 1, 2},
        {1, 0, 2},
        {2, 0, 1},
        {3},
        {4},
        {5, 6, 7, 8},
        {5, 6, 7, 8},
        {5, 6, 7, 8},
        {5, 6, 7, 8},
    ]
    adj_isol = [frozenset(st) for st in adj_isol]
    exp_cliques = [[0, 1, 2], [3], [4], [5, 6, 7, 8]]

    res = red_utils._find_cliques(adj_isol, strict=True)
    assert res == exp_cliques

    # Error if two cliques are not isolated
    adj_link = adj_isol
    adj_link[-1] = frozenset({5, 6, 7, 8, 1})

    with pytest.raises(ValueError, match="Non-isolated cliques found in graph."):
        red_utils._find_cliques(adj_link, strict=True)


@pytest.mark.parametrize("grid_alg", [True, False])
def test_upos_tol_reds(grid_alg):
    # Checks that the u-positive convention in get_antenna_redundancies
    # is enforced to the specificed tolerance.

    # Make a layout with two NS baselines, one with u ~ -2*eps, and another with u == 0
    # This would previously cause one to be flipped, when they should be redundant.

    eps = 1e-5
    tol = 3 * eps
    if grid_alg:
        tol = tol * 2

    ant_pos = np.array(
        [[-eps, 1.0, 0.0], [1.0, 1.0, 0.0], [eps, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    ant_nums = np.arange(4)

    red_grps, _, _ = utils.redundancy.get_antenna_redundancies(
        ant_nums, ant_pos, tol=tol, use_grid_alg=grid_alg
    )

    assert len(red_grps) == 4
