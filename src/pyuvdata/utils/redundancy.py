# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for working with redundant baselines."""

import warnings
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist

from .bls import antnums_to_baseline, baseline_index_flip


def _adj_list(vecs, tol, n_blocks=None):
    """Identify neighbors of each vec in vecs, to distance tol."""
    n_items = len(vecs)
    max_items = 2**10  # Max array size used is max_items**2. Avoid using > 1 GiB

    if n_blocks is None:
        n_blocks = max(n_items // max_items, 1)

    # We may sort blocks so that some pairs of blocks may be skipped.
    # Reorder vectors by x.

    order = np.argsort(vecs[:, 0])
    blocks = np.array_split(order, n_blocks)
    adj = [{k} for k in range(n_items)]  # Adjacency lists
    for b1 in blocks:
        for b2 in blocks:
            v1, v2 = vecs[b1], vecs[b2]
            # Check for no overlap, with tolerance.
            xmin1 = v1[0, 0] - tol
            xmax1 = v1[-1, 0] + tol
            xmin2 = v2[0, 0] - tol
            xmax2 = v2[-1, 0] + tol
            if max(xmin1, xmin2) > min(xmax1, xmax2):
                continue

            adj_mat = cdist(vecs[b1], vecs[b2]) < tol
            for bi, col in enumerate(adj_mat):
                adj[b1[bi]] = adj[b1[bi]].union(b2[col])
    return [frozenset(g) for g in adj]


def _find_cliques(adj, strict=False):
    n_items = len(adj)

    loc_gps = []
    visited = np.zeros(n_items, dtype=bool)
    for k in range(n_items):
        if visited[k]:
            continue
        a0 = adj[k]
        visited[k] = True
        if all(adj[it].__hash__() == a0.__hash__() for it in a0):
            group = list(a0)
            group.sort()
            visited[list(a0)] = True
            loc_gps.append(group)

    # Require all adjacency lists to be isolated maximal cliques:
    if strict and not all(sorted(st) in loc_gps for st in adj):
        raise ValueError("Non-isolated cliques found in graph.")

    return loc_gps


def find_clusters(*, location_ids, location_vectors, tol, strict=False):
    """
    Find clusters of vectors (e.g. redundant baselines, times).

    Parameters
    ----------
    location_ids : array_like of int
        ID labels for locations.
    location_vectors : array_like of float
        location vectors, can be multidimensional
    tol : float
        tolerance for clusters
    strict : bool
        Require that all adjacency lists be isolated maximal cliques.
        This ensures that vectors do not fall into multiple clusters.
        Default: False

    Returns
    -------
    list of list of location_ids

    """
    location_vectors = np.asarray(location_vectors)
    location_ids = np.asarray(location_ids)
    if location_vectors.ndim == 1:
        location_vectors = location_vectors[:, np.newaxis]

    adj = _adj_list(location_vectors, tol)  # adj = list of sets

    loc_gps = _find_cliques(adj, strict=strict)
    loc_gps = [np.sort(location_ids[gp]).tolist() for gp in loc_gps]
    return loc_gps


def find_clusters_grid(location_ids, location_vectors, tol=1.0):
    """
    Find redundant groups using a gridding algorithm developed by the HERA team.

    This is essentially a gridding approach, but it only keeps track of the grid
    points that have baselines assigned to them. It iterates through the
    baselines and assigns each baseline to a an existing group if it is within
    a grid spacing or makes a new group if there is no group. The location of
    the group is the baseline vector of the first baseline assigned to it, rounded
    to the grid spacing, so the resulting assigned grid point can depend on the
    order in which baseline vectors are passed to it. It is possible for a baseline
    to be assigned to a group that is up to but strictly less than 4 times the
    grid spacing from its true location, so we use a grid a factor of 4 smaller
    than the passed tolerance (`tol`). This method is quite robust for regular
    arrays if the tolerance is properly specified, but may not behave predictably
    for highly non-redundant arrays.

    Parameters
    ----------
    baselines : array_like of int
        Baseline numbers, shape (Nbls,)
    baseline_vecs : array_like of float
        Baseline vectors in meters, shape (Nbls, 3).
    tol : float
        Absolute tolerance of redundancy, in meters.

    Returns
    -------
    baseline_groups : list of lists of int
        list of lists of redundant baseline numbers
    baseline_ind_conj : list of int
        List of baselines that are redundant when reversed. Only returned if
        include_conjugates is True

    """
    bl_gps = {}
    # reduce the grid size to ensure baselines won't be assigned to a group
    # more than the tol away from their location. The factor of 4 is a personal
    # communication from Josh Dillon who developed this algorithm.
    grid_size = tol / 4.0

    p_or_m = (0, -1, 1)
    epsilons = [[dx, dy, dz] for dx in p_or_m for dy in p_or_m for dz in p_or_m]

    def check_neighbors(delta):
        # Check to make sure bl_gps doesn't have the key plus or minus rounding error
        for epsilon in epsilons:
            newKey = (
                delta[0] + epsilon[0],
                delta[1] + epsilon[1],
                delta[2] + epsilon[2],
            )
            if newKey in bl_gps:
                return newKey
        return

    baseline_ind_conj = []
    for bl_i, bl in enumerate(location_ids):
        delta = tuple(np.round(location_vectors[bl_i] / grid_size).astype(int))
        new_key = check_neighbors(delta)
        if new_key is not None:
            # this has a match
            bl_gps[new_key].append(bl)
        else:
            # this is a new group
            bl_gps[delta] = [bl]

    bl_list = [sorted(gv) for gv in bl_gps.values()]

    return bl_list, baseline_ind_conj


def get_baseline_redundancies(
    baselines, baseline_vecs, *, tol=1.0, include_conjugates=None, use_grid_alg=True
):
    """
    Find redundant baseline groups.

    Parameters
    ----------
    baselines : array_like of int
        Baseline numbers, shape (Nbls,)
    baseline_vecs : array_like of float
        Baseline vectors in meters, shape (Nbls, 3).
    tol : float
        Absolute tolerance of redundancy, in meters.
    include_conjugates : bool
        Option to include baselines that are redundant under conjugation.
        Only used if use_antpos is False. Default is currently False but will
        become True in version 3.4.
    use_grid_alg : bool
        Option to use the gridding based algorithm (developed by the HERA team)
        to find redundancies rather than the older clustering algorithm.

    Returns
    -------
    baseline_groups : list of lists of int
        list of lists of redundant baseline numbers
    vec_bin_centers : list of array_like of float
        List of vectors describing redundant group centers
    lengths : list of float
        List of redundant group baseline lengths in meters
    baseline_ind_conj : list of int
        List of baselines that are redundant when reversed. Only returned if
        include_conjugates is True

    """
    Nbls = baselines.shape[0]

    if not baseline_vecs.shape == (Nbls, 3):
        raise ValueError("Baseline vectors must be shape (Nbls, 3)")

    if include_conjugates is None:
        warnings.warn(
            "The include_conjugates parameter is not set. The default is "
            "currently False, which produces different groups than the groups "
            "produced when using the `compress_by_redundancy` method. "
            "The default will change to True in version 3.4.",
            DeprecationWarning,
        )
        include_conjugates = False

    baseline_vecs = deepcopy(baseline_vecs)  # Protect the vectors passed in.

    if include_conjugates:
        conjugates = []
        for bv in baseline_vecs:
            uneg = bv[0] < -tol
            uzer = np.isclose(bv[0], 0.0, atol=tol)
            vneg = bv[1] < -tol
            vzer = np.isclose(bv[1], 0.0, atol=tol)
            wneg = bv[2] < -tol
            conjugates.append(uneg or (uzer and vneg) or (uzer and vzer and wneg))

        conjugates = np.array(conjugates, dtype=bool)
        baseline_vecs[conjugates] *= -1
        baseline_ind_conj = baselines[conjugates]
        bl_gps, vec_bin_centers, lens = get_baseline_redundancies(
            baselines,
            baseline_vecs,
            tol=tol,
            include_conjugates=False,
            use_grid_alg=use_grid_alg,
        )
        return bl_gps, vec_bin_centers, lens, baseline_ind_conj

    if use_grid_alg:
        output = find_clusters_grid(
            location_ids=baselines, location_vectors=baseline_vecs, tol=tol
        )
        bl_gps, baseline_ind_conj = output
    else:
        try:
            bl_gps = find_clusters(
                location_ids=baselines,
                location_vectors=baseline_vecs,
                tol=tol,
                strict=True,
            )
        except ValueError as exc:
            raise ValueError(
                "Some baselines are falling into multiple redundant groups. "
                "Lower the tolerance to resolve ambiguity or use the gridding "
                "based algorithm (developed by the HERA team) to find redundancies "
                "by setting use_grid_alg=True."
            ) from exc

    n_unique = len(bl_gps)
    vec_bin_centers = np.zeros((n_unique, 3))
    for gi, gp in enumerate(bl_gps):
        inds = [np.where(i == baselines)[0] for i in gp]
        vec_bin_centers[gi] = np.mean(baseline_vecs[inds, :], axis=0)

    lens = np.sqrt(np.sum(vec_bin_centers**2, axis=1))
    return bl_gps, vec_bin_centers, lens


def get_antenna_redundancies(
    antenna_numbers,
    antenna_positions,
    *,
    tol=1.0,
    include_autos=False,
    use_grid_alg=True,
):
    """
    Find redundant baseline groups based on antenna positions.

    Parameters
    ----------
    antenna_numbers : array_like of int
        Antenna numbers, shape (Nants,).
    antenna_positions : array_like of float
        Antenna position vectors in the ENU (topocentric) frame in meters,
        shape (Nants, 3).
    tol : float
        Redundancy tolerance in meters.
    include_autos : bool
        Option to include autocorrelations.
    use_grid_alg : bool
        Option to use the gridding based algorithm (developed by the HERA team)
        to find redundancies rather than the older clustering algorithm.

    Returns
    -------
    baseline_groups : list of lists of int
        list of lists of redundant baseline numbers
    vec_bin_centers : list of array_like of float
        List of vectors describing redundant group centers
    lengths : list of float
        List of redundant group baseline lengths in meters

    Notes
    -----
    The baseline numbers refer to antenna pairs (a1, a2) such that
    the baseline vector formed from ENU antenna positions,
    blvec = enu[a1] - enu[a2]
    is close to the other baselines in the group.

    This is achieved by putting baselines in a form of the u>0
    convention, but with a tolerance in defining the signs of
    vector components.

    To guarantee that the same baseline numbers are present in a UVData
    object, ``UVData.conjugate_bls('u>0', uvw_tol=tol)``, where `tol` is
    the tolerance used here.

    """
    Nants = antenna_numbers.size

    bls = []
    bl_vecs = []

    for aj in range(Nants):
        mini = aj + 1
        if include_autos:
            mini = aj
        for ai in range(mini, Nants):
            anti, antj = antenna_numbers[ai], antenna_numbers[aj]
            bidx = antnums_to_baseline(antj, anti, Nants_telescope=Nants)
            bv = antenna_positions[ai] - antenna_positions[aj]
            bl_vecs.append(bv)
            bls.append(bidx)
    bls = np.array(bls)
    bl_vecs = np.array(bl_vecs)
    gps, vecs, lens, conjs = get_baseline_redundancies(
        bls, bl_vecs, tol=tol, include_conjugates=True, use_grid_alg=use_grid_alg
    )
    # Flip the baselines in the groups.
    for gi, gp in enumerate(gps):
        for bi, bl in enumerate(gp):
            if bl in conjs:
                gps[gi][bi] = baseline_index_flip(bl, Nants_telescope=Nants)

    return gps, vecs, lens
