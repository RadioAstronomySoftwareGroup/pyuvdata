# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Frequency related utilities."""
import warnings

import numpy as np

from .tools import _test_array_constant


def _check_flex_spw_contiguous(*, spw_array, flex_spw_id_array):
    """
    Check if the spectral windows are contiguous for multi-spw datasets.

    This checks the flex_spw_id_array to make sure that all channels for each
    spectral window are together in one block, versus being interspersed (e.g.,
    channel #1 and #3 is in spw #1, channels #2 and #4 are in spw #2). In theory,
    UVH5 and UVData objects can handle this, but MIRIAD, MIR, UVFITS, and MS file
    formats cannot, so we just consider it forbidden.

    Parameters
    ----------
    spw_array : array of integers
        Array of spectral window numbers, shape (Nspws,).
    flex_spw_id_array : array of integers
        Array of spectral window numbers per frequency channel, shape (Nfreqs,).

    """
    exp_spw_ids = np.unique(spw_array)
    # This is an internal consistency check to make sure that the indexes match
    # up as expected -- this shouldn't error unless someone is mucking with
    # settings they shouldn't be.
    assert np.all(np.unique(flex_spw_id_array) == exp_spw_ids), (
        "There are some entries in flex_spw_id_array that are not in spw_array. "
        "This is a bug, please report it in an issue."
    )

    n_breaks = np.sum(flex_spw_id_array[1:] != flex_spw_id_array[:-1])
    if (n_breaks + 1) != spw_array.size:
        raise ValueError(
            "Channels from different spectral windows are interspersed with "
            "one another, rather than being grouped together along the "
            "frequency axis. Most file formats do not support such "
            "non-grouping of data."
        )


def _check_freq_spacing(
    *,
    freq_array,
    freq_tols,
    channel_width,
    channel_width_tols,
    spw_array,
    flex_spw_id_array,
    raise_errors=True,
):
    """
    Check if frequencies are evenly spaced and separated by their channel width.

    This is a requirement for writing uvfits & miriad files.

    Parameters
    ----------
    freq_array : array of float
        Array of frequencies, shape (Nfreqs,).
    freq_tols : tuple of float
        freq_array tolerances (from uvobj._freq_array.tols).
    channel_width : array of float
        Channel widths, either a scalar or an array of shape (Nfreqs,).
    channel_width_tols : tuple of float
        channel_width tolerances (from uvobj._channel_width.tols).
    spw_array : array of integers or None
        Array of spectral window numbers, shape (Nspws,).
    flex_spw_id_array : array of integers or None
        Array of spectral window numbers per frequency channel, shape (Nfreqs,).
    raise_errors : bool
        Option to raise errors if the various checks do not pass.

    Returns
    -------
    spacing_error : bool
        Flag that channel spacings or channel widths are not equal.
    chanwidth_error : bool
        Flag that channel spacing does not match channel width.

    """
    spacing_error = False
    chanwidth_error = False

    # Check to make sure that the flexible spectral window has indicies set up
    # correctly (grouped together) for this check
    _check_flex_spw_contiguous(spw_array=spw_array, flex_spw_id_array=flex_spw_id_array)

    for spw_id in spw_array:
        mask = flex_spw_id_array == spw_id
        if sum(mask) > 1:
            freq_spacing = np.diff(freq_array[mask])
            freq_dir = -1.0 if all(freq_spacing < 0) else 1.0
            if not _test_array_constant(freq_spacing, tols=freq_tols):
                spacing_error = True
            if not _test_array_constant(channel_width[mask], tols=channel_width_tols):
                spacing_error = True
            elif not np.allclose(
                freq_spacing,
                np.mean(channel_width[mask]) * freq_dir,
                rtol=channel_width_tols[0],
                atol=channel_width_tols[1],
            ):
                chanwidth_error = True

    if raise_errors and spacing_error:
        raise ValueError(
            "The frequencies are not evenly spaced (probably because of a select "
            "operation) or has differing values of channel widths. Some file formats "
            "(e.g. uvfits, miriad) do not support unevenly spaced frequencies."
        )
    if raise_errors and chanwidth_error:
        raise ValueError(
            "The frequencies are separated by more than their channel width (probably "
            "because of a select operation). Some file formats (e.g. uvfits, miriad) "
            "do not support frequencies that are spaced by more than their channel "
            "width."
        )

    return spacing_error, chanwidth_error


def _sort_freq_helper(
    *,
    Nfreqs,  # noqa: N803
    freq_array,
    Nspws,  # noqa: N803
    spw_array,
    flex_spw_id_array,
    spw_order,
    channel_order,
    select_spw,
):
    """
    Figure out the frequency sorting order for object based frequency sorting.

    Parameters
    ----------
    Nfreqs :  int
        Number of frequencies, taken directly from the object parameter.
    freq_array :  array_like of float
        Frequency array, taken directly from the object parameter.
    Nfreqs :  int
        Number of spectral windows, taken directly from the object parameter.
    spw_array :  array_like of int
        Spectral window array, taken directly from the object parameter.
    flex_spw_id_array : array_like of int
        Array of SPW IDs for each channel, taken directly from the object parameter.
    spw_order : str or array_like of int
        A string describing the desired order of spectral windows along the
        frequency axis. Allowed strings include `number` (sort on spectral window
        number) and `freq` (sort on median frequency). A '-' can be prepended
        to signify descending order instead of the default ascending order,
        e.g., if you have SPW #1 and 2, and wanted them ordered as [2, 1],
        you would specify `-number`. Alternatively, one can supply an index array
        of length Nspws that specifies how to shuffle the spws (this is not the desired
        final spw order).  Default is to apply no sorting of spectral windows.
    channel_order : str or array_like of int
        A string describing the desired order of frequency channels within a
        spectral window. Allowed strings include `freq`, which will sort channels
        within a spectral window by frequency. A '-' can be optionally prepended
        to signify descending order instead of the default ascending order.
        Alternatively, one can supply an index array of length Nfreqs that
        specifies the new order. Default is to apply no sorting of channels
        within a single spectral window. Note that proving an array_like of ints
        will cause the values given to `spw_order` and `select_spw` to be ignored.
    select_spw : int or array_like of int
        An int or array_like of ints which specifies which spectral windows to
        apply sorting. Note that setting this argument will cause the value
        given to `spw_order` to be ignored.

    Returns
    -------
    index_array : ndarray of int
        Array giving the desired order of the channels to be used for sorting along the
        frequency axis

    Raises
    ------
    UserWarning
        Raised if providing arguments to select_spw and channel_order (the latter
        overrides the former).
    ValueError
        Raised if select_spw contains values not in spw_array, or if channel_order
        is not the same length as freq_array.

    """
    if (spw_order is None) and (channel_order is None):
        warnings.warn(
            "Not specifying either spw_order or channel_order causes "
            "no sorting actions to be applied. Returning object unchanged."
        )
        return

    # Check to see if there are arguments we should be ignoring
    if isinstance(channel_order, (np.ndarray, list, tuple)):
        if select_spw is not None:
            warnings.warn(
                "The select_spw argument is ignored when providing an "
                "array_like of int for channel_order"
            )
        if spw_order is not None:
            warnings.warn(
                "The spw_order argument is ignored when providing an "
                "array_like of int for channel_order"
            )
        channel_order = np.asarray(channel_order)
        if not channel_order.size == Nfreqs or not np.all(
            np.sort(channel_order) == np.arange(Nfreqs)
        ):
            raise ValueError(
                "Index array for channel_order must contain all indicies for "
                "the frequency axis, without duplicates."
            )
        index_array = channel_order
    else:
        index_array = np.arange(Nfreqs)
        # Multipy by 1.0 here to make a cheap copy of the array to manipulate
        temp_freqs = 1.0 * freq_array
        # Same trick for ints -- add 0 to make a cheap copy
        temp_spws = 0 + flex_spw_id_array

        # Check whether or not we need to sort the channels in individual windows
        sort_spw = {idx: channel_order is not None for idx in spw_array}
        if select_spw is not None:
            if spw_order is not None:
                warnings.warn(
                    "The spw_order argument is ignored when providing an "
                    "argument for select_spw"
                )
            if channel_order is None:
                warnings.warn(
                    "Specifying select_spw without providing channel_order causes "
                    "no sorting actions to be applied. Returning object unchanged."
                )
                return
            if isinstance(select_spw, (np.ndarray, list, tuple)):
                sort_spw = {idx: idx in select_spw for idx in spw_array}
            else:
                sort_spw = {idx: idx == select_spw for idx in spw_array}
        elif spw_order is not None:
            if isinstance(spw_order, (np.ndarray, list, tuple)):
                spw_order = np.asarray(spw_order)
                if not spw_order.size == Nspws or not np.all(
                    np.sort(spw_order) == np.arange(Nspws)
                ):
                    raise ValueError(
                        "Index array for spw_order must contain all indicies for "
                        "the spw_array, without duplicates."
                    )
            elif spw_order not in ["number", "freq", "-number", "-freq", None]:
                raise ValueError(
                    "spw_order can only be one of 'number', '-number', "
                    "'freq', '-freq', None or an index array of length Nspws"
                )
            elif Nspws > 1:
                # Only need to do this step if we actually have multiple spws.

                # If the string starts with a '-', then we will flip the order at
                # the end of the operation
                flip_spws = spw_order[0] == "-"

                if "number" in spw_order:
                    spw_order = np.argsort(spw_array)
                elif "freq" in spw_order:
                    spw_order = np.argsort(
                        [np.median(temp_freqs[temp_spws == idx]) for idx in spw_array]
                    )
                if flip_spws:
                    spw_order = np.flip(spw_order)
            else:
                spw_order = np.arange(Nspws)
            # Now that we know the spw order, we can apply the first sort
            index_array = np.concatenate(
                [index_array[temp_spws == spw] for spw in spw_array[spw_order]]
            )
            temp_freqs = temp_freqs[index_array]
            temp_spws = temp_spws[index_array]
        # Spectral windows are assumed sorted at this point
        if channel_order is not None:
            if channel_order not in ["freq", "-freq"]:
                raise ValueError(
                    "channel_order can only be one of 'freq' or '-freq' or an index "
                    "array of length Nfreqs"
                )
            for idx in spw_array:
                if sort_spw[idx]:
                    select_mask = temp_spws == idx
                    subsort_order = index_array[select_mask]
                    subsort_order = subsort_order[np.argsort(temp_freqs[select_mask])]
                    index_array[select_mask] = (
                        np.flip(subsort_order)
                        if channel_order[0] == "-"
                        else subsort_order
                    )
    if np.all(index_array[1:] > index_array[:-1]):
        # Nothing to do - the data are already sorted!
        return

    return index_array
