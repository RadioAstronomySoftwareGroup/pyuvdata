# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Utilities for collapsing arrays."""

import warnings
from copy import deepcopy

import numpy as np


def mean_collapse(
    arr, *, weights=None, axis=None, return_weights=False, return_weights_square=False
):
    """
    Collapse by averaging data.

    This is similar to np.average, except it handles infs (by giving them
    zero weight) and zero weight axes (by forcing result to be inf with zero
    output weight).

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        weights for average. If none, will default to equal weight for all
        non-infinite data.
    axis : int or tuple, optional
        Axis or axes to collapse (passed to np.sum). Default is all.
    return_weights : bool
        Whether to return sum of weights.
    return_weights_square: bool
        Whether to return the sum of the square of the weights. Default is False.

    """
    arr = deepcopy(arr)  # avoid changing outside
    if weights is None:
        weights = np.ones_like(arr)
    else:
        weights = deepcopy(weights)
    weights = weights * np.logical_not(np.isinf(arr))
    arr[np.isinf(arr)] = 0
    weight_out = np.sum(weights, axis=axis)
    if return_weights_square:
        weights_square = weights**2
        weights_square_out = np.sum(weights_square, axis=axis)
    out = np.sum(weights * arr, axis=axis)
    where = weight_out > 1e-10
    out = np.true_divide(out, weight_out, where=where)
    out = np.where(where, out, np.inf)
    if return_weights and return_weights_square:
        return out, weight_out, weights_square_out
    elif return_weights:
        return out, weight_out
    elif return_weights_square:
        return out, weights_square_out
    else:
        return out


def absmean_collapse(
    arr, *, weights=None, axis=None, return_weights=False, return_weights_square=False
):
    """
    Collapse by averaging absolute value of data.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        weights for average. If none, will default to equal weight for all
        non-infinite data.
    axis : int or tuple, optional
        Axis or axes to collapse (passed to np.sum). Default is all.
    return_weights : bool
        Whether to return sum of weights.
    return_weights_square: bool
        whether to return the sum of the squares of the weights. Default is False.

    """
    return mean_collapse(
        np.abs(arr),
        weights=weights,
        axis=axis,
        return_weights=return_weights,
        return_weights_square=return_weights_square,
    )


def quadmean_collapse(
    arr, *, weights=None, axis=None, return_weights=False, return_weights_square=False
):
    """
    Collapse by averaging in quadrature.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        weights for average. If none, will default to equal weight for all
        non-infinite data.
    axis : int or tuple, optional
        Axis or axes to collapse (passed to np.sum). Default is all.
    return_weights : bool
        Whether to return sum of weights.
    return_weights_square: bool
        whether to return the sum of the squares of the weights. Default is False.

    """
    out = mean_collapse(
        np.abs(arr) ** 2,
        weights=weights,
        axis=axis,
        return_weights=return_weights,
        return_weights_square=return_weights_square,
    )
    if return_weights and return_weights_square:
        return np.sqrt(out[0]), out[1], out[2]
    elif return_weights or return_weights_square:
        return np.sqrt(out[0]), out[1]
    else:
        return np.sqrt(out)


def or_collapse(
    arr, *, weights=None, axis=None, return_weights=False, return_weights_square=False
):
    """
    Collapse using OR operation.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        NOT USED, but kept for symmetry with other collapsing functions.
    axis : int or tuple, optional
        Axis or axes to collapse (take OR over). Default is all.
    return_weights : bool
        Whether to return dummy weights array.
        NOTE: the dummy weights will simply be an array of ones
    return_weights_square: bool
        NOT USED, but kept for symmetry with other collapsing functions.

    """
    if arr.dtype != np.bool_:
        raise ValueError("Input to or_collapse function must be boolean array")
    out = np.any(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn("Currently weights are not handled when OR-ing boolean arrays.")
    if return_weights:
        return out, np.ones_like(out, dtype=np.float64)
    else:
        return out


def and_collapse(
    arr, *, weights=None, axis=None, return_weights=False, return_weights_square=False
):
    """
    Collapse using AND operation.

    Parameters
    ----------
    arr : array
        Input array to process.
    weights: ndarray, optional
        NOT USED, but kept for symmetry with other collapsing functions.
    axis : int or tuple, optional
        Axis or axes to collapse (take AND over). Default is all.
    return_weights : bool
        Whether to return dummy weights array.
        NOTE: the dummy weights will simply be an array of ones
    return_weights_square: bool
        NOT USED, but kept for symmetry with other collapsing functions.

    """
    if arr.dtype != np.bool_:
        raise ValueError("Input to and_collapse function must be boolean array")
    out = np.all(arr, axis=axis)
    if (weights is not None) and not np.all(weights == weights.reshape(-1)[0]):
        warnings.warn("Currently weights are not handled when AND-ing boolean arrays.")
    if return_weights:
        return out, np.ones_like(out, dtype=np.float64)
    else:
        return out


def collapse(
    arr,
    alg,
    *,
    weights=None,
    axis=None,
    return_weights=False,
    return_weights_square=False,
):
    """
    Parent function to collapse an array with a given algorithm.

    Parameters
    ----------
    arr : array
        Input array to process.
    alg : str
        Algorithm to use. Must be defined in this function with
        corresponding subfunction above.
    weights: ndarray, optional
        weights for collapse operation (e.g. weighted mean).
        NOTE: Some subfunctions do not use the weights. See corresponding
        doc strings.
    axis : int or tuple, optional
        Axis or axes to collapse. Default is all.
    return_weights : bool
        Whether to return sum of weights.
    return_weights_square: bool
        Whether to return the sum of the squares of the weights. Default is False.

    """
    collapse_dict = {
        "mean": mean_collapse,
        "absmean": absmean_collapse,
        "quadmean": quadmean_collapse,
        "or": or_collapse,
        "and": and_collapse,
    }
    try:
        out = collapse_dict[alg](
            arr,
            weights=weights,
            axis=axis,
            return_weights=return_weights,
            return_weights_square=return_weights_square,
        )
    except KeyError as err:
        raise ValueError(
            "Collapse algorithm must be one of: "
            + ", ".join(collapse_dict.keys())
            + "."
        ) from err
    return out
