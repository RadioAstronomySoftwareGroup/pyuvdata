# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Testing for collapsing utilities."""

import numpy as np
import pytest

from pyuvdata.testing import check_warnings
from pyuvdata.utils import array_collapse


def test_collapse_mean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out = array_collapse.collapse(data, "mean", axis=0)
    out1 = array_collapse.mean_collapse(data, axis=0)
    # Actual values are tested in test_mean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_mean_returned_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out, wo = array_collapse.collapse(data, "mean", axis=0, return_weights=True)
    out1, wo1 = array_collapse.mean_collapse(data, axis=0, return_weights=True)
    # Actual values are tested in test_mean_no_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_mean_returned_with_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo = array_collapse.collapse(
        data, "mean", weights=w, axis=0, return_weights=True
    )
    out1, wo1 = array_collapse.mean_collapse(
        data, weights=w, axis=0, return_weights=True
    )
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_mean_returned_with_weights_and_weights_square():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo, wso = array_collapse.collapse(
        data, "mean", weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    out1, wo1, wso1 = array_collapse.mean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)
    assert np.array_equal(wso, wso1)


def test_collapse_mean_returned_with_weights_square_no_return_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wso = array_collapse.collapse(
        data,
        "mean",
        weights=w,
        axis=0,
        return_weights=False,
        return_weights_square=True,
    )
    out1, wso1 = array_collapse.mean_collapse(
        data, weights=w, axis=0, return_weights=False, return_weights_square=True
    )
    # Actual values are tested in test_mean_weights
    assert np.array_equal(out, out1)
    assert np.array_equal(wso, wso1)


def test_collapse_absmean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = (-1) ** i * np.ones_like(data[:, i])
    out = array_collapse.collapse(data, "absmean", axis=0)
    out1 = array_collapse.absmean_collapse(data, axis=0)
    # Actual values are tested in test_absmean_no_weights
    assert np.array_equal(out, out1)


def test_collapse_quadmean_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out = array_collapse.collapse(data, "quadmean", axis=0)
    out1 = array_collapse.quadmean_collapse(data, axis=0)
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)


def test_collapse_quadmean_returned_with_weights_and_weights_square():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo, wso = array_collapse.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=True,
        return_weights_square=True,
    )
    out1, wo1, wso1 = array_collapse.quadmean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)
    assert np.array_equal(wso, wso1)


def test_collapse_quadmean_returned_with_weights_square_no_return_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wso = array_collapse.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=False,
        return_weights_square=True,
    )
    out1, wso1 = array_collapse.quadmean_collapse(
        data, weights=w, axis=0, return_weights=False, return_weights_square=True
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wso, wso1)


def test_collapse_quadmean_returned_without_weights_square_with_return_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo = array_collapse.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=True,
        return_weights_square=False,
    )
    out1, wo1 = array_collapse.quadmean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=False
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_quadmean_returned_with_weights_square_without_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo = array_collapse.collapse(
        data,
        "quadmean",
        weights=w,
        axis=0,
        return_weights=False,
        return_weights_square=True,
    )
    out1, wo1 = array_collapse.quadmean_collapse(
        data, weights=w, axis=0, return_weights=False, return_weights_square=True
    )
    # Actual values are tested elsewhere?
    assert np.array_equal(out, out1)
    assert np.array_equal(wo, wo1)


def test_collapse_or_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, 8] = True
    o = array_collapse.collapse(data, "or", axis=0)
    o1 = array_collapse.or_collapse(data, axis=0)
    assert np.array_equal(o, o1)


def test_collapse_and_no_return_no_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, :] = True
    o = array_collapse.collapse(data, "and", axis=0)
    o1 = array_collapse.and_collapse(data, axis=0)
    assert np.array_equal(o, o1)


def test_collapse_error():
    pytest.raises(ValueError, array_collapse.collapse, np.ones((2, 3)), "fooboo")


def test_mean_no_weights():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    out, wo = array_collapse.mean_collapse(data, axis=0, return_weights=True)
    assert np.array_equal(out, np.arange(data.shape[1]))
    assert np.array_equal(wo, data.shape[0] * np.ones(data.shape[1]))
    out, wo = array_collapse.mean_collapse(data, axis=1, return_weights=True)
    assert np.all(out == np.mean(np.arange(data.shape[1])))
    assert len(out) == data.shape[0]
    assert np.array_equal(wo, data.shape[1] * np.ones(data.shape[0]))
    out, wo = array_collapse.mean_collapse(data, return_weights=True)
    assert out == np.mean(np.arange(data.shape[1]))
    assert wo == data.size
    out = array_collapse.mean_collapse(data)
    assert out == np.mean(np.arange(data.shape[1]))


def test_mean_weights_and_weights_square():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i]) + 1
    w = 1.0 / data
    out, wo, wso = array_collapse.mean_collapse(
        data, weights=w, axis=0, return_weights=True, return_weights_square=True
    )
    np.testing.assert_allclose(out * wo, data.shape[0])
    np.testing.assert_allclose(
        wo, float(data.shape[0]) / (np.arange(data.shape[1]) + 1)
    )
    np.testing.assert_allclose(
        wso, float(data.shape[0]) / (np.arange(data.shape[1]) + 1) ** 2
    )
    out, wo, wso = array_collapse.mean_collapse(
        data, weights=w, axis=1, return_weights=True, return_weights_square=True
    )
    np.testing.assert_allclose(out * wo, data.shape[1])
    np.testing.assert_allclose(wo, np.sum(1.0 / (np.arange(data.shape[1]) + 1)))
    np.testing.assert_allclose(wso, np.sum(1.0 / (np.arange(data.shape[1]) + 1) ** 2))

    # Zero weights
    w = np.ones_like(data)
    w[0, :] = 0
    w[:, 0] = 0
    out, wo = array_collapse.mean_collapse(data, weights=w, axis=0, return_weights=True)
    ans = np.arange(data.shape[1]).astype(np.float64) + 1
    ans[0] = np.inf
    assert np.array_equal(out, ans)
    ans = (data.shape[0] - 1) * np.ones(data.shape[1])
    ans[0] = 0
    assert np.all(wo == ans)
    out, wo = array_collapse.mean_collapse(data, weights=w, axis=1, return_weights=True)
    ans = np.mean(np.arange(data.shape[1])[1:] + 1) * np.ones(data.shape[0])
    ans[0] = np.inf
    assert np.all(out == ans)
    ans = (data.shape[1] - 1) * np.ones(data.shape[0])
    ans[0] = 0
    assert np.all(wo == ans)


def test_mean_infs():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    data[:, 0] = np.inf
    data[0, :] = np.inf
    out, wo = array_collapse.mean_collapse(data, axis=0, return_weights=True)
    ans = np.arange(data.shape[1]).astype(np.float64)
    ans[0] = np.inf
    assert np.array_equal(out, ans)
    ans = (data.shape[0] - 1) * np.ones(data.shape[1])
    ans[0] = 0
    assert np.all(wo == ans)
    out, wo = array_collapse.mean_collapse(data, axis=1, return_weights=True)
    ans = np.mean(np.arange(data.shape[1])[1:]) * np.ones(data.shape[0])
    ans[0] = np.inf
    assert np.all(out == ans)
    ans = (data.shape[1] - 1) * np.ones(data.shape[0])
    ans[0] = 0
    assert np.all(wo == ans)


def test_absmean():
    # Fake data
    data1 = np.zeros((50, 25))
    for i in range(data1.shape[1]):
        data1[:, i] = (-1) ** i * np.ones_like(data1[:, i])
    data2 = np.ones_like(data1)
    out1 = array_collapse.absmean_collapse(data1)
    out2 = array_collapse.absmean_collapse(data2)
    assert out1 == out2


def test_quadmean():
    # Fake data
    data = np.zeros((50, 25))
    for i in range(data.shape[1]):
        data[:, i] = i * np.ones_like(data[:, i])
    o1, w1 = array_collapse.quadmean_collapse(data, return_weights=True)
    o2, w2 = array_collapse.mean_collapse(np.abs(data) ** 2, return_weights=True)
    o3 = array_collapse.quadmean_collapse(data)  # without return_weights
    o2 = np.sqrt(o2)
    assert o1 == o2
    assert w1 == w2
    assert o1 == o3


def test_or_collapse():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, 8] = True
    o = array_collapse.or_collapse(data, axis=0)
    ans = np.zeros(25, np.bool_)
    ans[8] = True
    assert np.array_equal(o, ans)
    o = array_collapse.or_collapse(data, axis=1)
    ans = np.zeros(50, np.bool_)
    ans[0] = True
    assert np.array_equal(o, ans)
    o = array_collapse.or_collapse(data)
    assert o


def test_or_collapse_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, 8] = True
    w = np.ones_like(data, np.float64)
    o, wo = array_collapse.or_collapse(data, axis=0, weights=w, return_weights=True)
    ans = np.zeros(25, np.bool_)
    ans[8] = True
    assert np.array_equal(o, ans)
    assert np.array_equal(wo, np.ones_like(o, dtype=np.float64))
    w[0, 8] = 0.3
    with check_warnings(UserWarning, "Currently weights are"):
        o = array_collapse.or_collapse(data, axis=0, weights=w)
    assert np.array_equal(o, ans)


def test_or_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, array_collapse.or_collapse, data)


def test_and_collapse():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, :] = True
    o = array_collapse.and_collapse(data, axis=0)
    ans = np.zeros(25, np.bool_)
    assert np.array_equal(o, ans)
    o = array_collapse.and_collapse(data, axis=1)
    ans = np.zeros(50, np.bool_)
    ans[0] = True
    assert np.array_equal(o, ans)
    o = array_collapse.and_collapse(data)
    assert not o


def test_and_collapse_weights():
    # Fake data
    data = np.zeros((50, 25), np.bool_)
    data[0, :] = True
    w = np.ones_like(data, np.float64)
    o, wo = array_collapse.and_collapse(data, axis=0, weights=w, return_weights=True)
    ans = np.zeros(25, np.bool_)
    assert np.array_equal(o, ans)
    assert np.array_equal(wo, np.ones_like(o, dtype=np.float64))
    w[0, 8] = 0.3
    with check_warnings(UserWarning, "Currently weights are"):
        o = array_collapse.and_collapse(data, axis=0, weights=w)
    assert np.array_equal(o, ans)


def test_and_collapse_errors():
    data = np.zeros(5)
    pytest.raises(ValueError, array_collapse.and_collapse, data)
