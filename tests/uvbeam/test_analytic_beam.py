# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import numpy as np
import pytest

from pyuvdata import GaussianBeam


def test_chromatic_gaussian():
    """
    test_chromatic_gaussian
    Defining a gaussian beam with a spectral index and reference frequency.
    Check that beam width follows prescribed power law.
    """
    freqs = np.arange(120e6, 160e6, 4e6)
    Nfreqs = len(freqs)
    Npix = 1000
    alpha = -1.5
    sigma = np.radians(15.0)

    az = np.zeros(Npix)
    za = np.linspace(0, np.pi / 2.0, Npix)

    # Error if trying to define chromatic beam without a reference frequency
    with pytest.raises(
        ValueError, match="reference_freq must be set if `spectral_index` is not zero."
    ):
        GaussianBeam(sigma=sigma, spectral_index=alpha)

    A = GaussianBeam(sigma=sigma, reference_freq=freqs[0], spectral_index=alpha)

    # Get the widths at each frequency.

    vals = A.efield_eval(az, za, freqs)

    for fi in range(Nfreqs):
        # The beam peaks at 0.5 in each pol. Find where it drops by a factor of 2
        hwhm = za[np.argmin(np.abs(vals[fi] - 0.25))]
        print(hwhm)
        sig_f = sigma * (freqs[fi] / freqs[0]) ** alpha
        np.testing.assert_allclose(sig_f, 2 * hwhm / 2.355, atol=1e-3)
