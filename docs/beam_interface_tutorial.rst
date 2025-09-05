--------------
Beam Interface
--------------

The BeamInterface object is designed to provide a unified interface for UVBeam
and AnalyticBeam objects to compute beam response values. It can be constructed
with either a :class:`pyuvdata.UVBeam` or :class:`AnalyticBeam` and the beam
response can be calculated using the :meth:`pyuvdata.BeamInterface.compute_response`
method.

.. include:: tutorial_data_note.rst

Using BeamInterface
-------------------

The following code shows how to set up two BeamInterface objects, one with an
analytic beam and one with a UVBeam. Then each is evalated at the same frequency
and directions using the same call to :meth:`pyuvdata.BeamInterface.compute_response`.
The value of the BeamInterface object is that it unifies the interface so the
code calling it does not need to know if the beam that is attached to it is an
analytic beam or a UVBeam.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    from pyuvdata import ShortDipoleBeam, BeamInterface, UVBeam
    from pyuvdata.datasets import fetch_data

    filename = fetch_data("mwa_full_EE")

    dipole_beam = BeamInterface(ShortDipoleBeam(), beam_type="power")
    mwa_beam = BeamInterface(UVBeam.from_file(filename, pixels_per_deg=1), beam_type="power")

    # set up zenith angle, azimuth and frequency arrays to evaluate with
    az_grid = np.deg2rad(np.arange(0, 360))
    za_grid = np.deg2rad(np.arange(0, 91))
    az_array, za_array = np.meshgrid(az_grid, za_grid)

    az_array = az_array.flatten()
    za_array = za_array.flatten()

    # The MWA beam we have in our test data is small, it only has 3 frequencies,
    # so we will just get the value at one of those frequencies rather than
    # trying to interpolate to a new frequency.
    freqs = np.array([mwa_beam.beam.freq_array[-1]])

    dipole_beam_vals = dipole_beam.compute_response(
        az_array=az_array, za_array=za_array, freq_array=freqs
    )

    mwa_beam_vals = mwa_beam.compute_response(
        az_array=az_array, za_array=za_array, freq_array=freqs
    )
    assert dipole_beam_vals.shape == (1, 4, 1, 91 * 360)
    assert mwa_beam_vals.shape == (1, 4, 1, 91 * 360)
    assert dipole_beam_vals.dtype == np.complex128
    assert mwa_beam_vals.dtype == np.complex128
