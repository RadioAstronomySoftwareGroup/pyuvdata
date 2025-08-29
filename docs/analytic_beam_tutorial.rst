.. _analytic_beam_tutorial:

--------------
Analytic Beams
--------------

The analytic beams defined in pyuvdata are based on a base class,
:class:`pyuvdata.analytic_beam.AnalyticBeam`, which ensures a standard interface
and can be used to define other analytic beams in a consistent way.

Evaluating analytic beams
-------------------------

To evaluate an analytic beam at one or more frequencies and in in one or more
directions, use either the :meth:`pyuvdata.analytic_beam.AnalyticBeam.efield_eval`
or :meth:`pyuvdata.analytic_beam.AnalyticBeam.power_eval` methods as appropriate.

Evaluating an Airy Beam power response
**************************************

This code evaluates and plots an Airy beam power response. Note that we exclude
the cross polarizations, since this is an unpolarized beam, the cross polarizations
are identical to the auto polarization power beams. If the cross polarizations
are included, the array returned from the ``power_eval`` method will be complex.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    from pyuvdata import AiryBeam

    # Create an AiryBeam with a diameter of 14.5 meters
    airy_beam = AiryBeam(diameter=14.5, include_cross_pols=False)

    # set up zenith angle, azimuth and frequency arrays to evaluate with
    # make a regular grid in direction cosines for nice plots
    n_vals = 100
    zmax = np.radians(90)  # Degrees
    axis_arr = np.arange(-n_vals/2., n_vals/2.) / float(n_vals/2.)
    l_arr, m_arr = np.meshgrid(axis_arr, axis_arr)
    radius = np.sqrt(l_arr**2 + m_arr**2)
    za_array = radius * zmax
    az_array = np.arctan2(m_arr, l_arr)

    az_array = az_array.flatten()
    za_array = za_array.flatten()

    Nfreqs = 11
    freqs = np.linspace(100, 200, 11) * 1e6

    # find the values above the horizon so we don't evaluate beyond the horizon
    above_hor = np.nonzero(za_array <= np.pi / 2.)[0]

    # set up an output array that matches the expected shape, except that it
    # includes the points beyond the horizon, and fill it with infinity.
    # Then we will set the points above the horizon to the output of power_eval.
    beam_vals = np.full((1, airy_beam.Npols, Nfreqs, n_vals * n_vals), np.inf, dtype=float)

    beam_vals[:, :, :, above_hor] = airy_beam.power_eval(
        az_array=az_array[above_hor], za_array=za_array[above_hor], freq_array=freqs
    )

    beam_vals = np.reshape(beam_vals, (1, airy_beam.Npols, Nfreqs, n_vals, n_vals))

    fig, ax = plt.subplots(1, 2)
    bp_low = ax[0].imshow(
      beam_vals[0,0,0],
      norm=LogNorm(vmin = 1e-8, vmax =1),
      extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
      origin="lower",
    )
    _ = ax[0].set_title(f"Airy beam {freqs[0]*1e-6} MHz")
    _ = fig.colorbar(bp_low, ax=ax[0], fraction=0.046, pad=0.04, location="left")

    bp_high = ax[1].imshow(
      beam_vals[0,0,-1],
      norm=LogNorm(vmin = 1e-8, vmax =1),
      extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
      origin="lower",
    )
    _ = ax[1].set_title(f"Airy beam {freqs[-1]*1e-6} MHz")
    _ = fig.colorbar(bp_high, ax=ax[1], fraction=0.046, pad=0.04, location="left")

    for ind in range(2):
        _ = ax[ind].set_xticks([0], labels=["North"])
        _ = ax[ind].set_yticks([0], labels=["East"])
        _ = ax[ind].yaxis.set_label_position("right")
        _ = ax[ind].yaxis.tick_right()
        _ = ax[ind].xaxis.set_label_position("top")
        _ = ax[ind].xaxis.tick_top()

    fig.tight_layout()

.. skip: next

    plt.show()

    plt.savefig("Images/airy_beam.png", bbox_inches='tight')
    plt.clf()

.. image:: Images/airy_beam.png
    :width: 600


Evaluating a Short Dipole Beam E-Field response
***********************************************

This code evaluates and plots a short (Herzian) dipole beam E-field response
(also called the Jones matrix). Since it is the E-Field response, we have 4
effective maps because we have the response to each polarization basis vector
for each feed. In the case of a short dipole, these maps do not have an imaginary
part, but in general E-Field beams can be complex, so a complex array is returned.

.. clear-namespace

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    from pyuvdata import ShortDipoleBeam

    # Create an ShortDipoleBeam
    dipole_beam = ShortDipoleBeam()

    # set up zenith angle, azimuth and frequency arrays to evaluate with
    # make a regular grid in direction cosines for nice plots
    n_vals = 100
    zmax = np.radians(90)  # Degrees
    axis_arr = np.arange(-n_vals/2., n_vals/2.) / float(n_vals/2.)
    l_arr, m_arr = np.meshgrid(axis_arr, axis_arr)
    radius = np.sqrt(l_arr**2 + m_arr**2)
    za_array = radius * zmax
    az_array = np.arctan2(m_arr, l_arr)

    az_array = az_array.flatten()
    za_array = za_array.flatten()

    Nfreqs = 11
    freqs = np.linspace(100, 200, 11) * 1e8

    # find the values above the horizon so we don't evaluate beyond the horizon
    above_hor = np.nonzero(za_array <= np.pi / 2.)[0]

    # set up an output array that matches the expected shape except, that it
    # includes the points beyond the horizon, and fill it with infinity.
    # Then we will set the points above the horizon to the output of efield_eval.
    beam_vals = np.full(
        (dipole_beam.Naxes_vec, dipole_beam.Nfeeds, Nfreqs, n_vals * n_vals),
        np.inf,
        dtype=complex
    )

    beam_vals[:, :, :, above_hor] = dipole_beam.efield_eval(
        az_array=az_array[above_hor], za_array=za_array[above_hor], freq_array=freqs
    )

    beam_vals = np.reshape(
        beam_vals, (dipole_beam.Naxes_vec, dipole_beam.Nfeeds, Nfreqs, n_vals, n_vals)
    )

    fig, ax = plt.subplots(2, 2)

    be00 = ax[0,0].imshow(
        beam_vals[0,0,0].real,
        extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
        origin="lower",
    )
    _ = ax[0,0].set_title("E/W dipole azimuth response")
    _ = fig.colorbar(be00, ax=ax[0,0], location="left")

    be10 = ax[1,0].imshow(
        beam_vals[1,0,0].real,
        extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
        origin="lower",
    )
    _ = ax[1,0].set_title("E/W dipole zenith angle response")
    _ = fig.colorbar(be00, ax=ax[1,0], location="left")

    be01 = ax[0,1].imshow(
        beam_vals[0,1,0].real,
        extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
        origin="lower",
    )
    _ = ax[0,1].set_title("N/S dipole azimuth response")
    _ = fig.colorbar(be00, ax=ax[0,1], location="left")

    be11 = ax[1,1].imshow(
        beam_vals[1,1,0].real,
        extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
        origin="lower",
    )
    _ = ax[1,1].set_title("N/S dipole zenith angle response")
    _ = fig.colorbar(be00, ax=ax[1,1], location="left")

    for ind in range(2):
        for ind2 in range(2):
            _ = ax[ind, ind2].set_xticks([0], labels=["North"])
            _ = ax[ind, ind2].set_yticks([0], labels=["East"])
            _ = ax[ind, ind2].yaxis.set_label_position("right")
            _ = ax[ind, ind2].yaxis.tick_right()
            _ = ax[ind, ind2].xaxis.set_label_position("top")
            _ = ax[ind, ind2].xaxis.tick_top()

    fig.tight_layout()

.. skip: next

    plt.show()

    plt.savefig("Images/short_dipole_beam.png", bbox_inches='tight')
    plt.clf()

.. image:: Images/short_dipole_beam.png
    :width: 600


Defining new analytic beams
---------------------------

We have worked to make defining new analytic beams as straight forward as possible.
The new beam needs to inherit from either the :class:`pyuvdata.analytic_beam.AnalyticBeam`,
or the :class:`pyuvdata.analytic_beam.UnpolarizedAnalyticBeam`, which are base
classes that specify what needs to be defined on the new class. Unpolarized
beams (based on the ``UnpolarizedAnalyticBeam`` class) have fewer things that
need to be specified.

Note that while unpolarized beams are simpler to define and think about, they
are quite unphysical and can have results that may be surprising to radio
astronomers. Since unpolarized feeds respond equally to all orientations of the
E-field, if two feeds are specified they will have cross-feed power responses that
are more similar to typical auto-feed power responses (and they will be identical
to auto-feed power responses if the two feeds have the same beam shapes).

Setting parameters on the beam
******************************

If the new beam has any parameters that control the beam response (e.g. diameter),
The class must have an ``@dataclass`` decorator and the parameters must be listed
in the class definitions with type annotations and optionally defaults (these
are called ``fields`` in the dataclass, see the examples below and
`dataclass <https://docs.python.org/3/library/dataclasses.html>`_ for more details).

If you need to do some manipulation or validation of the parameters after they
are specified by the user, you can use the ``validate`` method to do that
(under the hood the ``validate`` method is called by the base object's dataclass
``__post_init__`` method, so the ``validate`` method will always be called
when the class is instantiated).
The gaussian beam example below shows how this can be done.

Polarized beams
***************

For polarized beams (based on the ``AnalyticBeam`` class), the following items
may be specified, the defaults on the ``AnalyticBeam`` class are noted:

  - ``feed_array``: This an array of feed strings (a list can also be passed,
    it will be converted to an array). The default is ``["x", "y"]``.
    This is a a dataclass field, so if it is specified, the class must have
    ``@dataclass`` decorator and it should be specified with type annotations
    and optionally a default (see examples below).

  - ``x_orientation``: For linear polarization feeds, this specifies what the
    ``x`` feed polarization correspond to, allowed values are ``"east"`` or
    ``"north"``, the default is ``"east"``. Should be set to ``None`` for
    circularly polarized feeds.
    This is a a dataclass field, so if it is specified, the class must have
    ``@dataclass`` decorator and it should be specified with type annotations
    and optionally a default (see examples below).

  - ``basis_vector_type``: This defines the coordinate system for the
    polarization basis vectors, the default is ``"az_za"``. Currently only
    ``"az_za"`` is supported, which specifies that there are 2 vector directions
    (i.e. ``Naxes_vec`` is 2).
    This should be defined as a class variable (see examples below).

Defining the beam response
**************************

At least one of the ``_efield_eval`` or ``_power_eval`` methods must be
defined to specify the response of the new beam. Defining ``_efield_eval`` is
the most general approach because it can represent complex and negative going
E-field beams (if only ``_efield_eval`` defined, power beams will be calculated
from the E-field beams). If only ``_power_eval`` is defined, the E-field beam is
defined as the square root of the auto polarization power beam, so the E-field
beam will be real and positive definite. Both methods can be specified, which
may allow for computational efficiencies in some cases.

The inputs to the ``_efield_eval`` and ``_power_eval`` methods are the same and
give the directions (azimuth and zenith angle) and frequencies to evaluate the
beam. All three inputs must be two-dimensional with the first axis having the
length of the number of frequencies and the second axis having the having the
length of the number of directions (these are essentially the output of an
``np.meshgrid`` on the direction and frequency vectors). The inputs are:

    - ``az_grid``: an array of azimuthal values in radians for the directions
      to evaluate the beam. Shape: (number of frequencies, number of directions)
    - ``za_array``: an array of zenith angle values in radians for the directions
      to evaluate the beam. Shape: (number of frequencies, number of directions)
    - ``freq_array``: an array of frequencies in Hz at which to evaluate the beam.
      Shape: (number of frequencies, number of directions)

The ``_efield_eval`` and ``_power_eval`` methods must return arrays with the beam
response. The shapes and types of the returned arrays are:

    - _efield_eval: a complex array of beam responses with shape:
      (``Naxes_vec``, ``Nfeeds``, ``freq_array.size``, ``az_array.size``).
      ``Naxes_vec`` is 2 for the ``"az_za"`` basis, and ``Nfeeds`` is typically 2.

    - ``_power_eval``: an array with shape: (1, ``Npols``, ``freq_array.size``,
      ``az_array.size``). ``Npols`` is equal to either ``Nfeeds`` squared if
      ``include_cross_pols`` was set to True (the default) when the beam was
      instantiated or ``Nfeeds`` if ``include_cross_pols`` was set to False. The
      array should be real if ``include_cross_pols`` was set to False and it can
      be complex if ``include_cross_pols`` was set to True (it will be cast to
      complex when it is called via the ``power_eval`` method on the base class).


Below we provide some examples of beams defined in pyuvdata to make this more
concrete.

Example: Defining simple unpolarized beams
******************************************

Airy beams are unpolarized but frequency dependent and require one parameter,
the dish diameter in meters. Since the Airy beam E-field response goes negative,
the ``_efield_eval`` method is specified in this beam. The definition in pyuvdata
for the AiryBeam object is:

.. literalinclude:: ../src/pyuvdata/analytic_beam.py
   :pyobject: AiryBeam

Below we show how to define a cosine shaped beam with a single width parameter,
which can be defined with just the ``_power_eval`` method.

.. clear-namespace

.. code-block:: python

    from dataclasses import dataclass

    import numpy as np
    from pyuvdata.analytic_beam import UnpolarizedAnalyticBeam
    from pyuvdata.utils.types import FloatArray

    @dataclass(kw_only=True)
    class CosBeam(UnpolarizedAnalyticBeam):
        """
        A variable-width zenith pointed cosine beam.

        Attributes
        ----------
        width : float
            Width parameter, E-field goes like a cosine of width * zenith angle,
            power goes like the same cosine squared.

        Parameters
        ----------
        width : float
            Width parameter, E-field goes like a cosine of width * zenith angle,
            power goes like the same cosine squared.
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
            the power beam.

        """

        width: float

        def _power_eval(
            self,
            *,
            az_grid: FloatArray,
            za_grid: FloatArray,
            f_grid: FloatArray,
        ) -> FloatArray:
            """Evaluate the power at the given coordinates."""

            data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")

            for pol_i in np.arange(self.Npols):
                data_array[0, pol_i, :, :] = np.cos(self.width * za_grid) ** 2

            return data_array

Defining a cosine beam with no free parameters is even simpler:

.. clear-namespace

.. code-block:: python

    import numpy as np
    from pyuvdata.analytic_beam import UnpolarizedAnalyticBeam
    from pyuvdata.utils.types import FloatArray

    class CosBeam(UnpolarizedAnalyticBeam):
        """
        A zenith pointed cosine beam.

        Parameters
        ----------
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
            the power beam.

        """

        def _power_eval(
            self,
            *,
            az_grid: FloatArray,
            za_grid: FloatArray,
            f_grid: FloatArray,
        ) -> FloatArray:
            """Evaluate the power at the given coordinates."""

            data_array = self._get_empty_data_array(az_grid.shape, beam_type="power")

            for pol_i in np.arange(self.Npols):
                data_array[0, pol_i, :, :] = np.cos(za_grid) ** 2

            return data_array


Example: Defining a beam with post init validation
**************************************************

The gaussian beam defined in pyuvdata is an unpolarized beam that has several
optional configurations that require some validation, which we do using the
``validate`` method.

Here is the definition in pyuvdata for the GaussianBeam object and the
``diameter_to_sigma`` function it uses:

.. literalinclude:: ../src/pyuvdata/analytic_beam.py
   :pyobject: diameter_to_sigma

.. literalinclude:: ../src/pyuvdata/analytic_beam.py
   :pyobject: GaussianBeam


Example: Defining a simple polarized beam
*****************************************

Short (Hertzian) dipole beams are polarized but frequency independent and do not
require any extra parameters. We just inherit the default values of ``feed_array``
and ``x_orientation`` from the ``AnalyticBeam`` class, so do not list them here.

Note that we define both the ``_efield_eval`` and ``_power_eval`` methods because
we can use some trig identities to reduce the number of cos/sin evaluations for
the power calculation, but it would give the same results if the ``_power_eval``
method was not defined (we have tests verifying this).

We handle the defaulting of the feed_array in the ``validate`` because dataclass
fields cannot have mutable defaults. We also do some other validation in that method.

The definition in pyuvdata for the ShortDipoleBeam object is:

.. literalinclude:: ../src/pyuvdata/analytic_beam.py
   :pyobject: ShortDipoleBeam
