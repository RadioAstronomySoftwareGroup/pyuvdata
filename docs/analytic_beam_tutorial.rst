.. _analytic_beam_tutorial:

--------------
Analytic Beams
--------------

The analytic beams defined in pyuvdata are based on an abstract base class,
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
the cross polarizations, since this is an unpolarized the cross polarizations
are identical to the auto polarization power beams. If the cross polarizations
are included the array returned from the ``power_eval`` method will be complex.

.. code-block:: python

  >>> import matplotlib.pyplot as plt
  >>> import numpy as np
  >>> from matplotlib.colors import LogNorm

  >>> from pyuvdata import AiryBeam

  >>> # Create an AiryBeam with a diameter of 14.5 meters
  >>> airy_beam = AiryBeam(diameter=14.5, include_cross_pols=False)

  >>> # set up zenith angle, azimuth and frequency arrays to evaluate with
  >>> # make a regular grid in direction cosines for nice plots
  >>> n_vals = 100
  >>> zmax = np.radians(90)  # Degrees
  >>> axis_arr = np.arange(-n_vals/2., n_vals/2.) / float(n_vals/2.)
  >>> l_arr, m_arr = np.meshgrid(axis_arr, axis_arr)
  >>> radius = np.sqrt(l_arr**2 + m_arr**2)
  >>> za_array = radius * zmax
  >>> az_array = np.arctan2(m_arr, l_arr)

  >>> az_array = az_array.flatten()
  >>> za_array = za_array.flatten()

  >>> Nfreqs = 11
  >>> freqs = np.linspace(100, 200, 11) * 1e6

  >>> # find the values above the horizon so we don't evaluate beyond the horizon
  >>> above_hor = np.nonzero(za_array <= np.pi / 2.)[0]

  >>> # set up an output array that matches the expected shape, except that it
  >>> # includes the points beyond the horizon, and fill it with infinity.
  >>> # Then we will set the points above the horizon to the output of power_eval.
  >>> beam_vals = np.full((1, airy_beam.Npols, Nfreqs, n_vals * n_vals), np.inf, dtype=float)

  >>> beam_vals[:, :, :, above_hor] = airy_beam.power_eval(
  ...     az_array=az_array[above_hor], za_array=za_array[above_hor], freq_array=freqs
  ... )

  >>> beam_vals = np.reshape(beam_vals, (1, airy_beam.Npols, Nfreqs, n_vals, n_vals))

  >>> fig, ax = plt.subplots(1, 2)
  >>> bp_low = ax[0].imshow(
  ...   beam_vals[0,0,0],
  ...   norm=LogNorm(vmin = 1e-8, vmax =1),
  ...   extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
  ... )
  >>> _ = ax[0].set_title(f"Airy beam {freqs[0]*1e-6} MHz")
  >>> _ = ax[0].set_xlabel("direction cosine l")
  >>> _ = ax[0].set_ylabel("direction cosine m")
  >>> _ = fig.colorbar(bp_low, ax=ax[0], fraction=0.046, pad=0.04)

  >>> bp_high = ax[1].imshow(
  ...   beam_vals[0,0,-1],
  ...   norm=LogNorm(vmin = 1e-8, vmax =1),
  ...   extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)],
  ... )
  >>> _ = ax[1].set_title(f"Airy beam {freqs[-1]*1e-6} MHz")
  >>> _ = ax[1].set_xlabel("direction cosine l")
  >>> _ = ax[1].set_ylabel("direction cosine m")
  >>> _ = fig.colorbar(bp_high, ax=ax[1], fraction=0.046, pad=0.04)
  >>> fig.tight_layout()
  >>> plt.show()  # doctest: +SKIP
  >>> plt.savefig("Images/airy_beam.png", bbox_inches='tight')
  >>> plt.clf()

.. image:: Images/airy_beam.png
  :width: 600


Evaluating a Short Dipole Beam E-Field response
***********************************************

This code evaluates and plots a short (Herzian) dipole beam E-field response
(also called the Jones matrix). Since it is the E-Field response, we have 4
effective maps because we have the response to each polarization basis vector
for each feed. In the case of a short dipole, these maps do not have an imaginary
part, but in general E-Field beams can be complex, so a complex array is returned.

.. code-block:: python

  >>> import matplotlib.pyplot as plt
  >>> import numpy as np

  >>> from pyuvdata import ShortDipoleBeam

  >>> # Create an ShortDipoleBeam
  >>> dipole_beam = ShortDipoleBeam()

  >>> # set up zenith angle, azimuth and frequency arrays to evaluate with
  >>> # make a regular grid in direction cosines for nice plots
  >>> n_vals = 100
  >>> zmax = np.radians(90)  # Degrees
  >>> axis_arr = np.arange(-n_vals/2., n_vals/2.) / float(n_vals/2.)
  >>> l_arr, m_arr = np.meshgrid(axis_arr, axis_arr)
  >>> radius = np.sqrt(l_arr**2 + m_arr**2)
  >>> za_array = radius * zmax
  >>> az_array = np.arctan2(m_arr, l_arr)

  >>> az_array = az_array.flatten()
  >>> za_array = za_array.flatten()

  >>> Nfreqs = 11
  >>> freqs = np.linspace(100, 200, 11) * 1e8

  >>> # find the values above the horizon so we don't evaluate beyond the horizon
  >>> above_hor = np.nonzero(za_array <= np.pi / 2.)[0]

  >>> # set up an output array that matches the expected shape except, that it
  >>> # includes the points beyond the horizon, and fill it with infinity.
  >>> # Then we will set the points above the horizon to the output of efield_eval.
  >>> beam_vals = np.full((dipole_beam.Naxes_vec, dipole_beam.Nfeeds, Nfreqs, n_vals * n_vals), np.inf, dtype=complex)

  >>> beam_vals[:, :, :, above_hor] = dipole_beam.efield_eval(
  ...     az_array=az_array[above_hor], za_array=za_array[above_hor], freq_array=freqs
  ... )

  >>> beam_vals = np.reshape(beam_vals, (dipole_beam.Naxes_vec, dipole_beam.Nfeeds, Nfreqs, n_vals, n_vals))

  >>> fig, ax = plt.subplots(2, 2)

  >>> be00 = ax[0,0].imshow(beam_vals[0,0,0].real, extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)])
  >>> _ = ax[0,0].set_title("E/W dipole azimuth response")
  >>> _ = ax[0,0].set_xlabel("direction cosine l")
  >>> _ = ax[0,0].set_ylabel("direction cosine m")
  >>> _ = fig.colorbar(be00, ax=ax[0,0])

  >>> be10 = ax[1,0].imshow(beam_vals[1,0,0].real, extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)])
  >>> _ = ax[1,0].set_title("E/W dipole zenith angle response")
  >>> _ = ax[1,0].set_xlabel("direction cosine l")
  >>> _ = ax[1,0].set_ylabel("direction cosine m")
  >>> _ = fig.colorbar(be00, ax=ax[1,0])

  >>> be01 = ax[0,1].imshow(beam_vals[0,1,0].real, extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)])
  >>> _ = ax[0,1].set_title("N/S dipole azimuth response")
  >>> _ = ax[0,1].set_xlabel("direction cosine l")
  >>> _ = ax[0,1].set_ylabel("direction cosine m")
  >>> _ = fig.colorbar(be00, ax=ax[0,1])

  >>> be11 = ax[1,1].imshow(beam_vals[1,1,0].real, extent=[np.min(l_arr), np.max(l_arr), np.min(m_arr), np.max(m_arr)])
  >>> _ = ax[1,1].set_title("N/S dipole zenith angle response")
  >>> _ = ax[1,1].set_xlabel("direction cosine l")
  >>> _ = ax[1,1].set_ylabel("direction cosine m")
  >>> _ = fig.colorbar(be00, ax=ax[1,1])

  >>> fig.tight_layout()
  >>> plt.show()  # doctest: +SKIP
  >>> plt.savefig("Images/short_dipole_beam.png", bbox_inches='tight')
  >>> plt.clf()

.. image:: Images/short_dipole_beam.png
  :width: 600


Defining new analytic beams
---------------------------

Defining new analytic beams is relatively straight forward. The new beam needs
to be defined as a `dataclass <https://docs.python.org/3/library/dataclasses.html>`_
that inherits from :class:`pyuvdata.analytic_beam.AnalyticBeam`, which is an
abstract base class that specifies what needs to be defined on the new class.

First any parameters that control the beam response (e.g. diameter) must be
listed with type annotations and optionally defaults to be picked up by the
dataclass constructor (these are called ``fields`` in the dataclass). In addition
to any fields specific to this new beam, the following fields must be specified:

  - ``feed_array``: this is an array of feed strings.

    - For unpolarized beams, this should be specified as::

          feed_array: npt.NDArray[str] | None = field(default=None, repr=False, compare=False)

      This specifies that the feed array can be either an array of strings or ``None``,
      that the default is ``None`` (which will be converted to the canonical
      ``[x, y]`` by the AnalyticBeam initialization) and that it shouldn't be
      included when representing or comparing beams of this class (which makes
      sense for unpolarized beams).

    - For polarized beams, this should be specified as required or hardcoded. For
      example, on the :class:`pyuvdata.ShortDipoleBeam` it is hardcoded as::

        feed_array = ["e", "n"]

  - ``x_orientation``: This species what the ``x`` feed polarization corresponds
    to, allowed values are ``"east"`` or ``"north"``.

    - For unpolarized beams, this should be specied as::

          x_orientation: Literal["east", "north"] = field(default="east", repr=False, compare=False)

      This specifies the allowed values for the x_orientation and also specifies
      that it shouldn't be included when representing or comparing beams of this
      class, which makes sense for unpolarized beams. The defaulting can be set
      to either "east" or "north" as you prefer.

    - For polarized beams, this should be specified as (again the choice of default
      is up to you)::

        x_orientation: Literal["east", "north"] = "east"

  - ``include_cross_pols``: This specifies whether or not the cross polarizations
    should be included when calculating the power response (essentially whether
    ``Npols`` is equal to ``Nfeeds`` squared or just ``Nfeeds``). It should only
    be used in the initialization, not set as a field on the object, so it should
    be specified using ``InitVar`` as (defaulting is your choice)::

      include_cross_pols: InitVar[bool] = True


Then there are three things that are requred to be defined on the new class to
actually calculate the response of the new beam:

  - ``basis_vector_type``: this should be defined as a class variable. It defines
    the coordinate system for the polarization basis vectors. Currently only
    ``"az_za"`` is supported, which specifies that there are 2 vector directions
    (i.e. ``Naxes_vec`` is 2).

  - ``_efield_eval``: this needs to be a method that returns the efield response
    for a given direction and frequency. The inputs must be:

      - ``az_array``: an array of azimuthal values in radians for the directions
        to evaluate the beam. Must be a 1D array the same length as the ``za_array``.
      - ``za_array``: an array of zenith angle values in radians for the directions
        to evaluate the beam. Must be a 1D array the same length as the ``az_array``.
      - ``freq_array``: an array of frequencies in Hz at which to evaluate the beam.
        Must be a 1D array.

    and it must return a complex array of beam responses with the shape:
    (``Naxes_vec``, ``Nfeeds``, ``freq_array.size``, ``az_array.size``).
    ``Naxes_vec`` is 2 for the ``"az_za"`` basis, and ``Nfeeds`` is typically 2.

  - ``_power_eval``: this needs to be a method that returns the power response
    for a given direction and frequency. The inputs must be:

      - ``az_array``: an array of azimuthal values in radians for the directions
        to evaluate the beam. Must be a 1D array the same length as the ``za_array``.
      - ``za_array``: an array of zenith angle values in radians for the directions
        to evaluate the beam. Must be a 1D array the same length as the ``az_array``.
      - ``freq_array``: an array of frequencies in Hz at which to evaluate the beam.
        Must be a 1D array.

    and it must return an array of beam responses with the shape:
    (1, ``Npols``, ``freq_array.size``, ``az_array.size``). The array can be complex
    if cross polarizations are included (if it is not complex it will be made
    complex if the cross polarizations are included when it is called via the
    ``power_eval`` method on the base class). ``Npols`` is equal to either
    ``Nfeeds`` squared if ``include_cross_pols`` is True (the default) or
    ``Nfeeds`` if ``include_cross_pols`` is False.

Note that if you need to do some manipulation or validation of the fields after
they are specified by the user, you can use the dataclass's ``__post_init__``
method to do that, being sure to call the super class ``__post_init__`` as well.
The gaussian beam example below shows how this can be done.

Below we provide some examples of beams defined in pyuvdata to make this more
concrete.

Defining a simple unpolarized beam
**********************************

Airy beams are unpolarized but frequency dependent and require one parameter,
the dish diameter in meters.

.. code-block:: python
  :linenos:

    import dataclasses
    from dataclasses import InitVar, dataclass, field
    from typing import Literal

    import numpy as np
    import numpy.typing as npt
    from astropy.constants import c as speed_of_light
    from scipy.special import j1
    from pyuvdata.analytic_beam import AnalyticBeam


    @dataclass(kw_only=True)
    class AiryBeam(AnalyticBeam):
        """
        A zenith pointed Airy beam.

        Airy beams are the diffraction pattern of a circular aperture, so represent
        an idealized dish. Requires a dish diameter in meters and is inherently
        chromatic and unpolarized.

        The unpolarized nature leads to some results that may be surprising to radio
        astronomers: if two feeds are specified they will have identical responses
        and the cross power beam between the two feeds will be identical to the
        power beam for a single feed.

        Attributes
        ----------
        diameter : float
            Dish diameter in meters.
        feed_array : np.ndarray of str
            Feeds to define this beam for, e.g. x & y or n & e (for "north" and "east")
            or r & l.
        x_orientation : str
            Physical orientation of the feed for the x feed. Not meaningful for
            AiryBeams, which are unpolarized.

        Parameters
        ----------
        diameter : float
            Dish diameter in meters.
        feed_array : np.ndarray of str
            Feeds to define this beam for, e.g. n & e or x & y or r & l.
        x_orientation : str
            Physical orientation of the feed for the x feed. Not meaningful for
            AiryBeams, which are unpolarized.
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
            the power beam.

        """

        diameter: float
        feed_array: npt.NDArray[str] | None = field(default=None, repr=False, compare=False)
        x_orientation: Literal["east", "north"] = field(
            default="east", repr=False, compare=False
        )

        include_cross_pols: InitVar[bool] = True

        basis_vector_type = "az_za"

        def _efield_eval(
            self,
            *,
            az_array: npt.NDArray[float],
            za_array: npt.NDArray[float],
            freq_array: npt.NDArray[float],
        ) -> npt.NDArray[float]:
            """Evaluate the efield at the given coordinates."""
            data_array = self._get_empty_data_array(az_array.size, freq_array.size)

            za_grid, f_grid = np.meshgrid(za_array, freq_array)
            kvals = (2.0 * np.pi) * f_grid / speed_of_light.to("m/s").value
            xvals = (self.diameter / 2.0) * np.sin(za_grid) * kvals
            values = np.zeros_like(xvals)
            nz = xvals != 0.0
            ze = xvals == 0.0
            values[nz] = 2.0 * j1(xvals[nz]) / xvals[nz]
            values[ze] = 1.0

            for fn in np.arange(self.Nfeeds):
                data_array[0, fn, :, :] = values / np.sqrt(2.0)
                data_array[1, fn, :, :] = values / np.sqrt(2.0)

            return data_array

        def _power_eval(
            self,
            *,
            az_array: npt.NDArray[float],
            za_array: npt.NDArray[float],
            freq_array: npt.NDArray[float],
        ) -> npt.NDArray[float]:
            """Evaluate the power at the given coordinates."""
            data_array = self._get_empty_data_array(
                az_array.size, freq_array.size, beam_type="power"
            )

            za_grid, f_grid = np.meshgrid(za_array, freq_array)
            kvals = (2.0 * np.pi) * f_grid / speed_of_light.to("m/s").value
            xvals = (self.diameter / 2.0) * np.sin(za_grid) * kvals
            values = np.zeros_like(xvals)
            nz = xvals != 0.0
            ze = xvals == 0.0
            values[nz] = (2.0 * j1(xvals[nz]) / xvals[nz]) ** 2
            values[ze] = 1.0

            for fn in np.arange(self.Npols):
                # For power beams the first axis is shallow because we don't have to worry
                # about polarization.
                data_array[0, fn, :, :] = values

            return data_array

Defining a simple polarized beam
********************************

Short (Hertzian) dipole beams are polarized but frequency independent and do not
require any extra parameters. Note that we hardcode the ``feed_array`` because
the eval methods assume that the first feed is a dipole aligned east/west and the
second is a dipole aligned north/south. The ``x_orientation`` field can be set
to control which feed is assigned to the ``x`` label, which is important when
writing simulated visibilities out to files (most visibility file types do not
support polarizations labelled as ``"e"`` or ``"n"``, they require them to be
labeled as ``"x"`` and ``"y"`` for linear polarization feeds).

.. code-block:: python
  :linenos:

    import dataclasses
    from dataclasses import InitVar, dataclass
    from typing import Literal

    import numpy as np
    import numpy.typing as npt
    from pyuvdata.analytic_beam import AnalyticBeam


    @dataclass(kw_only=True)
    class ShortDipoleBeam(AnalyticBeam):
        """
        A zenith pointed analytic short dipole beam with two crossed feeds.

        A classical short (Hertzian) dipole beam with two crossed feeds aligned east
        and north. Short dipole beams are intrinsically polarized but achromatic.
        Does not require any parameters, but the orientation of the dipole labelled
        as "x" can be specified to align "north" or "east" via the x_orientation
        parameter (matching the parameter of the same name on UVBeam and UVData
        objects).

        Attributes
        ----------
        feed_array : np.ndarray of str
            Feeds to define this beam for, e.g. x & y or n & e (for "north" and "east")
            or r & l.
        x_orientation : str
            The orientation of the dipole labeled 'x'. The default ("east") means
            that the x dipole is aligned east-west and that the y dipole is aligned
            north-south.

        Parameters
        ----------
        x_orientation : str
            The orientation of the dipole labeled 'x'. The default ("east") means
            that the x dipole is aligned east-west and that the y dipole is aligned
            north-south.
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne)
            for the power beam.

        """

        x_orientation: Literal["east", "north"] = "east"

        feed_array = ["e", "n"]

        include_cross_pols: InitVar[bool] = True

        basis_vector_type = "az_za"

        def _efield_eval(
            self,
            *,
            az_array: npt.NDArray[float],
            za_array: npt.NDArray[float],
            freq_array: npt.NDArray[float],
        ) -> npt.NDArray[float]:
            """Evaluate the efield at the given coordinates."""
            data_array = self._get_empty_data_array(az_array.size, freq_array.size)

            az_fgrid = np.repeat(az_array[np.newaxis], freq_array.size, axis=0)
            za_fgrid = np.repeat(za_array[np.newaxis], freq_array.size, axis=0)

            # The first dimension is for [azimuth, zenith angle] in that order
            # the second dimension is for feed [e, n] in that order
            data_array[0, 0] = -np.sin(az_fgrid)
            data_array[0, 1] = np.cos(az_fgrid)
            data_array[1, 0] = np.cos(za_fgrid) * np.cos(az_fgrid)
            data_array[1, 1] = np.cos(za_fgrid) * np.sin(az_fgrid)

            return data_array

        def _power_eval(
            self,
            *,
            az_array: npt.NDArray[float],
            za_array: npt.NDArray[float],
            freq_array: npt.NDArray[float],
        ) -> npt.NDArray[float]:
            """Evaluate the power at the given coordinates."""
            data_array = self._get_empty_data_array(
                az_array.size, freq_array.size, beam_type="power"
            )

            az_fgrid = np.repeat(az_array[np.newaxis], freq_array.size, axis=0)
            za_fgrid = np.repeat(za_array[np.newaxis], freq_array.size, axis=0)

            # these are just the sum in quadrature of the efield components.
            # some trig work is done to reduce the number of cos/sin evaluations
            data_array[0, 0] = 1 - (np.sin(za_fgrid) * np.cos(az_fgrid)) ** 2
            data_array[0, 1] = 1 - (np.sin(za_fgrid) * np.sin(az_fgrid)) ** 2

            if self.Npols > self.Nfeeds:
                # cross pols are included
                data_array[0, 2] = -(np.sin(za_fgrid) ** 2) * np.sin(2.0 * az_fgrid) / 2.0
                data_array[0, 3] = data_array[0, 2]

            return data_array


Defining a beam with post init validation
*****************************************

The gaussian beam defined in pyuvdata is an unpolarized beam that has several
optional configurations that require some validation, which we do using the
dataclass ``__post_init__`` method. Note that we call the ``super().__post_init__``
within that method to ensure that all the normal AnalyticBeam setup has been done.

.. code-block:: python
  :linenos:

    import dataclasses
    from dataclasses import InitVar, dataclass, field
    from typing import Literal

    import numpy as np
    import numpy.typing as npt
    from astropy.constants import c as speed_of_light
    from pyuvdata.analytic_beam import AnalyticBeam

    def diameter_to_sigma(diameter: float, freq_array: npt.NDArray[float]) -> float:
        """
        Find the sigma that gives a beam width similar to an Airy disk.

        Find the stddev of a gaussian with fwhm equal to that of
        an Airy disk's main lobe for a given diameter.

        Parameters
        ----------
        diameter : float
            Antenna diameter in meters
        freq_array : array of float
            Frequencies in Hz

        Returns
        -------
        sigma : float
            The standard deviation in zenith angle radians for a Gaussian beam
            with FWHM equal to that of an Airy disk's main lobe for an aperture
            with the given diameter.

        """
        wavelengths = speed_of_light.to("m/s").value / freq_array

        scalar = 2.2150894  # Found by fitting a Gaussian to an Airy disk function

        sigma = np.arcsin(scalar * wavelengths / (np.pi * diameter)) * 2 / 2.355

        return sigma


    @dataclass(kw_only=True)
    class GaussianBeam(AnalyticBeam):
        """
        A circular, zenith pointed Gaussian beam.

        Requires either a dish diameter in meters or a standard deviation sigma in
        radians. Gaussian beams specified by a diameter will have their width
        matched to an Airy beam at each simulated frequency, so are inherently
        chromatic. For Gaussian beams specified with sigma, the sigma_type defines
        whether the width specified by sigma specifies the width of the E-Field beam
        (default) or power beam in zenith angle. If only sigma is specified, the
        beam is achromatic, optionally both the spectral_index and reference_frequency
        parameters can be set to generate a chromatic beam with standard deviation
        defined by a power law:

        stddev(f) = sigma * (f/ref_freq)**(spectral_index)

        The unpolarized nature leads to some results that may be
        surprising to radio astronomers: if two feeds are specified they will have
        identical responses and the cross power beam between the two feeds will be
        identical to the power beam for a single feed.

        Attributes
        ----------
        sigma : float
            Standard deviation in radians for the gaussian beam. Only one of sigma
            and diameter should be set.
        sigma_type : str
            Either "efield" or "power" to indicate whether the sigma specifies the size of
            the efield or power beam. Ignored if `sigma` is None.
        diameter : float
            Dish diameter in meters to use to define the size of the gaussian beam, by
            matching the FWHM of the gaussian to the FWHM of an Airy disk. This will result
            in a frequency dependent beam.  Only one of sigma and diameter should be set.
        spectral_index : float
            Option to scale the gaussian beam width as a power law with frequency. If set
            to anything other than zero, the beam will be frequency dependent and the
            `reference_frequency` must be set. Ignored if `sigma` is None.
        reference_frequency : float
            The reference frequency for the beam width power law, required if `sigma` is not
            None and `spectral_index` is not zero. Ignored if `sigma` is None.
        feed_array : np.ndarray of str
            Feeds to define this beam for, e.g. x & y or n & e (for "north" and "east")
            or r & l.
        x_orientation : str
            Physical orientation of the feed for the x feed. Not meaningful for
            GaussianBeams, which are unpolarized.

        Parameters
        ----------
        sigma : float
            Standard deviation in radians for the gaussian beam. Only one of sigma
            and diameter should be set.
        sigma_type : str
            Either "efield" or "power" to indicate whether the sigma specifies the size of
            the efield or power beam. Ignored if `sigma` is None.
        diameter : float
            Dish diameter in meters to use to define the size of the gaussian beam, by
            matching the FWHM of the gaussian to the FWHM of an Airy disk. This will result
            in a frequency dependent beam.  Only one of sigma and diameter should be set.
        spectral_index : float
            Option to scale the gaussian beam width as a power law with frequency. If set
            to anything other than zero, the beam will be frequency dependent and the
            `reference_frequency` must be set. Ignored if `sigma` is None.
        reference_frequency : float
            The reference frequency for the beam width power law, required if `sigma` is not
            None and `spectral_index` is not zero. Ignored if `sigma` is None.
        feed_array : np.ndarray of str
            Feeds to define this beam for, e.g. n & e or x & y or r & l.
        x_orientation : str
            Physical orientation of the feed for the x feed. Not meaningful for
            GaussianBeams, which are unpolarized.
        include_cross_pols : bool
            Option to include the cross polarized beams (e.g. xy and yx or en and ne) for
            the power beam.

        """

        sigma: float | None = None
        sigma_type: Literal["efield", "power"] = "efield"
        diameter: float | None = None
        spectral_index: float = 0.0
        reference_frequency: float = None

        feed_array: npt.NDArray[str] | None = field(default=None, repr=False, compare=False)
        x_orientation: Literal["east", "north"] = field(
            default="east", repr=False, compare=False
        )

        include_cross_pols: InitVar[bool] = True

        basis_vector_type = "az_za"

        def __post_init__(self, include_cross_pols):
            """
            Post-initialization validation and conversions.

            Parameters
            ----------
            include_cross_pols : bool
                Option to include the cross polarized beams (e.g. xy and yx or en and ne)
                for the power beam.

            """
            if (self.diameter is None and self.sigma is None) or (
                self.diameter is not None and self.sigma is not None
            ):
                if self.diameter is None:
                    raise ValueError("Either diameter or sigma must be set.")
                else:
                    raise ValueError("Only one of diameter or sigma can be set.")

            if self.sigma is not None:
                if self.sigma_type != "efield":
                    self.sigma = np.sqrt(2) * self.sigma

                if self.spectral_index != 0.0 and self.reference_frequency is None:
                    raise ValueError(
                        "reference_frequency must be set if `spectral_index` is not zero."
                    )
                if self.reference_frequency is None:
                    self.reference_frequency = 1.0

            super().__post_init__(include_cross_pols=include_cross_pols)

        def get_sigmas(self, freq_array: npt.NDArray[float]) -> npt.NDArray[float]:
            """
            Get the sigmas for the gaussian beam using the diameter (if defined).

            Parameters
            ----------
            freq_array : array of floats
                Frequency values to get the sigmas for in Hertz.

            Returns
            -------
            sigmas : array_like of float
                Beam sigma values as a function of frequency. Size will match the
                freq_array size.

            """
            if self.diameter is not None:
                sigmas = diameter_to_sigma(self.diameter, freq_array)
            elif self.sigma is not None:
                sigmas = (
                    self.sigma
                    * (freq_array / self.reference_frequency) ** self.spectral_index
                )
            return sigmas

        def _efield_eval(
            self,
            *,
            az_array: npt.NDArray[float],
            za_array: npt.NDArray[float],
            freq_array: npt.NDArray[float],
        ) -> npt.NDArray[float]:
            """Evaluate the efield at the given coordinates."""
            sigmas = self.get_sigmas(freq_array)

            values = np.exp(
                -(za_array[np.newaxis, ...] ** 2) / (2 * sigmas[:, np.newaxis] ** 2)
            )
            data_array = self._get_empty_data_array(az_array.size, freq_array.size)
            for fn in np.arange(self.Nfeeds):
                data_array[0, fn, :, :] = values / np.sqrt(2.0)
                data_array[1, fn, :, :] = values / np.sqrt(2.0)

            return data_array

        def _power_eval(
            self,
            *,
            az_array: npt.NDArray[float],
            za_array: npt.NDArray[float],
            freq_array: npt.NDArray[float],
        ) -> npt.NDArray[float]:
            """Evaluate the power at the given coordinates."""
            sigmas = self.get_sigmas(freq_array)

            values = np.exp(
                -(za_array[np.newaxis, ...] ** 2) / (sigmas[:, np.newaxis] ** 2)
            )
            data_array = self._get_empty_data_array(
                az_array.size, freq_array.size, beam_type="power"
            )
            for fn in np.arange(self.Npols):
                # For power beams the first axis is shallow because we don't have to worry
                # about polarization.
                data_array[0, fn, :, :] = values

            return data_array
