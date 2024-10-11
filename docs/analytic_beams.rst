Analytic Beams
==============

pyuvdata defines several analytic primary beams for radio telescopes. While these
are not realistic models for true antennas (like those represented in
:class:`pyuvdata.UVBeam`), they can be useful in simulation because they are
lightweight and fast to evaluate (as opposed to having to interpolate).

The analytic beams defined in pyuvdata are based on a base class,
:class:`pyuvdata.analytic_beam.AnalyticBeam`, which ensures a standard interface
and can be used to define other analytic beams in a consistent way (see the
:ref:`analytic beam tutorial <analytic_beam_tutorial>`). To evaluate analytic
beams in particular directions at particular frequencies, use the
:meth:`pyuvdata.analytic_beam.AnalyticBeam.efield_eval`
or :meth:`pyuvdata.analytic_beam.AnalyticBeam.power_eval` methods as appropriate.

The ``AnalyticBeam`` base class also provides a yaml constructor that can enable
analytic beams to be instantiated directly from yaml files (see
:ref:`yaml_constructors`, similar constructors are also available for UVBeam
objects) and a plugin infrastructure that can automatically include any imported
subclass even if they are defined in other packages. This infrastructure, along
with the :class:`pyuvdata.BeamInterface` class, can simplify the setup for
simulations.

.. autoclass:: pyuvdata.analytic_beam.AnalyticBeam
   :members:


.. autoclass:: pyuvdata.AiryBeam
   :members:

.. autoclass:: pyuvdata.GaussianBeam
   :members:

.. autoclass:: pyuvdata.ShortDipoleBeam
   :members:

.. autoclass:: pyuvdata.UniformBeam
   :members:


.. _yaml_constructors:

yaml constructors
-----------------

Analytic beams can be instantiated directly from yaml files using the
``!AnalyticBeam`` tag. The ``class`` parameter must be specified and it can be
set to one of the pyuvdata provided analytic beams or to any AnalyticBeam
subclass. If the subclass is not defined in pyuvdata, either the subclass must
already be imported or it must be specified with the dot-pathed modules included
(i.e. ``my_module.MyAnalyticBeam``). Some analytic beams have required parameters
(e.g. ``diameter`` for AiryBeams), which must also be provided, see the object
definitions for details.

Some examples are provided below, note that the node key can be anything, it
does not need to be ``beam``:

A 16 meter diameter Airy beam::

    beam: !AnalyticBeam
        class: AiryBeam
        diameter: 16

A classical short dipole beam (the dot-pathed module notation is not required
for pyvudata beams but is shown here as an example)::

    beam: !AnalyticBeam
        class: pyuvdata.ShortDipoleBeam

A gaussian shaped beam with an E-Field beam sigma of 0.26 radians that has
width that scales as a power law with frequency::

    beam: !AnalyticBeam
        class: GaussianBeam
        reference_frequency: 120000000.
        spectral_index: -1.5
        sigma: 0.26
