Beam Interface
==============

The BeamInterface object is designed to provide a unified interface for UVBeam
and AnalyticBeam objects to compute beam response values. It can be constructed
with either a UVBeam or AnalyticBeam and it provides the
:meth:`pyuvdata.BeamInterface.compute_response` method, which can be used to
either evaluate the response of an AnalyticBeam or interpolate a UVBeam to get
the beam response. This interface provides a simplified way for simulators to
call either UVBeams or analytic beams.

.. autoclass:: pyuvdata.BeamInterface
   :members:
