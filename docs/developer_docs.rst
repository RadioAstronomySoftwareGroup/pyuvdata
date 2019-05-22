Developer Docs
==============
Documentation for all the under-the-hood classes and functions that most users
won't need to interact with.

Base Classes
------------
These classes are the under-the-hood classes that provide much of the
infrastructure for pyuvdata's user classes.

```UVParameter``` is the base class for the attributes on pyuvdata's user classes,
allowing the attributes to carry information like expected type and shape,
acceptable values or ranges and tolerances. We also have some type specific
parameter objects (e.g. for angles and earth locations) with extra functionality.

```UVBase``` is the base class for pyuvdata's user classes which has the common
infrastructure to turn the UVParameter attributes into properties and check the
attribute shapes and values.

.. autoclass:: pyuvdata.parameter.UVParameter
  :members:

.. autoclass:: pyuvdata.parameter.AntPositionParameter
  :members:

.. autoclass:: pyuvdata.parameter.AngleParameter
  :members:

.. autoclass:: pyuvdata.parameter.LocationParameter
  :members:

.. autoclass:: pyuvdata.uvbase.UVBase
  :members:

File Type Specific Classes
--------------------------
These classes inherit from pyuvdata's user classes and hold the file type
specific code. The read and write methods on the user classes convert between
the user classes and the file-specific classes automatically as needed, so users
generally do not need to interact with these classes, but developers may need to.

.. autoclass:: pyuvdata.uvfits.UVFITS
  :members:

.. autoclass:: pyuvdata.miriad.Miriad
  :members:

.. autoclass:: pyuvdata.fhd.FHD
  :members:

.. autoclass:: pyuvdata.ms.MS
  :members:

.. autoclass:: pyuvdata.uvh5.UVH5
  :members:

.. autoclass:: pyuvdata.fhd_cal.FHDCal
  :members:

.. autoclass:: pyuvdata.calfits.CALFITS
  :members:

.. autoclass:: pyuvdata.beamfits.BeamFITS
  :members:

.. autoclass:: pyuvdata.cst_beam.CSTBeam
  :members:


Functions
----------

.. autofunction:: pyuvdata.version.construct_version_info

.. autofunction:: pyuvdata.fhd.get_fhd_history

aipy extracts
_____________

.. automodule:: pyuvdata.aipy_extracts
  :members:
