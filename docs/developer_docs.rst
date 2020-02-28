Developer Docs
==============
Documentation for all the under-the-hood classes and functions that most users
won't need to interact with.

Base Classes
------------
These classes are the under-the-hood classes that provide much of the
infrastructure for pyuvdata's user classes.

``UVParameter`` is the base class for the attributes on pyuvdata's user classes,
allowing the attributes to carry information like expected type and shape,
acceptable values or ranges and tolerances. We also have some type specific
parameter objects (e.g. for angles and earth locations) with extra functionality.

``UVBase`` is the base class for pyuvdata's user classes which has the common
infrastructure to turn the UVParameter attributes into properties and check the
attribute shapes and values.

.. autoclass:: pyuvdata.parameter.UVParameter
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

.. autoclass:: pyuvdata.uvdata.fhd.FHD
  :members:

.. autoclass:: pyuvdata.uvdata.miriad.Miriad
  :members:

.. autoclass:: pyuvdata.uvdata.ms.MS
  :members:

.. autoclass:: pyuvdata.uvdata.mwa_corr_fits.MWACorrFITS
  :members:

.. autoclass:: pyuvdata.uvdata.uvfits.UVFITS
  :members:

.. autoclass:: pyuvdata.uvdata.uvh5.UVH5
  :members:

.. autoclass:: pyuvdata.uvcal.calfits.CALFITS
  :members:

.. autoclass:: pyuvdata.uvcal.fhd_cal.FHDCal
  :members:

.. autoclass:: pyuvdata.uvbeam.beamfits.BeamFITS
  :members:

.. autoclass:: pyuvdata.uvbeam.cst_beam.CSTBeam
  :members:

.. autoclass:: pyuvdata.uvbeam.mwa_beam.MWABeam
  :members:


Functions
----------

.. autofunction:: pyuvdata.uvdata.fhd.get_fhd_history

.. autofunction:: pyuvdata.uvbeam.mwa_beam.P1sin

.. autofunction:: pyuvdata.uvbeam.mwa_beam.P1sin_array

.. autofunction:: pyuvdata.uvflag.uvflag.and_rows_cols

.. autofunction:: pyuvdata.uvflag.uvflag.lst_from_uv

.. autofunction:: pyuvdata.uvflag.uvflag.flags2waterfall


aipy extracts
_____________

.. automodule:: pyuvdata.uvdata.aipy_extracts
  :members:
