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

.. autoclass:: pyuvdata.parameter.SkyCoordParameter
  :members:

.. autoclass:: pyuvdata.uvbase.UVBase
  :members:

File Type Specific Classes
--------------------------
These classes inherit from pyuvdata's user classes and hold the file type
specific code. The read and write methods on the user classes convert between
the user classes and the file-specific classes automatically as needed, so users
generally do not need to interact with these classes, but developers may need to.


UVData
******

.. autoclass:: pyuvdata.uvdata.fhd.FHD
  :members:

.. autoclass:: pyuvdata.uvdata.mir.Mir
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

UVCal
*****

.. autoclass:: pyuvdata.uvcal.calfits.CALFITS
  :members:

.. autoclass:: pyuvdata.uvcal.calh5.CalH5
  :members:

.. autoclass:: pyuvdata.uvcal.fhd_cal.FHDCal
  :members:

UVBeam
******

.. autoclass:: pyuvdata.uvbeam.beamfits.BeamFITS
  :members:

.. autoclass:: pyuvdata.uvbeam.cst_beam.CSTBeam
  :members:

.. autoclass:: pyuvdata.uvbeam.mwa_beam.MWABeam
  :members:


.. _Developer Docs Utility Functions:

Utility Functions
-----------------
Note that we are also listing private functions here (functions that start with
an underscore). While they are listed here, **they are not considered part of the
public API, so they can change without notice**. If you find that you need to rely
one of them let us know in a github issue and we can consider making it part of
the public API.


File I/O Utility Functions
**************************

Antenna position files
++++++++++++++++++++++

.. automodule:: pyuvdata.utils.io.antpos
  :members:
  :private-members:
  :undoc-members:

FHD files
+++++++++

.. automodule:: pyuvdata.utils.io.fhd
  :members:
  :private-members:
  :undoc-members:

FITS files
++++++++++

.. automodule:: pyuvdata.utils.io.fits
  :members:
  :private-members:
  :undoc-members:

HDF5 files
++++++++++

.. automodule:: pyuvdata.utils.io.hdf5
  :members:
  :private-members:
  :undoc-members:

Measurement Set files
+++++++++++++++++++++

.. automodule:: pyuvdata.utils.io.ms
  :members:
  :private-members:
  :undoc-members:

Array collapse functions for flags
**********************************

.. automodule:: pyuvdata.utils.array_collapse
  :members:
  :private-members:
  :undoc-members:

Functions for working with baseline numbers
*******************************************

.. automodule:: pyuvdata.utils.bls
  :members:
  :private-members:
  :undoc-members:
  :ignore-module-all:

Functions for working with the baseline-time axis
*************************************************

.. automodule:: pyuvdata.utils.bltaxis
  :members:
  :private-members:
  :undoc-members:

Functions for working with telescope coordinates
************************************************

.. automodule:: pyuvdata.utils.coordinates
  :members:
  :private-members:
  :undoc-members:
  :ignore-module-all:

Functions for working with the frequency axis
*********************************************

.. automodule:: pyuvdata.utils.frequency
  :members:
  :private-members:
  :undoc-members:

Functions for working with history
**********************************

.. automodule:: pyuvdata.utils.history
  :members:
  :private-members:
  :undoc-members:

Functions for working with phase center catalogs
************************************************

.. automodule:: pyuvdata.utils.phase_center_catalog
  :members:
  :private-members:
  :undoc-members:

Functions for working with phasing
**********************************

.. automodule:: pyuvdata.utils.phasing
  :members:
  :private-members:
  :undoc-members:

Functions for working with polarizations
****************************************

.. automodule:: pyuvdata.utils.pol
  :members:
  :private-members:
  :undoc-members:
  :ignore-module-all:

Functions for working with baseline redundancies
************************************************

.. automodule:: pyuvdata.utils.redundancy
  :members:
  :private-members:
  :undoc-members:

Functions for working with times and LSTs
*****************************************

.. automodule:: pyuvdata.utils.times
  :members:
  :private-members:
  :undoc-members:

General utility functions
*************************

.. automodule:: pyuvdata.utils.tools
  :members:
  :private-members:
  :undoc-members:

Mir Parser
----------
.. automodule:: pyuvdata.uvdata.mir_parser
  :members:

.. automodule:: pyuvdata.uvdata.mir_meta_data
  :members:


Other Functions
---------------

.. autofunction:: pyuvdata.uvbeam.mwa_beam.P1sin

.. autofunction:: pyuvdata.uvbeam.mwa_beam.P1sin_array

.. autofunction:: pyuvdata.uvflag.uvflag.and_rows_cols

.. autofunction:: pyuvdata.uvflag.uvflag.flags2waterfall


aipy extracts
-------------

.. automodule:: pyuvdata.uvdata.aipy_extracts
  :members:
