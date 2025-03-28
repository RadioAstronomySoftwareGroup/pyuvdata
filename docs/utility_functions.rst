Utility Functions
=================
Some of our utility functions are widely used and so are available to be imported
from the `pyuvdata.utils` namespace. These are shown here, for the full list of
all utility functions see: :ref:`utility subpackage`.

.. autofunction:: pyuvdata.utils.uvcalibrate

.. autofunction:: pyuvdata.utils.apply_uvflag

.. autofunction:: pyuvdata.utils.baseline_to_antnums
.. autofunction:: pyuvdata.utils.antnums_to_baseline

.. autofunction:: pyuvdata.utils.LatLonAlt_from_XYZ
.. autofunction:: pyuvdata.utils.XYZ_from_LatLonAlt
.. autofunction:: pyuvdata.utils.rotECEF_from_ECEF
.. autofunction:: pyuvdata.utils.ECEF_from_rotECEF
.. autofunction:: pyuvdata.utils.ENU_from_ECEF
.. autofunction:: pyuvdata.utils.ECEF_from_ENU

.. autofunction:: pyuvdata.utils.polstr2num
.. autofunction:: pyuvdata.utils.polnum2str
.. autofunction:: pyuvdata.utils.jstr2num
.. autofunction:: pyuvdata.utils.jnum2str
.. autofunction:: pyuvdata.utils.conj_pol
.. autofunction:: pyuvdata.utils.x_orientation_pol_map
.. autofunction:: pyuvdata.utils.parse_polstr
.. autofunction:: pyuvdata.utils.parse_jpolstr

.. autofunction:: pyuvdata.utils.get_lst_for_time

.. autofunction:: pyuvdata.utils.uvw_track_generator

.. autofunction:: pyuvdata.utils.collapse

Polarization Dictionaries
-------------------------
We also define some useful dictionaries for mapping polarizations:

  * ``pyuvdata.utils.POL_STR2NUM_DICT``: maps visibility polarization strings to polarization integers
  * ``pyuvdata.utils.POL_NUM2STR_DICT``: maps visibility polarization integers to polarization strings
  * ``pyuvdata.utils.JONES_STR2NUM_DICT``: maps calibration polarization strings to polarization integers
  * ``pyuvdata.utils.JONES_NUM2STR_DICT``: maps calibration polarization strings to polarization integers
  * ``pyuvdata.utils.CONJ_POL_DICT``: maps how visibility polarizations change when antennas are swapped (visibilities are conjugated)
  * ``pyuvdata.utils.XORIENTMAP``: maps x_orientation strings to cannonical names


.. _Utility subpackage:

Utils subpackage
----------------
This gives the full documentation of all functions inside the utils subpackage.
Unless they are also listed above, these functions must be imported using their
full subpackage/submodule path.

Note that we are also listing private functions here (functions that start with
an underscore). While they are listed here, **they are not considered part of the
public API, so they can change without notice**. If you find that you need to rely
one of them let us know in a github issue and we can consider making it part of
the public API.


File I/O Utility Functions
**************************

Antenna position files
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyuvdata.utils.io.antpos
  :members:
  :private-members:
  :undoc-members:

FHD files
~~~~~~~~~

.. automodule:: pyuvdata.utils.io.fhd
  :members:
  :private-members:
  :undoc-members:

FITS files
~~~~~~~~~~

.. automodule:: pyuvdata.utils.io.fits
  :members:
  :private-members:
  :undoc-members:

HDF5 files
~~~~~~~~~~

.. automodule:: pyuvdata.utils.io.hdf5
  :members:
  :private-members:
  :undoc-members:

Measurement Set files
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyuvdata.utils.io.ms
  :members:
  :private-members:
  :undoc-members:

Applying UVFlags to other objects
*********************************

.. automodule:: pyuvdata.utils.apply_uvflag.apply_uvflag
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

.. _coordinate_conversions:

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

Applying calibration solutions to data
**************************************

..
   Note: listing all functions here explicitly because using automodule causes
   a conflict with the earlier pyuvdat.utils.uvcalibrate function.

.. autofunction:: pyuvdata.utils.uvcalibrate.uvcalibrate
.. autofunction:: pyuvdata.utils.uvcalibrate._get_pol_conventions
.. autofunction:: pyuvdata.utils.uvcalibrate._apply_pol_convention_corrections
