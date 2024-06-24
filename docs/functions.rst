Useful Functions
================
There are some functions that interact with multiple types of objects to apply
calibrations solutions and flagging to other objects.

.. autofunction:: pyuvdata.uvcalibrate

.. autofunction:: pyuvdata.apply_uvflag


Utility Functions
-----------------
Some of our utility functions are widely used. The most commonly used ones are
noted here, for others see the developer docs: :ref:`developer docs utility functions`.

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
