------
UVCal
------

UVCal objects hold all of the metadata and data required to work with calibration
solutions for interferometric data sets. Calibration solutions are tied to antennas
rather than baselines. There are many different kinds of calibration solutions, UVCal
has support for many of the most common ones, but this flexibility leads to some
complexity in the definition of UVCal objects. The ``cal_type`` attribute on UVCal
objects indicates whether calibration solutions are "gain" (a complex number per
antenna, polarization and frequency) or "delay" (a real number per antenna and
polarization) type solutions. The ``cal_style`` attribute indicates whether the solution
came from a "sky" or "redundant" style of calibration solution. Some metadata items only
apply to one ``cal_type`` or ``cal_style``.

The antennas are described in two ways: with antenna numbers and antenna names. The
antenna numbers should **not** be confused with indices -- they are not required to start
at zero or to be contiguous, although it is not uncommon for some telescopes to number
them like indices. On UVCal objects, the names and numbers are held in the
``antenna_names`` and ``antenna_numbers`` attributes respectively. These are arranged
in the same order so that an antenna number can be used to identify an antenna name and
vice versa.
Note that not all the antennas listed in ``antenna_numbers`` and ``antenna_names`` are
guaranteed to have calibration solutions associated with them in the ``gain_array``
(or ``delay_array`` for delay type solutions). The antenna numbers associated with each
calibration solution is held in the ``ant_array`` attribute (which has the same length
as the ``gain_array`` or ``delay_array`` along the antenna axis).

For most users, the convenience methods for quick data access (see
`UVCal: Quick data access`_) are the easiest way to get data for particular antennas.
Those methods take the antenna numbers (i.e. numbers listed in ``antenna_numbers``)
as inputs.


UVCal: parameter shape changes
-------------------------------
As detailed in :ref:`uvdata_future_shapes`, UVData objects now support flexible spectral
windows and will have several of their parameter shapes change in version 3.0. They also
have a method to convert to the planned future array shapes now to support an orderly
conversion of code and packages that use UVData objects to the future shapes.

UVCal objects now also support flexible spectral windows and will see parameter shape
changes in version 3.0.

Spectral windows are implemented on standard "gain" type UVCal objects in a similar way
to the UVData implementation, where windows are defined as sets of frequency channels
with some extra parameters to track which channels are in each spectral window. This
allows for spectral windows to have arbitrary numbers of frequency channels and makes
the ``channel_width`` parameter be an array of length ``Nfreqs`` rather than a scalar,
but only when the UVCal object contains flexible spectral windows. Supporting multiple
spectral windows in this way removes the need for the spectral window axis on several
UVCal parameters, but the axis was left as a length 1 axis for backwards compatibility.

Spectral windows are treated a little differently for wide-band style UVCal objects,
which do not have an explicit frequency axis. For those objects, which include all
"delay" type UVCal objects as well as wide-band gain objects, the ``freq_array``
and ``channel_width`` parameters are not required but the ``freq_range`` parameter is
required. UVCal objects that are wide-band and use the future array shapes
can support multiple spectral windows, see details below.

In version 3.0, several parameters will change shape. For standard "gain" type
UVCal objects, the length 1 axis that was originally intended for the spectral windows
axis will be removed from the ``gain_array`` , ``flag_array``, ``quality_array``,
``input_flag_array``, ``total_quality_array`` and ``freq_array`` parameters and the
``channel_width`` parameter will always be an array of length ``Nfreqs``. For
wide-band and "delay" type UVCal objects, the spectral window axis will be retained but
the axis corresponding to the frequency axis will be removed from the ``gain_array`` ,
``delay_array``, ``flag_array``, ``quality_array``, ``input_flag_array`` and the
``total_quality_array`` and the ``freq_range`` parameter will gain a spectral window
axis. In addition, the ``integration_time`` parameter will always be an array of
length ``Ntimes``.

In order to support an orderly conversion of code and packages that use the ``UVCal``
object to these new parameter shapes, we have created the
:meth:`pyuvdata.UVCal.use_future_array_shapes` method which will change the parameters
listed above to have their future shapes. Users writing new code that uses ``UVCal``
objects are encouraged to call that method immediately after creating a UVCal object
or reading in data from a file to ensure that the code will be compatible with the
forthcoming changes. Developers and maintainers of existing code that uses ``UVCal``
objects are encouraged to similarly add that method call and convert their code to use
the new shapes at their earliest convenience to ensure future compatibility. The method
will be deprecated but not removed in version 3.0 (it will just become a no-op) so
that code that calls it will continue to function.


UVCal: Reading/writing
----------------------
Calibration files using UVCal.

a) Reading a cal fits gain calibration file.
********************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal.read_calfits(filename)

  >>> # Cal type:
  >>> print(cal.cal_type)
  gain

  >>> # number of antenna polarizations and polarization type.
  >>> print((cal.Njones, cal.jones_array))
  (1, array([-5]))

  >>> # Number of antennas with data
  >>> print(cal.Nants_data)
  19

  >>> # Number of frequencies
  >>> print(cal.Nfreqs)
  10

  >>> # Shape of the gain_array
  >>> print(cal.gain_array.shape)
  (19, 1, 10, 5, 1)

  >>> # plot abs of all gains for first time and first jones component.
  >>> for ant in range(cal.Nants_data): # doctest: +SKIP
  ...    plt.plot(cal.freq_array.flatten(), np.abs(cal.gain_array[ant, 0, :, 0, 0]))
  >>> plt.xlabel('Frequency (Hz)') # doctest: +SKIP
  >>> plt.ylabel('Abs(gains)') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP


b) FHD cal to cal fits
***********************
.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> obs_testfile = os.path.join(DATA_PATH, 'fhd_cal_data/1061316296_obs.sav')
  >>> cal_testfile = os.path.join(DATA_PATH, 'fhd_cal_data/1061316296_cal.sav')
  >>> settings_testfile = os.path.join(DATA_PATH, 'fhd_cal_data/1061316296_settings.txt')

  >>> fhd_cal = UVCal()
  >>> fhd_cal.read_fhd_cal(cal_testfile, obs_testfile, settings_file=settings_testfile)
  >>> fhd_cal.write_calfits(os.path.join('.', 'tutorial_cal.fits'), clobber=True)


UVCal: Initializing from a UVData object
----------------------------------------
The :meth:`pyuvdata.UVCal.initialize_from_uvdata` method allows you to initialize a UVCal
object from the metadata in a UVData object. This is useful for codes that are calculating
calibration solutions from UVData objects. There are many optional parameters to allow
users to specify additional metadata or changes from the uvdata metadata. By default,
this method creats a metadata only UVCal object, but it can optionally create the
data-like arrays as well, filled with zeros.

.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVData, UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> uvd_file = os.path.join(DATA_PATH, "zen.2458098.45361.HH.uvh5_downselected")
  >>> uvd = UVData.from_file(uvd_file, file_type="uvh5")
  >>> uvc = UVCal.initialize_from_uvdata(uvd, "multiply", "redundant")
  >>> print(uvc.ant_array)
  [ 0  1 11 12 13 23 24 25]


UVCal: Quick data access
------------------------
Method for quick data access, similar to those on :class:`pyuvdata.UVData`
(:ref:`quick_access`), are available for :class:`pyuvdata.UVCal`.
There are three specific methods that will return numpy arrays:
:meth:`pyuvdata.UVCal.get_gains`, :meth:`pyuvdata.UVCal.get_flags`, and
:meth:`pyuvdata.UVCal.get_quality`. When possible, these methods will return numpy
MemoryView objects, which is relatively fast and adds minimal memory overhead.

a) Data for a single antenna and instrumental polarization
**********************************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> UVC = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
  >>> UVC.read_calfits(filename)
  >>> gain = UVC.get_gains(9, 'Jxx')  # gain for ant=9, pol='Jxx'

  >>> # One can equivalently make any of these calls with the input wrapped in a tuple.
  >>> gain = UVC.get_gains((9, 'Jxx'))

  >>> # If no polarization is fed, then all polarizations are returned
  >>> gain = UVC.get_gains(9)

  >>> # One can also request flags and quality arrays in a similar manner
  >>> flags = UVC.get_flags(9, 'Jxx')
  >>> quals = UVC.get_quality(9, 'Jxx')

UVCal: Calibrating UVData
-------------------------
Calibration solutions in a :class:`pyuvdata.UVCal` object can be applied to a
:class:`pyuvdata.UVData` object using the :func:`pyuvdata.utils.uvcalibrate` function.


a) Calibration of UVData by UVCal
*********************************
.. code-block:: python

  >>> # We can calibrate directly using a UVCal object
  >>> import os
  >>> from pyuvdata import UVData, UVCal, utils
  >>> from pyuvdata.data import DATA_PATH
  >>> uvd = UVData()
  >>> uvd.read(os.path.join(DATA_PATH, "zen.2458098.45361.HH.uvh5_downselected"), file_type="uvh5")
  >>> uvc = UVCal()
  >>> uvc.read_calfits(os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected"))
  >>> # this is an old calfits file which has the wrong antenna names, so we need to fix them first.
  >>> # fix the antenna names in the uvcal object to match the uvdata object
  >>> uvc.antenna_names = np.array(
  ...     [name.replace("ant", "HH") for name in uvc.antenna_names]
  ... )
  >>> uvd_calibrated = utils.uvcalibrate(uvd, uvc, inplace=False)

  >>> # We can also un-calibrate using the same UVCal
  >>> uvd_uncalibrated = utils.uvcalibrate(uvd_calibrated, uvc, inplace=False, undo=True)


UVCal: Selecting data
---------------------
The :meth:`pyuvdata.UVCal.select` method lets you select specific antennas
(by number or name), frequencies (in Hz or by channel number), times or jones components
(by number or string) to keep in the object while removing others.

a) Select antennas to keep on UVCal object using the antenna number.
********************************************************************
.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected")
  >>> cal.read_calfits(filename)

  >>> # print all the antennas numbers with data in the original file
  >>> print(cal.ant_array)
  [ 0  1 11 12 13 23 24 25]
  >>> cal.select(antenna_nums=[1, 13, 25])

  >>> # print all the antennas numbers with data after the select
  >>> print(cal.ant_array)
  [ 1 13 25]

b) Select antennas to keep using the antenna names, also select frequencies to keep.
************************************************************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected")
  >>> cal.read_calfits(filename)

  >>> # print all the antenna names with data in the original file
  >>> print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array])
  ['ant0', 'ant1', 'ant11', 'ant12', 'ant13', 'ant23', 'ant24', 'ant25']

  >>> # print the first 10 frequencies in the original file
  >>> print(cal.freq_array[0, 0:10])
  [1.000000e+08 1.015625e+08 1.031250e+08 1.046875e+08 1.062500e+08
   1.078125e+08 1.093750e+08 1.109375e+08 1.125000e+08 1.140625e+08]
  >>> cal.select(antenna_names=['ant11', 'ant13', 'ant25'], freq_chans=np.arange(0, 4))

  >>> # print all the antenna names with data after the select
  >>> print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array])
  ['ant11', 'ant13', 'ant25']

  >>> # print all the frequencies after the select
  >>> print(cal.freq_array)
  [[1.000000e+08 1.015625e+08 1.031250e+08 1.046875e+08]]

d) Select times
***************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected")
  >>> cal.read_calfits(filename)

  >>> # print all the times in the original file
  >>> print(cal.time_array)
  [2458098.45677626 2458098.45690053 2458098.45702481 2458098.45714908
   2458098.45727336 2458098.45739763 2458098.45752191 2458098.45764619
   2458098.45777046 2458098.45789474]

  >>> # select the first 3 times
  >>> cal.select(times=cal.time_array[0:3])

  >>> print(cal.time_array)
  [2458098.45677626 2458098.45690053 2458098.45702481]

d) Select Jones components
**************************
Selecting on Jones component can be done either using the component numbers or
the component strings (e.g. "Jxx" or "Jyy" for linear polarizations or "Jrr" or
"Jll" for circular polarizations). If ``x_orientation`` is set on the object, strings
represting the physical orientation of the dipole can also be used (e.g. "Jnn" or "ee).

.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> import pyuvdata.utils as uvutils
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, "zen.2458098.45361.HH.omni.calfits_downselected")
  >>> cal.read_calfits(filename)

  >>> # Jones component numbers can be found in the jones_array
  >>> print(cal.jones_array)
  [-5 -6]

  >>> # Jones component numbers can be converted to strings using a utility function
  >>> print(uvutils.jnum2str(cal.jones_array))
  ['Jxx', 'Jyy']

  >>> # make a copy of the object and select Jones components using the component numbers
  >>> cal2 = cal.copy()
  >>> cal2.select(jones=[-5])

  >>> # print Jones component numbers and strings after select
  >>> print(cal2.jones_array)
  [-5]
  >>> print(uvutils.jnum2str(cal2.jones_array))
  ['Jxx']

  >>> # make a copy of the object and select Jones components using the component strings
  >>> cal2 = cal.copy()
  >>> cal2.select(jones=["Jxx"])

  >>> # print Jones component numbers and strings after select
  >>> print(cal2.jones_array)
  [-5]
  >>> print(uvutils.jnum2str(cal2.jones_array))
  ['Jxx']

  >>> # print x_orientation
  >>> print(cal.x_orientation)
  east

  >>> # make a copy of the object and select Jones components using the physical orientation strings
  >>> cal2 = cal.copy()
  >>> cal2.select(jones=["Jee"])

  >>> # print Jones component numbers and strings after select
  >>> print(cal2.jones_array)
  [-5]
  >>> print(uvutils.jnum2str(cal2.jones_array))
  ['Jxx']

UVCal: Adding data
------------------
The :meth:`~pyuvdata.UVCal.__add__` method lets you combine UVCal objects along
the antenna, time, frequency, and/or polarization axis.

a) Add frequencies.
*******************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal1 = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()

  >>> # Downselect frequencies to recombine
  >>> cal1.select(freq_chans=np.arange(0, 5))
  >>> cal2.select(freq_chans=np.arange(5, 10))
  >>> cal3 = cal1 + cal2
  >>> print((cal1.Nfreqs, cal2.Nfreqs, cal3.Nfreqs))
  (5, 5, 10)

b) Add times.
****************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal1 = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()

  >>> # Downselect times to recombine
  >>> times = np.unique(cal1.time_array)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal3 = cal1 + cal2
  >>> print((cal1.Ntimes, cal2.Ntimes, cal3.Ntimes))
  (2, 3, 5)

c) Adding in place.
*******************
The following two commands are equivalent, and act on cal1
directly without creating a third uvcal object.

.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal1 = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()
  >>> times = np.unique(cal1.time_array)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal1.__add__(cal2, inplace=True)

  >>> cal1.read_calfits(filename)
  >>> cal2 = cal1.copy()
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal1 += cal2

d) Reading multiple files.
**************************
If any of the read methods (:meth:`pyuvdata.UVCal.read_calfits`,
:meth:`pyuvdata.UVCal.read_fhd_cal`) are given a list of files,
each file will be read in succession and added to the previous.

.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal = UVCal()
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.gain.calfits')
  >>> cal.read_calfits(filename)
  >>> cal1 = cal.select(freq_chans=np.arange(0, 2), inplace=False)
  >>> cal2 = cal.select(freq_chans=np.arange(2, 4), inplace=False)
  >>> cal3 = cal.select(freq_chans=np.arange(4, 7), inplace=False)
  >>> cal1.write_calfits(os.path.join('.', 'tutorial1.fits'))
  >>> cal2.write_calfits(os.path.join('.', 'tutorial2.fits'))
  >>> cal3.write_calfits(os.path.join('.', 'tutorial3.fits'))
  >>> filenames = [os.path.join('.', f) for f
  ...              in ['tutorial1.fits', 'tutorial2.fits', 'tutorial3.fits']]
  >>> cal.read_calfits(filenames)

  >>> # For FHD cal datasets pass lists for each file type
  >>> fhd_cal = UVCal()
  >>> obs_testfiles = [os.path.join(DATA_PATH, f) for f
  ...                  in ['fhd_cal_data/1061316296_obs.sav',
  ...                      'fhd_cal_data/set2/1061316296_obs.sav']]
  >>> cal_testfiles = [os.path.join(DATA_PATH, f) for f
  ...                  in ['fhd_cal_data/1061316296_cal.sav',
  ...                      'fhd_cal_data/set2/1061316296_cal.sav']]
  >>> settings_testfiles = [os.path.join(DATA_PATH, f) for f
  ...                       in ['fhd_cal_data/1061316296_settings.txt',
  ...                           'fhd_cal_data/set2/1061316296_settings.txt']]
  >>> fhd_cal.read_fhd_cal(cal_testfiles, obs_testfiles, settings_file=settings_testfiles)
  diffuse_model parameter value is a string, values are different

UVCal: Changing cal_type from 'delay' to 'gain'
-----------------------------------------------
UVCal includes the method :meth:`pyuvdata.UVCal.convert_to_gain`, which changes a
UVCal object's ``cal_type`` parameter from "delay" to "gain", and accordingly sets the
object's ``gain_array`` to an array consistent with its pre-existing ``delay_array``.

.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVCal
  >>> from pyuvdata.data import DATA_PATH
  >>> cal = UVCal()

  >>> # This file has a cal_type of 'delay'.
  >>> filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.delay.calfits')
  >>> cal.read_calfits(filename)
  >>> print(cal.cal_type)
  delay

  >>> # But we can convert it to a 'gain' type calibration.
  >>> cal.convert_to_gain()
  >>> print(cal.cal_type)
  gain

  >>> # If we want the calibration to use a positive value in its exponent, rather
  >>> # than the default negative value:
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal = cal.convert_to_gain(delay_convention='plus')

  >>> # Convert to gain *without* running the default check that internal arrays are
  >>> # of compatible shapes:
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal.convert_to_gain(run_check=False)

  >>> # Convert to gain *without* running the default check that optional parameters
  >>> # are properly shaped and typed:
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal.convert_to_gain(check_extra=False)

  >>> # Convert to gain *without* running the default checks on the reasonableness
  >>> # of the resulting calibration's parameters.
  >>> cal = UVCal()
  >>> cal.read_calfits(filename)
  >>> cal.convert_to_gain(run_check_acceptability=False)
