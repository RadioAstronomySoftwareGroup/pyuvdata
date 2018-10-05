Tutorial
========

.. testsetup::
   from __future__ import absolute_import, division, print_function

------
UVData
------

UVData: File conversion
-----------------------
Converting between tested data formats

a) miriad -> uvfits
*******************
::

  >>> from pyuvdata import UVData
  >>> UV = UVData()

  # This miriad file is known to be a drift scan.
  # Use either the file type specific read_miriad or the generic read function,
  # optionally specify the file type
  >>> miriad_file = 'pyuvdata/data/new.uvA'
  >>> UV.read_miriad(miriad_file)
  >>> UV.read(miriad_file, file_type='miriad')
  >>> UV.read(miriad_file)

  # Write out the uvfits file
  >>> UV.write_uvfits('tutorial.uvfits', force_phase=True, spoof_nonessential=True)
  The data are in drift mode and do not have a defined phase center. Phasing to zenith of the first timestamp.

b) uvfits -> miriad
*******************
::

  >>> from pyuvdata import UVData
  >>> import shutil
  >>> import os
  >>> UV = UVData()
  >>> uvfits_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'

  # Use either the file type specific read_uvfits or the generic read function,
  # optionally specify the file type
  >>> UV.read_uvfits(uvfits_file)
  >>> UV.read(uvfits_file, file_type='uvfits')
  >>> UV.read(uvfits_file)

  # Write out the miriad file
  >>> write_file = 'tutorial.uv'
  >>> if os.path.exists(write_file):
  ...    shutil.rmtree(write_file)
  >>> UV.write_miriad(write_file)

c) FHD -> uvfits
****************
When reading FHD format, we need to point to several files.
::

  >>> from pyuvdata import UVData
  >>> UV = UVData()

  # Construct the list of files
  >>> fhd_prefix = 'pyuvdata/data/fhd_vis_data/1061316296_'
  >>> fhd_files = [fhd_prefix + f for f in ['flags.sav', 'vis_XX.sav', 'params.sav',
  ...                                       'vis_YY.sav', 'vis_model_XX.sav',
  ...                                       'vis_model_YY.sav', 'settings.txt']]

  # Use either the file type specific read_fhd or the generic read function,
  # optionally specify the file type
  >>> UV.read_fhd(fhd_files)
  >>> UV.read(fhd_files, file_type='fhd')
  >>> UV.read(fhd_files)
  >>> UV.write_uvfits('tutorial.uvfits', spoof_nonessential=True)

d) FHD -> miriad
****************
::

  >>> from pyuvdata import UVData
  >>> import shutil
  >>> import os
  >>> UV = UVData()
  >>> fhd_prefix = 'pyuvdata/data/fhd_vis_data/1061316296_'

  # Construct the list of files
  >>> fhd_prefix = 'pyuvdata/data/fhd_vis_data/1061316296_'
  >>> fhd_files = [fhd_prefix + f for f in ['flags.sav', 'vis_XX.sav', 'params.sav',
  ...                                       'vis_YY.sav', 'vis_model_XX.sav',
  ...                                       'vis_model_YY.sav', 'settings.txt']]
  >>> UV.read(fhd_files)
  >>> write_file = 'tutorial.uv'
  >>> if os.path.exists(write_file):
  ...    shutil.rmtree(write_file)
  >>> UV.write_miriad(write_file)

e) CASA -> uvfits
******************
::

  >>> from pyuvdata import UVData
  >>> UV = UVData()
  >>> ms_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms'

  # Use either the file type specific read_ms or the generic read function,
  # optionally specify the file type
  # note that reading CASA measurement sets requires casacore to be installed
  >>> UV.read_ms(ms_file)
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/SPECTRAL_WINDOW: 14 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms: 23 columns, 1360 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/POLARIZATION: 4 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/ANTENNA: 8 columns, 28 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/OBSERVATION: 9 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/FIELD: 9 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/HISTORY: 9 columns, 6 rows

  # Write out uvfits file
  >>> UV.write_uvfits('tutorial.uvfits', spoof_nonessential=True)

f) CASA -> miriad
*****************
::

  >>> from pyuvdata import UVData
  >>> import shutil
  >>> import os
  >>> UV=UVData()
  >>> ms_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms'

  # note that reading CASA measurement sets requires casacore to be installed
  >>> UV.read(ms_file)
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/SPECTRAL_WINDOW: 14 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms: 23 columns, 1360 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/POLARIZATION: 4 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/ANTENNA: 8 columns, 28 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/OBSERVATION: 9 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/FIELD: 9 columns, 1 rows
  Successful readonly open of default-locked table pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.ms/HISTORY: 9 columns, 6 rows

  # Write out Miriad file
  >>> write_file = 'tutorial.uv'
  >>> if os.path.exists(write_file):
  ...    shutil.rmtree(write_file)
  >>> UV.write_miriad(write_file)

g) miriad -> uvh5
*****************
::

  >>> from pyuvdata import UVData
  >>> UV = UVData()

  # This miriad file is known to be a drift scan.
  >>> miriad_file = 'pyuvdata/data/new.uvA'
  >>> UV.read(miriad_file)

  # Write out the uvh5 file
  # note that writing uvh5 files requires h5py to be installed
  >>> UV.write_uvh5('tutorial.uvh5')

h) uvfits -> uvh5
*****************
::

   >>> from pyuvdata import UVData
   >>> import os
   >>> UV = UVData()
   >>> uvfits_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
   >>> UV.read(uvfits_file)

   # Write out the uvh5 file
   # note that writing uvh5 files requires h5py to be installed
   >>> write_file = 'tutorial.uvh5'
   >>> if os.path.exists(write_file):
   ...    os.remove(write_file)
   >>> UV.write_uvh5(write_file)

   # Read the uvh5 file back in. Use either the file type specific read_uvh5 or
   # the generic read function, optionally specify the file type
   >>> UV.read_uvh5(write_file)
   >>> UV.read(write_file, file_type='uvh5')
   >>> UV.read(write_file)


UVData: Quick data access
--------------------------
A small suite of functions are available to quickly access numpy arrays of data,
flags, and nsamples.

a) Data for single antenna pair / polarization combination.
************************************************************
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)
  >>> data = UV.get_data(1, 2, 'rr')  # data for ant1=1, ant2=2, pol='rr'
  >>> times = UV.get_times(1, 2)  # times corresponding to 0th axis in data
  >>> print(data.shape)
  (9, 64)
  >>> print(times.shape)
  (9,)

  # One can equivalently make any of these calls with the input wrapped in a tuple.
  >>> data = UV.get_data((1, 2, 'rr'))
  >>> times = UV.get_times((1, 2))

b) Flags and nsamples for above data.
*********************************************
::

  >>> flags = UV.get_flags(1, 2, 'rr')
  >>> nsamples = UV.get_nsamples(1, 2, 'rr')
  >>> print(flags.shape)
  (9, 64)
  >>> print(nsamples.shape)
  (9, 64)

c) Data for single antenna pair, all polarizations.
************************************************************
::

  >>> data = UV.get_data(1, 2)
  >>> print(data.shape)
  (9, 64, 4)

  # Can also give baseline number
  >>> data2 = UV.get_data(UV.antnums_to_baseline(1, 2))
  >>> print(np.all(data == data2))
  True

d) Data for single polarization, all baselines.
************************************************************
::

  >>> data = UV.get_data('rr')
  >>> print(data.shape)
  (1360, 64)

e) Iterate over all antenna pair / polarizations.
************************************************************
::

  >>> for key, data in UV.antpairpol_iter():
  ...  flags = UV.get_flags(key)
  ...  nsamples = UV.get_nsamples(key)

    # Do something with the data, flags, nsamples

f) Convenience functions to ask what antennas, baselines, and pols are in the data.
******************************************************************************************
::

  # Get all unique antennas in data
  >>> print(UV.get_ants())
  [ 0  1  2  3  6  7  8 11 14 18 19 20 21 22 23 24 26 27]

  # Get all baseline nums in data, print first 10.
  >>> print(UV.get_baseline_nums()[0:10])
  [67586 67587 67588 67591 67592 67593 67596 67599 67603 67604]

  # Get all (ordered) antenna pairs in data (same info as baseline_nums), print first 10.
  >>> print(UV.get_antpairs()[0:10])
  [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (0, 11), (0, 14), (0, 18), (0, 19)]

  # Get all antenna pairs and polariations, i.e. keys produced in UV.antpairpol_iter(), print first 5.
  >>> print(UV.get_antpairpols()[0:5])
  [(0, 1, 'rr'), (0, 1, 'll'), (0, 1, 'rl'), (0, 1, 'lr'), (0, 2, 'rr')]

g) Quick access to file attributes of a UV* object (UVData, UVCal, UVBeam)
******************************************************************************************
::

  ## in bash ##
  pyuvdata_inspect.py --attr=data_array.shape <uv*_file> # will print data_array.shape to stdout

  pyuvdata_inspect.py --attr=Ntimes,Nfreqs,Nbls <uv*_file> # will print Ntimes,Nfreqs,Nbls to stdout

  pyuvdata_inspect.py -i <uv*_file> # will load object to instance name "uv" and will remain in interpreter

UVData: Phasing
-----------------------
Phasing/unphasing data
::

  >>> from pyuvdata import UVData
  >>> from astropy.time import Time
  >>> UV = UVData()
  >>> miriad_file = 'pyuvdata/data/new.uvA'
  >>> UV.read(miriad_file)
  >>> print(UV.phase_type)
  drift

  # Phase the data to the zenith at first time step
  >>> UV.phase_to_time(Time(UV.time_array[0], format='jd'))
  >>> print(UV.phase_type)
  phased

  # Undo phasing to try another phase center
  >>> UV.unphase_to_drift()

  # Phase to a specific ra/dec/epoch (in radians)
  >>> UV.phase(5.23368, 0.710940, epoch="J2000")

UVData: Plotting
------------------
Making a simple waterfall plot.

Note: there is now support for reading in only part of a uvfits, uvh5 or miriad file
(see :ref:`UVData: Working with large files`), so you need not read in the
entire file to plot one waterfall.
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)
  >>> print(UV.data_array.shape)
  (1360, 1, 64, 4)
  >>> print(UV.Ntimes)
  15
  >>> print(UV.Nfreqs)
  64
  >>> bl = UV.antnums_to_baseline(1, 2)
  >>> print(bl)
  69635
  >>> bl_ind = np.where(UV.baseline_array == bl)[0]

  # Amplitude waterfall for 0th spectral window and 0th polarization
  >>> plt.imshow(np.abs(UV.data_array[bl_ind, 0, :, 0])) # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

  # Update: With new UI features, making waterfalls is easier than ever!
  >>> plt.imshow(np.abs(UV.get_data((1, 2, UV.polarization_array[0])))) # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP


UVData: Location conversions
--------------------------------
A number of conversion methods exist to map between different coordinate systems
for locations on the earth.

a) Getting antenna positions in topocentric frame in units of meters
***************************************************************************
::

  # directly from UVData object
  >>> from pyuvdata import UVData
  >>> uvd = UVData()
  >>> uvd.read('pyuvdata/data/new.uvA')
  >>> antpos, ants = uvd.get_ENU_antpos()

  # using utils
  >>> from pyuvdata import utils
  >>> uvd = UVData()
  >>> uvd.read('pyuvdata/data/new.uvA')

  # get antennas positions in ECEF
  >>> antpos = uvd.antenna_positions + uvd.telescope_location

  # convert to topocentric (East, North, Up or ENU) coords.
  >>> antpos = utils.ENU_from_ECEF(antpos, *uvd.telescope_location_lat_lon_alt)

UVData: Selecting data
-----------------------
The select method lets you select specific antennas (by number or name),
antenna pairs, frequencies (in Hz or by channel number), times or polarizations
to keep in the object while removing others.

Note: The same select interface is now supported on the read for uvfits, uvh5
and miriad files (see :ref:`UVData: Working with large files`), so you need not
read in the entire file before doing the select.

a) Select 3 antennas to keep using the antenna number.
********************************************************
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # print all the antennas numbers with data in the original file
  >>> print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  [ 0  1  2  3  6  7  8 11 14 18 19 20 21 22 23 24 26 27]
  >>> UV.select(antenna_nums=[0, 11, 20])

  # print all the antennas numbers with data after the select
  >>> print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  [ 0 11 20]

b) Select 3 antennas to keep using the antenna names, also select 5 frequencies to keep.
*****************************************************************************************
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # print all the antenna names with data in the original file
  >>> unique_ants = np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist())
  >>> print([UV.antenna_names[np.where(UV.antenna_numbers==a)[0][0]] for a in unique_ants])
  ['W09', 'E02', 'E09', 'W01', 'N06', 'N01', 'E06', 'E08', 'W06', 'W04', 'N05', 'E01', 'N04', 'E07', 'W05', 'N02', 'E03', 'N08']

  # print how many frequencies in the original file
  >>> print(UV.freq_array.size)
  64
  >>> UV.select(antenna_names=['N02', 'E09', 'W06'], frequencies=UV.freq_array[0,0:4])

  # print all the antenna names with data after the select
  >>> unique_ants = np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist())
  >>> print([UV.antenna_names[np.where(UV.antenna_numbers==a)[0][0]] for a in unique_ants])
  ['E09', 'W06', 'N02']

  # print all the frequencies after the select
  >>> print(UV.freq_array)
  [[3.6304542e+10 3.6304667e+10 3.6304792e+10 3.6304917e+10]]

c) Select a few antenna pairs to keep
******************************************
::

  >>> from pyuvdata import UVData
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # print how many antenna pairs with data in the original file
  >>> print(len(set(zip(UV.ant_1_array, UV.ant_2_array))))
  153
  >>> UV.select(bls=[(0, 2), (6, 0), (0, 21)])

  # note that order of the values in the pair does not matter
  # print all the antenna pairs after the select
  >>> print(list(set(zip(UV.ant_1_array, UV.ant_2_array))))
  [(0, 6), (0, 21), (0, 2)]

d) Select antenna pairs and polarizations using ant_str argument
********************************************************************

Basic options are 'auto', 'cross', or 'all'. 'auto' returns just the
autocorrelations (all pols), while 'cross' returns just the cross-correlations
(all pols).  The ant_str can also contain:

1. Individual antenna number(s):
________________________________

- 1: returns all antenna pairs containing antenna number 1 (including the auto correlation)
- 1,2: returns all antenna pairs containing antennas 1 and/or 2

::

  >>> from pyuvdata import UVData
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # Print the number of antenna pairs in the original file
  >>> print(len(UV.get_antpairs()))
  153

  # Apply select to UV object
  >>> UV.select(ant_str='1,2,3')

  # Print the number of antenna pairs after the select
  >>> print(len(UV.get_antpairs()))
  48

2. Individual baseline(s):
___________________________

- 1_2: returns only the antenna pair (1,2)
- 1_2,1_3,1_10: returns antenna pairs (1,2),(1,3),(1,10)
- (1,2)_3: returns antenna pairs (1,3),(2,3)
- 1_(2,3): returns antenna pairs (1,2),(1,3)

::

  >>> from pyuvdata import UVData
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # Print the number of antenna pairs in the original file
  >>> print(len(UV.get_antpairs()))
  153

  # Apply select to UV object
  >>> UV.select(ant_str='(1,2)_(3,6)')

  # Print the antennas pairs with data after the select
  >>> print(UV.get_antpairs())
  [(1, 3), (1, 6), (2, 3), (2, 6)]

3. Antenna number(s) and polarization(s):
__________________________________________

When polarization information is passed with antenna numbers,
all antenna pairs kept in the object will retain data for each specified polarization

- 1x: returns all antenna pairs containing antenna number 1 and polarizations xx and xy
- 2x_3y: returns the antenna pair (2,3) and polarization xy
- 1r_2l,1l_3l,1r_4r: returns antenna pairs (1,2), (1,3), (1,4) and polarizations rr, ll, and rl.  This yields a complete list of baselines with polarizations of 1r_2l, 1l_2l, 1r_2r, 1r_3l, 1l_3l, 1r_3r, 1r_11l, 1l_11l, and 1r_11r.
- (1x,2y)_(3x,4y): returns antenna pairs (1,3),(1,4),(2,3),(2,4) and polarizations xx, yy, xy, and yx
- 2l_3: returns antenna pair (2,3) and polarizations ll and lr
- 2r_3: returns antenna pair (2,3) and polarizations rr and rl
- 1l_3,2x_3: returns antenna pairs (1,3), (2,3) and polarizations ll, lr, xx, and xy
- 1_3l,2_3x: returns antenna pairs (1,3), (2,3) and polarizations ll, rl, xx, and yx

::

  >>> from pyuvdata import UVData
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # Print the number of antennas and polarizations with data in the original file
  >>> print((len(UV.get_antpairs()), UV.get_pols()))
  (153, ['rr', 'll', 'rl', 'lr'])

  # Apply select to UV object
  >>> UV.select(ant_str='1r_2l,1l_3l,1r_6r')

  # Print all the antennas numbers and polarizations with data after the select
  >>> print((UV.get_antpairs(), UV.get_pols()))
  ([(1, 2), (1, 3), (1, 6)], ['rr', 'll', 'rl'])

4. Stokes parameter(s):
________________________

Can be passed lowercase or uppercase

- i,I: keeps only Stokes I
- q,V: keeps both Stokes Q and V

5. Minus sign(s):
________________________

If a minus sign is present in front of an antenna number, it will not be kept in the data

- 1,-3: returns all antenna pairs containing antenna 1, but removes any containing antenna 3
- 1,-1_3: returns all antenna pairs containing antenna 1, except the antenna pair (1,3)
- 1x_(-3y,10x): returns antenna pair (1,10) and polarization xx

::

  >>> from pyuvdata import UVData
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)

  # Print the number of antenna pairs in the original file
  >>> print(len(UV.get_antpairs()))
  153

  # Apply select to UV object
  >>> UV.select(ant_str='1,-1_3')

  # Print the number of antenna pairs with data after the select
  >>> print(len(UV.get_antpairs()))
  16

e) Select data and return new object (leaving original intact).
********************************************************************
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> UV = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> UV.read(filename)
  >>> UV2 = UV.select(antenna_nums=[0, 11, 20], inplace=False)

  # print all the antennas numbers with data in the original file
  >>> print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  [ 0  1  2  3  6  7  8 11 14 18 19 20 21 22 23 24 26 27]

  # print all the antennas numbers with data after the select
  >>> print(np.unique(UV2.ant_1_array.tolist() + UV2.ant_2_array.tolist()))
  [ 0 11 20]

UVData: Adding data
-----------------------
The __add__ method lets you combine UVData objects along
the baseline-time, frequency, and/or polarization axis.

a) Add frequencies.
*********************
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> import copy
  >>> uv1 = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> uv1.read(filename)
  >>> uv2 = copy.deepcopy(uv1)

  # Downselect frequencies to recombine
  >>> uv1.select(freq_chans=np.arange(0, 32))
  >>> uv2.select(freq_chans=np.arange(32, 64))
  >>> uv3 = uv1 + uv2
  >>> print((uv1.Nfreqs, uv2.Nfreqs, uv3.Nfreqs))
  (32, 32, 64)

b) Add times.
****************
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> import copy
  >>> uv1 = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> uv1.read(filename)
  >>> uv2 = copy.deepcopy(uv1)

  # Downselect times to recombine
  >>> times = np.unique(uv1.time_array)
  >>> uv1.select(times=times[0:len(times) // 2])
  >>> uv2.select(times=times[len(times) // 2:])
  >>> uv3 = uv1 + uv2
  >>> print((uv1.Ntimes, uv2.Ntimes, uv3.Ntimes))
  (7, 8, 15)
  >>> print((uv1.Nblts, uv2.Nblts, uv3.Nblts))
  (459, 901, 1360)

c) Adding in place.
*******************
The following two commands are equivalent, and act on uv1
directly without creating a third uvdata object.
::

  >>> from pyuvdata import UVData
  >>> import numpy as np
  >>> import copy
  >>> uv1 = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> uv1.read(filename)
  >>> uv2 = copy.deepcopy(uv1)
  >>> uv1.select(times=times[0:len(times) // 2])
  >>> uv2.select(times=times[len(times) // 2:])
  >>> uv1.__add__(uv2, inplace=True)

  >>> uv1.read(filename)
  >>> uv2 = copy.deepcopy(uv1)
  >>> uv1.select(times=times[0:len(times) // 2])
  >>> uv2.select(times=times[len(times) // 2:])
  >>> uv1 += uv2

d) Reading multiple files.
****************************
If any of the read methods are given a list of files
(or list of lists for FHD datasets), each file will be read in succession
and added to the previous.
::

  >>> from pyuvdata import UVData
  >>> uv = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> uv.read(filename)
  >>> uv1 = uv.select(freq_chans=np.arange(0, 20), inplace=False)
  >>> uv2 = uv.select(freq_chans=np.arange(20, 40), inplace=False)
  >>> uv3 = uv.select(freq_chans=np.arange(40, 64), inplace=False)
  >>> uv1.write_uvfits('tutorial1.uvfits')
  >>> uv2.write_uvfits('tutorial2.uvfits')
  >>> uv3.write_uvfits('tutorial3.uvfits')
  >>> filenames = ['tutorial1.uvfits', 'tutorial2.uvfits', 'tutorial3.uvfits']
  >>> uv.read(filenames)

UVData: Working with large files
----------------------------------------------
To save on memory and time, pyuvdata supports reading only parts of uvfits, uvh5 and
miriad files.

a) Reading just the header of a uvfits file
******************************************
This option is only available for uvfits files, which separate the header which
is very lightweight to read from the metadata which takes a little more memory.
When only the header info is read in, the UVData object is not fully specified,
so only some of the expected attributes are filled out.

The read_metadata keyword is ignored for other file types.
::

  >>> from pyuvdata import UVData
  >>> uv = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> uv.read(filename, read_data=False, read_metadata=False)
  >>> print((uv.Nblts, uv.Nfreqs, uv.Npols))
  (1360, 64, 4)

  >>> print(uv.freq_array.size)
  64

  >>> print(uv.time_array)
  None

  >>> print(uv.data_array)
  None

b) Reading the metadata of a uvfits, uvh5 or miriad file
******************************************
For uvh5 and uvfits files, reading in the metadata results in a UVData object
that is still not fully specified, but every attribute except the data_array,
flag_array and nsample_array are filled out. For Miriad files, less of the
metadata can be read without reading the data, but many of the attributes
are available. For uvfits files, the metadata can be read in at the same time
as the header, or you can read in the header followed by the metadata
(both shown below).

FHD and measurement set (ms) files do not support reading only the metadata
(the read_data keyword is ignored for these file types).
::

  >>> from pyuvdata import UVData
  >>> uv = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'

  # read the header and metadata but not the data
  >>> uv.read(filename, read_data=False)

  # read the header first, then the metadata but not the data
  >>> uv.read(filename, read_data=False, read_metadata=False)
  >>> uv.read(filename, read_data=False)

  >>> print(uv.time_array.size)
  1360

  >>> print(uv.data_array)
  None

  # If the data_array, flag_array or nsample_array are needed later, they can be
  # read into the existing object:
  >>> uv.read(filename)
  >>> print(uv.data_array.shape)
  (1360, 1, 64, 4)

c) Reading only parts of uvfits, uvh5 or miriad data
****************************************************
The same options that are available for the select function can also be passed to
the read method to do the select on the read, saving memory and time if only a
portion of the data are needed.

Note that these keywords can be used for any file type, but for FHD and
measurement set (ms) files, the select is done after the read, which does not
save memory. Miriad only supports some of the selections on the read, the
unsupported ones are done after the read. Note that miriad supports a select on
read for a time range, while uvfits and uvh5 support a list of times to include.
Any of the select keywords can be used for any file type, but selects for keywords
that are not supported by the select on read for a given file type will be
done after the read, which does not save memory.
::

  >>> import numpy as np
  >>> from pyuvdata import UVData
  >>> uv = UVData()
  >>> filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  >>> uv.read(filename, freq_chans=np.arange(32))
  >>> print(uv.data_array.shape)
  (1360, 1, 32, 4)

  # Reading in the header and metadata can help with specifying what data to read in
  >>> uv = UVData()
  >>> uv.read(filename, read_data=False)
  >>> unique_times = np.unique(uv.time_array)
  >>> print(unique_times.shape)
  (15,)

  >>> times_to_keep = unique_times[[0, 2, 4]]
  >>> uv.read(filename, times=times_to_keep)
  >>> print(uv.data_array.shape)
  (179, 1, 64, 4)

  # Select a few baselines from a miriad file
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.HH.uvcA'
  >>> uv.read(filename, bls=[(9, 10), (9, 20)])
  >>> print(uv.get_antpairs())
  [(9, 10), (9, 20)]


UVData: Finding Redundant Baselines
-----------------------------------
pyuvdata.utils contains functions for finding redundant groups of baselines in an array, either by antenna positions or uvw coordinates. Baselines are considered redundant if they are within a specified tolerance distance (default is 1 meter).

The ``get_baseline_redundancies`` function accepts an array of baseline indices and an array of baseline vectors (ie, uvw coordinates) as input, and finds redundancies among the vectors as given. If the ``with_conjugates`` option is selected, it will flip baselines such that ``u > 0``, or ``v > 0 if u = 0``, or ``w > 0 if u = v = 0``. In this case, a list of ``conjugates`` is returned as well, which contains indices for the baselines that were flipped for the redundant groups. In either mode of operation, this will only return baseline indices that are in the list passed in.

The ``get_antenna_redundancies`` function accepts an array of antenna indices and an array of antenna positions as input, defines baseline vectors and indices under the positive-u condition described above, and runs ``get_baseline_redundancies`` to find redundant baselines. This is similar to running get_baseline_redundancies with the ``with_conjugates`` option, except that the baseline indices are returned such that none need to be flipped to be redundant. This is more like what's defined in the ``hera_cal`` package.

::
    >>> import numpy as np
    >>> from pyuvdata import UVData
    >>> from pyuvdata import utils as uvutils
    >>> uvd = UVData()

    # This file contains a HERA19 layout.
    >>> uvd.read_uvfits("pyuvdata/data/hera19_8hrs_uncomp_10MHz_000_05.003111-05.033750.uvfits")
    >>> uvd.unphase_to_drift(use_ant_pos=True)
    >>> tol = 0.05  # Tolerance in meters
    >>> uvd.select(times=uvd.time_array[0])

    # Returned values: list of redundant groups, corresponding mean baseline vectors, baseline lengths.
    >>> baseline_groups, vec_bin_centers, lengths = uvutils.get_baseline_redundancies(uvd.baseline_array, uvd.uvw_array, tol=tol)
    >>> print(len(baseline_groups))
    52

    # The with_conjugates option includes baselines that are redundant when reversed.
    # If used, the conjugates list will contain a list of indices of baselines that must be flipped to be redundant.
    >>> baseline_groups, vec_bin_centers, lengths, conjugates = uvutils.get_baseline_redundancies(uvd.baseline_array, uvd.uvw_array, tol=tol, with_conjugates=True)
    >>> print(len(baseline_groups))
    31

    # Using antenna positions instead
    >>> antpos, antnums = uvd.get_ENU_antpos()
    >>> baseline_groups, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(antnums, antpos, tol=tol, include_autos=True)
    >>> print(len(baseline_groups))
    31

    # get_antenna_redundancies has the option to ignore autocorrelations.
    >>> baseline_groups, vec_bin_centers, lengths = uvutils.get_antenna_redundancies(antnums, antpos, tol=tol, include_autos=False)
    >>> print(len(baseline_groups))
    30


------
UVCal
------

UVCal: Reading/writing
-----------------------
Calibration files using UVCal.

a) Reading a cal fits gain calibration file.
*************************************
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> cal = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal.read_calfits(filename)

  # Cal type:
  >>> print(cal.cal_type)
  gain

  # number of antenna polarizations and polarization type.
  >>> print((cal.Njones, cal.jones_array))
  (1, array([-5]))

  # Number of antennas with data
  >>> print(cal.Nants_data)
  19

  # Number of frequencies
  >>> print(cal.Nfreqs)
  1024

  # Shape of the gain_array
  >>> print(cal.gain_array.shape)
  (19, 1, 1024, 56, 1)

  # plot abs of all gains for first time and first jones polarization.
  >>> for ant in range(cal.Nants_data): # doctest: +SKIP
  ...    plt.plot(cal.freq_array.flatten(), np.abs(cal.gain_array[ant, 0, :, 0, 0]))
  >>> plt.xlabel('Frequency (Hz)') # doctest: +SKIP
  >>> plt.ylabel('Abs(gains)') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP


b) FHD cal to cal fits
***********************
::

  >>> from pyuvdata import UVCal
  >>> import os
  >>> obs_testfile = 'pyuvdata/data/fhd_cal_data/1061316296_obs.sav'
  >>> cal_testfile = 'pyuvdata/data/fhd_cal_data/1061316296_cal.sav'
  >>> settings_testfile = 'pyuvdata/data/fhd_cal_data/1061316296_settings.txt'

  >>> fhd_cal = UVCal()
  >>> fhd_cal.read_fhd_cal(cal_testfile, obs_testfile, settings_file=settings_testfile)
  >>> fhd_cal.write_calfits('tutorial_cal.fits', clobber=True)


UVCal: Selecting data
-----------------------
The select method lets you select specific antennas (by number or name),
frequencies (in Hz or by channel number), times or polarizations
to keep in the object while removing others.

a) Select 3 antennas to keep using the antenna number.
********************************************************************
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> cal = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal.read_calfits(filename)

  # print all the antennas numbers with data in the original file
  >>> print(cal.ant_array)
  [  9  10  20  22  31  43  53  64  65  72  80  81  88  89  96  97 104 105
   112]
  >>> cal.select(antenna_nums=[9, 22, 64])

  # print all the antennas numbers with data after the select
  >>> print(cal.ant_array)
  [ 9 22 64]

b) Select 3 antennas to keep using the antenna names, also select 5 frequencies to keep.
**********************************************************************************************
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> cal = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal.read_calfits(filename)

  # print all the antenna names with data in the original file
  >>> print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array[0:9]])
  ['ant9', 'ant10', 'ant20', 'ant22', 'ant31', 'ant43', 'ant53', 'ant64', 'ant65']

  # print all the frequencies in the original file
  >>> print(cal.freq_array)
  [[1.00000000e+08 1.00097656e+08 1.00195312e+08 ... 1.99707031e+08
    1.99804688e+08 1.99902344e+08]]
  >>> cal.select(antenna_names=['ant31', 'ant81', 'ant104'], freq_chans=np.arange(0, 4))

  # print all the antenna names with data after the select
  >>> print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array])
  ['ant31', 'ant81', 'ant104']

  # print all the frequencies after the select
  >>> print(cal.freq_array)
  [[1.00000000e+08 1.00097656e+08 1.00195312e+08 1.00292969e+08]]


UVCal: Adding data
-----------------------
The __add__ method lets you combine UVCal objects along
the antenna, time, frequency, and/or polarization axis.

a) Add frequencies.
*********************
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> import copy
  >>> cal1 = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal1.read_calfits(filename)
  >>> cal2 = copy.deepcopy(cal1)

  # Downselect frequencies to recombine
  >>> cal1.select(freq_chans=np.arange(0, 512))
  >>> cal2.select(freq_chans=np.arange(512, 1024))
  >>> cal3 = cal1 + cal2
  >>> print((cal1.Nfreqs, cal2.Nfreqs, cal3.Nfreqs))
  (512, 512, 1024)

b) Add times.
****************
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> import copy
  >>> cal1 = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal1.read_calfits(filename)
  >>> cal2 = copy.deepcopy(cal1)

  # Downselect times to recombine
  >>> times = np.unique(cal1.time_array)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal3 = cal1 + cal2
  >>> print((cal1.Ntimes, cal2.Ntimes, cal3.Ntimes))
  (28, 28, 56)

c) Adding in place.
*******************
The following two commands are equivalent, and act on cal1
directly without creating a third uvcal object.
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> import copy
  >>> cal1 = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal1.read_calfits(filename)
  >>> cal2 = copy.deepcopy(cal1)
  >>> times = np.unique(cal1.time_array)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal1.__add__(cal2, inplace=True)

  >>> cal1.read_calfits(filename)
  >>> cal2 = copy.deepcopy(cal1)
  >>> cal1.select(times=times[0:len(times) // 2])
  >>> cal2.select(times=times[len(times) // 2:])
  >>> cal1 += cal2

d) Reading multiple files.
****************************
If any of the read methods (read_calfits, read_fhd_cal) are given a list of files,
each file will be read in succession and added to the previous.
::

  >>> from pyuvdata import UVCal
  >>> import numpy as np
  >>> import copy
  >>> cal = UVCal()
  >>> filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  >>> cal.read_calfits(filename)
  >>> cal1 = cal.select(freq_chans=np.arange(0, 20), inplace=False)
  >>> cal2 = cal.select(freq_chans=np.arange(20, 40), inplace=False)
  >>> cal3 = cal.select(freq_chans=np.arange(40, 64), inplace=False)
  >>> cal1.write_calfits('tutorial1.fits')
  >>> cal2.write_calfits('tutorial2.fits')
  >>> cal3.write_calfits('tutorial3.fits')
  >>> filenames = ['tutorial1.fits', 'tutorial2.fits', 'tutorial3.fits']
  >>> cal.read_calfits(filenames)

  # For FHD cal datasets pass lists for each file type
  >>> fhd_cal = UVCal()
  >>> obs_testfiles = ['pyuvdata/data/fhd_cal_data/1061316296_obs.sav', 'pyuvdata/data/fhd_cal_data/set2/1061316296_obs.sav']
  >>> cal_testfiles = ['pyuvdata/data/fhd_cal_data/1061316296_cal.sav', 'pyuvdata/data/fhd_cal_data/set2/1061316296_cal.sav']
  >>> settings_testfiles = ['pyuvdata/data/fhd_cal_data/1061316296_settings.txt', 'pyuvdata/data/fhd_cal_data/set2/1061316296_settings.txt']
  >>> fhd_cal.read_fhd_cal(cal_testfiles, obs_testfiles, settings_file=settings_testfiles)
  diffuse_model parameter value is a string, values are different

------
UVBeam
------


UVBeam: Reading/writing
-----------------------
Reading and writing beam files using UVBeam.

a) Reading a CST power beam file
******************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> beam = UVBeam()

  # you can pass several filenames and the objects from each file will be
  # combined across the appropriate axis -- in this case frequency
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']

  # have to specify the telescope_name, feed_name, feed_version, model_name
  # and model_version because they are not included in the file
  # specify the polarization that the file represents and set rotate_pol to
  # generate the other polarization by rotating by 90 degrees.
  >>> beam.read_cst_beam(filenames, beam_type='power', frequency=[150e6, 123e6],
  ...                    feed_pol='x', rotate_pol=True, telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> print(beam.beam_type)
  power
  >>> print(beam.pixel_coordinate_system)
  az_za
  >>> print(beam.data_normalization)
  physical

  # number of beam polarizations and polarization type.
  >>> print((beam.Npols, beam.polarization_array))
  (2, array([-5, -6]))
  >>> print(beam.Nfreqs)
  2
  >>> print(beam.data_array.shape)
  (1, 1, 2, 2, 181, 360)

  # plot zenith angle cut through beam
  >>> plt.plot(beam.axis2_array, beam.data_array[0, 0, 0, 0, :, 0]) # doctest: +SKIP
  >>> plt.xscale('log') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Power') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

b) Reading a CST E-field beam file
******************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> beam = UVBeam()

  # you can pass several filenames and the objects from each file will be
  # combined across the appropriate axis -- in this case frequency
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']

  # have to specify the telescope_name, feed_name, feed_version, model_name
  # and model_version because they are not included in the file
  >>> beam.read_cst_beam(filenames, beam_type='efield', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> print(beam.beam_type)
  efield

c) Writing a regularly gridded beam FITS file
**********************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='power', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> beam.write_beamfits('tutorial.fits', clobber=True)

d) Writing a HEALPix beam FITS file
******************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='power', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')

  # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'

  # note that the `to_healpix` method requires healpy to be installed
  >>> beam.to_healpix()
  >>> beam.write_beamfits('tutorial.fits', clobber=True)

UVBeam: Selecting data
-----------------------
The select method lets you select specific image axis indices (or pixels if
pixel_coordinate_system is HEALPix), frequencies and feeds (or polarizations if
beam_type is power) to keep in the object while removing others.

a) Selecting a range of Zenith Angles
******************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='power', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> new_beam = beam.select(axis2_inds=np.arange(0, 20), inplace=False)

  # plot zenith angle cut through beams
  >>> plt.plot(beam.axis2_array, beam.data_array[0, 0, 0, 0, :, 0], # doctest: +SKIP
  ...         new_beam.axis2_array, new_beam.data_array[0, 0, 0, 0, :, 0], 'r')
  >>> plt.xscale('log') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Power') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

UVBeam: Converting to beam types and coordinate systems
---------------------------------------------------------------------

a) Convert a regularly gridded az_za power beam to HEALpix (leaving original intact).
********************************************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> import healpy as hp
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='power', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')

  # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'
  >>> hpx_beam = beam.to_healpix(inplace=False)
  >>> hp.mollview(hpx_beam.data_array[0,0,0,0,:]) # doctest: +SKIP

b) Convert a regularly gridded az_za efield beam to HEALpix (leaving original intact).
********************************************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> import healpy as hp
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='efield', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')

  # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'
  >>> hpx_beam = beam.to_healpix(inplace=False)
  >>> hp.mollview(np.abs(hpx_beam.data_array[0,0,0,0,:])) # doctest: +SKIP


c) Convert a regularly gridded efield beam to a power beam (leaving original intact).
********************************************************************
::

  >>> from pyuvdata import UVBeam
  >>> import copy
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='efield', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> new_beam = beam.efield_to_power(inplace=False)

  # plot zenith angle cut through the beams
  >>> plt.plot(beam.axis2_array, beam.data_array[1, 0, 0, 0, :, 0].real, label='E-field real') # doctest: +SKIP
  >>> plt.plot(beam.axis2_array, beam.data_array[1, 0, 0, 0, :, 0].imag, 'r', label='E-field imaginary') # doctest: +SKIP
  >>> plt.plot(new_beam.axis2_array, np.sqrt(new_beam.data_array[0, 0, 0, 0, :, 0]), 'black', label='sqrt Power') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Magnitude') # doctest: +SKIP
  >>> plt.legend() # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

Generating pseudo Stokes ('pI', 'pQ', 'pU', 'pV') beams
********************************************************************
::

  >>> from pyuvdata import UVBeam
  >>> from pyuvdata import utils as uvutils
  >>> import numpy as np
  >>> import healpy as hp
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='efield', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> beam.interpolation_function = 'az_za_simple'
  >>> pstokes_beam = beam.to_healpix(inplace=False)
  >>> pstokes_beam.efield_to_pstokes()
  >>> pstokes_beam.peak_normalize()

  # plotting pseudo-stokes I
  >>> pol_array = pstokes_beam.polarization_array
  >>> pstokes = uvutils.polstr2num('pI')
  >>> pstokes_ind = np.where(np.isin(pol_array, pstokes))[0][0]
  >>> hp.mollview(np.abs(pstokes_beam.data_array[0, 0, pstokes_ind, 0, :])) # doctest: +SKIP

Calculating pseudo Stokes ('pI', 'pQ', 'pU', 'pV') beam area and beam squared area
********************************************************************
::

  >>> from pyuvdata import UVBeam
  >>> import numpy as np
  >>> beam = UVBeam()
  >>> filenames = ['pyuvdata/data/HERA_NicCST_150MHz.txt', 'pyuvdata/data/HERA_NicCST_123MHz.txt']
  >>> beam.read_cst_beam(filenames, beam_type='efield', telescope_name='HERA',
  ...                    feed_name='PAPER_dipole', feed_version='0.1',
  ...                    model_name='E-field pattern - Rigging height 4.9m',
  ...                    model_version='1.0')
  >>> beam.interpolation_function = 'az_za_simple'

  # note that the `to_healpix` method requires healpy to be installed
  >>> pstokes_beam = beam.to_healpix(inplace=False)
  >>> pstokes_beam.efield_to_pstokes()
  >>> pstokes_beam.peak_normalize()

  # calculating beam area
  >>> freqs = pstokes_beam.freq_array
  >>> pI_area = pstokes_beam.get_beam_area('pI')
  >>> pQ_area = pstokes_beam.get_beam_area('pQ')
  >>> pU_area = pstokes_beam.get_beam_area('pU')
  >>> pV_area = pstokes_beam.get_beam_area('pV')
  >>> pI_area1, pI_area2 = round(pI_area[0].real, 5), round(pI_area[1].real, 5)
  >>> pQ_area1, pQ_area2 = round(pQ_area[0].real, 5), round(pQ_area[1].real, 5)
  >>> pU_area1, pU_area2 = round(pU_area[0].real, 5), round(pU_area[1].real, 5)
  >>> pV_area1, pV_area2 = round(pV_area[0].real, 5), round(pV_area[1].real, 5)

  >>> print ('Beam area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][0]*1e-6, pI_area1, pU_area1, pU_area1, pV_area1))
  Beam area at 123.0 MHz for pseudo-stokes
  I: 0.05734
  Q: 0.03339
  U: 0.03339
  V: 0.05372

  >>> print ('Beam area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][1]*1e-6, pI_area2, pU_area2, pU_area2, pV_area2))
  Beam area at 150.0 MHz for pseudo-stokes
  I: 0.03965
  Q: 0.02265
  U: 0.02265
  V: 0.03664

  # calculating beam squared area
  >>> freqs = pstokes_beam.freq_array
  >>> pI_sq_area = pstokes_beam.get_beam_sq_area('pI')
  >>> pQ_sq_area = pstokes_beam.get_beam_sq_area('pQ')
  >>> pU_sq_area = pstokes_beam.get_beam_sq_area('pU')
  >>> pV_sq_area = pstokes_beam.get_beam_sq_area('pV')
  >>> pI_sq_area1, pI_sq_area2 = round(pI_sq_area[0].real, 5), round(pI_sq_area[1].real, 5)
  >>> pQ_sq_area1, pQ_sq_area2 = round(pQ_sq_area[0].real, 5), round(pQ_sq_area[1].real, 5)
  >>> pU_sq_area1, pU_sq_area2 = round(pU_sq_area[0].real, 5), round(pU_sq_area[1].real, 5)
  >>> pV_sq_area1, pV_sq_area2 = round(pV_sq_area[0].real, 5), round(pV_sq_area[1].real, 5)

  >>> print ('Beam squared area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][0]*1e-6, pI_sq_area1, pU_sq_area1, pU_sq_area1, pV_sq_area1))
  Beam squared area at 123.0 MHz for pseudo-stokes
  I: 0.02439
  Q: 0.01161
  U: 0.01161
  V: 0.02426

  >>> print ('Beam squared area at {} MHz for pseudo-stokes\nI: {}\nQ: {}\nU: {}\nV: {}'.format(freqs[0][1]*1e-6, pI_sq_area2, pU_sq_area2, pU_sq_area2, pV_sq_area2))
  Beam squared area at 150.0 MHz for pseudo-stokes
  I: 0.01693
  Q: 0.0079
  U: 0.0079
  V: 0.01683

-----------------
Tutorial Cleanup
-----------------
::

  # delete all written files
  >>> import shutil
  >>> import os
  >>> import glob
  >>> filelist = glob.glob('tutorial*fits') + glob.glob('tutorial*.uvh5')
  >>> for f in filelist:
  ...     os.remove(f)
  >>> shutil.rmtree('tutorial.uv')
