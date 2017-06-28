Tutorial
========

UVData: File conversion
---------
Converting between tested data formats

a) miriad (aipy) -> uvfits
**************************
::

  from pyuvdata import UVData
  UV = UVData()
  miriad_file = 'pyuvdata/data/new.uvA'
  UV.read_miriad(miriad_file)  # this miriad file is known to be a drift scan
  UV.write_uvfits('new.uvfits', force_phase=True, spoof_nonessential=True)  # write out the uvfits file

b) uvfits -> miriad (aipy)
**************************
::

  from pyuvdata import UVData
  UV = UVData()
  uvfits_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(uvfits_file)
  UV.write_miriad('day2_TDEM0003_10s_norx_1src_1spw.uv')  # write out the miriad file

c) FHD -> uvfits
****************
When reading FHD format, we need to point to several files.::

  from pyuvdata import UVData
  UV = UVData()
  fhd_prefix = 'pyuvdata/data/fhd_vis_data/1061316296_'
  # Construct the list of files
  fhd_files = [fhd_prefix + f for f in ['flags.sav', 'vis_XX.sav', 'params.sav',
                                        'vis_YY.sav', 'vis_model_XX.sav',
                                        'vis_model_YY.sav', 'settings.txt']]
  UV.read_fhd(fhd_files)
  UV.write_uvfits('1061316296.uvfits', spoof_nonessential=True)

d) FHD -> miriad (aipy)
****************
::

  from pyuvdata import UVData
  UV = UVData()
  fhd_prefix = 'pyuvdata/data/fhd_vis_data/1061316296_'
  # Construct the list of files
  fhd_files = [fhd_prefix + f for f in ['flags.sav', 'vis_XX.sav', 'params.sav',
                                        'vis_YY.sav', 'vis_model_XX.sav',
                                        'vis_model_YY.sav', 'settings.txt']]
  UV.read_fhd(fhd_files)
  UV.write_uvfits('1061316296.uvfits')

e) CASA -> uvfits
******************
::
   from pyuvdata import UVData
   UV=UVData()
   ms_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1scan.ms'
   UV.read_ms(ms_file)
   UV.write_uvfits('new.uvfits')#write out uvfits file

f) CASA -> miriad (aipy)
******************
::
   from pyuvdata import UVData
   UV=UVData()
   ms_file = 'pyuvdata/data/day2_TDEM0003_10s_norx_1scan.ms'
   UV.read_ms(ms_file)
   UV.write_miriad('new.uvfits')#write out miriad file



UVData: Phasing
---------
Phasing/unphasing data::

  from pyuvdata import UVData
  import ephem
  UV = UVData()
  miriad_file = 'pyuvdata/data/new.uvA'
  UV.read_miriad(miriad_file)
  print(UV.phase_type)  # Data is a drift scan
  UV.phase_to_time(UV.time_array[0])  # Phases the data to the zenith at first time step
  print(UV.phase_type)  # Data should now be phased
  UV.unphase_to_drift()  # Undo phasing to try another phase center
  UV.phase(5.23368, 0.710940, ephem.J2000)  # Phase to a specific ra/dec/epoch (in radians)

UVData: Plotting
---------
Making a simple waterfall plot::

  from pyuvdata import UVData
  import numpy as np
  import matplotlib.pyplot as plt
  UV = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(filename)
  print(UV.data_array.shape)  # Data should have shape (Nblts, Nspws, Nfreqs, Npols)
  print(UV.Ntimes)  # Number of time samples in data
  print(UV.Nfreqs)  # Number of frequency channels in data
  bl = UV.antnums_to_baseline(1, 2)  # Convert antenna numbers  (e.g. 1, 2) to baseline number
  bl_ind = np.where(UV.baseline_array == bl)[0]  # Indices corresponding to baseline
  plt.imshow(np.abs(UV.data_array[bl_ind, 0, :, 0]))  # Amplitude waterfall for 0th spectral window and 0th polarization
  plt.show()

Update: With new UI features, making waterfalls is easier than ever!::

  plt.imshow(np.abs(UV.get_data((1, 2, UV.polarization_array[0]))))
  plt.show()

UVData: Quick data access
---------
A small suite of functions are available to quickly access numpy arrays of data,
flags, and nsamples.

a) Data for single antenna pair / polarization combination.
***************
::

  from pyuvdata import UVData
  import numpy as np
  UV = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(filename)
  data = UV.get_data((1, 2, 'rr'))  # data for ant1=1, ant2=2, pol='rr'
  times = UV.get_times((1, 2))  # times corresponding to 0th axis in data
  print(data.shape)
  print(times.shape)

b) Flags and nsamples for above data.
***************
::

  flags = UV.get_flags((1, 2, 'rr'))
  nsamples = UV.get_nsamples((1, 2, 'rr'))
  print(flags.shape)
  print(nsamples.shape)

c) Data for single antenna pair, all polarizations.
***************
::

  data = UV.get_data((1, 2))
  print(data.shape)
  data2 = UV.get_data(UV.antnums_to_baseline(1, 2))  # Can also give baseline number
  print(np.all(data == data2))

d) Data for single polarization, all baselines.
***************
::

  data = UV.get_data('rr')
  print(data.shape)

e) Iterate over all antenna pair / polarizations.
***************
::

  for key, data in UV.antpairpol_iter():
    print(key)
    flags = UV.get_flags(key)
    nsamples = UV.get_nsamples(key)
    # Do something with the data, flags, nsamples

f) Convenience functions to ask what antennas, baselines, and pols are in the data.
***************
::

  print(UV.get_ants())  # All unique antennas in data
  print(UV.get_baseline_nums())  # All baseline nums in data
  print(UV.get_antpairs())  # All (ordered) antenna pairs in data (same info as baseline_nums)
  print(UV.get_antpairpols)  # All antenna pairs and polariations.
                             # ie, keys produced in UV.antpairpol_iter().

UVData: Selecting data
---------
The select method lets you select specific antennas (by number or name),
antenna pairs, frequencies (in Hz or by channel number), times or polarizations
to keep in the object while removing others.

a) Select 3 antennas to keep using the antenna number.
****************
::

  from pyuvdata import UVData
  import numpy as np
  UV = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(filename)
  # print all the antennas numbers with data in the original file
  print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  UV.select(antenna_nums=[0, 11, 20])
  # print all the antennas numbers with data after the select
  print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))

b) Select 3 antennas to keep using the antenna names, also select 5 frequencies to keep.
****************
::

  from pyuvdata import UVData
  import numpy as np
  UV = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(filename)
  # print all the antenna names with data in the original file
  unique_ants = np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist())
  print([UV.antenna_names[np.where(UV.antenna_numbers==a)[0][0]] for a in unique_ants])
  # print all the frequencies in the original file
  print(UV.freq_array)
  UV.select(antenna_names=['N02', 'E09', 'W06'], frequencies=UV.freq_array[0,0:4])
  # print all the antenna names with data after the select
  unique_ants = np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist())
  print([UV.antenna_names[np.where(UV.antenna_numbers==a)[0][0]] for a in unique_ants])
  # print all the frequencies after the select
  print(UV.freq_array)

c) Select a few antenna pairs to keep
****************
::

  from pyuvdata import UVData
  UV = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(filename)
  # print all the antenna pairs with data in the original file
  print(set(zip(UV.ant_1_array, UV.ant_2_array)))
  UV.select(ant_pairs_nums=[(0, 2), (6, 0), (0, 21)])
  # note that order of the values in the pair does not matter
  # print all the antenna pairs after the select
  print(set(zip(UV.ant_1_array, UV.ant_2_array)))

d) Select data and return new object (leaving original in tact).
****************
::

  from pyuvdata import UVData
  import numpy as np
  UV = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  UV.read_uvfits(filename)
  UV2 = UV.select(antenna_nums=[0, 11, 20], inplace=False)
  # print all the antennas numbers with data in the original file
  print(np.unique(UV.ant_1_array.tolist() + UV.ant_2_array.tolist()))
  # print all the antennas numbers with data after the select
  print(np.unique(UV2.ant_1_array.tolist() + UV2.ant_2_array.tolist()))

UVData: Adding data
---------
The __add__ method lets you combine UVData objects along
the baseline-time, frequency, and/or polarization axis.

a) Add frequencies.
****************
::

  from pyuvdata import UVData
  import numpy as np
  import copy
  uv1 = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  uv1.read_uvfits(filename)
  uv2 = copy.deepcopy(uv1)
  # Downselect frequencies to recombine
  uv1.select(freq_chans=np.arange(0, 32))
  uv2.select(freq_chans=np.arange(32, 64))
  uv3 = uv1 + uv2
  print(uv1.Nfreqs, uv2.Nfreqs, uv3.Nfreqs)

b) Add times.
****************
::

  from pyuvdata import UVData
  import numpy as np
  import copy
  uv1 = UVData()
  filename = 'pyuvdata/data/day2_TDEM0003_10s_norx_1src_1spw.uvfits'
  uv1.read_uvfits(filename)
  uv2 = copy.deepcopy(uv1)
  # Downselect times to recombine
  times = np.unique(uv1.time_array)
  uv1.select(times=times[0:len(times) / 2])
  uv2.select(times=times[len(times) / 2:])
  uv3 = uv1 + uv2
  print(uv1.Ntimes, uv2.Ntimes, uv3.Ntimes)
  print(uv1.Nblts, uv2.Nblts, uv3.Nblts)

c) Adding in place.
****************
The following two commands are equivalent, and act on uv1
directly without creating a third uvdata object.
::

  uv1.__add__(uv2, inplace=True)
  uv1 += uv2

d) Reading multiple files.
****************
If any of the read methods are given a list of files
(or list of lists in the case of read_fhd), each file will be read in succession
and added to the previous.
::

  from pyuvdata import UVData
  uv = UVData()
  filenames = ['file1.uvfits', 'file2.uvfits', 'file3.uvfits']
  uv.read_uvfits(filenames)


UVCal: Reading/writing
---------
Calibration files using UVCal.

a) Reading a gain calibration file.
****************
::

  from pyuvdata import UVCal
  import numpy as np
  import matplotlib.pyplot as plt
  cal = UVCal()
  filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  cal.read_calfits(filename)
  print 'Cal Type = ', cal.cal_type  # should print out 'gains'
  print 'Number of jones parameters = ', cal.Njones, cal.jones_array  # number of antenna polarizations and polarization type.
  print 'Number of antennas with data = ', cal.Nants_data
  print 'Number of frequencies = ', cal.Nfreqs
  print 'Shape of the gain_array', cal.gain_array.shape  # (cal.Nants_data, cal.Nfreqs, cal.Ntimes, cal.Njones)
  for ant in range(cal.Nants_data):
      plt.plot(cal.freq_array.flatten(), np.abs(cal.gain_array[ant, 0, :, 0, 0]))  # plot abs of all gains for first time and first jones polarization.
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Abs(gains)')
  plt.show()


b) Writing a gain calibration file.
****************
::

  from pyuvdata import UVCal
  import numpy as np
  time_array = 2457698 + np.linspace(.2, .3, 16)  # time_array in JD
  Ntimes = len(time_array)
  freq_array = np.linspace(1e6, 2e6, 1024)  # frequency array in Hz
  Nfreqs = len(freq_array)
  jones_array = np.array([-5, -6])  #  only 2 jones parameters.
  Njones = len(jones_array)
  ant_array = np.arange(19)
  Nants_data = len(ant_array)
  antenna_names = np.array(['ant{0}.format(ant)' for ant in ant_array])
  Nspws = 1  # only 1 spw is supported
  # Generate fake data
  gains = (np.random.randn(Nants_data, Nspws, Nfreqs, Ntimes, Njones)
           + 1j*np.random.randn(Nants_data, Nspws, Nfreqs, Ntimes, Njones))
  flags = np.ones_like(gains, dtype=np.bool)
  chisq = np.random.randn(Nants_data, Nspws, Nfreqs, Ntimes, Njones)

  cal = UVCal()
  cal.set_gain()
  cal.Nfreqs = Nfreqs
  cal.Njones = Njones
  cal.Ntimes = Ntimes
  cal.history = 'This is an example file generated from tutorial 5b of pycaldata.'
  cal.Nspws = 1
  cal.freq_array = freq_array.reshape(cal.Nspws, -1)
  cal.freq_range = [freq_array[0], freq_array[-1]]  # valid frequencies for solutions.
  cal.channel_width = np.diff(freq_array)[0]
  cal.jones_array = jones_array
  cal.time_array = time_array
  cal.integration_time = np.diff(time_array)[0]
  cal.gain_convention = 'divide'  # Use this operation to apply gain solution.
  cal.x_orientation = 'east'  # orientation of 1st jones parameter.
  cal.time_range = [time_array[0], time_array[-1]]
  cal.telescope_name = 'Fake Telescope'
  cal.Nants_data = Nants_data
  cal.Nants_telescope = Nants_data  # have solutions for all antennas in array.
  cal.ant_array = ant_array
  cal.antenna_names = antenna_names
  cal.antenna_numbers = ant_array
  cal.flag_array = flags
  cal.gain_array = gains
  cal.quality_array = chisq

  cal.write_calfits('tutorial5b.fits')

UVCal: Selecting data
---------
The select method lets you select specific antennas (by number or name),
frequencies (in Hz or by channel number), times or polarizations
to keep in the object while removing others.

a) Select 3 antennas to keep using the antenna number.
****************
::

  from pyuvdata import UVCal
  import numpy as np
  cal = UVCal()
  filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  cal.read_calfits(filename)
  # print all the antennas numbers with data in the original file
  print(cal.ant_array)
  cal.select(antenna_nums=[9, 22, 64])
  # print all the antennas numbers with data after the select
  print(cal.ant_array)

b) Select 3 antennas to keep using the antenna names, also select 5 frequencies to keep.
****************
::

  from pyuvdata import UVCal
  import numpy as np
  cal = UVCal()
  filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
  cal.read_calfits(filename)
  # print all the antenna names with data in the original file
  print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array])
  # print all the frequencies in the original file
  print(cal.freq_array)
  cal.select(antenna_names=['ant31', 'ant81', 'ant104'], freq_chans=np.arange(0, 4))
  # print all the antenna names with data after the select
  print([cal.antenna_names[np.where(cal.antenna_numbers==a)[0][0]] for a in cal.ant_array])
  # print all the frequencies after the select
  print(cal.freq_array)
