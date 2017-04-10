Tutorial
========

Example 1
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
  UV.write_uvfits('day2_TDEM0003_10s_norx_1src_1spw.uv')  # write out the miriad file

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

Example 2
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

Example 3
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

Example 4
---------
Selecting data: The select method lets you select specific antennas, frequencies,
times or polarizations to keep in the object while removing others.

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

  Example 5
  ---------
  Calibration files using UVCal.

  a) Reading a gain calibration file.
  ****************
  ::

    from pyuvdata import UVCal
    import numpy as np
    import matplotlib.pyplot as plt
    UV = UVCal()
    filename = 'pyuvdata/data/zen.2457698.40355.xx.fitsA'
    UV.read_calfits(filename)
    print 'Cal Type = ', UV.cal_type  # should print out 'gains'
    print 'Number of jones parameters = ', UV.Njones, UV.jones_array  # number of antenna polarizations and polarization type.
    print 'Number of antennas with data = ', UV.Nants_data
    print 'Number of frequencies = ', UV.Nfreqs
    print 'Shape of the gain_array', UV.gain_array.shape  # (UV.Nants_data, UV.Nfreqs, UV.Ntimes, UV.Njones)
    for ant in range(UV.Nants_data):
        plt.plot(UV.freq_array.flatten(), np.abs(UV.gain_array[ant, :, 0, 0]))  # plot abs of all gains for first time and first jones polarization.
    plt.xlabel('Frequency (GHz)')
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
    jones_array = np.array([-5])  #  only 1 jones parameter (E-W).
    Njones = len(jones_array)
    antenna_numbers = np.arange(19)
    Nants_data = len(antenna_numbers)
    antenna_names = np.array(['ant{0}.format(ant)' for ant in antenna_numbers])
    # Generate fake data to process into correct format. Gains formatted with gains[pol][ant].
    gains = {}
    flags = {}
    chisq = {}
    for pol in jones_array:
        if not gains.has_key(pol):
            gains[pol] = {}
            flags[pol] = {}
            chisq[pol] = {}
        for ant in antenna_numbers:
            gains[pol][ant] = np.random.randn(Ntimes, Nfreqs) + 1j*np.random.randn(Ntimes, Nfreqs)
            flags[pol][ant] = np.ones_like(gains[pol][ant], dtype=np.bool)
            chisq[pol][ant] = np.random.randn(Ntimes, Nfreqs)

    gainarray = []
    flagarray = []
    chisqarray = []
    for pol in jones_array:
        dd = []
        fl = []
        ch = []
        for ant in antenna_numbers:
            dd.append(gains[pol][ant])
            fl.append(flags[pol][ant])
            ch.append(chisq[pol][ant])
        gainarray.append(dd)
        flagarray.append(fl)
        chisqarray.append(ch)

    gainarray = np.array(gainarray).swapaxes(0,3).swapaxes(0,1) # get it into format so shape is correct.
    flagarray = np.array(flagarray).swapaxes(0,3).swapaxes(0,1)
    chisqarray = np.array(chisqarray).swapaxes(0,3).swapaxes(0,1)

    UV = UVCal()
    UV.set_gain()
    UV.Nfreqs = Nfreqs
    UV.Njones = Njones
    UV.Ntimes = Ntimes
    UV.history = 'This is an example file generated from tutorial 5b of pyuvdata.'
    UV.Nspws = 1
    UV.freq_array = freq_array.reshape(UV.Nspws, -1)
    UV.freq_range = [freq_array[0], freq_array[-1]]  # valid frequencies for solutions.
    UV.channel_width = np.diff(freq_array)[0]
    UV.jones_array = jones_array
    UV.time_array = time_array
    UV.integration_time = np.diff(freq_array)[0]
    UV.gain_convention = 'divide'  # Use this operation to apply gain solution.
    UV.x_orientation = 'east'  # orientation of 1st jones parameter.
    UV.time_range = [time_array[0], time_array[-1]]
    UV.telescope_name = 'Fake Telescope'
    UV.Nants_data = Nants_data
    UV.Nants_telescope = Nants_data  # have solutions for all antennas in array.
    UV.antenna_names = antenna_names
    UV.antenna_numbers = antenna_numbers
    UV.flag_array = flagarray
    UV.gain_array = gainarray
    UV.quality_array = chisqarray

    UV.write_calfits('tutorial5b.fits')
