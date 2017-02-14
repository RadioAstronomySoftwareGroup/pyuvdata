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
