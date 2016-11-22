Tutorial
========

Example 1: Converting a miriad data set to uvfits (taken from miriad_to_uvfits.py)::

  import uvdata
  UV = UVData()
  UV.read_miriad('mydata.uv', 'miriad')      #this miriad file is known to be a drift scan
  UV.phase_to_time(UV.time_array[0])         #uvfits can't do drift scans, so phase to the ra/dec of the first integration
  UV.write_uvfits('mydata.uvfits', 'uvfits') #write out the uvfits file
