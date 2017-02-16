Cal Parameters
==============
These are the standard attributes of UVCal objects.

Under the hood they are actually properties based on UVParameter objects.

Required
----------------
These parameters are required to have a sensible UVCal object and 
are required for most kinds of uv cal files.

**Nants_data**
     Number of antennas with data present (i.e. number of unique entries in ant_1_array and ant_2_array). May be smaller than the number of antennas in the array

**Nants_telescope**
     Number of antennas in the array. May be larger than the number of antennas with data

**Nfreqs**
     Number of frequency channels

**Npols**
     Number of polarizations

**Nspws**
     Number of spectral windows (ie non-contiguous spectral chunks). More than one spectral window is not currently supported.

**Ntimes**
     Number of times

**antenna_names**
     List of antenna names, shape (Nants_telescope), with numbers given by antenna_numbers (which can be matched to ant_1_array and ant_2_array). There must be one entry here for each unique entry in ant_1_array and ant_2_array, but there may be extras as well.

**antenna_numbers**
     List of integer antenna numbers corresponding to antenna_names,shape (Nants_telescope). There must be one entry here for each unique entry in ant_1_array and ant_2_array, but there may be extras as well.

**cal_type**
     cal type parameter. Values are delay, gain or unknown.

**flag_array**
     Array of flags, True is flagged.shape: (Nants_data, Nfreqs, Ntimes, Npols), type = bool.

**freq_array**
     Array of frequencies, shape (Nspws, Nfreqs), units Hz

**gain_convention**
     The convention for applying he calibration solutions to data.Indicates that to calibrate one should divide or multiply uncalibrated data by gains.

**history**
     String of history, units English

**polarization_array**
     Array of polarization integers, shape (Npols). AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)

**quality_array**
     Array of qualities, shape: (Nants_data, Nfreqs, Ntimes, Npols), type = float.

**time_array**
     Array of times, center of integration, shape (Ntimes), units Julian Date

**x_orientation**
     Orientation of the physical dipole corresponding to what is labelled as the x polarization. Values are east (east/west orientation),  north (north/south orientation) or unknown.

Optional
----------------
These parameters are defined by one or more file standard but are not always required.
Some of them are required depending on the cal_type (as noted below).

**delay_array**
     Array of delays. shape: (Nants_data, Ntimes, Npols), type = float

**gain_array**
     Array of gains, shape: (Nants_data, Nfreqs, Ntimes, Npols), type = complex float.

**input_flag_array**
     Array of input flags, True is flagged. shape: (Nants_data, Nfreqs, Ntimes, Npols), type = bool.

last updated: 2017-02-16