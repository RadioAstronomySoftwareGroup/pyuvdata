Parameters
==============
These are the standard attributes of UVData objects.

Under the hood they are actually properties based on UVParameter objects.

Angle type attributes also have convenience properties named the same thing 
with '_degrees' appended through which you can get or set the value in degrees.

Similarly location type attributes (which are given in topocentric xyz coordinates) 
have convenience properties named the same thing with '_lat_lon_alt' and 
'_lat_lon_alt_degrees' appended through which you can get or set the values using 
latitude, longitude and altitude values in radians or degrees and meters.

Required
----------------
These parameters are required to have a sensible UVData object and 
are required for most kinds of uv data files.

**Nants_data**
     number of antennas with data present. May be smaller than the number of antennas in the array

**Nants_telescope**
     number of antennas in the array. May be larger than the number of antennas with data

**Nbls**
     number of baselines

**Nblts**
     Ntimes * Nbls

**Nfreqs**
     number of frequency channels

**Npols**
     number of polarizations

**Nspws**
     number of spectral windows (ie non-contiguous spectral chunks)

**Ntimes**
     Number of times

**ant_1_array**
     array of first antenna indices, shape (Nblts), type = int, 0 indexed

**ant_2_array**
     array of second antenna indices, shape (Nblts), type = int, 0 indexed

**antenna_names**
     list of antenna names, shape (Nants_telescope), with numbers given by antenna_numbers (which can be matched to ant_1_array and ant_2_array). There must be one entry here for each unique entry in ant_1_array and ant_2_array, but there may be extras as well.

**antenna_numbers**
     integer antenna number corresponding to antenna_names, shape (Nants_telescope). There must be one entry here for each unique entry in self.ant_1_array and self.ant_2_array, but there may be extras as well.

**baseline_array**
     array of baseline indices, shape (Nblts), type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16

**channel_width**
     width of channel (Hz)

**data_array**
     array of the visibility data, shape: (Nblts, Nspws, Nfreqs, Npols), type = complex float, in units of self.vis_units

**flag_array**
     boolean flag, True is flagged, same shape as data_array.

**freq_array**
     array of frequencies, shape (Nspws, Nfreqs), units Hz

**history**
     string of history, units English

**instrument**
     receiver or backend.

**integration_time**
     length of the integration (s)

**lst_array**
     array of lsts, center of integration, shape (Nblts), units radians

**nsample_array**
     number of data points averaged into each data element, type = int, same shape as data_array

**object_name**
     source or field observed (string)

**phase_type**
     string indicating phasing type. Allowed values are "drift", "phased" and "unknown"

**polarization_array**
     array of polarization integers, shape (Npols). AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)

**spw_array**
     array of spectral window numbers, shape (Nspws)

**telescope_location**
     telescope location: xyz in ITRF (earth-centered frame). Can also be set using telescope_location_lat_lon_alt or telescope_location_lat_lon_alt_degrees properties

**telescope_name**
     name of telescope (string)

**time_array**
     array of times, center of integration, shape (Nblts), units Julian Date

**uvw_array**
     Projected baseline vectors relative to phase center, shape (3, Nblts), units meters

**vis_units**
     Visibility units, options are: "uncalib", "Jy" or "K str"

Optional
----------------
These parameters are defined by one or more file standard but are not always required.
Some of them are required depending on the phase_type (as noted below).

**antenna_positions**
     array giving coordinates of antennas relative to telescope_location (ITRF frame), shape (Nants_telescope, 3)

**dut1**
     DUT1 (google it) AIPS 117 calls it UT1UTC

**earth_omega**
     earth's rotation rate in degrees per day

**extra_keywords**
     any user supplied extra keywords, type=dict

**gst0**
     Greenwich sidereal time at midnight on reference date

**phase_center_dec**
     Required if phase_type = "phased". Declination of phase center (see uvw_array), units radians

**phase_center_epoch**
     Required if phase_type = "phased". Epoch year of the phase applied to the data (eg 2000.)

**phase_center_ra**
     Required if phase_type = "phased". Right ascension of phase center (see uvw_array), units radians

**rdate**
     date for which the GST0 or whatever... applies

**timesys**
     We only support UTC

**uvplane_reference_time**
     FHD thing we do not understand, something about the time at which the phase center is normal to the chosen UV plane for phasing

**zenith_dec**
     Required if phase_type = "drift". Declination of zenith. units: radians, shape (Nblts)

**zenith_ra**
     Required if phase_type = "drift". Right ascension of zenith. units: radians, shape (Nblts)

last updated: 2016-09-01