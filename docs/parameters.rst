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
     Number of antennas with data present (i.e. number of unique entries in ant_1_array and ant_2_array). May be smaller than the number of antennas in the array

**Nants_telescope**
     Number of antennas in the array. May be larger than the number of antennas with data

**Nbls**
     Number of baselines

**Nblts**
     Number of baseline-times (i.e. number of spectra). Not necessarily equal to Nbls * Ntimes

**Nfreqs**
     Number of frequency channels

**Npols**
     Number of polarizations

**Nspws**
     Number of spectral windows (ie non-contiguous spectral chunks). More than one spectral window is not currently supported.

**Ntimes**
     Number of times

**ant_1_array**
     Array of first antenna indices, shape (Nblts), type = int, 0 indexed

**ant_2_array**
     Array of second antenna indices, shape (Nblts), type = int, 0 indexed

**antenna_names**
     List of antenna names, shape (Nants_telescope), with numbers given by antenna_numbers (which can be matched to ant_1_array and ant_2_array). There must be one entry here for each unique entry in ant_1_array and ant_2_array, but there may be extras as well.

**antenna_numbers**
     List of integer antenna numbers corresponding to antenna_names, shape (Nants_telescope). There must be one entry here for each unique entry in ant_1_array and ant_2_array, but there may be extras as well.

**baseline_array**
     Array of baseline indices, shape (Nblts), type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16

**channel_width**
     Width of frequency channels (Hz)

**data_array**
     Array of the visibility data, shape: (Nblts, Nspws, Nfreqs, Npols), type = complex float, in units of self.vis_units

**flag_array**
     Boolean flag, True is flagged, same shape as data_array.

**freq_array**
     Array of frequencies, shape (Nspws, Nfreqs), units Hz

**history**
     String of history, units English

**instrument**
     Receiver or backend. Sometimes identical to telescope_name

**integration_time**
     Length of the integration (s)

**lst_array**
     Array of lsts, center of integration, shape (Nblts), units radians

**nsample_array**
     Number of data points averaged into each data element, NOT required to be an integer. type = float, same shape as data_array

**object_name**
     Source or field observed (string)

**phase_type**
     String indicating phasing type. Allowed values are "drift", "phased" and "unknown"

**polarization_array**
     Array of polarization integers, shape (Npols). AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)

**spw_array**
     Array of spectral window Numbers, shape (Nspws)

**telescope_location**
     Telescope location: xyz in ITRF (earth-centered frame). Can also be accessed using telescope_location_lat_lon_alt or telescope_location_lat_lon_alt_degrees properties

**telescope_name**
     Name of telescope (string)

**time_array**
     Array of times, center of integration, shape (Nblts), units Julian Date

**uvw_array**
     Projected baseline vectors relative to phase center, shape (Nblts, 3), units meters

**vis_units**
     Visibility units, options are: "uncalib", "Jy" or "K str"

Optional
----------------
These parameters are defined by one or more file standard but are not always required.
Some of them are required depending on the phase_type (as noted below).

**antenna_positions**
     Array giving coordinates of antennas relative to telescope_location (ITRF frame), shape (Nants_telescope, 3)

**dut1**
     DUT1 (google it) AIPS 117 calls it UT1UTC

**earth_omega**
     Earth's rotation rate in degrees per day

**extra_keywords**
     Any user supplied extra keywords, type=dict

**gst0**
     Greenwich sidereal time at midnight on reference date

**phase_center_dec**
     Required if phase_type = "phased". Declination of phase center (see uvw_array), units radians. Can also be accessed using phase_center_dec_degrees.

**phase_center_epoch**
     Required if phase_type = "phased". Epoch year of the phase applied to the data (eg 2000.)

**phase_center_ra**
     Required if phase_type = "phased". Right ascension of phase center (see uvw_array), units radians. Can also be accessed using phase_center_ra_degrees.

**rdate**
     Date for which the GST0 or whatever... applies

**timesys**
     We only support UTC

**uvplane_reference_time**
     FHD thing we do not understand, something about the time at which the phase center is normal to the chosen UV plane for phasing

**zenith_dec**
     Required if phase_type = "drift". Declination of zenith. units: radians, shape (Nblts). Can also be accessed using zenith_dec_degrees.

**zenith_ra**
     Required if phase_type = "drift". Right ascension of zenith. units: radians, shape (Nblts). Can also be accessed using zenith_ra_degrees.

last updated: 2017-02-15