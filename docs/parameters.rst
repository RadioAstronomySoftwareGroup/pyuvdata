Parameters
==============
Required
----------------
These parameters are required to make most kinds of uv data files.  In many cases there are default values available which let you sidestep the requirement (see spoof values).

            Parameters are implimented using the UVParameter object. See the uvbase documentation for more

            

*UVData.*\ **Nants_data**
     number of antennas with data present. May be smaller than the number of antennas in the array

*UVData.*\ **Nants_telescope**
     number of antennas in the array. May be larger than the number of antennas with data

*UVData.*\ **Nbls**
     number of baselines

*UVData.*\ **Nblts**
     Ntimes * Nbls

*UVData.*\ **Nfreqs**
     number of frequency channels

*UVData.*\ **Npols**
     number of polarizations

*UVData.*\ **Nspws**
     number of spectral windows (ie non-contiguous spectral chunks)

*UVData.*\ **Ntimes**
     Number of times

*UVData.*\ **ant_1_array**
     array of first antenna indices, shape (Nblts), type = int, 0 indexed

*UVData.*\ **ant_2_array**
     array of second antenna indices, shape (Nblts), type = int, 0 indexed

*UVData.*\ **antenna_names**
     list of antenna names, shape (Nants_telescope), with numbers given by antenna_numbers (which can be matched to ant_1_array and ant_2_array). There must be one entry here for each unique entry in ant_1_array and ant_2_array, but there may be extras as well.

*UVData.*\ **antenna_numbers**
     integer antenna number corresponding to antenna_names, shape (Nants_telescope). There must be one entry here for each unique entry in self.ant_1_array and self.ant_2_array, but there may be extras as well.

*UVData.*\ **baseline_array**
     array of baseline indices, shape (Nblts), type = int; baseline = 2048 * (ant2+1) + (ant1+1) + 2^16 (may this break casa?)

*UVData.*\ **channel_width**
     width of channel (Hz)

*UVData.*\ **data_array**
     array of the visibility data, shape: (Nblts, Nspws, Nfreqs, Npols), type = complex float, in units of self.vis_units

*UVData.*\ **flag_array**
     boolean flag, True is flagged, same shape as data_array.

*UVData.*\ **freq_array**
     array of frequencies, shape (Nspws,Nfreqs), units Hz

*UVData.*\ **history**
     string of history, units English

*UVData.*\ **instrument**
     receiver or backend.

*UVData.*\ **integration_time**
     length of the integration (s)

*UVData.*\ **is_phased**
     true/false whether data is phased (true) or drift scanning (false)

*UVData.*\ **lst_array**
     array of lsts, center of integration, shape (Nblts), units radians

*UVData.*\ **nsample_array**
     number of data points averaged into each data element, type = int, same shape as data_array

*UVData.*\ **object_name**
     source or field observed (string)

*UVData.*\ **polarization_array**
     array of polarization integers (Npols). AIPS Memo 117 says: stokes 1:4 (I,Q,U,V);  circular -1:-4 (RR,LL,RL,LR); linear -5:-8 (XX,YY,XY,YX)

*UVData.*\ **spw_array**
     array of spectral window numbers

*UVData.*\ **telescope_location**
     telescope location: xyz in ITRF (earth-centered frame). Can also be set using telescope_location_lat_lon_alt or telescope_location_lat_lon_alt_degrees properties

*UVData.*\ **telescope_name**
     name of telescope (string)

*UVData.*\ **time_array**
     array of times, center of integration, shape (Nblts), units Julian Date

*UVData.*\ **uvw_array**
     Projected baseline vectors relative to phase center, (3,Nblts), units meters

*UVData.*\ **vis_units**
     Visibility units, options ["uncalib","Jy","K str"]

Not required
----------------
These parameters are defined by one or more file standard but are not officially required.


*UVData.*\ **antenna_positions**
     array giving coordinates of antennas relative to telescope_location (ITRF frame), (Nants_telescope, 3)

*UVData.*\ **dateobs**
     date of observation

*UVData.*\ **dut1**
     DUT1 (google it) AIPS 117 calls it UT1UTC

*UVData.*\ **earth_omega**
     earth's rotation rate in degrees per day

*UVData.*\ **extra_keywords**
     any user supplied extra keywords, type=dict

*UVData.*\ **gst0**
     Greenwich sidereal time at midnight on reference date

*UVData.*\ **phase_center_dec**
     declination of phase center (see uvw_array), units radians

*UVData.*\ **phase_center_epoch**
     epoch year of the phase applied to the data (eg 2000.)

*UVData.*\ **phase_center_ra**
     right ascension of phase center (see uvw_array), units radians

*UVData.*\ **rdate**
     date for which the GST0 or whatever... applies

*UVData.*\ **timesys**
     We only support UTC

*UVData.*\ **uvplane_reference_time**
     FHD thing we do not understand, something about the time at which the phase center is normal to the chosen UV plane for phasing

*UVData.*\ **zenith_dec**
     dec of zenith. units: radians, shape (Nblts)

*UVData.*\ **zenith_ra**
     ra of zenith. units: radians, shape (Nblts)

last updated: 2016-08-30