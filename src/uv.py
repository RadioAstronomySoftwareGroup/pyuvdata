class UVData:
    def __init__(self):
        #Basic information required by class
        # they are populated on file read or by the calling code 
        self.Ntimes             =   None    # Number of times
        self.Nbls               =   None    # number of baselines
        self.Nblts              =   None    # Ntimes * Nbls
        self.Nfreqs             =   None    # number of frequency channels
        self.Npols              =   None    # number of polarizations
        self.Nspws              =   None    # number of spectral windows (ie non-contiguous spectral chunks)
        self.uvw_array          =   None    # phase center projected baseline vectors  , dimensions (3,Nblts), units meters
        self.time_array         =   None    # array of times, center of integration, dimensions (Nblts), julian date
        self.ant_1_array        =   None    # array of antenna indices, dimensions (Nblts), type=int, 0 indexed
        self.ant_2_array        =   None    # array of antenna indices, dimensions (Nblts), type=int, 0 indexed
        self.baseline_array     =   None    # array of baseline indices, dimensions (Nblts), type=int 
                                            # baseline = 2048 * (ant2+1) + (ant1+1) + 2^16 #does this break casa?
        self.freq_array         =   None    # array of frequencies, dimensions (Nfreqs), units Hz
        self.polarization_array =   None    # array of polarization integers (Npols)
                                            # stokes 1:4 (I,Q,U,V); -1:-4 (RR,LL,RL,LR); -5:-8 (XX,YY,XY,YX) AIPS Memo 117
        self.spw_array          =   None    # array of spectral window numbers
        self.phase_center_ra    =   None    # right ascension of phase center (see uvw_array), units degrees
        self.phase_center_dec   =   None    # declination of phase center (see uvw_array), units degrees

        #these are not strictly necessary and assume homogeniety of time and frequency.
        self.integration_time   =   None    # length of the integration in seconds
        self.channel_width      =   None    # width of channel in Hz

        #observation information

        self.object_name        =   None    # this is the source or field that the telescope has observed. type=string
        self.telescope          =   None    # string name of telescope
        self.instrument         =   None    # receiver or whatever attached to the backend.
        self.dateobs            =   None    # date of observation start, units JD.  
        self.history            =   None    # string o' history units English
        self.vis_units          =   None    # Visibility units, options ['uncalib','Jy','K str']
        self.phase_center_epoch =   None    # epoch of the phase applied to the data (ex 'J2000')

        #antenna information
        self.Nants              =   None    # number of antennas
        self.antenna_names      =   None    # list of antenna names, dimensions (Nants), 
                                            # indexed by self.ant_1_array, self.ant_2_array, self.antenna_indices
        self.antenna_indices    =   None    # integer index into antenna_names, dimensions (Nants), 
                                            # there must be one entry here for each
                                            # unique entry in self.ant_1_array and self.ant_2_array
                                            # indexes into antenna_names (sort order must be preserved)
        self.antenna_frame      =   None    # coordinate frame for antenna positions (eg 'ITRF' -also google ECEF)         
        self.x_array            =   None    # x coordinate of array center in meters in frame self.antenna_frame
        self.y_array            =   None    # y coordinate of array center in meters in frame self.antenna_frame
        self.z_array            =   None    # z coordinate of array center in meters in frame self.antenna_frame
                                            # NB: ECEF has x running through long=0 and z through the north pole
        # the below are copied from AIPS memo 117, but we don't quite know why. 
        #  when available they are populated. user beware?
        self.GST0               =   None    # Greenwich sidereal time at midnight on reference date
        self.Rdate              =   None    # date for which the GST0 or orbits or whatever... applies
        self.earth_omega        =   None    # earth's rotation rate in degrees per day
        # stuff about reference frequencies and handedness is left out here as a favor to our children 
        return True

    def write(self,filename):
        self.check()
        "filename ending in .uvfits gets written as a uvfits"
        if filename.endswith('.uvfits'):
            self.write_uvfits(self,filename)
    def read(self,filename):
        if filename.endswith('.uvfits'):
            self.read_uvfits(self,filename)
        self.check()
    def check(self):
        return True
    def read_uvfits(self,filename):
        F = fits.open(filename)
        D = F[0]
        self.times = D.data.field('DATE')
        bls = D.data.field('BASELINE')
        uvws  = n.array(zip(D.data.field('UU'),D.data.field('VV'),D.data.field('WW')))*1e9 #bl vectors in ns
        DATA =D.data.field('DATA').squeeze()[:,:,:,0] + 1j*D.data.field('DATA').squeeze()[:,:,:,1]
        MASK = (D.data.field('DATA').squeeze()[:,:,:,2]==0)
        freqs = gethdufreqs(D)
        Nfreqs = D.data.field('DATA').shape[3]
        Npol = D.data.field('DATA').shape[4]
        Nblt = D.data.field('DATA').shape[0] 
        del(D)
        ant2    = n.array(bls%256).astype(int)-1
        ant1    = n.array((bls-ant2)/256).astype(int)-1        

