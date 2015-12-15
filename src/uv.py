from astropy import constants as const
from astropy.time import Time
import numpy as n
class UVData:
    def __init__(self):
        #Basic information required by class
        self.data_array         =   None    # array of the data (Nblts,Nspws,Nfreqs,Npols), type=complex float, 
                                            # in units of self.vis_units
        self.nsample_array      =   None    # number of data points averaged into each data element, type=int, same shape as data_array
        self.flag_array         =   None    # boolean flag, True is flagged, same shape as data_array
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
        self.freq_array         =   None    # array of frequencies, dimensions (Nspws,Nfreqs), units Hz
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
        self.telescope_name          =   None    # string name of telescope
        self.instrument         =   None    # receiver or whatever attached to the backend.
        self.latitude           =   None    # latitude of telescope, units degrees
        self.longitude          =   None    # longitude of telescope, units degrees
        self.altitude           =   None    # altitude of telescope, units meters
        self.dateobs            =   None    # date of observation start, units JD.  
        self.history            =   None    # string o' history units English
        self.vis_units          =   None    # Visibility units, options ['uncalib','Jy','K str']
        self.phase_center_epoch =   None    # epoch year of the phase applied to the data (eg 2000)

        #antenna information
        self.Nants              =   None    # number of antennas
        self.antenna_names      =   None    # list of antenna names, dimensions (Nants), 
                                            # indexed by self.ant_1_array, self.ant_2_array, self.antenna_indices
        self.antenna_indices    =   None    # integer index into antenna_names, dimensions (Nants), 
                                            # there must be one entry here for each
                                            # unique entry in self.ant_1_array and self.ant_2_array
                                            # indexes into antenna_names (sort order must be preserved)
        self.xyz_telescope_frame=   None    # coordinate frame for antenna positions (eg 'ITRF' -also google ECEF)         
        self.x_telescope        =   None    # x coordinate of array center in meters in frame self.antenna_frame
        self.y_telescope        =   None    # y coordinate of array center in meters in frame self.antenna_frame
        self.z_telescope        =   None    # z coordinate of array center in meters in frame self.antenna_frame
                                            # NB: ECEF has x running through long=0 and z through the north pole
        self.antenna_positions  =   None    # array giving coordinates of antennas (Nants,3)
                                            # coordinates relative to {x,y,z}_telescope in the same frame
        # the below are copied from AIPS memo 117, but could be revised to merge with other sources of data 
        #  when available they are populated. user beware?
        self.GST0               =   None    # Greenwich sidereal time at midnight on reference date
        self.Rdate              =   None    # date for which the GST0 or orbits or whatever... applies
        self.earth_omega        =   360.985 # earth's rotation rate in degrees per day (might not be enough sigfigs)
        self.DUT1               =   0.0     # DUT1 (google it) AIPS 117 calls it UT1UTC
        self.TIMESYS            =   'UTC'   #
        # stuff about reference frequencies and handedness is left out here as a favor to our children 

        self.fits_extra_keywords=   None    # the user supplied extra keywords in fits header, type=dict

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
    def _gethduaxis(self,D,axis):
        ax = str(axis)
        N = D.header['NAXIS'+ax]
        X0 = D.header['CRVAL'+ax]
        dX = D.header['CDELT'+ax]
        Xi0 = D.header['CRPIX'+ax]
        return n.arange(X0-dX*Xi0,X0-dX*Xi0+N*dX,dX)

    def read_uvfits(self,filename):
        F = fits.open(filename)
        D = F[0]
        #check that we have a single source file!! (TODO)

        self.time_array = D.data.field('DATE') #astropy.io fits reader scales date according to relevant PZER0 (?)
        self.baseline_array = D.data.field('BASELINE')

        #check if we have an spw dimension
        if D.header['NAXIS']==7: 
            self.Nspws = D.header['NAXIS5']
            self.data_array =D.data.field('DATA')[:,0,0,:,:,:,0] + 1j*D.data.field('DATA')[:,0,0,:,:,:,1]
            self.flag_array = (D.data.field('DATA')[:,0,0,:,:,:,2]<=0)
            self.nsample_array = n.abs(D.data.field('DATA')[:,0,0,:,:,:,2])
            self.Nspws = D.header['NAXIS5']
            assert(self.Nspws == self.data_array.shape[1])
            self.spw_array = gethduaxis(D,5)
            #the axis number for phase center depends on if the spw exists
            self.phase_center_ra = D.header['CRVAL6']
            self.phase_center_dec =D.header['CRVAL7']
        else:
            #in many uvfits files the spw axis is left out, here we put it back in so the dimensionality stays the same
            self.data_array =D.data.field('DATA')[:,0,0,:,:,0] + 1j*D.data.field('DATA')[:,0,0,:,:,1]
            self.data_array = self.data_array[:,n.newaxis,:,:]
            self.flag_array = (D.data.field('DATA')[:,0,0,:,:,2]<=0)
            self.flag_array = self.flag_array[:,n.newaxis,:,:]
            self.nsample_array = n.abs(D.data.field('DATA')[:,0,0,:,:,2])
            self.nsample_array = self.nsample_array[:,n.newaxis,:,:]
            self.Nspws = 1
            #the axis number for phase center depends on if the spw exists
            self.phase_center_ra = D.header['CRVAL5']
            self.phase_center_dec =D.header['CRVAL6']

        #get my dimension sizes
        self.Nfreqs = D.header['NAXIS4']
        assert(self.Nfreqs == self.data_array.shape[2])
        self.Npols = D.header['NAXIS3']
        assert(self.Npols == self.data_array.shape[3])
        self.Nblts = D.header['GCOUNT']
        assert(self.Nblts == self.data_array.shape[0])


        # read baseline vectors in units of seconds, return in meters
        self.uvw_array  = n.array(zip(D.data.field('UU'),D.data.field('VV'),D.data.field('WW')))*const.c.to('m/s').value 
        self.freq_array = self.gethduaxis(D,4)

        #here we account for two standard methods of forming a single integer index based
        #   on two integer antenna numbers
        if n.max(self.baseline_array)<=2**16:  #for 255 and fewer antennas
            self.ant_2_array    = n.array(self.baseline_array%2**8).astype(int)-1
            self.ant_1_array    = n.array((self.baseline_array-self.ant_2_array)/2**8).astype(int)-1 
            self.baseline_array = (self.ant_2_array+1)*2**11 + self.ant_1_array+1 + 2**16
        elif n.min(self.baseline_array)>=2**16: # for 2047 and fewer antennas
            bls = self.baseline_array - 2**16
            self.ant_2_array    = n.array(bls%2**11).astype(int)-1
            self.ant_1_array    = n.array((bls-self.ant_2_array)/2**11).astype(int)-1 
        #todo build SKA and or FFTT

        self.polarization_array = gethduaxis(D,3)
        
        #other info
        try:
            self.object_name            = D.header['OBJECT']
            self.telescope_name         = D.header['TELESCOP']
            self.instrument             = D.header['INSTRUME']
            self.latitude               = D.header['LAT']
            self.longitude              = D.header['LON']
            self.altitude               = D.header['ALT']
            self.dataobs                = D.header['DATE-OBS']
            self.history                = D.header['HISTORY']
            self.vis_units              = D.header['BUNIT']
            self.phase_center_epoch     = D.header['EPOCH']
        except(KeyError):
            print "WARNING (todo make actual warning), importing of non-essential data failed"
        #find all the header items after the history and keep them as a dictionary
        etcpointer = 0
        for thing in D.header:
            etcpointer +=1 
            if thing=='HISTORY':break
        #self.fits = {} #todo change this to fits_extra_keywords!
        for key in D.header[etcpointer:]:
            self.fits_extra_keywords[key] = D.header[key]
        
        #READ the antenna table
        #TODO FINISH
        ant_hdu = F[1]
        #stuff in columns
        self.antenna_names = ant_hdu.data.field('ANNAME')
        self.antenna_indices = ant_hdu.data.field('NOSTA')
        self.antenna_positions = ant_hdu.data.field('STABXYZ')
        #stuff in the header
        if self.telescope_name is None: self.telescope_name=ant_hdu.header['ARRNAM'] 
        self.xyz_telescope_frame = ant_hdu.header['FRAME']
        self.x_telescope    =   ant_hdu.header['ARRAYX']
        self.y_telescope    =   ant_hdu.header['ARRAYY']
        self.z_telescope    =   ant_hdu.header['ARRAYZ']
        self.GST0           =   ant_hdu.header['GSTIA0']
        self.Rdate          =   ant_hdu.header['RDATE']     
        self.earth_omega    =   ant_hdu.header['DEGPDY']
        self.DUT1           =   ant_hdu.header['UT1UTC']
        self.TIMESYS        =   ant_hdu.header['TIMESYS']

            
        del(D)
        print "LOG (todo make actual log): file load did not fail"
        return True
    def write_uvfits(self,filename):
        weights_array = self.nsample_array * n.where(self.flag_array, -1, 1)
        data_array = self.data_array[:,n.newaxis,n.newaxis,:,:,:]
        weights_array = weights_array[:,n.newaxis,n.newaxis,:,:,:]
        uvfits_array_data = n.concatenate(data_array,weights_array,axis=6)
        #uvfits_array_data shape is (Nblts,1,1,[Nspws],Nfreqs,Npols,3)
        uvw_array_sec = self.uvw_array/const.c.to('m/s').value
        jd_midnight = n.floor(self.time_array[0]-0.5)+0.5
        time_array = self.time_array - jd_midnight       
        #uvfits convention is that time_array + jd_midnight = actual JD
        #jd_midnight is julian midnight on first day of observation
        group_parameter_list = [uvw_array_sec[0],uvw_array_sec[1],uvw_array_sec[2],time_array,self.baseline_array]
        #list contains arrays of [u,v,w,date,baseline]; each array has shape (Nblts) 
        hdu = fits.GroupData(uvfits_array_data,parnames=['UU      ','VV      ','WW      ','DATE    ','BASELINE'], 
            pardata=group_parameter_list, bitpix=-32)
        hdu = fits.GroupsHDU(hdu)
        
        hdu.header['PTYPE1  '] = 'UU      '
        hdu.header['PSCAL1  '] = 1.0
        hdu.header['PZERO1  '] = 0.0
     
        hdu.header['PTYPE2  '] = 'VV      '
        hdu.header['PSCAL2  '] = 1.0
        hdu.header['PZERO2  '] = 0.0
     
        hdu.header['PTYPE3  '] = 'WW      '
        hdu.header['PSCAL3  '] = 1.0
        hdu.header['PZERO3  '] = 0.0
     
        hdu.header['PTYPE4  '] = 'DATE    '
        hdu.header['PSCAL4  '] = 1.0
        hdu.header['PZERO4  '] = jd_midnight 
    
        hdu.header['PTYPE5  '] = 'BASELINE'
        hdu.header['PSCAL5  '] = 1.0
        hdu.header['PZERO5  '] = 0.0
 
        hdu.header['DATE-OBS']= Time(self.time_array[0],scale='utc',format='jd').iso 
        #ISO string of first time in self.time_array
     
        hdu.header['CTYPE2  '] = 'COMPLEX '
        hdu.header['CRVAL2  '] = 1.0
        hdu.header['CRPIX2  '] = 1.0
        hdu.header['CDELT2  '] = 1.0
        
        hdu.header['CTYPE3  '] = 'STOKES  '
        hdu.header['CRVAL3  '] = self.polarization_array[0]
        hdu.header['CRPIX3  '] = 1.0
        hdu.header['CDELT3  '] = n.diff(self.polarization_array)[0]
        
        hdu.header['CTYPE4  '] = 'FREQ    '
        hdu.header['CRVAL4  '] = self.freq_array[0,0]
        hdu.header['CRPIX4  '] = 1.0
        hdu.header['CDELT4  '] = n.diff(self.freq_array[0])[0]
        
        hdu.header['CTYPE5  '] = 'IF      '
        hdu.header['CRVAL5  '] = 1.0
        hdu.header['CRPIX5  '] = 1.0
        hdu.header['CDELT5  '] = 1.0
    
        hdu.header['CTYPE6  '] = 'RA'
        hdu.header['CRVAL6  '] = self.phase_center_ra

        hdu.header['CTYPE7  '] = 'DEC'
        hdu.header['CRVAL7  '] = self.phase_center_dec

        hdu.header['BUNIT   '] = self.vis_units
        hdu.header['BSCALE  '] = 1.0
        hdu.header['BZERO   '] = 0.0

        hdu.header['OBJECT  '] = self.object_name
        hdu.header['TELESCOP'] = self.telescope
        hdu.header['LAT     '] = self.latitude
        hdu.header['LON     '] = self.longitude
        hdu.header['ALT     '] = self.altitude
        hdu.header['INSTRUME'] = self.instrument
        hdu.header['EPOCH   '] = self.phase_center_epoch
     
        hdu.header['HISTORY '] = self.history
        #end standard keywords; begin user-defined keywords
     
        for key, value in self.fits_extra_keywords.iteritems():
            keyword = key[:8].upper() #header keywords have to be 8 characters or less
            hdu.header[keyword] = value
 
        #ADD the ANTENNA table 
        mntsta = [0] * self.Nants #0 specifies alt-az, 6 would specify a phased array
        staxof = [0] * self.Nants
        poltya = ['X'] * self.Nants  #beware, X can mean just about anything
        polaa = [90.0] * self.Nants     
        poltyb = ['Y'] * self.Nants
        polab = [0.0] * self.Nants
        
        
        col1 = fits.Column(name='ANNAME', format='8A', array=self.antenna_names)
        col2 = fits.Column(name='STABXYZ', format='3D', array=self.antenna_positions) 
        col3 = fits.Column(name='NOSTA', format='1J', array=self.antenna_indices)
        col4 = fits.Column(name='MNTSTA', format='1J', array=mntsta) 
        col5 = fits.Column(name='STAXOF', format='1E', array=staxof) 
        col6 = fits.Column(name='POLTYA', format='1A', array=poltya)
        col7 = fits.Column(name='POLAA', format='1E', array=polaa) 
        #col8 = fits.Column(name='POLCALA', format='3E', array=polcala)
        col9 = fits.Column(name='POLTYB', format='1A', array=poltyb) 
        col10 = fits.Column(name='POLAB', format='1E', array=polab)
        #col11 = fits.Column(name='POLCALB', format='3E', array=polcalb) 
        #note ORBPARM is technically required, but we didn't put it

        cols = fits.ColDefs([col1, col2,col3, col4,col5, col6,col7,col9, col10,])
        # This only works for astropy 0.4 which is not available from pip
        ant_hdu = fits.BinTableHDU.from_columns(cols)
        
        
        ant_hdu.header['EXTNAME'] = 'AIPS AN'

        ant_hdu.header['ARRAYX'] = self.x_telescope
        ant_hdu.header['ARRAYY'] = self.y_telescope
        ant_hdu.header['ARRAYZ'] = self.z_telescope
        ant_hdu.header['FRAME'] = self.xyz_telescope_frame
        ant_hdu.header['GSTIAO'] = self.GST0
        ant_hdu.header['FREQ'] = self.freq_array[0]
        ant_hdu.header['RDATE'] = self.RDate
        ant_hdu.header['UT1UTC'] = self.UT1UTC

        ant_hdu.header['TIMSYS'] = self.TIMESYS
        if self.TIMESYS=='IAT': print "WARNING: FILES WITH TIME SYSTEM 'IAT' ARE NOT SUPPORTED" 
        ant_hdu.header['ARRNAM'] = self.telescope_name
        ant_hdu.header['NO_IF'] = self.Nspws
        #ant_hdu.header['IATUTC'] = 35. 

        #set mandatory parameters which are not supported by this object (or that we just don't understand)
        ant_hdu.header['DEGPDY'] = 360.985 #if a comet stikes the earth, update.
        ant_hdu.header['NUMORB'] = 0
        ant_hdu.header['NOPCAL'] = 0  #note: Bart had this set to 3. We've set it 0 after aips 117. -jph
        ant_hdu.header['POLTYPE'] = 'X-Y LIN'
        ant_hdu.header['FREQID'] = -1
        #note: we do not support the concept of "frequency setups" lists of spws given in a SU table. 
        ant_hdu.header['POLARX'] = 0.0  #if there are offsets in images, this could be the culprit
        ant_hdu.header['POLARY'] = 0.0
        ant_hdu.header['DATUTC'] = 0 #ONLY UTC SUPPORTED
        ant_hdu.header['XYZHAND'] = 'RIGHT' #we always output right handed coordinates

        #ADD the FQ table
        # skipping for now and limiting to a single spw
        
        #write the file
        hdulist = fits.HDUList(hdus=[hdu,ant_hdu])
        hdulist.writeto(filename,clobber=True)

        return True

