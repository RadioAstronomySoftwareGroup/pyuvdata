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
        self.telescope          =   None    # string name of telescope
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
#    def gethdufreqs(self,D):
#        nfreqs = D.header['NAXIS4']
#        f0  = D.header['CRVAL4']
#        df = D.header['CDELT4']
#        fi0 = D.header['CRPIX4']-1  #convert 1 indexed to 0 indexed
#        return n.arange(f0-df*fi0,f0-df*fi0 + nfreqs*df,df)
#    def gethdupols(self,D):
#        npols = D.header['NAXIS3']
#        pol0 = D.header['CRVAL3']
#        dpol = D.header['CDELT3']
#        poli0 = D.header['CRPIX3']-1
#        return n.arange(pol0-dpol*poli0,pol0-dpol*poli0+npols*dpol,dpol)
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
            self.telescope              = D.header['TELESCOP']
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
        self.etc_items = {}
        for key in D.header[etcpointer:]:
            self.etc_items[key] = D.header[key]

            
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
 
    
