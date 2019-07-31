# - * - coding: utf-8 - * -
"""Class for reading MWA correlator FITS files

"""
from astropy.io import fits
import numpy as np
import warnings
from astropy.time import Time

from . import UVData
from . import utils as uvutils

def input_output_mapping():
    '''
    Builds a mapping dictionary from the pfb output numbers (the correlator indices for
    antenna number and polarization) to the pfb input numbers. These input numbers
    can be mapped to antenna numbers using metadata.
    '''
    #this comes from mwa_build_lfiles/mwac_utils.c 
    #inputs are mapped to the indices of pfb_mapper as follows (from mwa_build_lfiles/antenna_mapping.h):
    #floor(index/4) + index%4 * 16 = input
    pfb_mapper = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
    #pfb_mapper maps the first 64 inputs; use it to build a mapper for all 256 inputs
    pfb_inputs_to_outputs = {}
    for p in range(4):
        for i in range(64):            
            pfb_inputs_to_outputs[pfb_mapper[i] + p * 64] = p * 64 + i
    return pfb_inputs_to_outputs

#TODO: ask Bryna: do we want to allow reading in a subset of files from an observation? 
#or do we want to force all the files from an observation to be included?

class MWACorrFITS(UVData):
    """
    Defines a MWA correlator fits-specific subclass of UVData for reading MWA 
    correlator fits files. This class should not be interacted with directly, 
    instead use the read_mwa_corr_fits method on the UVData class.
    """
    def read_mwa_corr_fits(self, filelist, run_check = True, check_extra = True, 
                 run_check_acceptability = True):
        """
        Read in data from a list of MWA correlator fits files. 

        Args:
            filelist: The list of MWA correlator files to read from. Currently only
                a single observation is supported. Must include one metafits file.
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
        """
        
        metafits_file = None
        file_dict = {}
        
        for file in filelist:
            if file.lower().endswith('.metafits'):
                #check that have a metafits file
                #TODO: * *  * fix this so that it allows for multiple metafits files! * * * 
                #for now, force only one metafits file
                if metafits_file is not None:
                    raise ValueError('multiple metafits files in filelist')
                metafits_file = file
            #organize data files
            #TOOD: what to do with the metafits_ppds file?
            elif file.lower().endswith('00.fits'):
                if '00' not in file_dict.keys():
                    file_dict['00'] = [file]
                else:
                    file_dict['00'].append(file)
            elif file.lower().endswith('01.fits'):
                if '01' not in file_dict.keys():
                    file_dict['01'] = [file]
                else:
                    file_dict['01'].append(file)
            elif file.lower().endswith('.mwaf'):
                if 'flags' not in file_dict.keys():
                    file_dict['flags'] = [file]
                else:
                    file_dict['flags'].append(file)        
            else:
                continue
        
        #checks:
        if '01' not in file_dict.keys() and '00' not in file_dict.keys():
            raise ValueError('no fits files submitted')
        if '01' not in file_dict.keys() or '00' not in file_dict.keys():
            raise ValueError('this reader currently does not support reading in a subset of the observation')
        if len(file_dict['00'])!= len(file_dict['01']):
            raise ValueError('coarse band file missing')
        if 'flags' not in file_dict.keys():
            warnings.warn('no flag files submitted')
        
        #first set parameters that are always true
        self.Nspws = 1
        self.spw_array = np.array([0])
        self.phase_type = 'drift'
        self.vis_units = 'uncalib'
        self.Npols = 4 
        
        #get information from metafits file
        with fits.open(metafits_file, memmap = True) as meta:
            meta_hdr = meta[0].header
            
            chans = meta_hdr['CHANNELS'].split(', ') 
            
            #TODO: fix this with a better time array solution
            start_time = Time(meta_hdr['GPSTIME'] + 2, format = 'gps', scale = 'utc')
            start_time = start_time.unix
            inttime = meta_hdr['INTTIME'] 
            
            #get parameters from header
            self.Nfreqs = meta_hdr['NCHANS']
            self.Ntimes = meta_hdr['NSCANS']
            self.channel_width = meta_hdr['FINECHAN'] * 1000#this assumes no averaging by this code so will need to be updated
            self.history = str(meta_hdr['HISTORY'])
            if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
                self.history  += self.pyuvdata_version_str
            self.instrument = meta_hdr['INSTRUME']
            self.telescope_name = meta_hdr['TELESCOP']
            self.object_name = meta_hdr['FILENAME']
            #TODO: remove these keys and store remaining keys in extra keywords
            
            #get data from metafits file table
            meta_tbl = meta[1].data
            
            self.antenna_numbers = meta_tbl['Antenna'][1::2]#because of polarization, each antenna # is listed twice
            self.antenna_names = meta_tbl['TileName'][1::2]
            
            #TODO: self.antenna_positions
            #TODO: self.antenna_diameters
            #TODO: self.x_orientation
            
        self.Nants_data = len(self.antenna_numbers)
        self.Nants_telescope = len(self.antenna_numbers)
        self.Nbls = len(self.antenna_numbers) * (len(self.antenna_numbers + 1))/2
        self.Nblts = self.Nbls * self.Ntimes
        self.integration_time = np.array([inttime for i in range(self.Nblts)])#assumes no averaging
        
        #create an array of channel numbers
        chans = np.array([int(i) for i in chans])
        
        #TODO: fix this for a subset of included coarse channels?
        #build frequency array
        self.frequency_array = np.zeros((self.Nspws, self.Nfreqs))
        #each coarse channel is split into 128 fine channels of width 10 kHz. The first fine channel for 
        #each coarse channel is centered on the lower bound frequency of that channel and it's center
        #frequency is computed as fine_center = coarse_channel_number * 1280-640 (kHz).
        #If the fine channels have been averaged (added) by some factor, the center of the resulting channel
        #is found by averaging the centers of the first and last fine channels it is made up of.
        #That is, avg_fine_center=(lowest_fine_center+highest_fine_center)/2
        #where highest_fine_center=lowest_fine_center+(avg_factor-1)*10 kHz
        #so avg_fine_center=(lowest_fine_center+lowest_fine_center+(avg_factor-1)*10)/2
        #                   =lowest_fine_center+((avg_factor-1)*10)/2
        #We calculate an offset=((avg_factor-1)*10)/2 to build the frequency array
        #center frequency is (0 + 30)/2 = 15kHz.
        avg_factor = self.channel_width/10000
        num_avg_chans = 128/avg_factor
        width = self.channel_width/1000
        offset = (avg_factor-1) * 10/2
        for i in range(len(chans)):
            #get the lowest fine freq going into the lowest averaged channel(kHz)
            lower_fine_freq = chans[i] * 1280 - 640
            #find the center of the lowest averaged channel
            first_center = lower_fine_freq + offset
            #add the channel centers for this coarse channel into the frequency array (converting from kHz to Hz)
            self.frequency_array[0, int(i * 128/avg_factor):int((i + 1) * 128/avg_factor)] = np.arange(first_center, first_center + num_avg_chans * width, width) * 1000
        
        #check that coarse channels are contiguous.
        #TODO: look at a data file where the coarse channels aren't contiguous to make sure this works
        chans = np.array([int(i) for i in chans])
        for i in np.diff(chans):
            if i!= 1:
                warnings.warn('coarse channels are not contiguous for this observation')
                break
        
        #check that all channels are included
        #make an array of channels from file names
        file_chans = np.array([int(file.split('_')[-2][-2:]) for file in file_dict['00']])
        if len(chans)!= len(file_chans):
            warnings.warn('only a subset of coarse channels are being processed')        
            #if not all channels are included, check that included channels are contiguous
            #actually this doesn't work, because the ordering is weird
            for i in np.diff(file_chans):
                if i!= 1:
                    warnings.warn('coarse channels are not contiguous for included data')
                    break
        
        #make dictionary for coarse channel ordering
        #this breaks if coarse channel files are missing
        #channels in group 0-128 go in order; channels in group 129-155 go in reverse order
        #that is, if the lowest channel is 127, it will be assigned to the first file
        #channel 128 will be assigned to the second file
        #then the highest channel will be assigned to the third file
        #and the next hightest channel assigned to the fourth file, and so on
        count = 0   
        #count the number of channels that are in group 0-128
        for i in chans:
            if i <= 128:
                count += 1
        chan_order = {i + 1:i if i<count else (len(chans) + count-i-1) for i in range(len(chans))}
        
        #read in the data!
        #read into a data array with dimensions (time, frequency, baselines/pols)
        #and a times array holds the unix timestamp from each hdu
        #TODO: fix this to deal with a frequency or time subset?
        data_dump = np.zeros((self.Ntimes, self.Nfreqs, self.Nbls * self.Npols), dtype = np.complex)
        times = np.arange(start_time, start_time + self.Ntimes * 0.5, 0.5)
        
        #build time array
        #TODO: convert time_array from unix to julian dates
        self.time_array = np.array([i for i in times for j in range(self.Nbls)])
        
        #read the earlier time files
        for file in file_dict['00']:
            #get the course channel index from the file name
            coarse=int(file.split('_')[-2][-2:])
            with fits.open(file,memmap=False,do_not_scale_image_data=False) as hdu_list:
                end_time=len(hdu_list)
                #get number of fine channels
                fine=hdu_list[1].header['NAXIS2']
                freq_ind=chan_order[coarse]*fine
                for i in range(1,end_time):
                    time=hdu_list[i].header['TIME']+hdu_list[i].header['MILLITIM']/1000.0
                    time_ind=np.where(times==time)[0][0]
                    #this takes data from real to complex numbers
                    data_dump[time_ind,freq_ind:freq_ind+fine,:]=hdu_list[i].data[:,0::2]+1j*hdu_list[i].data[:,1::2]
        #read the later time files
        for file in file_dict['01']:
            #get the coarse channel index from the file name
            coarse=int(file.split('_')[-2][-2:])
            with fits.open(file,memmap=False,do_not_scale_image_data=False) as hdu_list:
                end_time=len(hdu_list)
                #find frequency index
                freq_ind=chan_order[coarse]*fine
                for i in range(1,end_time):
                    time=hdu_list[i].header['TIME']+hdu_list[i].header['MILLITIM']/1000.0
                    time_ind=np.where(times==time)[0][0]
                    #this takes data from real to complex numbers
                    data_dump[time_ind,freq_ind:freq_ind+fine,:] = hdu_list[i].data[:,0::2] + 1j*hdu_list[i].data[:,1::2]
        
        #TODO: add flagging for missing times
        #TODO: read in flag files

        #TODO: check conjugation
        #build new data array
        #polarizations are ordered yy, yx, xy, xx
        self.polarization_array = np.array([-6, -8, -7, -5])
        data_reorder = np.zeros((self.Ntimes, self.Nbls, self.Nfreqs, self.Npols), dtype = np.complex)
        
        #build mapper from antenna numbers and polarizations to pfb inputs
        corr_ants_to_pfb_inputs = {}
        for i in range(len(self.antenna_numbers)):
            for p in range(2):
                corr_ants_to_pfb_inputs[(self.antenna_numbers[i], p)] = 2 * i + p
                
        #for mapping, start with a pair of antennas/polarizations 
        #this is the pair we want to find the data for
        #map the pair to the corresponding pfb input indices 
        #map the pfb input indices to the pfb output indices
        #these are the indices for the data corresponding to the initial antenna/pol pair
        pfb_inputs_to_outputs=input_output_mapping()
        for ant1 in range(128):
            for ant2 in range(ant1,128):
                for p1 in range(2):
                    for p2 in range(2):
                        #generate the indices in data_reorder for this combination 
                        #baselines are ordered (0,0),(0,1),...,(0,127),(1,1),.....
                        #polarizion of 0 (1) corresponds to y (x)
                        pol_ind=2*p1+p2
                        bls_ind=128*ant1-ant1*(ant1+1)/2+ant2
                        #find the pfb output indices for this combination
                        ind1_1,ind1_2=corr_ants_to_pfb_inputs[(ant1,p1)],corr_ants_to_pfb_inputs[(ant2,p2)]
                        ind2_1,ind2_2=pfb_inputs_to_outputs[(ind1_1)],pfb_inputs_to_outputs[(ind1_2)]
                        out_ant1=int(ind2_1/2)
                        out_ant2=int(ind2_2/2)
                        out_p1=ind2_1%2
                        out_p2=ind2_2%2
                        #the correlator has antenna 1 >= antenna2, so check if ind2_1 and ind2_2 satisfy this
                        #get the index for the data
                        if out_ant1<out_ant2:
                            data_index=2*out_ant2*(out_ant2+1)+4*out_ant1+2*out_p2+out_p1
                            #need to take the complex conjugate of the data
                            data_reorder[:,bls_ind,:,pol_ind]=np.conj(data_dump[:,:,data_index])
                        else:
                            data_index=2*out_ant1*(out_ant1+1)+4*out_ant2+2*out_p1+out_p2
                            data_reorder[:,bls_ind,:,pol_ind]=data_dump[:,:,data_index] 
        
        #add spectral window index
        #assign as data array        
        
        
        
        
        #TODO:
        self.baseline_array = np.zeros(self.Nblts, dtype = np.complex)
        #TODO:
        self.ant_1_array = np.zeros(self.Nblts)
        #TODO:
        self.ant_2_array = np.zeros(self.Nblts)
        #TODO:
        self.data_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs, self.Npols))
        #TODO:
        self.flag_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs, self.Npols))
        #TODO:
        self.lst_array = np.zeros(self.Nblts)
        #TODO:
        self.nsample_array = np.zeros((self.Nblts, self.Nspws, self.Nfreqs, self.Npols))
        #TODO
        self.polarization_array = np.zeros(self.Npols)
        #TODO ask Bryna what is the shape of this?
        #self.telescope_location
        #TODO talk to Bryna about this!
        self.time_array = np.zeros(self.Nblts)
        #TODO
        self.uvw_array = np.zeros((self.Nblts, 3))
            
            
            
            
            
            
    