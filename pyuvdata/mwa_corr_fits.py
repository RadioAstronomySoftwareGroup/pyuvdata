# -*- coding: utf-8 -*-
"""Class for reading MWA correlator FITS files

"""
from astropy.io import fits
import numpy as np
import warnings

from . import UVData
from . import utils as uvutils

def input_output_mapping():
    '''
    Builds a mapping dictionary from the pfb output numbers (the correlator indices for
    antenna number and polarization) to the pfb input numbers and polarizations. The input numbers
    can be mapped to tile numbers using metadata.
    The MWA correlator indexes by antenna1, antenna2, polarization1, polarization2,real/imag,
    where the rightmost index moves fastest.
    '''
    #this comes from mwa_build_lfiles/mwac_utils.c 
    #here's the mapper for pfb outputs:
    pfb_mapper=[0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]
    #build a mapping matrix
    #build a dictionary whose keys are tuples (index1,index2)
    #index1 and index2 are pfb output numbers, and should range from 0 to nstation
    pfb_outputs_to_inputs={}
    for p in range(4):
        for i in range(64):
            
            pfb_outputs_to_inputs[p*64+i]=pfb_mapper[i]+p*64
    #now build a dictionary that maps the correlator antenna indices and polarizations
    #to the pfb input antenna indices and polarizations by mapping the key
    #(index1,index2) to a list: [stn,stn, pol1,pol2]
    #these can then be mapped to antenna tiles using the metadata
    corr_mapper={}
    #iterate over the number of antennas (128) and x and y (2)
    for inp1 in range(128):
        for inp2 in range(128):
            for p1 in range(2):
                for p2 in range(2):
                    index1=inp1*2+p1
                    index2=inp2*2+p2
                    corr_mapper[(pfb_outputs_to_inputs[index1],pfb_outputs_to_inputs[index2])]=[inp1,inp2,p1,p2]
    return corr_mapper
#what to do with the metafits_ppds file?
#also flag files?

class MWACorrFITS(UVData):
    """
    Defines a MWA correlator fits-specific subclass of UVData for reading MWA 
    correlator fits files. This class should not be interacted with directly, 
    instead use the read_mwa_corr_fits method on the UVData class.
    """
    def read_mwa_corr_fits(self, filelist, run_check=True, check_extra=True, 
                 run_check_acceptability=True):
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
        
        #TODO: check that data for all the contiguous coarse channels are present
        
        #check that have a metafits file
        #TODO: ***fix this so that it allows for multiple metafits files!***
        #for now, force only one metafits file
        metafits_file=None
        for file in filelist:
            if file.lower().endswith('.metafits'):
                if metafits_file is not None:
                    raise ValueError('multiple metafits files in filelist')
                metafits_file = file
            else:
                continue #need this?
                
        #TODO: check that visibility files are included
        #TODO: look for flag files; not required but give a warning if they don't exist
        
        #first set parameters that are always true
        self.Nspws=1
        self.spw_array = np.array([0])
        self.phase_type='drift'
        self.vis_units='uncalib'
        self.Npols=4 #TODO ask Bryna: should this be allowed to vary?
        
        #pull parameters from metafits file
        with fits.open(metafits_file, memmap=True) as meta:
            meta_hdr=meta[0].header
            #get data from header
            self.Nfreqs=meta_hdr['NCHANS']
            self.Ntimes=meta_hdr['NSCANS']
            self.channel_width=meta_hdr['FINECHAN']#this assumes no averaging by this code so will need to be updated
            self.history = str(meta_hdr['HISTORY'])
            if not uvutils._check_history_version(self.history, self.pyuvdata_version_str):
                self.history += self.pyuvdata_version_str
            self.integration_time=meta_hdr['INTTIME'] #TODO: talk to Bryna re: parameter definition)
            self.instrument=meta_hdr['INSTRUME']
            self.telescope_name=meta_hdr['TELESCOP']
            self.object_name=meta_hdr['FILENAME']
            #TODO: remove these keys and store remaining keys in extra keywords
            #TODO: check that coarse channels are contiguous
#            chans=meta[0].header['CHANNELS'].split(',') 
#            chans=np.array([int(i) for i in chans])
            
            #get data from metafits file table
            meta_tbl=meta[1].data
            self.antenna_numbers=meta_tbl['Tile'][1::2]#because of polarization, each antenna # is listed twice
            self.antenna_names=meta_tbl['TileName'][1::2]
            self.Nants_data=len(self.antenna_numbers)
            self.Nants_telescope=len(self.antenna_numbers)
            self.Nbls=len(self.antenna_numbers)*(len(self.antenna_numbers+1))/2
            self.Nblts=self.Nbls*self.Ntimes
            #TODO: self.freq_array
            
            #TODO: self.antenna_positions
            #TODO: self.antenna_diameters
            #TODO: self.x_orientation
            
            #TODO:
            self.baseline_array=np.zeros(self.Nblts)
            #TODO:
            self.ant_1_array=np.zeros(self.Nblts)
            #TODO:
            self.ant_2_array=np.zeros(self.Nblts)
            #TODO:
            self.data_array=np.zeros((self.Nblts,self.Nspws,self.Nfreqs,self.Npols))
            #TODO:
            self.flag_array=np.zeros((self.Nblts,self.Nspws,self.Nfreqs,self.Npols))
            #TODO:
            self.lst_array=np.zeros(self.Nblts)
            #TODO:
            self.nsample_array=np.zeros((self.Nblts,self.Nspws,self.Nfreqs,self.Npols))
            #TODO
            self.polarization_array=np.zeros(self.Npols)
            #TODO ask Bryna what is the shape of this?
            #self.telescope_location
            #TODO
            self.time_array=np.zeros(self.Nblts)
            #TODO
            self.uvw_array=np.zeros((self.Nblts,3))
            
            
            
            
            
            
    