"""Class for reading and writing casa measurement sets."""
from astropy import constants as const
import astropy.time as time
import numpy as np
import os
import warnings
from pyuvdata import UVData
import parameter as uvp
import casacore.tables as tables
import telescopes


polDict={1:1,2:2,3:3,4:4,5:-1,6:-3,7:-4,8:-2,9:-5,10:-7,11:-8,12:-6}

#convert from casa stokes integers to pyuvdata
class MS(UVData):
    """
    Defines a class for reading and writing casa measurement sets.
    Attributs:
      ms_required_extra: Names of optional MSParameters that are required for casa ms
    """
    ms_required_extra=['data_column','antenna_positions']
    def _ms_hist_to_string(self,history_table):
        '''
        converts a CASA history table into a string that can be stored as the uvdata history parameter.
        Returns this string
        Args: history_table, a casa table object
        Returns: casa history table converted to a string with \n denoting new lines and     denoting column breaks
        '''
        history_str='APP_PARAMS;CLI_COMMAND;APPLICATION;MESSAGE;OBJECT_ID;OBSERVATION_ID;ORIGIN;PRIORITY;TIME\n'
        app_params=history_table.getcol('APP_PARAMS')['array']
        cli_command=history_table.getcol('CLI_COMMAND')['array']
        application=history_table.getcol('APPLICATION')
        message=history_table.getcol('MESSAGE')
        obj_id=history_table.getcol('OBJECT_ID')
        obs_id=history_table.getcol('OBSERVATION_ID')
        origin=history_table.getcol('ORIGIN')
        priority=history_table.getcol('PRIORITY')
        times=history_table.getcol('TIME')
        #Now loop through columns and generate history string
        for tbrow in range(len(times)):
            newline=str(app_params[tbrow]) \
            +';'+str(cli_command[tbrow]) \
            +';'+str(application[tbrow]) \
            +';'+str(message[tbrow]) \
            +';'+str(obj_id[tbrow]) \
            +';'+str(obs_id[tbrow]) \
            +';'+str(origin[tbrow]) \
            +';'+str(priority[tbrow]) \
            +';'+str(times[tbrow])+'\n'
            history_str+=newline
        return history_str




    def write_ms(self):
        '''
        writing ms is not yet supported
        '''



    def read_ms(self,filepath,run_check=True,run_sanity_check=True,data_column='DATA'):
        '''
        read in a casa measurement set
        ARGS:
        filename: name of the measurement set folder
        run_check:
        '''
        if not os.path.exists(filepath):
            raise(IOError, filepath + ' not found')
        #set visibility units
        if(data_column=='DATA'):
            self.vis_units="UNCALIB"
        elif(data_column=='CORRECTED_DATA'):
            self.vis_units="JY"
        elif(data_column=='MODEL'):
            self.vis_units="JY"
        self.data_column=data_column
        #get spectral window information
        tb_spws=tables.table(filepath+'/SPECTRAL_WINDOW')
        freqs=tb_spws.getcol('CHAN_FREQ')
        self.freq_array=freqs
        self.Nfreqs=int(freqs.shape[1])
        self.channel_width=tb_spws.getcol('CHAN_WIDTH')[0,0]
        self.Nspws=int(freqs.shape[0])
        self.spw_array=np.arange(self.Nspws)
        tb_spws.close()
        #now get the data
        tb=tables.table(filepath)
        times_unique=time.Time(np.unique(tb.getcol('TIME')/(3600.*24.)),format='mjd').jd
        self.Ntimes=int(len(times_unique))
        data_array=tb.getcol(data_column)
        self.Nblts=int(data_array.shape[0])
        flag_array=tb.getcol('FLAG')
        #CASA stores data in complex array with dimension NbltsxNfreqsxNpols
        #-!-What about multiple spws?-!-
        if(len(data_array.shape)==3):
            data_array=np.expand_dims(data_array,axis=1)
            flag_array=np.expand_dims(flag_array,axis=1)
        self.data_array=data_array
        self.flag_array=flag_array
        self.Npols=int(data_array.shape[-1])
        self.uvw_array=tb.getcol('UVW')
        self.ant_1_array=tb.getcol('ANTENNA1').astype(np.int32)
        self.ant_2_array=tb.getcol('ANTENNA2').astype(np.int32)
        self.Nants_data=len(np.unique(np.concatenate((np.unique(self.ant_1_array),np.unique(self.ant_2_array)))))
        self.baseline_array=self.antnums_to_baseline(self.ant_1_array,self.ant_2_array)
        self.Nbls=len(np.unique(self.baseline_array))
        #Get times. MS that I'm used to are modified Julian dates in seconds (thanks to Danny Jacobs for figuring out the proper conversion)
        self.time_array=time.Time(tb.getcol('TIME')/(3600.*24.),format='mjd').jd
        #Polarization array
        tbPol=tables.table(filepath+'/POLARIZATION')
        polList=tbPol.getcol('CORR_TYPE')[0]#list of lists, probably with each list corresponding to SPW. 
        self.polarization_array=np.zeros(len(polList),dtype=np.int32)
        for polnum in range(len(polList)):
            self.polarization_array[polnum]=int(polDict[polList[polnum]])
        tbPol.close()
        #Integration time
        #use first interval and assume rest are constant (though measurement set has all integration times for each Nblt )
        #self.integration_time=tb.getcol('INTERVAL')[0]
        #for some reason, interval ends up larger than the difference between times...
        self.integration_time=(times_unique[1]-times_unique[0])*3600.*24.

        #open table with antenna location information
        tbAnt=tables.table(filepath+'/ANTENNA')
        tbObs=tables.table(filepath+'/OBSERVATION')
        self.telescope_name=tbObs.getcol('TELESCOPE_NAME')[0]
        self.instrument=tbObs.getcol('TELESCOPE_NAME')[0]
        tbObs.close()
        #Use Telescopes.py dictionary to set array position
        self.antenna_positions=tbAnt.getcol('POSITION')
        xyz_telescope_frame = tbAnt.getcolkeyword('POSITION','MEASINFO')['Ref']
        antFlags=np.empty(len(self.antenna_positions),dtype=bool)
        antFlags[:]=False
        for antnum in range(len(antFlags)):
            antFlags[antnum]=np.all(self.antenna_positions[antnum,:]==0)
        try:
            self.set_telescope_params()
        except:
            if(xyz_telescope_frame=='ITRF'):
                self.telescope_location=np.array(np.mean(self.antenna_positions[np.invert(antFlags),:],axis=0))
                #antenna names
        ant_names=tbAnt.getcol('STATION')
        self.Nants_telescope=len(antFlags[np.invert(antFlags)])
        test_name=ant_names[0]
        names_same=True
        for antnum in range(len(ant_names)):
            if(not(ant_names[antnum]==test_name)):
                names_same=False
        if(not(names_same)):
            self.antenna_names=ant_names#cotter measurement sets store antenna names in the NAMES column. 
        else:
            self.antenna_names=tbAnt.getcol('NAME')#importuvfits measurement sets store antenna namesin the STATION column.
        self.antenna_numbers=np.arange(len(self.antenna_names)).astype(int)
        nAntOrig=len(self.antenna_names)
        ant_names=[]
        for antNum in range(len(self.antenna_names)):
            if not(antFlags[antNum]):
                ant_names.append(self.antenna_names[antNum])
        self.antenna_names=ant_names
        self.antenna_numbers=self.antenna_numbers[np.invert(antFlags)]
        self.antenna_positions=self.antenna_positions[np.invert(antFlags),:]
        '''
        #remove blank names
        for axnum in range(self.antenna_positions.shape[1]):
            self.antenna_positions[:,axnum]-=np.mean(self.antenna_positions[:,axnum])
        try:
            thisTelescope=telescopes.get_telescope(self.instrument)
            self.telescope_location_lat_lon_alt_degrees=(np.degrees(thisTelescope['latitude']),np.degrees(thisTelescope['longitude']),thisTelescope['altitude'])
            #self.telescope_location=np.array(np.mean(tbAnt.getcol('POSITION'),axis=0))
            print 'Telescope %s is known. Using stored values.'%(self.instrument)
        except:
            #If Telescope is unknown, use mean ITRF Positions of antennas
            self.telescope_location=np.array(np.mean(tbAnt.getcol('POSITION'),axis=0))
        '''
        #self.telescope_location=np.array(np.mean(tbAnt.getcol('POSITION'),axis=0))
        #Warning: the value one gets with set_telescope_params is different from the mean of antenna locations.
        #        try:
        #except:            
        tbAnt.close()
        tbField=tables.table(filepath+'/FIELD')
        #print 'shape='+str(tbField.getcol('PHASE_DIR').shape[1])
        if(tbField.getcol('PHASE_DIR').shape[1]==2):
            self.phase_type='drift'
            self.set_drift()
        elif(tbField.getcol('PHASE_DIR').shape[1]==1):
            self.phase_type='phased'
            self.phase_center_epoch=float(tb.getcolkeyword('UVW','MEASINFO')['Ref'][1:])#MSv2.0 appears to assume J2000. Not sure how to specifiy otherwise
            self.phase_center_ra=tbField.getcol('PHASE_DIR')[0][0][0]
            self.phase_center_dec=tbField.getcol('PHASE_DIR')[0][0][1]
            self.set_phased()
        #else:
        #    self.phase_type='unknown'
        #set LST array from times and itrf
        self.set_lsts_from_time_array()

        #set the history parameter
        #as a string with \t indicating column breaks
        #\n indicating row breaks.
        self.history=self._ms_hist_to_string(tables.table(filepath+'/HISTORY'))
        #CASA weights column keeps track of number of data points averaged.
        self.nsample_array=tb.getcol('WEIGHT_SPECTRUM')
        if(len(self.nsample_array.shape)==3):
            self.nsample_array=np.expand_dims(self.nsample_array,axis=1)
        self.object_name=tbField.getcol('NAME')[0]
        tbField.close()
        tb.close()
        if run_check:
            self.check(run_sanity_check=run_sanity_check)
