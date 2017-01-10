"""Class for reading and writing casa measurement sets."""
from astropy import constants as const
from astropy.time import Time
import numpy as np
import warnings
from uvdata import UVData
import parameter as uvp
import casacore.tables as tables
import telescopes

polDict={1:1,2:2,3:3,4:4,5:-1,6:-2,7:-3,8:-4,9:-5,10:-6,11:-7,12:-8}
#convert from casa stokes integers to pyuvdata
class MS(UVData):
    """
    Defines a class for reading and writing casa measurement sets.
    Attributs:
      ms_required_extra: Names of optional MSParameters that are required for casa ms
    """
    def _ms_hist_to_string(history_table):
        '''
        converts a CASA history table into a string that can be stored as the uvdata history parameter.
        Returns this string
        Args: history_table, a casa table object
        Returns: casa history table converted to a string with \n denoting new lines and \t denoting column breaks
        '''
        history_str='APP_PARAMS\tCLI_COMMAND\tAPPLICATION\tMESSAGE\tOBJECT_ID\tOBSERVATION_ID\tORIGIN\tPRIORITY\tTIME\n'
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
            +'\t'+str(cli_command[tbrow]) \
            +'\t'+str(application[tbrow]) \
            +'\t'+str(message[tbrow]) \
            +'\t'+str(obj_id[tbrow]) \
            +'\t'+str(obs_id[tbrow]) \
            +'\t'+str(origin[tbrow]) \
            +'\t'+str(priority[tbrow]) \
            +'\t'+str(times[tbrow])+'\n'
            history_str+=newline
        return history_str




    def write_ms():
        '''
        writing ms is not yet supported
        '''



    def read_ms(self,filename,run_check=True,run_sanity_check=True,data_column='DATA'):
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
                self.vis_units.value="uncalib"
            elif(data_column=='CORRECTED_DATA'):
                self.vis_units.value="Jy"
            elif(data_column=='MODEL'):
                self.vis_units.value="Jy"
            self.data_column.value=data_column
            #get spectral window information
            tb_spws=casacore.tables.table(filename+'/SPECTRAL_WINDOW')
            freqs=tb_spws.getcol('CHAN_FREQ')
            self.freq_array.value=freqs
            self.Nfreqs.value=int(freqs.shape[1])
            self.channel_width.value=tb_spws.getcol('CHAN_WIDTH')[0,0]
            self.Nspws.value=int(freqs.shape[0])
            self.spw_array.value=n.arange(self.Nspws.value)
            tb_spws.close()
            #now get the data
            tb=casacore.tables.table(filename)
            times_unique=np.unique(tb.getcol('TIME'))
            self.Ntimes.value=int(len(times_unique))
            data_array=tb.getcol(data_column)
            flag_array=tb.getcol('FLAG')
            self.Nbls.value=int(self.Nblts.value/self.nTimes.value)
            #CASA stores data in complex array with dimension NbltsxNfreqsxNpols
            #-!-What about multiple spws?-!-
            if(len(data_array.shape)==3):
                data_array=np.expand_dims(data_array,axis=1)
                flag_array=np.expand_dims(flag_array,axis=1)
            self.data_array.value=data_array
            self.Npols.value=int(data_array.shape[2])
            self.uvw_array.value=tb.getcol('UVW').T
            self.ant_1_array.value=tb.getcol('ANTENNA1').astype(int32)
            self.ant_2_array.value=tb.getcol('ANTENNA2').astype(int32)
            self.Nants_data.value=len(np.unique(self.ant_1_array))
            self.basline_array.value=self.antnums_to_baseline(self.ant_1_array.value,self.ant2_array.value)
            #Get times. MS that I'm used to are modified Julian dates in seconds (thanks to Danny Jacobs for figuring out the proper conversion)
            self.time_array.value=time.Time(tb.getcol('TIME')/(3600.*24.),format='mjd').jd
            #Polarization array
            tbPol=tables.table(filename+'.ms/POLARIZATION')
            self.polarization_array.value=polDict[tbPol.getcol('CORR_TYPE')].astype(int)
            tbPol.close()
            #Integration time
            #use first interval and assume rest are constant (though measurement set has all integration times for each Nblt )
            self.integration_time.value=tb.getcol('INTERVAL')[0]
            #open table with antenna location information
            tbAnt=tables.table(filename+'/ANTENNA')
            #antenna names
            self.antenna_names.value=tbAnt.getcol('NAME')
            self.Nants_telescope.value=len(self.antenna_names)
            self.dish_diameters.value=tbAnt.getcol('DISH_DIAMETER')
            self.flag_row.value=tbAnt.getcol('FLAG_ROW')
            self.mount.value=tbAnt.getcol('MOUNT')
            #Source Field
            self.antenna_numbers.value=np.unique(tbAnt.getcol('ANTENNA1')).astype(int)
            #Telescope Name
            #Instrument
            self.instrument.value=tbAnt.getcol('STATION')[0]
            self.telescope_name.value=tbAnt.getcol('STATION')[0]
            #Use Telescopes.py dictionary to set array position
            self.antenna_positions=np.array(tbAnt.getcol('POSITION')-np.mean(tbAnt.getcol('POSITION'),axis=1))
            try:
                thisTelescope=telescopes.get_telescope(self.instrument.value)
                self.telescope_location_lat_lon_alt_degrees.value=(np.degrees(thisTelescope['latitude']),np.degrees(thisTelescope['longitutde']),thisTelescope['altitude'])
                self.telescope_location.value=np.array(np.mean(tbAnt.getcol('POSITION'),axis=1))
            except:
                #If Telescope is unknown, use mean ITRF Positions of antennas
                self.telescope_location.value=np.array(np.mean(tbAnt.getcol('POSITION'),axis=1))
            tbAnt.close()
            tbField=tables.table(filename+'/FIELD')
            if(tbField.getcol('PHASE_DIR').shape[1]==2):
                self.phase_type.value='drift'
            elif(tbField.getcol('PHASE_DIR').shape[1]==1):
                self.phase_type.value='phased'
                self.phase_center_epoch=2000.#MSv2.0 appears to assume J2000. Not sure how to specifiy otherwise
                self.phase_center_ra=tbField.getcol('PHASE_DIR')[0][0]
                self.phase_center_dec=tbField.getcol('PHASE_DIR')[0][1]
            #else:
            #    self.phase_type.value='unknown'
            #set LST array from times and itrf
            self.set_lsts_from_time_array()

            #set the history parameter
            #as a string with \t indicating column breaks
            #\n indicating row breaks.
            self.history=self._ms_hist_to_string(tables.table(filename+'/HISTORY'))
            #CASA weights column keeps track of number of data points averaged.
            self.nsample_array=tb.getcol('WEIGHT')
            self.object_name=''
            tb.close()
            if run_check:
                self.check(run_sanity_check=run_sanity_check)
