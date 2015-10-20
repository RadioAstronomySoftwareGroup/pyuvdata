import unittest
import inspect
from uvdata.uv import UVData

class TestUVDataInit(unittest.TestCase):
    def setUp(self):
        self.default_attributes = [
            'data_array','nsample_array','flag_array','Ntimes','Nbls','Nblts',
            'Nfreqs','Npols','Nspws','uvw_array','time_array','ant_1_array','ant_2_array',
            'baseline_array','freq_array','polarization_array','spw_array','phase_center_ra',
            'phase_center_dec','integration_time','channel_width','object_name','telescope',
            'instrument','dateobs','history','vis_units','phase_center_epoch','Nants',
            'antenna_names','antenna_indices','antenna_frame','x_array','y_array','z_array',
            'GST0','Rdate','earth_omega','fits_extra_keywords']    
        self.uv_object = UVData()
    def tearDown(self):
        del(self.uv_object)
    def test_default_attributes_exist(self):
        for attribute in self.default_attributes:
            self.assertTrue(hasattr(self.uv_object, attribute), 
                            msg = 'expected attribute ' + attribute + ' does not exist')
    def test_default_attribute_values(self):
        for attribute in self.default_attributes:
            self.assertIsNone(getattr(self.uv_object, attribute), 
                              msg = 'attribute ' + attribute + ' is not None as expected')
    def test_unexpected_attributes(self):
        attributes = [i for i in self.uv_object.__dict__.keys() if i[0] != '_']
        for attribute in attributes:
            self.assertTrue(attribute in self.default_attributes, 
                            msg = 'unexpected attribute ' + attribute + ' found in UVData')

class TestReadUVFits(unittest.TestCase):
    def setUp(self):
        import os





if __name__=='__main__':
    unittest.main()
