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
            'instrument','latitude','longitude','altitude','dateobs','history','vis_units','phase_center_epoch','Nants',
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
        self.test_file_directory = './uvdata_test_temp'
        if not os.path.exists(self.test_file_directory):
            os.mkdir(self.test_file_directory)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_file_directory)
    def test_ReadNRAO(self):
        testfile='../data/day2_TDEM0003_10s_norx.uvfits'
        UV = UVData()
        test = UV.read_uvfits(testfile)
        self.assertTrue(test)
class TestWriteUVFits(unittest.TestCase):
    def test_writeNRAO(self):
        testfile='../data/day2_TDEM0003_10s_norx.uvfits'
        UV = UVData()
        UV.read_uvfits(testfile)
        test = UV.write_uvfits('outtest.uvfits')
        self.assertTrue(test)




if __name__=='__main__':
    unittest.main()
