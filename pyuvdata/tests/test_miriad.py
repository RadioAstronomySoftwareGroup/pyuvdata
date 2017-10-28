"""Tests for Miriad object."""
import os
import shutil
import copy
import numpy as np
import ephem
import nose.tools as nt
import aipy
from pyuvdata import UVData
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import aipy.miriad as amiriad
from astropy import constants as const
import warnings


def test_ReadWriteReadATCA():
    uv_in = UVData()
    uv_out = UVData()
    atca_file = os.path.join(DATA_PATH, 'atca_miriad')
    testfile = os.path.join(DATA_PATH, 'test/outtest_atca_miriad.uv')
    uvtest.checkWarnings(uv_in.read_miriad, [atca_file],
                         nwarnings=3, category=[UserWarning, UserWarning, UserWarning],
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope ATCA is not in known_telescopes. '
                                  'Telescope location',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna position',
                                  'Telescope ATCA is not in known_telescopes.'])

    uv_in.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read_miriad, [testfile],
                         message='Telescope ATCA is not in known_telescopes.')
    nt.assert_equal(uv_in, uv_out)


def test_ReadNRAOWriteMiriadReadMiriad():
    """Test reading in a CASA tutorial uvfits file, writing and reading as miriad"""
    uvfits_uv = UVData()
    miriad_uv = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                               'RESTFREQ', 'ORIGIN']
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [testfile], message='Telescope EVLA is not')
    uvfits_uv.write_miriad(testfile + '.uv', clobber=True)
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile + '.uv'], message='Telescope EVLA is not')
    nt.assert_equal(uvfits_uv, miriad_uv)
    del(uvfits_uv)
    del(miriad_uv)


def test_ReadMiriadWriteUVFits():
    """
    Miriad to uvfits loopback test.

    Read in Miriad files, write out as uvfits, read back in and check for
    object equality.
    """
    miriad_uv = UVData()
    uvfits_uv = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uvfits')
    uvtest.checkWarnings(miriad_uv.read_miriad, [miriad_file],
                         known_warning='miriad')
    miriad_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    # these are not equal because miriad_uv still retains zenith_dec and
    # zenith_ra, which are not present in uvfits_uv
    nt.assert_false(miriad_uv == uvfits_uv)
    # they are equal if only required parameters are checked:
    nt.assert_true(miriad_uv.__eq__(uvfits_uv, check_extra=False))

    # remove zenith_ra and zenith_dec to test that the rest of the objects are equal
    miriad_uv.zenith_ra = None
    miriad_uv.zenith_dec = None
    nt.assert_equal(miriad_uv, uvfits_uv)

    # check error if phase_type is wrong and force_phase not set
    uvtest.checkWarnings(miriad_uv.read_miriad, [miriad_file],
                         known_warning='miriad')
    nt.assert_raises(ValueError, miriad_uv.write_uvfits, testfile, spoof_nonessential=True)
    miriad_uv.set_unknown_phase_type()
    nt.assert_raises(ValueError, miriad_uv.write_uvfits, testfile, spoof_nonessential=True)

    # check error if spoof_nonessential not set
    uvtest.checkWarnings(miriad_uv.read_miriad, [miriad_file],
                         known_warning='miriad')
    nt.assert_raises(ValueError, miriad_uv.write_uvfits, testfile, force_phase=True)

    # check warning when correct_lat_lon is set to False
    uvtest.checkWarnings(miriad_uv.read_miriad, [miriad_file],
                         {'correct_lat_lon': False},
                         message=['Altitude is not present in Miriad file, '
                                  'using known location altitude'])

    # check that setting the phase_type to something wrong errors
    nt.assert_raises(ValueError, uvtest.checkWarnings, miriad_uv.read_miriad,
                     [miriad_file], {'phase_type': 'phased'})
    nt.assert_raises(ValueError, uvtest.checkWarnings, miriad_uv.read_miriad,
                     [miriad_file], {'phase_type': 'foo'})

    del(miriad_uv)
    del(uvfits_uv)


def test_wronglatlon():
    """
    Check for appropriate warnings with incorrect lat/lon values or missing telescope

    To test this, we needed files without altitudes and with wrong lat, lon or telescope values.
    These test files were made commenting out the line in miriad.py that adds altitude
    to the file and running the following code:
    import os
    import numpy as np
    from pyuvdata import UVData
    from pyuvdata.data import DATA_PATH
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    latfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglat.xy.uvcRREAA')
    lonfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglon.xy.uvcRREAA')
    telescopefile = os.path.join(DATA_PATH, 'zen.2456865.60537_wrongtelecope.xy.uvcRREAA')
    uv_in.read_miriad(miriad_file)
    uv_in.select(times=uv_in.time_array[0])
    uv_in.select(freq_chans=[0])

    lat, lon, alt = uv_in.telescope_location_lat_lon_alt
    lat_wrong = lat + 10 * np.pi / 180.
    uv_in.telescope_location_lat_lon_alt = (lat_wrong, lon, alt)
    uv_in.write_miriad(latfile)
    uv_out.read_miriad(latfile)

    lon_wrong = lon + 10 * np.pi / 180.
    uv_in.telescope_location_lat_lon_alt = (lat, lon_wrong, alt)
    uv_in.write_miriad(lonfile)
    uv_out.read_miriad(lonfile)

    uv_in.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_in.telescope_name = 'foo'
    uv_in.write_miriad(telescopefile)
    uv_out.read_miriad(telescopefile, run_check=False)
    """
    uv_in = UVData()
    latfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglat.xy.uvcRREAA')
    lonfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglon.xy.uvcRREAA')
    telescopefile = os.path.join(DATA_PATH, 'zen.2456865.60537_wrongtelecope.xy.uvcRREAA')

    uvtest.checkWarnings(uv_in.read_miriad, [latfile], nwarnings=2,
                         message=['Altitude is not present in file and latitude value does not match',
                                  latfile + ' was written with an old version of pyuvdata'])
    uvtest.checkWarnings(uv_in.read_miriad, [lonfile], nwarnings=2,
                         message=['Altitude is not present in file and longitude value does not match',
                                  lonfile + ' was written with an old version of pyuvdata'])
    uvtest.checkWarnings(uv_in.read_miriad, [telescopefile], {'run_check': False},
                         nwarnings=4,
                         message=['Altitude is not present in Miriad file, and telescope',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna position',
                                  telescopefile + ' was written with an old version of pyuvdata',
                                  'Telescope foo is not in known_telescopes.'])


def test_miriad_location_handling():
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testdir = os.path.join(DATA_PATH, 'test/')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    aipy_uv = aipy.miriad.UV(miriad_file)

    if os.path.exists(testfile):
        shutil.rmtree(testfile)

    # Test for using antenna positions to get telescope position
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         message=['Altitude is not present in Miriad file, using known '
                                  'location values for PAPER.'])
    # extract antenna positions and rotate them for miriad
    nants = aipy_uv['nants']
    rel_ecef_antpos = np.zeros((nants, 3), dtype=uv_in.antenna_positions.dtype)
    for ai, num in enumerate(uv_in.antenna_numbers):
        rel_ecef_antpos[num, :] = uv_in.antenna_positions[ai, :]

    # find zeros so antpos can be zeroed there too
    antpos_length = np.sqrt(np.sum(np.abs(rel_ecef_antpos)**2, axis=1))

    ecef_antpos = rel_ecef_antpos + uv_in.telescope_location
    longitude = uv_in.telescope_location_lat_lon_alt[1]
    antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)

    # zero out bad locations (these are checked on read)
    antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    antpos = antpos.T.flatten() / const.c.to('m/ns').value

    # make new file
    aipy_uv2 = aipy.miriad.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': antpos})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read_miriad, [testfile], nwarnings=3,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is not set, but antenna '
                                  'positions are present. Mean antenna latitude '
                                  'and longitude values match file values, so '
                                  'telescope_position will be set using the mean '
                                  'of the antenna altitudes',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions have a different mean latitude than the file latitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy.miriad.UV(miriad_file)
    aipy_uv2 = aipy.miriad.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude
    new_lat = aipy_uv['latitud'] * 1.5
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': antpos,
                                             'latitud': new_lat})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read_miriad, [testfile], nwarnings=3,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna latitude '
                                  'value does not match',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions have a different mean longitude than the file longitude
    # this is harder because of the rotation that's done on the antenna positions
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy.miriad.UV(miriad_file)
    aipy_uv2 = aipy.miriad.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file longitude
    new_lon = aipy_uv['longitu'] + np.pi
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': antpos,
                                             'longitu': new_lon})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read_miriad, [testfile], nwarnings=3,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna longitude '
                                  'value does not match',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions have a different mean longitude &
    # latitude than the file longitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy.miriad.UV(miriad_file)
    aipy_uv2 = aipy.miriad.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude and longitude
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': antpos,
                                             'latitud': new_lat, 'longitu': new_lon})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read_miriad, [testfile], nwarnings=3,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna latitude and '
                                  'longitude values do not match',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions are far enough apart to make the
    # mean position inside the earth

    good_antpos = np.where(antpos_length > 0)[0]
    rot_ants = good_antpos[:len(good_antpos) / 2]
    rot_antpos = uvutils.rotECEF_from_ECEF(ecef_antpos[rot_ants, :], longitude + np.pi)
    modified_antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)
    modified_antpos[rot_ants, :] = rot_antpos
    # zero out bad locations (these are checked on read)
    modified_antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    modified_antpos = modified_antpos.T.flatten() / const.c.to('m/ns').value

    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy.miriad.UV(miriad_file)
    aipy_uv2 = aipy.miriad.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use modified absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': modified_antpos})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read_miriad, [testfile], nwarnings=3,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna position '
                                  'does not give a telescope_location on the '
                                  'surface of the earth.',
                                  'Telescope foo is not in known_telescopes.'])


def test_singletimeselect_drift():
    """
    Check behavior with writing & reading after selecting a single time from a drift file.

    """
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')

    uv_in.select(times=uv_in.time_array[0])
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read_miriad(testfile)
    nt.assert_equal(uv_in, uv_out)

    # check that setting the phase_type works
    uv_out.read_miriad(testfile, phase_type='drift')
    nt.assert_equal(uv_in, uv_out)


def test_poltoind():
    miriad_uv = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uvtest.checkWarnings(miriad_uv.read_miriad, [miriad_file], known_warning='miriad')
    pol_arr = miriad_uv.polarization_array

    miriad = miriad_uv._convert_to_filetype('miriad')
    miriad.polarization_array = None
    nt.assert_raises(ValueError, miriad._pol_to_ind, pol_arr[0])

    miriad.polarization_array = [pol_arr[0], pol_arr[0]]
    nt.assert_raises(ValueError, miriad._pol_to_ind, pol_arr[0])


def test_miriad_extra_keywords():
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords['testdict'] = {'testkey': 23}
    uvtest.checkWarnings(uv_in.check, message=['testdict in extra_keywords is a '
                                               'list, array or dict'])
    nt.assert_raises(TypeError, uv_in.write_miriad, testfile, clobber=True,
                     run_check=False)
    uv_in.extra_keywords.pop('testdict')

    uv_in.extra_keywords['testlist'] = [12, 14, 90]
    uvtest.checkWarnings(uv_in.check, message=['testlist in extra_keywords is a '
                                               'list, array or dict'])
    nt.assert_raises(TypeError, uv_in.write_miriad, testfile, clobber=True,
                     run_check=False)
    uv_in.extra_keywords.pop('testlist')

    uv_in.extra_keywords['testarr'] = np.array([12, 14, 90])
    uvtest.checkWarnings(uv_in.check, message=['testarr in extra_keywords is a '
                                               'list, array or dict'])
    nt.assert_raises(TypeError, uv_in.write_miriad, testfile, clobber=True,
                     run_check=False)
    uv_in.extra_keywords.pop('testarr')

    # check for warnings with extra_keywords keys that are too long
    uv_in.extra_keywords['test_long_key'] = True
    uvtest.checkWarnings(uv_in.check, message=['key test_long_key in extra_keywords '
                                               'is longer than 8 characters'])
    uvtest.checkWarnings(uv_in.write_miriad, [testfile], {'clobber': True, 'run_check': False},
                         message=['key test_long_key in extra_keywords is longer than 8 characters'])
    uv_in.extra_keywords.pop('test_long_key')

    # check handling of boolean keywords
    uv_in.extra_keywords['bool'] = True
    uv_in.extra_keywords['bool2'] = False
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read_miriad(testfile)

    nt.assert_equal(uv_in, uv_out)
    uv_in.extra_keywords.pop('bool')
    uv_in.extra_keywords.pop('bool2')

    # check handling of int-like keywords
    uv_in.extra_keywords['int1'] = np.int(5)
    uv_in.extra_keywords['int2'] = 7
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read_miriad(testfile)

    nt.assert_equal(uv_in, uv_out)
    uv_in.extra_keywords.pop('int1')
    uv_in.extra_keywords.pop('int2')

    # check handling of float-like keywords
    uv_in.extra_keywords['float1'] = np.int64(5.3)
    uv_in.extra_keywords['float2'] = 6.9
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read_miriad(testfile)

    nt.assert_equal(uv_in, uv_out)
    uv_in.extra_keywords.pop('float1')
    uv_in.extra_keywords.pop('float2')

    # check handling of complex-like keywords
    # currently they are NOT supported
    uv_in.extra_keywords['complex1'] = np.complex64(5.3 + 1.2j)
    uv_in.extra_keywords['complex2'] = 6.9 + 4.6j
    nt.assert_raises(TypeError, uv_in.write_miriad, testfile, clobber=True)


def test_breakReadMiriad():
    """Test Miriad file checking."""
    uv_in = UVData()
    uv_out = UVData()
    nt.assert_raises(IOError, uv_in.read_miriad, 'foo')

    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')
    uv_in.Npols += 1
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read_miriad, [testfile], {'run_check': False},
                         message=['npols=2 but found 1 pols in data file'])

    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')
    uv_in.Nblts += 10
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read_miriad, [testfile], {'run_check': False},
                         message=['Nblts does not match the number of unique blts in the data'])

    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')
    uv_in.Nbls += 10
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read_miriad, [testfile], {'run_check': False},
                         message=['Nbls does not match the number of unique baselines in the data'])

    uvtest.checkWarnings(uv_in.read_miriad, [miriad_file],
                         known_warning='miriad')
    uv_in.Ntimes += 10
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read_miriad, [testfile], {'run_check': False},
                         message=['Ntimes does not match the number of unique times in the data'])


def test_readWriteReadMiriad():
    """
    PAPER file Miriad loopback test.

    Read in Miriad PAPER file, write out as new Miriad file, read back in and
    check for object equality.
    """
    uv_in = UVData()
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read_miriad, [testfile], known_warning='miriad')
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)

    nt.assert_equal(uv_in, uv_out)

    # check that trying to overwrite without clobber raises an error
    nt.assert_raises(ValueError, uv_in.write_miriad, write_file)

    # check that if x_orientation is set, it's read back out properly
    uv_in.x_orientation = 'east'
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)
    nt.assert_equal(uv_in, uv_out)

    # check that if antenna_diameters is set, it's read back out properly
    uvtest.checkWarnings(uv_in.read_miriad, [testfile], known_warning='miriad')
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float) + 14.0
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)
    nt.assert_equal(uv_in, uv_out)

    # check that antenna diameters get written if not exactly float
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float32) + 14.0
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)
    nt.assert_equal(uv_in, uv_out)

    # check that trying to write a file with unknown phasing raises an error
    uv_in.set_unknown_phase_type()
    nt.assert_raises(ValueError, uv_in.write_miriad, write_file, clobber=True)

    # check for backwards compatibility with old keyword 'diameter' for antenna diameters
    testfile_diameters = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
    uv_in.read_miriad(testfile_diameters)
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read_miriad(write_file)
    nt.assert_equal(uv_in, uv_out)

    # check that variables 'ischan' and 'nschan' were written to new file
    # need to use aipy, since pyuvdata is not currently capturing these variables
    uv_in.read_miriad(write_file)
    uv_aipy = amiriad.UV(write_file)
    nfreqs = uv_in.Nfreqs
    nschan = uv_aipy['nschan']
    ischan = uv_aipy['ischan']
    nt.assert_equal(nschan, nfreqs)
    nt.assert_equal(ischan, 1)

    del(uv_in)
    del(uv_out)
    del(uv_aipy)


def test_readMSWriteMiriad_CASAHistory():
    """
    read in .ms file.
    Write to a miriad file, read back in and check for history parameter
    """
    ms_uv = UVData()
    miriad_uv = UVData()
    ms_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.ms')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad')
    uvtest.checkWarnings(ms_uv.read_ms, [ms_file],
                         message='Telescope EVLA is not',
                         nwarnings=0)
    ms_uv.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(miriad_uv.read_miriad, [testfile],
                         message='Telescope EVLA is not')

    nt.assert_equal(miriad_uv, ms_uv)


def test_rwrMiriad_antpos_issues():
    """
    test warnings and errors associated with antenna position issues in Miriad files

    Read in Miriad PAPER file, mess with various antpos issues and write out as
    a new Miriad file, read back in and check for appropriate behavior.
    """
    uv_in = UVData()
    uv_out = UVData()
    testfile = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    write_file = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read_miriad, [testfile], known_warning='miriad')
    uv_in.antenna_positions = None
    uv_in.write_miriad(write_file, clobber=True)
    uvtest.checkWarnings(uv_out.read_miriad, [write_file],
                         message=['Antenna positions are not present in the file.'])

    nt.assert_equal(uv_in, uv_out)

    uvtest.checkWarnings(uv_in.read_miriad, [testfile], known_warning='miriad')
    ants_with_data = list(set(uv_in.ant_1_array).union(uv_in.ant_2_array))
    ant_ind = np.where(uv_in.antenna_numbers == ants_with_data[0])[0]
    uv_in.antenna_positions[ant_ind, :] = [0, 0, 0]
    uv_in.write_miriad(write_file, clobber=True, no_antnums=True)
    uvtest.checkWarnings(uv_out.read_miriad, [write_file], message=['antenna number'])

    nt.assert_equal(uv_in, uv_out)

    uvtest.checkWarnings(uv_in.read_miriad, [testfile], known_warning='miriad')
    uv_in.antenna_positions = None
    ants_with_data = sorted(list(set(uv_in.ant_1_array).union(uv_in.ant_2_array)))
    new_nums = []
    new_names = []
    for a in ants_with_data:
        new_nums.append(a)
        ind = np.where(uv_in.antenna_numbers == a)[0][0]
        new_names.append(uv_in.antenna_names[ind])
    uv_in.antenna_numbers = np.array(new_nums)
    uv_in.antenna_names = new_names
    uv_in.Nants_telescope = len(uv_in.antenna_numbers)
    uv_in.write_miriad(write_file, clobber=True, no_antnums=True)
    uvtest.checkWarnings(uv_out.read_miriad, [write_file],
                         message=['Antenna positions are not present in the file.'])

    nt.assert_equal(uv_in, uv_out)


'''
This test is commented out since we no longer believe AIPY phases correctly
to the astrometric ra/dec.  Hopefully we can reinstitute it one day.
def test_ReadMiriadPhase():
    unphasedfile = os.path.join(DATA_PATH, 'new.uvA')
    phasedfile = os.path.join(DATA_PATH, 'new.uvA.phased')
    unphased_uv = UVData()
    phased_uv = UVData()
    # test that phasing makes files equal
    uvtest.checkWarnings(unphased.read, [unphasedfile, 'miriad'], known_warning='miriad')
    unphased.phase(ra=0.0, dec=0.0, epoch=ephem.J2000)
    uvtest.checkWarnings(phased.read, [phasedfile, 'miriad'], known_warning='miriad')
    nt.assert_equal(unphased, phased)
'''


def test_multi_files():
    """
    Reading multiple files at once.
    """
    uv_full = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile1 = os.path.join(DATA_PATH, 'test/uv1')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv_full.unphase_to_drift()
    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_miriad(testfile1, clobber=True)
    uv2.write_miriad(testfile2, clobber=True)
    uvtest.checkWarnings(uv1.read_miriad, [[testfile1, testfile2]], nwarnings=2,
                         message=['Telescope EVLA is not', 'Telescope EVLA is not'])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_true(uvutils.check_histories(uv_full.history + '  Downselected to '
                                           'specific frequencies using pyuvdata. '
                                           'Combined data along frequency axis using'
                                           ' pyuvdata.', uv1.history))
    uv1.history = uv_full.history
    nt.assert_equal(uv1, uv_full)


def test_antpos_units():
    """
    Read uvfits, write miriad. Check written antpos are in ns.
    """
    uv = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test/uv_antpos_units')
    uvtest.checkWarnings(uv.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv.write_miriad(testfile, clobber=True)
    auv = amiriad.UV(testfile)
    aantpos = auv['antpos'].reshape(3, -1).T * const.c.to('m/ns').value
    aantpos = aantpos[uv.antenna_numbers, :]
    aantpos = (uvutils.ECEF_from_rotECEF(aantpos, uv.telescope_location_lat_lon_alt[1]) -
               uv.telescope_location)
    nt.assert_true(np.allclose(aantpos, uv.antenna_positions))
