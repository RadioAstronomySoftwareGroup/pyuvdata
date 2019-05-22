# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""Tests for Miriad object.

"""
from __future__ import absolute_import, division, print_function

import os
import shutil
import copy
import six
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from astropy import constants as const
from astropy.utils import iers

from pyuvdata import UVData
from pyuvdata.miriad import Miriad
import pyuvdata.utils as uvutils
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH

from .. import aipy_extracts


def test_ReadWriteReadATCA():
    uv_in = UVData()
    uv_out = UVData()
    atca_file = os.path.join(DATA_PATH, 'atca_miriad')
    testfile = os.path.join(DATA_PATH, 'test/outtest_atca_miriad.uv')
    uvtest.checkWarnings(uv_in.read, [atca_file],
                         nwarnings=4, category=[UserWarning, UserWarning, UserWarning, UserWarning],
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope ATCA is not in known_telescopes. ',
                                  'Altitude is not present',
                                  'Telescope location is set at sealevel at the file lat/lon '
                                  'coordinates. Antenna positions are present, but the mean antenna '
                                  'position does not give a telescope_location on the surface of the '
                                  'earth. Antenna positions do not appear to be on the surface of the '
                                  'earth and will be treated as relative.',
                                  'Telescope ATCA is not in known_telescopes.'])

    uv_in.write_miriad(testfile, clobber=True)
    uvtest.checkWarnings(uv_out.read, [testfile],
                         message='Telescope ATCA is not in known_telescopes.')
    assert uv_in == uv_out


def test_ReadNRAOWriteMiriadReadMiriad():
    """Test reading in a CASA tutorial uvfits file, writing and reading as miriad"""
    uvfits_uv = UVData()
    miriad_uv = UVData()
    testfile = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    writefile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    expected_extra_keywords = ['OBSERVER', 'SORTORD', 'SPECSYS',
                               'RESTFREQ', 'ORIGIN']
    uvtest.checkWarnings(uvfits_uv.read_uvfits, [testfile], message='Telescope EVLA is not')
    uvfits_uv.write_miriad(writefile, clobber=True)
    uvtest.checkWarnings(miriad_uv.read, [writefile], message='Telescope EVLA is not')
    assert uvfits_uv == miriad_uv
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
    uvtest.checkWarnings(miriad_uv.read, [miriad_file],
                         known_warning='miriad')
    miriad_uv.write_uvfits(testfile, spoof_nonessential=True, force_phase=True)
    uvfits_uv.read_uvfits(testfile)
    assert miriad_uv == uvfits_uv

    # check error if phase_type is wrong and force_phase not set
    uvtest.checkWarnings(miriad_uv.read, [miriad_file],
                         known_warning='miriad')
    pytest.raises(ValueError, miriad_uv.write_uvfits, testfile, spoof_nonessential=True)
    miriad_uv.set_unknown_phase_type()
    pytest.raises(ValueError, miriad_uv.write_uvfits, testfile, spoof_nonessential=True)

    # check error if spoof_nonessential not set
    uvtest.checkWarnings(miriad_uv.read, [miriad_file],
                         known_warning='miriad')
    pytest.raises(ValueError, miriad_uv.write_uvfits, testfile, force_phase=True)

    # check warning when correct_lat_lon is set to False
    uvtest.checkWarnings(miriad_uv.read, [miriad_file],
                         {'correct_lat_lon': False}, nwarnings=1,
                         message=['Altitude is not present in Miriad file, using known location altitude value for PAPER and lat/lon from file.'])

    # check that setting the phase_type to something wrong errors
    pytest.raises(ValueError, uvtest.checkWarnings, miriad_uv.read,
                  [miriad_file], {'phase_type': 'phased'})
    pytest.raises(ValueError, uvtest.checkWarnings, miriad_uv.read,
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
    uv_in.read(miriad_file)
    uv_in.select(times=uv_in.time_array[0])
    uv_in.select(freq_chans=[0])

    lat, lon, alt = uv_in.telescope_location_lat_lon_alt
    lat_wrong = lat + 10 * np.pi / 180.
    uv_in.telescope_location_lat_lon_alt = (lat_wrong, lon, alt)
    uv_in.write_miriad(latfile)
    uv_out.read(latfile)

    lon_wrong = lon + 10 * np.pi / 180.
    uv_in.telescope_location_lat_lon_alt = (lat, lon_wrong, alt)
    uv_in.write_miriad(lonfile)
    uv_out.read(lonfile)

    uv_in.telescope_location_lat_lon_alt = (lat, lon, alt)
    uv_in.telescope_name = 'foo'
    uv_in.write_miriad(telescopefile)
    uv_out.read(telescopefile, run_check=False)
    """
    uv_in = UVData()
    latfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglat.xy.uvcRREAA')
    lonfile = os.path.join(DATA_PATH, 'zen.2456865.60537_wronglon.xy.uvcRREAA')
    telescopefile = os.path.join(DATA_PATH, 'zen.2456865.60537_wrongtelecope.xy.uvcRREAA')

    uvtest.checkWarnings(uv_in.read, [latfile], nwarnings=3,
                         message=['Altitude is not present in file and latitude value does not match',
                                  'This file was written with an old version of pyuvdata',
                                  'This file was written with an old version of pyuvdata'],
                         category=[UserWarning, DeprecationWarning, DeprecationWarning])
    uvtest.checkWarnings(uv_in.read, [lonfile], nwarnings=3,
                         message=['Altitude is not present in file and longitude value does not match',
                                  'This file was written with an old version of pyuvdata',
                                  'This file was written with an old version of pyuvdata'],
                         category=[UserWarning, DeprecationWarning, DeprecationWarning])
    uvtest.checkWarnings(uv_in.read, [telescopefile], {'run_check': False},
                         nwarnings=6,
                         message=['Altitude is not present in Miriad file, and telescope',
                                  'Altitude is not present in Miriad file, and telescope',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna position',
                                  'This file was written with an old version of pyuvdata',
                                  'This file was written with an old version of pyuvdata',
                                  'Telescope foo is not in known_telescopes.'],
                         category=(3 * [UserWarning] + 2 * [DeprecationWarning]
                                   + [UserWarning]))


def test_miriad_location_handling():
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testdir = os.path.join(DATA_PATH, 'test/')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    aipy_uv = aipy_extracts.UV(miriad_file)

    if os.path.exists(testfile):
        shutil.rmtree(testfile)

    # Test for using antenna positions to get telescope position
    uvtest.checkWarnings(uv_in.read, [miriad_file],
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
    aipy_uv2 = aipy_extracts.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': antpos})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read, [testfile], nwarnings=4,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Altitude is not present ',
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
    aipy_uv = aipy_extracts.UV(miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status='new')
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

    uvtest.checkWarnings(uv_out.read, [testfile], nwarnings=5,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna latitude '
                                  'value does not match',
                                  'drift RA, Dec is off from lst, latitude by more than 1.0 deg',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions have a different mean longitude than the file longitude
    # this is harder because of the rotation that's done on the antenna positions
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status='new')
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

    uvtest.checkWarnings(uv_out.read, [testfile], nwarnings=5,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna longitude '
                                  'value does not match',
                                  'drift RA, Dec is off from lst, latitude by more than 1.0 deg',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions have a different mean longitude &
    # latitude than the file longitude
    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use absolute antenna positions, change file latitude and longitude
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': antpos,
                                             'latitud': new_lat, 'longitu': new_lon})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read, [testfile], nwarnings=5,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Telescope location is set at sealevel at the '
                                  'file lat/lon coordinates. Antenna positions '
                                  'are present, but the mean antenna latitude and '
                                  'longitude values do not match',
                                  'drift RA, Dec is off from lst, latitude by more than 1.0 deg',
                                  'Telescope foo is not in known_telescopes.'])

    # Test for handling when antenna positions are far enough apart to make the
    # mean position inside the earth

    good_antpos = np.where(antpos_length > 0)[0]
    rot_ants = good_antpos[:len(good_antpos) // 2]
    rot_antpos = uvutils.rotECEF_from_ECEF(ecef_antpos[rot_ants, :], longitude + np.pi)
    modified_antpos = uvutils.rotECEF_from_ECEF(ecef_antpos, longitude)
    modified_antpos[rot_ants, :] = rot_antpos
    # zero out bad locations (these are checked on read)
    modified_antpos[np.where(antpos_length == 0), :] = [0, 0, 0]
    modified_antpos = modified_antpos.T.flatten() / const.c.to('m/ns').value

    # make new file
    if os.path.exists(testfile):
        shutil.rmtree(testfile)
    aipy_uv = aipy_extracts.UV(miriad_file)
    aipy_uv2 = aipy_extracts.UV(testfile, status='new')
    # initialize headers from old file
    # change telescope name (so the position isn't set from known_telescopes)
    # and use modified absolute antenna positions
    aipy_uv2.init_from_uv(aipy_uv, override={'telescop': 'foo', 'antpos': modified_antpos})
    # copy data from old file
    aipy_uv2.pipe(aipy_uv)
    # close file properly
    del(aipy_uv2)

    uvtest.checkWarnings(uv_out.read, [testfile], nwarnings=4,
                         message=['Altitude is not present in Miriad file, and '
                                  'telescope foo is not in known_telescopes. '
                                  'Telescope location will be set using antenna positions.',
                                  'Altitude is not present ',
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
    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')

    uv_in.select(times=uv_in.time_array[0])
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # check that setting the phase_type works
    uv_out.read(testfile, phase_type='drift')
    assert uv_in == uv_out

    # check again with more than one time but only 1 unflagged time
    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')
    time_gt0_array = np.where(uv_in.time_array > uv_in.time_array[0])[0]
    uv_in.flag_array[time_gt0_array, :, :, :] = True

    # get unflagged blts
    blt_good = np.where(~np.all(uv_in.flag_array, axis=(1, 2, 3)))
    assert np.isclose(np.mean(np.diff(uv_in.time_array[blt_good])), 0.)

    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out

    # check that setting the phase_type works
    uv_out.read(testfile, phase_type='drift')
    assert uv_in == uv_out


def test_poltoind():
    miriad_uv = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    uvtest.checkWarnings(miriad_uv.read, [miriad_file], known_warning='miriad')
    pol_arr = miriad_uv.polarization_array

    miriad = miriad_uv._convert_to_filetype('miriad')
    miriad.polarization_array = None
    pytest.raises(ValueError, miriad._pol_to_ind, pol_arr[0])

    miriad.polarization_array = [pol_arr[0], pol_arr[0]]
    pytest.raises(ValueError, miriad._pol_to_ind, pol_arr[0])


def test_miriad_extra_keywords():
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')

    # check for warnings & errors with extra_keywords that are dicts, lists or arrays
    uv_in.extra_keywords['testdict'] = {'testkey': 23}
    uvtest.checkWarnings(uv_in.check, message=['testdict in extra_keywords is a '
                                               'list, array or dict'])
    pytest.raises(TypeError, uv_in.write_miriad, testfile, clobber=True,
                  run_check=False)
    uv_in.extra_keywords.pop('testdict')

    uv_in.extra_keywords['testlist'] = [12, 14, 90]
    uvtest.checkWarnings(uv_in.check, message=['testlist in extra_keywords is a '
                                               'list, array or dict'])
    pytest.raises(TypeError, uv_in.write_miriad, testfile, clobber=True,
                  run_check=False)
    uv_in.extra_keywords.pop('testlist')

    uv_in.extra_keywords['testarr'] = np.array([12, 14, 90])
    uvtest.checkWarnings(uv_in.check, message=['testarr in extra_keywords is a '
                                               'list, array or dict'])
    pytest.raises(TypeError, uv_in.write_miriad, testfile, clobber=True,
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
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('bool')
    uv_in.extra_keywords.pop('bool2')

    # check handling of int-like keywords
    uv_in.extra_keywords['int1'] = np.int(5)
    uv_in.extra_keywords['int2'] = 7
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('int1')
    uv_in.extra_keywords.pop('int2')

    # check handling of float-like keywords
    uv_in.extra_keywords['float1'] = np.int64(5.3)
    uv_in.extra_keywords['float2'] = 6.9
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out
    uv_in.extra_keywords.pop('float1')
    uv_in.extra_keywords.pop('float2')

    # check handling of very long strings
    long_string = 'this is a very long string ' * 1000
    uv_in.extra_keywords['longstr'] = long_string
    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)
    assert uv_in == uv_out
    uv_in.extra_keywords.pop('longstr')

    # check handling of complex-like keywords
    # currently they are NOT supported
    uv_in.extra_keywords['complex1'] = np.complex64(5.3 + 1.2j)
    uv_in.extra_keywords['complex2'] = 6.9 + 4.6j
    pytest.raises(TypeError, uv_in.write_miriad, testfile, clobber=True)


def test_roundtrip_optional_params():
    uv_in = UVData()
    uv_out = UVData()
    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')

    uv_in.x_orientation = 'east'
    uv_in.reorder_blts()

    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out

    # test with bda as well (single entry in tuple)
    uv_in.reorder_blts(order='bda')

    uv_in.write_miriad(testfile, clobber=True)
    uv_out.read(testfile)

    assert uv_in == uv_out


def test_breakReadMiriad():
    """Test Miriad file checking."""
    uv_in = UVData()
    uv_out = UVData()
    pytest.raises(IOError, uv_in.read, 'foo', file_type='miriad')

    miriad_file = os.path.join(DATA_PATH, 'zen.2456865.60537.xy.uvcRREAA')
    testfile = os.path.join(DATA_PATH, 'test/outtest_miriad.uv')
    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')

    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')
    uv_in.Nblts += 10
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read, [testfile], {'run_check': False},
                         message=['Nblts does not match the number of unique blts in the data'])

    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')
    uv_in.Nbls += 10
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read, [testfile], {'run_check': False},
                         message=['Nbls does not match the number of unique baselines in the data'])

    uvtest.checkWarnings(uv_in.read, [miriad_file],
                         known_warning='miriad')
    uv_in.Ntimes += 10
    uv_in.write_miriad(testfile, clobber=True, run_check=False)
    uvtest.checkWarnings(uv_out.read, [testfile], {'run_check': False},
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
    write_file2 = os.path.join(DATA_PATH, 'test/outtest_miriad2.uv')
    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)

    assert uv_in == uv_out

    # check that we can read & write phased data
    uv_in2 = copy.deepcopy(uv_in)
    uv_in2.phase_to_time(Time(np.mean(uv_in2.time_array), format='jd'))
    uv_in2.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)

    assert uv_in2 == uv_out

    # check that trying to overwrite without clobber raises an error
    pytest.raises(IOError, uv_in.write_miriad, write_file, clobber=False)

    # check that if x_orientation is set, it's read back out properly
    uv_in.x_orientation = 'east'
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that if antenna_diameters is set, it's read back out properly
    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float) + 14.0
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that antenna diameters get written if not exactly float
    uv_in.antenna_diameters = np.zeros((uv_in.Nants_telescope,), dtype=np.float32) + 14.0
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that trying to write a file with unknown phasing raises an error
    uv_in.set_unknown_phase_type()
    pytest.raises(ValueError, uv_in.write_miriad, write_file, clobber=True)

    # check for backwards compatibility with old keyword 'diameter' for antenna diameters
    testfile_diameters = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
    uv_in.read(testfile_diameters)
    uv_in.write_miriad(write_file, clobber=True)
    uv_out.read(write_file)
    assert uv_in == uv_out

    # check that variables 'ischan' and 'nschan' were written to new file
    # need to use aipy, since pyuvdata is not currently capturing these variables
    uv_in.read(write_file)
    uv_aipy = aipy_extracts.UV(write_file)  # on enterprise, this line makes it so you cant delete the file
    nfreqs = uv_in.Nfreqs
    nschan = uv_aipy['nschan']
    ischan = uv_aipy['ischan']
    assert nschan == nfreqs
    assert ischan == 1
    del(uv_aipy)  # close the file so it can be used later

    # check partial IO selections
    full = UVData()
    uvtest.checkWarnings(full.read, [testfile], known_warning='miriad')
    full.write_miriad(write_file, clobber=True)
    uv_in = UVData()

    # test only specified bls were read, and that flipped antpair is loaded too
    uv_in.read(write_file, bls=[(0, 0), (0, 1), (4, 2)])
    assert uv_in.get_antpairs() == [(0, 0), (0, 1), (2, 4)]
    exp_uv = full.select(bls=[(0, 0), (0, 1), (4, 2)], inplace=False)
    assert uv_in == exp_uv

    # test all bls w/ 0 are loaded
    uv_in.read(write_file, antenna_nums=[0])
    diff = set(full.get_antpairs()) - set(uv_in.get_antpairs())
    assert 0 not in np.unique(diff)
    exp_uv = full.select(antenna_nums=[0], inplace=False)
    assert np.max(exp_uv.ant_1_array) == 0
    assert np.max(exp_uv.ant_2_array) == 0
    assert uv_in == exp_uv

    uv_in.read(write_file, antenna_nums=[0], bls=[(2, 4)])
    assert np.array([bl in uv_in.get_antpairs() for bl in [(0, 0), (2, 4)]]).all()
    exp_uv = full.select(antenna_nums=[0], bls=[(2, 4)], inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, bls=[(2, 4, 'xy')])
    assert np.array([bl in uv_in.get_antpairs() for bl in [(2, 4)]]).all()
    exp_uv = full.select(bls=[(2, 4, 'xy')], inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, bls=[(4, 2, 'yx')])
    assert np.array([bl in uv_in.get_antpairs() for bl in [(2, 4)]]).all()
    exp_uv = full.select(bls=[(4, 2, 'yx')], inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, bls=(4, 2, 'yx'))
    assert np.array([bl in uv_in.get_antpairs() for bl in [(2, 4)]]).all()
    exp_uv = full.select(bls=[(4, 2, 'yx')], inplace=False)
    assert uv_in == exp_uv

    # test time loading
    uv_in.read(write_file, time_range=[2456865.607, 2456865.609])
    full_times = np.unique(full.time_array[(full.time_array > 2456865.607) & (full.time_array < 2456865.609)])
    assert np.isclose(np.unique(uv_in.time_array), full_times).all()
    exp_uv = full.select(times=full_times, inplace=False)
    assert uv_in == exp_uv

    # test polarization loading
    uv_in.read(write_file, polarizations=['xy'])
    assert full.polarization_array == uv_in.polarization_array
    exp_uv = full.select(polarizations=['xy'], inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, polarizations=[-7])
    assert full.polarization_array == uv_in.polarization_array
    exp_uv = full.select(polarizations=[-7], inplace=False)
    uv_in == exp_uv

    # test ant_str
    uv_in.read(write_file, ant_str='auto')
    assert np.array([blp[0] == blp[1] for blp in uv_in.get_antpairs()]).all()
    exp_uv = full.select(ant_str='auto', inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, ant_str='cross')
    assert np.array([blp[0] != blp[1] for blp in uv_in.get_antpairs()]).all()
    exp_uv = full.select(ant_str='cross', inplace=False)
    assert uv_in == exp_uv

    uv_in.read(write_file, ant_str='all')
    assert uv_in == full
    pytest.raises(AssertionError, uv_in.read, write_file, ant_str='auto', antenna_nums=[0, 1])

    # assert exceptions
    pytest.raises(ValueError, uv_in.read, write_file, bls='foo')
    pytest.raises(ValueError, uv_in.read, write_file, bls=[[0, 1]])
    pytest.raises(ValueError, uv_in.read, write_file, bls=[('foo', 'bar')])
    pytest.raises(ValueError, uv_in.read, write_file, bls=[('foo', )])
    pytest.raises(ValueError, uv_in.read, write_file, bls=[(1, 2), (2, 3, 'xx')])
    pytest.raises(ValueError, uv_in.read, write_file, bls=[(2, 4, 0)])
    pytest.raises(ValueError, uv_in.read, write_file, bls=[(2, 4, 'xy')], polarizations=['xy'])
    pytest.raises(AssertionError, uv_in.read, write_file, antenna_nums=np.array([(0, 10)]))
    pytest.raises(AssertionError, uv_in.read, write_file, polarizations='xx')
    pytest.raises((AssertionError, ValueError), uv_in.read, write_file, polarizations=[1.0])
    pytest.raises(ValueError, uv_in.read, write_file, polarizations=['yy'])
    pytest.raises(AssertionError, uv_in.read, write_file, time_range='foo')
    pytest.raises(AssertionError, uv_in.read, write_file, time_range=[1, 2, 3])
    pytest.raises(AssertionError, uv_in.read, write_file, time_range=['foo', 'bar'])
    pytest.raises(ValueError, uv_in.read, write_file, time_range=[10.1, 10.2])
    pytest.raises(AssertionError, uv_in.read, write_file, ant_str=0)

    # assert partial-read and select are same
    uv_in.read(write_file, polarizations=[-7], bls=[(4, 4)])
    exp_uv = full.select(polarizations=[-7], bls=[(4, 4)], inplace=False)
    assert uv_in == exp_uv

    # assert partial-read and select are same
    uv_in.read(write_file, bls=[(4, 4, 'xy')])
    exp_uv = full.select(bls=[(4, 4, 'xy')], inplace=False)
    assert uv_in == exp_uv

    # assert partial-read and select are same
    unique_times = np.unique(full.time_array)
    time_range = [2456865.607, 2456865.609]
    times_to_keep = unique_times[((unique_times > 2456865.607)
                                 & (unique_times < 2456865.609))]
    uv_in.read(write_file, antenna_nums=[0], time_range=time_range)
    exp_uv = full.select(antenna_nums=[0], times=times_to_keep, inplace=False)
    assert uv_in == exp_uv

    # assert partial-read and select are same
    uv_in.read(write_file, polarizations=[-7], time_range=time_range)
    exp_uv = full.select(polarizations=[-7], times=times_to_keep, inplace=False)
    assert uv_in == exp_uv

    # check handling for generic read selections unsupported by read_miriad
    uvtest.checkWarnings(uv_in.read, [write_file], {'times': times_to_keep},
                         message=['Warning: a select on read keyword is set'])
    exp_uv = full.select(times=times_to_keep, inplace=False)
    assert uv_in == exp_uv

    # check handling for generic read selections unsupported by read_miriad
    blts_select = np.where(full.time_array == unique_times[0])[0]
    ants_keep = [0, 2, 4]
    uvtest.checkWarnings(uv_in.read, [write_file], {'blt_inds': blts_select, 'antenna_nums': ants_keep},
                         message=['Warning: blt_inds is set along with select on read'])
    exp_uv = full.select(blt_inds=blts_select, antenna_nums=ants_keep, inplace=False)
    assert uv_in != exp_uv

    del(uv_in)
    del(uv_out)
    del(full)

    # try metadata only read
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read, [testfile], {'read_data': False}, known_warning='miriad')
    assert uv_in.time_array is None
    assert uv_in.data_array is None
    assert uv_in.integration_time is None
    metadata = ['antenna_positions', 'antenna_names', 'antenna_positions', 'channel_width',
                'history', 'vis_units', 'telescope_location']
    for m in metadata:
        assert getattr(uv_in, m) is not None

    # test exceptions
    # multiple file read-in
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    new_uv = uv_in.select(freq_chans=np.arange(5), inplace=False)
    new_uv.write_miriad(write_file, clobber=True)
    new_uv = uv_in.select(freq_chans=np.arange(5) + 5, inplace=False)
    new_uv.write_miriad(write_file2, clobber=True)
    pytest.raises(ValueError, uv_in.read, [write_file, write_file2], read_data=False)
    # read-in when data already exists
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    pytest.raises(ValueError, uv_in.read, testfile, read_data=False)

    # test load_telescope_coords w/ blank Miriad
    uv_in = Miriad()
    uv = aipy_extracts.UV(testfile)
    uvtest.checkWarnings(uv_in._load_telescope_coords, [uv], known_warning='miriad')
    assert uv_in.telescope_location_lat_lon_alt is not None
    # test load_antpos w/ blank Miriad
    uv_in = Miriad()
    uv = aipy_extracts.UV(testfile)
    uvtest.checkWarnings(uv_in._load_antpos, [uv], known_warning='miriad')
    assert uv_in.antenna_positions is not None

    # test that changing precision of integraiton_time is okay
    # tolerance of integration_time (1e-3) is larger than floating point type conversions
    uv_in = UVData()
    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    uv_in.integration_time = uv_in.integration_time.astype(np.float32)
    uv_in.write_miriad(write_file, clobber=True)
    new_uv = UVData()
    new_uv.read(write_file)
    assert uv_in == new_uv


@uvtest.skipIf_no_casa
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
    uvtest.checkWarnings(miriad_uv.read, [testfile],
                         message='Telescope EVLA is not')

    assert miriad_uv == ms_uv


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
    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    uv_in.antenna_positions = None
    uvtest.checkWarnings(uv_in.write_miriad, [write_file], {'clobber': True},
                         message=['antenna_positions are not defined.'],
                         category=DeprecationWarning)
    uvtest.checkWarnings(uv_out.read, [write_file], nwarnings=3,
                         message=['Antenna positions are not present in the file.',
                                  'Antenna positions are not present in the file.',
                                  'antenna_positions are not defined.'],
                         category=[UserWarning, UserWarning, DeprecationWarning])

    assert uv_in == uv_out

    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
    ants_with_data = list(set(uv_in.ant_1_array).union(uv_in.ant_2_array))
    ant_ind = np.where(uv_in.antenna_numbers == ants_with_data[0])[0]
    uv_in.antenna_positions[ant_ind, :] = [0, 0, 0]
    uv_in.write_miriad(write_file, clobber=True, no_antnums=True)
    uvtest.checkWarnings(uv_out.read, [write_file], message=['antenna number'])

    assert uv_in == uv_out

    uvtest.checkWarnings(uv_in.read, [testfile], known_warning='miriad')
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
    uvtest.checkWarnings(uv_in.write_miriad, [write_file], {'clobber': True, 'no_antnums': True},
                         message=['antenna_positions are not defined.'],
                         category=DeprecationWarning)
    uvtest.checkWarnings(uv_out.read, [write_file], nwarnings=3,
                         message=['Antenna positions are not present in the file.',
                                  'Antenna positions are not present in the file.',
                                  'antenna_positions are not defined.'],
                         category=[UserWarning, UserWarning, DeprecationWarning])

    assert uv_in == uv_out


def test_multi_files():
    """
    Reading multiple files at once.
    """
    uv_full = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile1 = os.path.join(DATA_PATH, 'test/uv1')
    testfile2 = os.path.join(DATA_PATH, 'test/uv2')
    uvtest.checkWarnings(uv_full.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uvtest.checkWarnings(uv_full.unphase_to_drift, category=DeprecationWarning,
                         message='The xyz array in ENU_from_ECEF is being '
                                 'interpreted as (Npts, 3)')
    uv_full.conjugate_bls('ant1<ant2')

    uv1 = copy.deepcopy(uv_full)
    uv2 = copy.deepcopy(uv_full)
    uv1.select(freq_chans=np.arange(0, 32))
    uv2.select(freq_chans=np.arange(32, 64))
    uv1.write_miriad(testfile1, clobber=True)
    uv2.write_miriad(testfile2, clobber=True)
    uvtest.checkWarnings(uv1.read, [[testfile1, testfile2]], nwarnings=2,
                         message=['Telescope EVLA is not', 'Telescope EVLA is not'])
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full

    # again, setting axis
    uvtest.checkWarnings(uv1.read, [[testfile1, testfile2]], {'axis': 'freq'},
                         nwarnings=2, message=['Telescope EVLA is not'])
    # Check history is correct, before replacing and doing a full object check
    assert uvutils._check_histories(uv_full.history + '  Downselected to '
                                    'specific frequencies using pyuvdata. '
                                    'Combined data along frequency axis using'
                                    ' pyuvdata.', uv1.history)
    uv1.history = uv_full.history
    assert uv1 == uv_full


def test_antpos_units():
    """
    Read uvfits, write miriad. Check written antpos are in ns.
    """
    uv = UVData()
    uvfits_file = os.path.join(DATA_PATH, 'day2_TDEM0003_10s_norx_1src_1spw.uvfits')
    testfile = os.path.join(DATA_PATH, 'test/uv_antpos_units')
    uvtest.checkWarnings(uv.read_uvfits, [uvfits_file], message='Telescope EVLA is not')
    uv.write_miriad(testfile, clobber=True)
    auv = aipy_extracts.UV(testfile)
    aantpos = auv['antpos'].reshape(3, -1).T * const.c.to('m/ns').value
    aantpos = aantpos[uv.antenna_numbers, :]
    aantpos = (uvutils.ECEF_from_rotECEF(aantpos, uv.telescope_location_lat_lon_alt[1])
               - uv.telescope_location)
    assert np.allclose(aantpos, uv.antenna_positions)


def test_readMiriadwriteMiriad_check_time_format():
    """
    test time_array is converted properly from Miriad format
    """
    # test read-in
    fname = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
    uvd = UVData()
    uvd.read(fname)
    uvd_t = uvd.time_array.min()
    uvd_l = uvd.lst_array.min()
    uv = aipy_extracts.UV(fname)
    uv_t = uv['time'] + uv['inttime'] / (24 * 3600.) / 2

    lat, lon, alt = uvd.telescope_location_lat_lon_alt
    t1 = Time(uv['time'], format='jd', location=(lon, lat))
    dt = TimeDelta(uv['inttime'] / 2, format='sec')
    t2 = t1 + dt
    lsts = uvutils.get_lst_for_time(np.array([t1.jd, t2.jd]), lat, lon, alt)
    delta_lst = lsts[1] - lsts[0]
    uv_l = uv['lst'] + delta_lst

    # assert starting time array and lst array are shifted by half integration
    assert np.isclose(uvd_t, uv_t)

    # avoid errors if IERS table is too old (if the iers url is down)
    if iers.conf.auto_max_age is None and six.PY2:
        tolerance = 2e-5
    else:
        tolerance = 1e-8
    assert np.allclose(uvd_l, uv_l, atol=tolerance)
    # test write-out
    fout = os.path.join(DATA_PATH, 'ex_miriad')
    uvd.write_miriad(fout, clobber=True)
    # assert equal to original miriad time
    uv2 = aipy_extracts.UV(fout)
    assert np.isclose(uv['time'], uv2['time'])
    assert np.isclose(uv['lst'], uv2['lst'], atol=tolerance)
    if os.path.exists(fout):
        shutil.rmtree(fout)


def test_file_with_bad_extra_words():
    """Test file with bad extra words is iterated and popped correctly."""
    fname = os.path.join(DATA_PATH, 'test_miriad_changing_extra.uv')
    uv = UVData()
    warn_message = ['Altitude is not present in Miriad file, '
                    'using known location values for PAPER.',
                    'Mean of empty slice.',
                    'invalid value encountered in double_scalars',
                    'npols=4 but found 1 pols in data file',
                    'Mean of empty slice.',
                    'invalid value encountered in double_scalars',
                    'antenna number 0 has visibilities associated with it, '
                    'but it has a position of (0,0,0)',
                    'antenna number 26 has visibilities associated with it, '
                    'but it has a position of (0,0,0)',
                    ]
    warn_category = ([UserWarning] + [RuntimeWarning] * 2
                     + [UserWarning] + [RuntimeWarning] * 2
                     + [UserWarning] * 2)
    # This is an old PAPER file, run_check must be set to false
    # The antenna positions is (0, 0, 0) vector
    uv = uvtest.checkWarnings(uv.read_miriad, func_args=[fname],
                              func_kwargs={'run_check': False},
                              category=warn_category,
                              nwarnings=len(warn_message),
                              message=warn_message
                              )
