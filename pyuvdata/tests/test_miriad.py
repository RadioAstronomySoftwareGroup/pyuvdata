"""Tests for Miriad object."""
import nose.tools as nt
import os
import numpy as np
import ephem
from pyuvdata import UVData
import pyuvdata.tests as uvtest
from pyuvdata.data import DATA_PATH
import copy


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

    uvtest.checkWarnings(uv_in.read_miriad, [latfile],
                         message=['Altitude is not present in file and latitude value does not match'])
    uvtest.checkWarnings(uv_in.read_miriad, [lonfile],
                         message=['Altitude is not present in file and longitude value does not match'])
    uvtest.checkWarnings(uv_in.read_miriad, [telescopefile], {'run_check': False},
                         nwarnings=2, category=[UserWarning, UserWarning],
                         message=['Altitude is not present in Miriad file, and telescope',
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

    # check that trying to write a file with unknown phasing raises an error
    uv_in.set_unknown_phase_type()
    nt.assert_raises(ValueError, uv_in.write_miriad, write_file, clobber=True)

    del(uv_in)
    del(uv_out)


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

    # the objects will not be equal because extra_keywords are not written to
    # or read from miriad files
    nt.assert_false(miriad_uv == ms_uv)
    # they are equal if only required parameters are checked:
    nt.assert_true(miriad_uv.__eq__(ms_uv, check_extra=False))

    # remove the extra_keywords to check that the rest of the objects are equal
    ms_uv.extra_keywords = {}
    nt.assert_equal(miriad_uv, ms_uv)

    nt.assert_equal(ms_uv.history, miriad_uv.history)
    del(miriad_uv)
    del(ms_uv)


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
    uv_out.read_miriad(write_file)

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
    uv_out.read_miriad(write_file)

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
                         category=[UserWarning, UserWarning],
                         message=['Telescope EVLA is not', 'Telescope EVLA is not'])
    # Check history is correct, before replacing and doing a full object check
    nt.assert_equal(uv_full.history + '  Downselected to specific frequencies'
                    ' using pyuvdata. Combined data along frequency axis using'
                    ' pyuvdata.', uv1.history.replace('\n', ''))
    uv1.history = uv_full.history

    # the objects will not be equal because extra_keywords are not written to
    # or read from miriad files
    nt.assert_false(uv1 == uv_full)
    # they are equal if only required parameters are checked:
    nt.assert_true(uv1.__eq__(uv_full, check_extra=False))

    # remove the extra_keywords to check that the rest of the objects are equal
    uv_full.extra_keywords = {}
    nt.assert_equal(uv1, uv_full)
