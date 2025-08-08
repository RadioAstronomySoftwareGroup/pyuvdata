# Copyright (c) 2025 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License
"""Test data setup with Pooch."""

import importlib
import warnings

import pooch
from pooch import Untar

import pyuvdata

# set the data version. This should be updated when the test data are changed.
data_version = "v0.0.4"

pup = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("pyuvdata"),
    # The remote data is on Github
    base_url="https://github.com/RadioAstronomySoftwareGroup/rasg-datasets/raw/{version}",
    version=data_version,
    # If this is a development version, get the data from the "main" branch
    version_dev="main",
    registry=None,
)

# Get registry file from package_data
registry_file = importlib.resources.files(pyuvdata) / "test_data_registry.txt"
# Load this registry file
pup.load_registry(str(registry_file))

fetch_dict = {
    # Visibility data
    "alma_ms": (
        "visibility_data",
        "ALMA",
        "X5707_1spw_1scan_10chan_1time_1bl_noatm.ms.tar.gz",
    ),
    "ata_uvh5_352": (
        "visibility_data",
        "ATA",
        "ata.LoA.C0352.uvh5_60647_62965_9760406_3c286_0001.uvh5",
    ),
    "ata_uvh5_544": (
        "visibility_data",
        "ATA",
        "ata.LoA.C0544.uvh5_60647_62965_9760406_3c286_0001.uvh5",
    ),
    "atca_miriad": ("visibility_data", "ATCA", "atca_miriad.tar.gz"),
    "carma_miriad": ("visibility_data", "CARMA", "carma_miriad.tar.gz"),
    "hera_old_miriad": (
        "visibility_data",
        "HERA",
        "zen.2457698.40355.xx.HH.uvcAA.tar.gz",
    ),
    "hera_uvcalibrate_uvh5": (
        "visibility_data",
        "HERA",
        "zen.2458098.45361.HH_downselected.uvh5",
    ),
    "hera_h2c_uvh5": ("visibility_data", "HERA", "zen.2458432.34569.uvh5"),
    "hera_h3c_uvh5": ("visibility_data", "HERA", "zen.2458661.23480.HH.uvh5"),
    "ovro_lwa": (
        "visibility_data",
        "LWA",
        "2018-03-21-01_26_33_0004384620257280_000000_downselected.ms.tar.gz",
    ),
    "lwasv": (
        "visibility_data",
        "LWA",
        "test_adp4_0_00300673800807520000_58342_05_00_14.ms.tar.gz",
    ),
    "mwa_2013_uvfits": ("visibility_data", "MWA", "1061316296.uvfits"),
    "mwa_2013_nearfield_uvw": ("visibility_data", "MWA", "1061316296_nearfield_w.npy"),
    "mwa_birli_ms": ("visibility_data", "MWA", "1090008640_birli.ms.tar.gz"),
    "mwa_cotter_ms": ("visibility_data", "MWA", "1102865728_small.ms.tar.gz"),
    "mwa_cotter_phase_test1": ("visibility_data", "MWA", "1133866760.uvfits"),
    "mwa_cotter_phase_test2": ("visibility_data", "MWA", "1133866760_rephase.uvfits"),
    "mwa_fhd": ("visibility_data", "MWA", "fhd_vis_data.tar.gz"),
    "mwa_2013_raw_gpubox": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1061315448_20130823175130_mini_gpubox07_01.fits",
    ),
    "mwa_2013_vv": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1061315448_20130823175130_mini_vv_07_01.uvh5",
    ),
    "mwa_2013_metafits": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1061315448.metafits",
    ),
    "mwa_2015_raw_gpubox01": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_20151116182537_mini_gpubox01_00.fits",
    ),
    "mwa_2015_raw_gpubox06": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_20151116182637_mini_gpubox06_01.fits",
    ),
    "mwa_2015_ppds": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_metafits_ppds.fits",
    ),
    "mwa_2015_mwaf01": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_mini_01.mwaf",
    ),
    "mwa_2015_mwaf06": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_mini_06.mwaf",
    ),
    "mwa_2015_cotter_uvfits": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_mini_cotter.uvfits",
    ),
    "mwa_2015_metafits": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552.metafits",
    ),
    "mwa_2015_metafits_mod": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1131733552_mod.metafits",
    ),
    "mwax_2021_raw_gpubox": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1320409688_20211108122750_mini_ch137_000.fits",
    ),
    "mwax_2021_metafits": (
        "visibility_data",
        "MWA",
        "mwa_corr_fits_testfiles/1320409688.metafits",
    ),
    "paper_2012_miriad": ("visibility_data", "PAPER", "new.uvA.tar.gz"),
    "paper_redundant": ("visibility_data", "PAPER", "test_redundant_array.uvfits"),
    "paper_miriad_changing_extra": (
        "visibility_data",
        "PAPER",
        "test_miriad_changing_extra.uv.tar.gz",
    ),
    "paper_2014_miriad": (
        "visibility_data",
        "PAPER",
        "zen.2456865.60537.xy.uvcRREAA.tar.gz",
    ),
    "paper_2014_uvfits": (
        "visibility_data",
        "PAPER",
        "zen.2456865.60537.xy.uvcRREAAM.uvfits",
    ),
    "paper_2014_ms": (
        "visibility_data",
        "PAPER",
        "zen.2456865.60537.xy.uvcRREAAM.ms.tar.gz",
    ),
    "sim_airy_hex": (
        "visibility_data",
        "Simulated",
        "fewant_randsrc_airybeam_Nsrc100_10MHz.uvfits",
    ),
    "sim_uniform_imaging_uvfits": (
        "visibility_data",
        "Simulated",
        "ref_1.1_uniform.uvfits",
    ),
    "sim_uniform_imaging_fhd": ("visibility_data", "Simulated", "refsim1.1_fhd.tar.gz"),
    "sim_bda": ("visibility_data", "Simulated", "simulated_bda_file.uvh5"),
    "sma_mir": ("visibility_data", "SMA", "sma_test.mir.tar.gz"),
    "vla_casa_tutorial_uvfits": (
        "visibility_data",
        "VLA",
        "day2_TDEM0003_10s_norx_1src_1spw.uvfits",
    ),
    "vla_casa_tutorial_ms": (
        "visibility_data",
        "VLA",
        "day2_TDEM0003_10s_norx_1src_1spw.ms.tar.gz",
    ),
    "vlba_mojave_uvfits": ("visibility_data", "VLBA", "mojave.uvfits"),
    # Calibration solutions
    "hera_omnical1": (
        "calibration_solutions",
        "HERA",
        "zen.2457555.42443.HH.uvcA.omni.calfits",
    ),
    "hera_firstcal_delay": (
        "calibration_solutions",
        "HERA",
        "zen.2457698.40355.xx.delay.calfits",
    ),
    "hera_omnical2": (
        "calibration_solutions",
        "HERA",
        "zen.2457698.40355.xx.gain.calfits",
    ),
    "hera_uvcalibrate_calfits": (
        "calibration_solutions",
        "HERA",
        "zen.2458098.45361.HH.omni_downselected.calfits",
    ),
    "mwa_fhd_cal": ("calibration_solutions", "MWA", "fhd_cal_data.tar.gz"),
    "sma_amp_gcal": ("calibration_solutions", "SMA", "sma.ms.amp.gcal.tar.gz"),
    "sma_bcal": ("calibration_solutions", "SMA", "sma.ms.bcal.tar.gz"),
    "sma_dcal": ("calibration_solutions", "SMA", "sma.ms.dcal.tar.gz"),
    "sma_dterms_pcal": ("calibration_solutions", "SMA", "sma.ms.dterms.pcal.tar.gz"),
    "sma_pha_gcal": ("calibration_solutions", "SMA", "sma.ms.pha.gcal.tar.gz"),
    "sma_tcal": ("calibration_solutions", "SMA", "sma.ms.tcal.tar.gz"),
    # Primary Beams
    "hera_fagnoni_vivaldi_150": (
        "primary_beams",
        "HERA",
        "NicCSTbeams/farfield_150MHz.txt",
    ),
    "hera_fagnoni_dipole_123": (
        "primary_beams",
        "HERA",
        "NicCSTbeams/HERA_NicCST_123MHz.txt",
    ),
    "hera_fagnoni_dipole_150": (
        "primary_beams",
        "HERA",
        "NicCSTbeams/HERA_NicCST_150MHz.txt",
    ),
    "hera_fagnoni_vivaldi_yaml": (
        "primary_beams",
        "HERA",
        "NicCSTbeams/HERA_Vivaldi_CST_beams.yaml",
    ),
    "hera_fagnoni_dipole_yaml": (
        "primary_beams",
        "HERA",
        "NicCSTbeams/NicCSTbeams.yaml",
    ),
    "hera_casa_beam": ("primary_beams", "HERA", "HERABEAM.FITS"),
    "ovro_lwa_feko_x": ("primary_beams", "LWA", "OVRO_LWA_x.ffe"),
    "ovro_lwa_feko_y": ("primary_beams", "LWA", "OVRO_LWA_y.ffe"),
    "mwa_full_EE": ("primary_beams", "MWA", "mwa_full_EE_test.h5"),
    # Flagsets
    "hera_flags": ("flagsets", "HERA", "zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5"),
}


def registry_name(data_type, telescope, filename):
    """Construct the registry name for a file."""
    return "/".join([data_type, telescope, filename])


def fetch_data(name: str | list):
    """
    Fetch a dataset by key in the fetch_dict.

    Parameters
    ----------
    name : str or list of str
        Name of the dataset (key in fetch_dict).

    Returns
    -------
    filename : str or list of str
        Location of the file(s) or folder on the local computer.
    """
    input_list = True
    if isinstance(name, str):
        input_list = False
        name = [name]
    fname = []
    for dname in name:
        data_type, telescope, filename = fetch_dict[dname]
        # The data will be downloaded automatically the first time this is run for it.
        # Afterwards, Pooch finds it in the local cache and doesn't repeat the download.
        reg_name = registry_name(data_type, telescope, filename)
        if ".tar" in filename:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Python 3.14 will, by default, filter extracted "
                    "tar archives",
                )
                pup.fetch(reg_name, processor=Untar(extract_dir="."))
            folder_name = str(pup.abspath / reg_name.split(".tar")[0])
            fname.append(folder_name)
        else:
            fname.append(pup.fetch(reg_name))
    if len(fname) == 1 and not input_list:
        return fname[0]
    return fname


def fetch_oldproj_uvw(antpos: bool, frame: str):
    """Fetch numpy files with uvws from old projection code."""
    if antpos:
        uvw_path = f"oldproj_antpos_{frame}_uvw.npy"
    else:
        uvw_path = f"oldproj_{frame}_uvw.npy"

    reg_name = registry_name("visibility_data", "HERA", uvw_path)
    fname = pup.fetch(reg_name)
    return fname
