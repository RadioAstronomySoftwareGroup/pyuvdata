# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [2.4.1] - 2023-10-13

### Added
- Added a `uvw_track_generator` method within `utils` for calculating the expected
uvws (and a few other values) without needing to instantiate a whole `UVData` object.
- Added a convenience function called `compare_value` in `UVParameter` that enables
one to do value checking with tolerances accounted for.
- New `mwa_metafits_file` and `telescope_name` optional parameters to `UVFlag.read` and
`UVFlag.__init__` to help with setting telescope metadata for old UVFlag files that are
missing it.
- MWA antenna metadata to our known telescopes to allow them to be filled in for old
UVFlag files. This is a stopgap solution, a better approach is to pass an MWA metafits
file to the new `mwa_metafits_file` parameter.
- Support for recarrays in `UVParameter` objects and in `UVBase`, needed for pyradiosky.
- Support for setting the astrometry library for various object methods including `set_lsts_from_time_array`, file read methods and others.
- Properly round-trip the telescope frame through UVH5, UVFITS and MS files.

### Fixed
- A bug in apparent coordinate calculation that resulted in small errors/loss of
precision due to the way times were passed when using `erfa` astrometry library.
- A small correction due to polar drift for LST calculation when using the `erfa`
astrometry library.
- Fixed a bug in `utils.calc_app_coords` that occurred when supplying an astropy `Time`
object for the `time_array` argument.
- Fixed a bug in `UVData.write_ms` that caused the writer to crash if prior measurement
set history was not formatted in the currently expected fashion.

## [2.4] - 2023-07-12

### Added
- Support for multiple spectral windows in `UVData.frequency_average`, including a new
parameter `respect_spws` for controlling whether averaging crosses spectral window
boundaries.
- Better handling in `UVData.frequency_average` when averaging by a number of channels
that does not divide evenly into the number of channels in each spectral window.
- Compatibility with Python 3.11

### Changed
- The following `UVFlag` parameters are now required: `Nspws`, `channel_width`,
`spw_array`, `telescope_name`, `telescope_location`, `antenna_names`, `antenna_numbers`,
and `antenna_positions`.
- The `quality_array` on UVCal objects is no longer required.

### Deprecated
- The `sky_field` attribute on `UVCal`.

### Fixed
- A bug in LST calculation that led to small discontinuities in LSTs calculated using
the `erfa` or `novas` astrometry libraries.
- Error when setting `UVBeam.freq_interp_kind` to an integer.
- Error when reading `mwa_corr_fits` files from the new MWAX correlator

### Removed
- The `phase_uvw` and `unphase_uvw` utility methods associated with old style phasing.
- The `with_conjugates` option to the `get_baseline_redundancies` utility method in
favor of the `include_conjugates` option.
- Support for the `interpolation_function` attribute on UVBeams.
- Support for the "unphased" catalog type in UVData.phase_center_catalog in favor of
the "unprojected" catalog type.
- The `UVData.unphase_to_drift` method in favor of `UVData.unproject_phase` method.
- Support for using old style phasing on UVData objects (the `UVData.fix_phase` method
to fix datasets that were phased with the old style of phasing remains as well as the
`fix_old_proj` option on read for uvfits, miriad and uvh5 file types).
- Phasing related keywords in the following `UVData` methods: `__add__`, `__iadd__`,
`fast_concat`, `read` and `from_file`. Multiple phase centers are now fully supported
so datasets can always be combined without phasing. If desired, the full object can be
rephased to a single phase center afterwards.
- The `use_cotter_flags` and `flag_small_sig_ants` options to the `UVData.read` and
`UVData.from_file` methods for MWA correlator fits files in favor of the
`use_aoflagger_flags` and `flag_small_auto_ants` options respectively.
- The `spoof_nonessential` option to `UVData.write_uvfits` method as the previously
spoofed parameters are now properly calculated.

## [2.3.3] - 2023-05-25

### Added
- New dependency `docstring-parser` to enable dynamic docstring generation for
methods that mostly wrap other methods.
- A new `UVCal.new()` method (based on new function `new_uvcal`) that creates a new,
self-consistent `UVCal` object from scratch from a set of flexible input parameters.
- A new `UVData.new()` method (based on new function `new_uvdata`) that creates a new,
self-consistent `UVData` object from scratch from a set of flexible input parameters.
- A new `fast_concat` method on `UVCal`.
- A new generic `read` method on `UVCal` that, like the `read` methods on our other
objects, supports all file types and a new `from_file` class method to allow one-line
reading of UVCal files.

### Deprecated
- The `input_flag_array` attribute on `UVCal`.
- Support for the 'unknown' cal_type in UVCal.
- Reading in multiple files to `UVCal` using file-type specific read methods
(e.g. `read_calfits`) in favor of the generic `read` method.

### Fixed
- Removed error when `time_axis_faster_than_bls=True` and `Ntimes=1`. In the case of
`Ntimes=1`, it is inconsequential whether time-axis moves first or not, so it should
not be an error.

## [2.3.2] - 2023-04-10

### Added
- The `catalog_name` keyword has been added to the `UVData.read` and `UVData.select`
methods to allow users to select on source name.

### Changed
- The keywords for `UVData.read` have been changed for MIR-specific values to more
generally match those founds with other data types.
- The `UVData.read_mir` method has been overhauled to mak reading in of MIR data more
efficiency in memory/processor usage.

### Fixed
- Frequency frame information for MS datasets is now correctly recorded as "TOPO".

## [2.3.1] - 2023-04-03

### Added
- A new `FastUVH5Meta` object that enables quick partial reading of uvh5 file metadata.
This is used by default in the `UVData.read_uvh5` method to initialize a `UVData`
object, without any apparent change for the user. To get the benefits of the new class,
use it directly to interface with the metadata portion of a `.uvh5` file.

### Changed
- Updated numpy requirements to `>=1.20`.
- Updated scipy requirements to `>=1.5`.

### Fixed
- Handling of antenna_names and antenna_numbers in `read_fhd` and `read_fhd_cal`.

## [2.3.0] - 2023-03-13

### Added
- A new `interpolation_function` parameter to `UVBeam.interp` and `UVBeam.to_healpix`
to allow the function name to be passed into the methods, with sensible defaulting.
- Support for selecting on phase center IDs, including on read.
- Several new attributes to UVFlag: `telescope_name`, `telescope_location`,
`antenna_names`, `antenna_numbers`, `antenna_positions`, `channel_width`, `spw_array`,
`flex_spw_id_array`, enabling support for flexible spectral windows. All of these except
`flex_spw_id_array` will eventually be required for all UVFlag object types, as will
`Nspws` (which used to only be required for baseline and antenna types). These
parameters all already exist on UVData and UVCal objects and UVFlag objects initialized
from them will inherit them. UVFlag objects will also inherit these parameters when
they are converted between types using UVData and UVCalobjects.
- The `UVFlag.set_telescope_params` method, similar to the ones on UVData and UVCal,
to set several of these new parameters.

### Changed
- Increases the h5py minimum version to 3.1.
- Increase the minimum compatible version for lunarsky to 0.2.1 to fix astropy
deprecation warnings.

### Fixed
- Fixed some bugs related to handling of the `UVCal.time_range` parameter in
`UVCal.__add__` and other methods. Ensured that it is a list when
`UVCal.future_array_shapes` is False.
- Fixed a bug in `UVCal.__add__` when data are sorted differently along any axis or
interleaved along an axis.
- Fixed a bug in reading MS files with non-UTC time scales.
- Fixed some bugs in UVData, UVFlag and UVCal `__add__` and `select` methods for objects
with only one spectral window and `flex_spw_id_array` defined. Also fixed some handling
in those methods on UVFlag for objects with multiple spectral windows.
- Fix a bug when reading FHD files into UVData objects when the layout file was not
passed but the antenna_positions are in the known telescopes.
- Fixed a bug where the beamfits partial read parameters were not available in the
`UVBeam.from_file` class method and were not passed properly when reading multiple files.
- Fixed a bug where objects created from scratch with the old phase attributes weren't
properly getting converted to the new `phase_center_catalog` representation. That
conversion now happens in the check so will happen before any `write`. There could still
be errors if some methods are called before a check is done (or if the check is turned
off).
- Fix a bug in how `frame_pa` was calculated in `phase` and `_set_app_coords_helper` for
multi_phase_center objects (it was using the old `phase_center_frame` attribute).
- Fix a bug where trying to select lsts or lst_ranges on read didn't work for some file
types.
- Severe performance hit when calling `polnum2str` and its variants for many baselines.

### Deprecated
- Reading files into objects without setting `use_future_array_shapes` now results in
deprecation warnings.
- The utility functions `phase_uvw` and `unphase_uvw` associated with the deprecated
old phasing method.
- The `flex_spw_id_array` will be required on all UVData and UVFlag and all
non-wide-band UVCal objects in version 3.0.
- Deprecated the `interpolation_function` attribute on UVBeams.
- Deprecated the older phase attributes (`phase_type`, `phase_center_ra`,
`phase_center_dec`, `phase_center_frame`, `phase_center_epoch`, `object_name`) in favor
of the `phase_center_catalog`. The older phase attributes will be removed in version 3.0.
- The `lst_from_uv` function in the uvflag module.

### Removed
- Removed deprecated handling for UVCal objects without `telescope_location`,
`antenna_positions` and `lst_array` parameters defined.

## [2.2.12] - 2022-12-07

### Changed
- Updated how wheels for PyPI are built.

## [2.2.11] - 2022-11-30

### Added
- A frame attribute to the telescope_location parameters on UVData, UVCal and UVFlag
objects to support observatories on the moon (or anywhere not on earth).

### Fixed
- A bug in `UVData.read_uvh5` where `multi_phase_center` was being used to imply the
existence of the `phase_center_catalog` header item, rather than checking for the
presence of that item. This conflicted with the uvh5 memo.

## [2.2.10] - 2022-10-20

### Added
- A new subclass of UVParameter for SkyCoord objects, needed in pyradiosky.
- Added `UVData.convert_to_flex_pol` method to enable writing uvh5 files with the
polarization axis changing slowest as desired by HERA. Also updated
`UVData.remove_flex_pol` to properly undo the `UVData.convert_to_flex_pol` operation.
- Added an option to `UVData.read` and `UVData.from_file` methods when reading uvh5
type files to call `remove_flex_pol` as part of the read. This is defaulted to `True`
so that when HERA data is written with flex_pol the change will not disrupt users, but
other users of uvh5 may want to set it to `False`.
- Added support to make bitshuffle compression easier when writing UVData objects to
uvh5 files
- Support for partial reads on beamfits files. This can be done on frequency, azimuth
or zenith angle (if beam is in az/za coordinates).
- Logging statements (INFO level) in `UVData.check`.

### Changed
- Significantly increased uvh5 reading and writing speed.
- Added `use_future_array_shapes` method to UVBeam to allow users to convert to using
the future array shapes now, with support throughout UVBeam methods and related utility
functions. Also added `use_current_array_shapes` to revert to the standard shapes.
- Major performance improvement in the `UVBeam.to_healpix` method.
- Performance improvement when doing a `UVData.select` using the `bls` parameter (~70%
improvement in the `_select_preprocess` function)

### Fixed
- A bug in `UVData.check` when `check_autos` is True and the object is a flex pol object.

## [2.2.9] - 2022-8-23

### Added
- Added `use_future_array_shapes` method to UVFlag to allow users to convert to using the
future array shapes now, with support throughout UVFlag methods and related utility
functions. Also added a `use_future_array_shapes` parameter to UVFlag's `__init__`,
`read`, `from_uvdata` and `from_uvcal` methods to allow the conversion to be done at the
time the object is made. Added `use_current_array_shapes` to revert to the standard shapes.
- Baseline number calculations for antenna numbers > 2047.
- New uvh5 version 1.1 definition in memo and code to use the future phasing info
(i.e. the phase_center_catalog) rather than the older parameters.
- Added the `normalize_by_autos` method to `UVData`, which allows one to normalize
cross-correlations (i.e., covert values from arbitrary scale to correlation coefficients)
if auto-correlations are present in the data.
- Added the `write` method to `MirParser`, which allows for `MirParser` to write out
Mir-formatted data to disk.
- Added the `select` method to `MirParser`, which provides a much simpler interface for
selecting subsets of data to load.
- Added the `rechunk` method to `MirParser`, which allows data to be spectrally averaged
(either after already loading or to be averaged "on-the-fly" when data are read from
disk).
- Added the `redoppler_data` method to `MirParser`, which enables frequency shifting of
the data.
- Added new `__add__` and `__iadd__` methods to `MirParser` for combining data sets.
- Added new functionality to `UVBeam.check` to verify that power beams for the auto
polarizations (and pstokes) are real-only, along with an option to force them to be
real-only if non-zero imaginary components are detected.
- Coarse band correction for mwax to `mwa_corr_fits`
- Reordering methods for UVCal: `reorder_antennas`, `reorder_freqs`, `reorder_times`
and `reorder_jones`.

### Changed
- Always write out uvh5 version 1.0 or greater files (which use the future array shapes) with `UVData.write_uvh5`.
- Use "unprojected" rather than "unphased" for the cat_type in the phase_center_catalog
to describe zenith driftscan data that have no w-projections applied.
- The `MirParser` class has been significantly overhauled, which significantly reduces
memory usage and improves processing speed.
- Updated minimum dependency versions: numpy>=1.19, scipy>=1.3, optional dependencies:
python-casacore>=3.3, pyyaml>=5.1, astropy-healpix>=0.6
- `UVBase` object now require that individual attribute names match that given in
`UVParameter._name`.
- `UVData.fix_phase` now raises a warning when called.
- Changed `UVData.write_uvfits` to allow for one to write out datasets in UVFITS format
without "spoofing" (via setting `spoof_nonessential=True`) UVFITS-specific values.
- Methods that ensure that attributes are set up properly on `UVBeam` are now called in
the `UVBeam.check` method, making it easier to fill in UVBeam objects from scratch.
- Changed uvfits reading and writing to no longer subtract/add 1 to the antenna
numbers. The subtraction/addition of 1 was initially done because FITS uses one-based
indexing but antenna numbers are physical numbers not indices, so the numbers should not
be adjusted.
- Changed the way baseline numbers are calculated from antenna numbers to conform to the
standard in uvfits and miriad. The inconsistency with those files was noticed when we
corrected the handling of antenna numbers in uvfits.
- The `spw_order` parameter for the `UVData.reorder_freqs` method now accepts an index
array rather than an array of spw numbers, making it match the other reorder methods.
- Updated the astropy requirement to >= 5.0.4
- Dropped support for python 3.7

### Fixed
- Store the phase_center_catalog as a set of nested datasets in uvh5 files rather than
as a json blob.
- A bug in `UVData.__add__` where flexible spectral window datasets were not combined
correctly if they contained overlapping frequencies in different spectral windows.
- A bug in `UVData.print_phase_center_info` that occasionally resulted in incorrect
values being reported for RA/Az/Longitudinal coordinates.
- A bug in `UVData.select` that could cause `UVData.check` to fail if
`UVData._scan_number_array` was set.
- A bug in `UVBeam.select` where after selecting down to only auto polarization power
beams the `UVBeam.data_array` remained complex instead of real.
- A bug in `UVBeam.__add__` where adding an object with cross pol power beams to an
object with only auto pol power beams could result in loss of the imaginary part of the
cross pol power beams.
- Testing of piping keywords through `read` by `test_mwa_corr_fits`.
- Incorrect piping of `flag_dc_offset` keyword.
- `mwa_corr_fits` handing of the `BSCALE` keyword in gpubox files.

### Deprecated
- The "unphased" cat_type in phase_center_catalog.

## [2.2.8] - 2022-2-15

### Added
- Parameter `check_azza_domain` in `UVBeam.interp()`, which can turn off the (relatively
  expensive) check on input az/za values.

## [2.2.7] - 2022-02-09

### Added
- The `UVCal.initialize_from_uvdata` method to initialize a UVCal object using metadata
from a UVData object.
- An option to `UVData.reorder_blts` to sort the autos before the crosses.
- Added new functionality to `UVData.check` to verify that auto-correlations are real-only,
along with an option to force them to be real-only if non-zero imaginary components are detected.

### Changed
- `UVFlag.to_baseline()` uses internal time tolerances (both rtol and atol) stored in the UVParameter
  `_time_array.tols` to check whether times in the UVFlag object and the input UVFlag/UVData object match.

### Deprecated
- The `with_conjugates` keyword in favor of the `include_conjugates` keyword in the
`utils.get_baseline_redundancies` function.

## [2.2.6] - 2022-01-12

### Added
- A new attribute to `UVData` called `flex_spw_polarization_array`, of type=int and shape=(`Nspws`,),
  which allows for individual spectral windows to carry different polarization data.
- Added the `_make_flex_pol` and `remove_flex_pol` methods to `UVData`, which allows for
  one to convert a standard `UVData` object to one with "flexible-polarization".
- Added a new method to `MirParser` called `_apply_tsys`, which will convert MIR visibility data
  from correlation coefficients to pseudo-Jy.
- Added a method to `Mir` called `_init_from_mir_parser`, which allows for one to pass
  a `MirParser` object to be converted into a UVData object (rather than reading from
  a file on disk).
- Added the `flex_spw` attribute to the `UVCal` class, which can be set to True by using
the new `_set_flex_spw` method.
- Added the optional `flex_spw_id_array` attribute to UVCal class, of type=int and
shape=(`Nfreqs`,), which indexes individual channels along the frequency axis to `spw_array`.
- Added `use_future_array_shapes` method to UVCal to allow users to convert to using the
future array shapes now, with support throughout UVCal methods and related utility
functions. Also added `use_current_array_shapes` to revert to the standard shapes.
- Added support for wide-band gain calibrations via the `wide_band` attribute on the
`UVCal` class, which can be set using the new `_set_wide_band` method.
- Added `time_range`, `lsts`, and `lst_range` kwargs from UVH5.write_uvh5_part() to UVData.write_uvh5_part().
- Added `time_range`, `lsts`, and `lst_range` kwargs from UVH5.write_uvh5_part() to UVData.write_uvh5_part().

## Changed
- General performance improvements in the `read_mir` method.
- Enabled `read_mir` to read in dual- and full-polarization data.

### Fixed
- Fixed a bug where `UVData.compress_by_redundancy` sometimes produced incorrectly
  conjugated visibilities.
- Fixed a bug where `lsts` and `lst_range` got ignored when doing partial i/o with multiple files.

## [2.2.5] - 2021-12-21

### Added
- Support for reading MWAX correlator fits files.
- Added a warning when using `UVData.write_uvfits` when the `vis_units` attribute is
anything other than `Jy`.
- Added a warning when using `UVData.write_uvfits` if a UVData object has > 256 antennas,
recommending use of `UVData.write_ms` if intending to import the data into CASA.
- Reading writing of scan numbers for MS files as `UVData.scan_number_array`.
- Grouping of contiguous integrations for a phase center into "scan numbers" in `UVData._set_scan_numbers`.
This grouping defines `UVData.scan_number_array` when not originally present in the data (e.g. reading in
a non-MS file) and is used when writing to an MS file.
- Added a `filename` attribute to UVCal, UVBeam and UVFlag objects.
- Support for reading MWAX/birli uvfits files.
- Flexible spectral windows to `read_mwa_corr_fits`.

### Changed
- Changed the `UVData.write_ms` to reduce processing time and memory footprint.
- Changed the order in which data are written in Measurement Set format via the method
  `UVData.write_ms`, where data are now first grouped together by `scan_number_array`.
- Changed the defaults by which the `UVData.antenna_names` attribute is set when reading
measurement sets (used to default to station names, now uses antenna names if available
and the MS file was not written using the CASA `importuvfits` routine).
- Fixed a bug where `UVData.x_orientation` and `UVData.vis_units` were not being written
 to / read from measurement sets.
- Updated the astroquery requirement to >= 0.4.4, due to changes in the API handling for
calls to JPL Horizons.
- Assumes uvfits files are in ITRF frame unless explicitly stated otherwise. Consistent with AIPS 117.
- Improved readability, functionality, and memory usage in `read_mwa_corr_fits`.

### Fixed
- A bug in writing calfits files when the optional `time_range` parameter is not set.
- A bug where the `TIME_CENTROID` field was not being filled in data sets written by
  `UVData.write_ms`, which caused odd behavior in some CASA routines (e.g., `gaincal`).
- A bug in reading in uvfits files with baseline coordinates that have suffixes of
'---SIN' or '---NCP' which are allowed in uvfits files.
- A bug that could cause some routines in CASA to fail when using data sets written by
  `UVData.write_ms`.
- A bug that could have resulted in `UVData.__add__` combining objects together incorrectly
  when containing overlapping time-baselines/polarization/frequency channels together
  that were ordered differently for the two `UVData` objects.
- A bug that prevented `extra_keywords` keys with a value of `None` from being
  saved to UVH5 files.
- A bug that resulted in the wrong expected shapes for data-like arrays when metadata
only UVData objects were set to use future array shapes.
- A bug that caused an error when writing non-double antenna diameters to measurement set files
- Added a warning in `utils.uvcalibrate` when uvdata x_orientation is not set.
- Fixed a bug in `UVBeam.efield_to_power` when there is only one feed.

## [2.2.4] - 2021-10-14

### Fixed
- Fixed a bug in converting non-crosspol power beams to real in `UVBeam.efield_to_power`.

## [2.2.3] - 2021-10-13

### Fixed
- A bug that resulted in coordinate queries via for solar-system objects `utils.lookup_jplhorizons` to throw an error. The workaround temporarily requires the precision for ephemerides to be limited to 0.05 arcseconds.
- A bug that resulted in a potential error when attempting to write out multi-phase-center data sets into the UVH5 file format.


## [2.2.2] - 2021-9-30

### Added
- Added the `UVData.write_ms` method, a UVData measurement set writer.
- Added an option to the `UVData.check` method to flip the uvws and conjugate the
data_array to align with the `UVData` convention on baseline orientation.
- Added the `flip_gain_conj` option to `utils.uvcalibrate` to use the opposite gain
conjugation convention in calibration.
- Support for selecting on polarization strings on `UVData`, `UVCal`, `UVBeam` and `UVFlag`.
- `from_file` class function on UVData and UVBeam. Allows users to instantiate an object from a file directly from each Class.
- Generic 'read' function for UVBeam objects.
- UVData methods `set_data`, `set_flags`, and `set_nsamples` which allow the user to
update values in the corresponding arrays.
- pyERFA was added as an explicit dependency (previously it was implicit because it is
an astropy dependency).

### Fixed
- A bug in the way times were written to uvfits files that could cause a loss of precision.
- A bug that allowed the check to pass on `UVData`, `UVCal` and `UVFlag` objects if the
`flag_array` contained integers instead of booleans.
- A bug in equality testing for `UVParameter` objects with strict types when the
parameters contain arrays.
- Provide a more useful error message if `UVCal.read_fhd_cal` is called with
`read_data=False` and a settings file is not provided.
- Fixed a bug in `UVCal.read_fhd_cal` where the reader crashed on a metadata only read
if the settings file had an empty field for the diffuse model.
- Fixed a bug in `UVBeam.efield_to_power` where a the `data_array` remained complex
rather than real because the tolerance was too low in the numpy `real_if_close` call.

## [2.2.1] - 2021-7-21

### Changed
- The `__eq__` method on UVBase objects (and subclasses) now supports an
`allowed_failures` keyword for parameters which are allowed to be unequal
without failing the check. This includes the `filename` parameter of
`UVData` objects by default.


## [2.2.0] - 2021-6-26

### Added
- Added `filename` attribute for UVData objects.
- Added support for multiple sources/phase centers in a single UVData object, which can
be enabled using the `_set_multi_phase_center` method. Using this method will set the
new attribute `multi_phase_center` to `True` (otherwise set to `False`.).
- Added the optional attribute `Nphase`, of type `int` and required if
`multi_phase_center=True`, which records the number of different phase centers recorded
in the UVData object.
- Added the optional attribute `phase_center_catalog`, a dictionary of dictionaries
that is required if `multi_phase_center=True`, which records various meta-information
about the different phase centers used within the UVData object. The keys of individual
entries of phase_center_catalog are strings representing the names of the individual
phase centers. These keys are matched to individual dictionaries which store various
information about the phase centers, including coordinate information (e.g., `cat_lat`,
`cat_lon`, `cat_frame`, `cat_epoch`) and a unique identification number (`cat_id`, type
`int`).  Catalog information can be printed to the terminal using the method
`print_phase_center_info`.
- Added the optional attribute `phase_center_id_array`, of type `int` and shape
`(Nblts,)` required if `multi_phase_center=True`, which records which phase center the
individual baseline-time record is phased to. The individual values of are matched to
`cat_id` from the individual dictionary entries of `phase_center_catalog`.
- Added the attributes `phase_center_app_ra` and `phase_center_app_dec`, which records
he topocentric apparent positions of the phase center as seen from the position
`telescope_location`. Both are of type `float`, shape `(Nblts,)`.
- Added the attribute `phase_center_frame_pa`, of type `float`, shape `(Nblts,)`, which
encodes the position angle between the topocentric apparent frame and the celestial
frame of phase center coordinates, as recorded in `phase_center_frame` (e.g., FK5,
ICRS). UV-coordinates are rotated by `phase_center_frame_pa`, such that
- Added a switch called `ignore_name` to `__add__` which allows one to ignore whether
the `object_name` attribute matches when attempting to combine objects  (default is
`ignore_name=False`, which will cause an error to be thrown if `object_name` does not
match). This parameter is also available with the `read` method, when reading in
multiple files.
- Added a switch called `make_multi_phase` to the `__add__` method, which will convert
the data to a multi-phase-center-enabled dataset, allowing for UVData objects with
different phase centers to be combined. This parameter is also available with the `read`
method, when reading in multiple files.
- Added the `fix_phase` method, which converts UVData objects from the "old" phasing
framework to the "new' phasing framework.
- Added support for looking up ephemeris information of solar system objects from
JPL-Horizons via the optionally required `astroquery` packaged.
- Added option to select based on LST or an LST range in UVData objects.
- Added `get_lsts` method on UVData objects for retrieving LST corresponding to data.

### Changed
- `utils.uvcalibrate` will error rather than warn if some of the newly added checks do
not pass, including if antenna names do not match between the objects.
- Phasing methods (e.g., `UVData.phase`, `UVData.unphase_to_drift`,
`UVData.set_uvws_from_antenna_positions`) have undergone a significant revision, which
change the way in UVW coordinates are calculated. Specifically, these methods will now
use the topocentric apparent positions (stored in the attributes `phase_center_app_ra`,
`phase_center_app_dec`). As a result, phasing of data and calculating of uvw-coordinates
are significantly faster.
- `phase` method now supports phase centers of different types, specified by the
`cat_type` parameter. Supported types include "sidereal" (default; fixed position in
RA/Dec), "ephem" (position in RA/Dec which moves with time), and "driftscan" (fixed
position in Az/El, NOT the same as `phase_type`=“drift”). The "ephem" and "driftscan"
types can only be used when `multi_phase_center=True`.
- `phase` now uses antenna positions for deriving uvw-coordinates by default.
- Decreased the standard tolerances for angle-based parameters to 1 mas (was 10 mas).
- Updated the astropy requirement to >= 4.2.1
- Unknown phasing types (i.e., `phase_type="unknown"`) are no longer supported, `read`
(and the associated file-reading methods) will default `phase_type` to "drift" when
unable to definitively ascertain the phasing type.

### Fixed
- A bug in `aipy_extracts.pipe` that caused an error when using `raw=True`.
- A bug in `aipy_extracts.write` which caused an error when using certain masked arrays.
- Phasing accuracy/UVW coordinate calculation no longer limited to 2 cm on a 3 km
baseline (i.e.,~1 part in 1e5; new accuracy is better than 1 part in 1e8).
- A bug in reading DSB data from Mir filetypes, that resulted in visibilities being
matched to the wrong baselines.
- A bug in `UVData.phase_to_time` which prevented users from rephasing phased objects.
- A bug in `UVBeam.read_cst_beam` when specifying beams with a single file using a yaml.

### Removed
- The UVBeam methods `set_cs_params`, `set_efield`, `set_power`, `set_simple`, and
`set_phased_array` have been removed.
- The UVCal methods `set_gain`, `set_delay`, `set_unknown_cal_type`, `set_sky`, and
`set_redundant` have been removed.
- The UVData methods `set_phased` and `set_drift` have been removed.
- The functions `utils._str_to_bytes` and `utils._bytes_to_str` have been removed.
- The `flag_missing` keyword of the `utils.uvcalibrate` function has been removed.


## [2.1.5] - 2021-4-02

### Added
- Added `use_future_array_shapes` method to allow users to convert to using the future
array shapes now, with support throughout UVData methods and related utility functions.
Also added `use_current_array_shapes` to revert to the standard shapes.
- Added versioning to UVH5 files. Files with future array shapes are version 1.0,
files with the current shapes are are version 0.1.
- Added `UVData.reorder_freqs` method to reorder the data along the frequency axis.
- Better re-initialization of UVParameters on UVBase objects when using pickle. Improves compatibility of UVBase objects with MPI.
- Added option to apply a Van Vleck correction to mwa_corr_fits files.

### Fixed
- A bug in mwa_corr_fits that resulted in an error when using `memap=true`
while reading files containing the 'BSCALE' keyword.
- A bug in mwa_corr_fits that didn't properly test against 2015 data, and so didn't
catch the error when using `memap=true` while reading files containing the 'BSCALE' keyword.
- A bug in mwa_corr_fits that didn't round start_flag to the nearest multiple of
integration time when using goodtime, resulting in an error when reading data
taken at 2 seconds.
- A bug in reading in the MWA beam that caused the beam to be rotated by 90 degrees.

## [2.1.4] - 2021-2-04

### Added
- Added `propagate_coarse_flags` option to `read_mwa_corr_fits`, as well as new flagging defaults.
- Added support for `extra_keywords` in UVFlag construction from UVData objects (and paths)

### Changed
- Improved memory usage in reading MWA correlator fits files.
- Speed improvement in redundant group finder.

### Fixed
- Fixed a bug in `compress_by_redundancy` with `method='average'` that resulted in more unique times on the compressed object than on the original object.
- Fixed bug causing MemoryError when finding redundant baselines for over 100 antennas.

## [2.1.3] - 2020-12-15

### Added
- Added support for `telescope_location`, `antenna_positions` and `lst_array` in UVCal objects and file types.
- Added support for a `metadata_only` mode in UVCal, including options to only read the metadata when reading in calibration files.
- Added a `copy` method for UVCal objects.
- Added `antenna_positions`, `antenna_names` and `antenna_numbers` as optional metadata to our known telescopes list and added HERA antenna position information.
- Added handling for `extra_keywords` in UVFlag objects, including read/write to/from files.
- Added handling for `object_name` and `extra_keywords` to `sum_vis` and `diff_vis` methods and added the `override_params` option to override other parameters.

### Changed
- Changed to use Astropy sites for telescope locations when avaliable. This results in a small change for our known position for the MWA.
- Modified `UVData.read` to do faster concatenation of files, changed the interface to `UVData.fast_concat` to allow lists of `UVData` objects to be passed in.

### Fixed
- Fixed a bug where telescope positions from MWA uvfits files created by Cotter were not identified as being in the ITRF frame because of a missing FITS keyword.
- Fixed a bug where `antenna_positions` from FHD files were interpreted as being in the relative ECEF frame rather than the rotated ECEF frame
- Fixed a bug where the `lst_array` was not updated in `compress_by_redundancy` when using `method='average'`.
- Fixed an undefined parameter bug when interpolating beams that do not cover the sky.
- Fixed an indexing error bug in `compress_by_redundancy` when using `method='average'`.

### Deprecated
- UVCal objects now require `telescope_location`, `antenna_positions` and `lst_array` parameters. Not setting them will cause an error starting in pyuvdata 2.3.

## [2.1.2] - 2020-10-07

### Added
- Added support for multiple spectral windows for UVH5, UVFITS, MIR, and MIRIAD files.
- Added the `flex_spw` attribute to the `UVData` class, which can be set to True by using the new `_set_flex_spw` method.
- Added the optional `flex_spw_id_array` attribute, of type=int and shape=(`Nfreqs`,), which indexes individual channels along the frequency axis to `spw_array`.
- Adjustment to digital gain removal from mwa_corr_fits files to account for a division by 64 due to a bit selection shift.
- Options to remove coarse band shape and digital gains from mwa_corr_fits files.
- Support for cotter flags in mwa_corr_fits files.

### Changed
- When `flex_spw=True`, the `channel_width` attribute of UVData is expected to be an array of type=float and shape=(`Nfreqs`,). Individual channels are allowed to have different channel widths.
- Changed `freq_array` to be of shape (1, `Nfreqs`) (formerley (`Nspws`, `Nfreqs`)).
- Changed  `data_array`, `flag_array`, `nsample_array` to be of shape (`Nblts`, 1, `Nfreqs`, `Npols`) (formerly (`Nblts`, `Nspws`, `Nfreqs`, `Npols`)).
- Changed UVH5 metadata byte conversion method from tobytes() to bytes()

## [2.1.1] - 2020-8-07

### Added
- Added read-only support for MIR files, adding the `Mir` class to `UVData`.
- Added the `MirParser` class, which allows for python access to MIR files.
- A new `check_warnings` method in our test module that behaves like `pytest.warns`
but adds the ability to check for multiple warnings.
- The 'ant_str' option when making selects on UVFlag objects
- A check that the uvws match the antenna positions, as part of the acceptability checking.

### Changed
- Updated the numpy requirement to >= 1.18
- Moved `parse_ants` to `utils.py` to allow any UVBased object the potential to use it.

### Deprecated
- The `checkWarnings` function in our test module, use the new `check_warnings`
function instead.

## [2.1.0] - 2020-7-08

### Added
- `UVFlag` can now take `PosixPath` objects as input.
- An option to average rather than select a single baseline in `UVData.compress_by_redundancy`,
controlled by the `method` keyword.
- `nsample_array_dtype` keyword for `UVData.read`, which sets the datatype for the nsample_array. Currently only used for mwa_corr_fits files.
- `utils` cython module to speed up some common utility functions.
- Added support for installing the package on Windows.
- 'background_lsts' keyword in `uvdata.read` to calculate lsts in the background while reading. Defaults to True.
- `background` keyword in `uvdata.set_lsts_from_time_array` to allow lst calculation in a background thread.
- `read_data` keyword to read_mwa_corr_fits, allows from metadata only reads of MWA correlator fits files.
- `propagate_flags` keyword for `UVData.frequency_average` which flags averaged samples if any contributing samples were flagged

### Changed
- Nants data calculation changed to use numpy functions for small speed up.
- Changed `input` variable to `indata` in UVFlag to avoid shadowing python builtin.
- `select` now also accepts a list of baseline indices for the `bls` parameter.
- `data_array_dtype` keyword for `UVData.read` is now also respected by mwa_corr_fits files. Previously it was only used by uvh5 files.
- The `time_range` parameter on UVCal is no longer required and it is suggested to only set it if the Ntimes axis is length one.
- FHD now supports metadata only reads.
- Updated formatting and added more explicit typing in corr_fits.pyx
- Updated Nants calculations for speed up.
- updated HERA telescope location to match the HERA defined center of array.
- `utils.uvcalibrate` now incorporates many more consistency checks between the uvcal and uvdata object. The new keywords `time_check` and `ant_check` were added to control some of these checks.

### Deprecated
- The UVBeam methods `set_cs_params`, `set_efield`, `set_power`, `set_simple`, and `set_phased_array` have been made private. The public methods will be removed in version 2.2.
- The UVCal methods `set_gain`, `set_delay`, `set_unknown_cal_type`, `set_sky`, and `set_redundant` have been made private. The public methods will be removed in version 2.2.
- `utils.uvcalibrate` will error rather than warn if some of the newly added checks do not pass, including if antenna names do not match between the objects, starting in version 2.2. In addition, the `flag_missing` keyword is deprecated and will be removed in version 2.2.

### Fixed
- UVFlag.__add__ now properly concatenates all existing data-like parameters of the object, including optional ones.
- A bug in `UVData.downsample_in_time` where the data were not being properly weighted by the nsample array and the nsample_array was not being properly weighted by the integration times.
- A bug in `UVData.downsample_in_time` that lead to duplicated data on the final object if a baseline had varying integration times and some integration times were greater than or equal to the requested minimum integration time.

## [2.0.2] - 2020-4-29

### Added
- New `_corr_fits` C extension for performing the index and conjugation mapping calculation for read_mwa_corr_fits
- `copy` method to UVCal and UVBeam objects
- `chunks` to `UVH5.write_uvh5` and `UVData.write_uvh5` for HDF5 dataset chunking.
- `multidim_index` to `UVH5.read_uvh5` and `UVData.read` for multidimensional slicing into HDF5 datasets

### Changed
- Various serialized calculations in uvdata.py, uvbeam.py, and utils.py to be vectorized with numpy
- Miriad interface was re-written from a hand written CPython interface to a Cython interface which dynamically creates the CPython during package setup/build.
- uvfits, calfits and beamfits files should now be read faster due to a simplified handling of fits header objects, especially when the history is very long.
- The UVData methods `set_drift`, `set_phased`, and `set_unknown_phase_type` have been made private. The public methods will be removed in version 2.2.

## [2.0.1] - 2020-3-24

### Changed
- UVData allows conjugation of metadata only objects.
- Handling of strings in UVFlag files has been made more widely compatible.

### Added
- Option to provide parameters for RectBivariatespline through interp
- `weights_square_array` (optional) parameter on UVFlag - stores sum of squares of weights when converting to waterfall
- `frequency_average` method on UVData to average data along the frequency axis.

### Fixed
- A bug in `select` where baseline-polarization tuples would not be conjugated correctly in `UVData` and `UVFlag`, and select on read for Miriad files.
- `metafits_ppds.fits` files can now be passed to `mwa_corr_fits.read` without throwing an error.
- UVParameters that are array_like and have NaNs are now properly identified as equal if the NaNs are in the same locations and all non-NaN entries are equal.
- A bug in `mwa_corr_fits.read` in filling flag and nsample arrays.
- A bug in `UVData.downsample_in_time` in calculating the number of new blts.

## [2.0.0] - 2020-2-12

### Changed
- All references to Python 2 removed from codebase.

### Added
- Routine flagging of MWA coarse band edges and center channels, as well as beginning and end integrations. Only done during `read_mwa_corr_fits`.
- Added a return_weights_square option to `utils.collapse` function, which returns the sum of the squares of the weights.

### Removed
- Previously deprecated code marked for removal in version > 1.5:
 - reading multiple files with file format specific read functions (e.g.  read_mirad). Multi-file reads can only be performed with `uvdata.read`
 - `read_metadata` keyword in various specific read functions (e.g. `read_miriad`)
 - `metadata_only` keyword in `select` and `get_redundancies`
 - `uvdata.miriad.read_miriad_metadata`
 - `phase_center` keyword in `uvdata.read_mwa_corr_fits`
 - `phase_data` keyword in `uvdata.read_mwa_corr_fits`
 - `uvdata.get_antenna_redundancies`
 - `uvdata.get_baseline_redundancies`
 - `uvdata.uvfits.read_uvfits_metadata`
 - `uvdata.uvfits.read_uvfits_data`

## [1.5.0] - 2020-1-15

### Added
- Support for reading the MWA full embedded element beam into a UVBeam object.
- New `time_range` keyword to `select` so exact times don't need to be specified (also added to `read` methods for select on read).
- Support for rephasing phased data including on `read`, `__add__` and `fast_concat` so that files with different phasing can be read in together.
- `sum_vis` and `diff_vis` for summing or differencing visibilities in the data_array.
- `read_mwa_corr_fits` for reading in MWA correlator gpubox files and applying cable corrections.
- `eq_coeffs` for storing equalization coefficients and `remove_eq_coeffs` for removing them.
- `utils.apply_uvflag` for applying UVFlag objects to UVData objects

### Fixed
- Fixed `utils.uvcalibrate` to handle `x_orientation` attribute
- Arrays of file names can now be passed to `read` (not just lists).
- run_check is no longer turned off in `read_uvfits` and `read_uvh5` when `read_data` is False.
- Redundancy finder will now error if any baselines appear in multiple groups.
- A bug in `UVCal` objects that prevented them from properly getting data with `ee`/`nn`-style polarizations.
- a bug in `UVFlag` where `x_orientation` was not set during initialization.
- A bug in `UVCal.read_fhd_cal` that caused calibration solutions to be approximately doubled.
- A bug in UVFlag where polarization array states were not updated when using `force_pol` keyword in `to_antenna` and `to_baseline`
- A bug in UVFlag.to_baseline() where force_pol kwarg did not work for UVData Npols > 1
- `UVData.read_uvfits` no longer breaks if there are non-ascii bytes in antenna names (which CASA sometimes writes).

### Deprecated
- Reading in multiple files (or file sets) using file-type specific read methods (e.g. `read_uvfits`) in favor of the generic `read` method.
- The `phase_center` and `phase_data` keywords to `read_mwa_corr_fits` in favor of `phase_to_pointing_center` and the `phase_center_radec` keyword in the generic `read` method.
- Support for reading only the header (not all the metadata) of uvfits files.
- The `read_uvfits_metadata` and `read_uvfits_data` methods on the UVFITS object.
- The `read_miriad_metadata` method on the Mirad object.

## [1.4.2] - 2019-10-15

### Added
- `copy` method for `UVData`, which can optionally make a copy of just metadata
- `upsample_in_time`, `downsample_in_time`, and `resample_in_time` methods on `UVData` objects
- `utils.uvcalibrate(.., undo=True)` kwarg for undo-ing a calibration.
- `utils.uvcalibrate` updates `UVData.vis_units` if `UVCal.gain_scale` is set.
- `UVCal.gain_scale` non-required attribute
- UVData.get_redundancies method to replace old get_baseline_redundancies and get_antenna_redundancies.
- option for `UVBeam.interp` to return a new beam object.
- `UVFlag` information on Read The Docs

### Changed
- `UVData.phase_to_time` now accepts a float as an input. Assumes float represents a JD.
- Added optional acceptability check for `utils.LatLonAlt_from_XYZ`
- Use `astropy_healpix` rather than `healpy` for HEALPix functionality because `healpy` has a GPL license which is incompatible with ours.
- `h5py` is now a required package instead of an optional one.
- Phasing now supports metadata only `UVData` objects
- utils.get_baseline_redundancies uses scipy pdist functions instead of for loops (faster)
- UVData.get_antenna_redundancies will no longer automatically conjugate baselines.
- UVData.get_baseline_redundancies and UVData.get_antenna_redundancies have been combined.
- `UVFlag` inherits from `UVBase` object.
- `UVFlag` objects can now convert from antenna type to baseline type
- `UVFlag` objects can now be initialized without inputs

### Fixed
- A bug in UVBeam._interp_freq where kind parameter was not passed for real-only beams
- A bug in get_antenna_redundancies for nearly N-S baselines.
- A bug where `conj_pol` could not handle cardinal direction polarizations.
- A bug that gave the wrong error message when calling `UVData.phase_to_time` without an Astropy Time object.

### Deprecated
- UVData.get_baseline_redundancies and UVData.get_antenna_redundancies will be deprecated in version 1.6.1

## [1.4.1] - 2019-08-2

### Added
- `metadata_only` property on `UVData` to automatically detect if data-like arrays are present
- support for combining metadata only objects and reading in multiple files as metadata only
- `utils.uvcalibrate` flag propagation bug fix
- `UVCal.ant2ind` indexing bug fix
- `UVCal.get_*` methods for accessing data arrays with antenna-polarization keys
- `utils.uvcalibrate` for automated calibration of `UVData` by `UVCal`

### Fixed
- Fixed a bug in select that caused bls and antenna_names/numbers to be or'ed rather than and'ed together.
- Fixed a bug where `baseline_to_antnums` could accept a numpy array as input but not other array_like objects.

### Changed
- removed `new_object` keyword from `UVBeam._interp_freq` in favor of new functionality in `UVBeam.interp`

## [1.4.0] - 2019-05-23

### Added
- Option in UVBase.check() to ignore whether required parameters are set.
- made an option to not save the `lst_array` to uvfits files.
- `conjugate_bls` option to `UVData.get_antenna_redundancies`
- `UVData.conjugate_bls` method to conjugate baselines to get the desired baseline directions.
- `UVData.reorder_blts` method to reorder the data along the blt axis (and optionally also conjugate baselines), and a new `blt_order` optional parameter on UVData objects to track the ordering (including through read/writes).
- `lst_array` is now saved to UVFITS files (even though it's not a standard parameter) so that it doesn't have to be recalculated

### Fixed
- Fixed init logic in UVFlag.
- Fixed a bug in how FHD uvw vectors were oriented (and visibilities were conjugated)
- Fixed a bug in `UVData.inflate_by_redundancy` when Nblts is not equal to Nbls * Ntimes.
- Fixed a bug in UVData when reading in an FHD file with a single time integration.
- Fixed a bug in how the longitudinal branch cut was handled in beam interpolation
- Changed the way interpolation splines are saved in UVBeam to fix errors related to polarization selections.
- Python 3: `np.string_` call now uses `keepdims=True` to guard against `antenna_names` being cast as an array.

### Changed
- Testing framework changed from `nose` to `pytest`.
- `uvdata.set_lsts_from_time_array` only calculates lsts for unique elements in `time_array`.

## [1.3.8] - 2019-05-01

### Added
- Optional `x_orientation` parameter to utils functions polstr2num, polnum2str, jstr2num and jnum2str to allow for E/N based polarization strings (rather than just x/y based ones)
- New optional `x_orientation` parameter on UVBeam (paralleling UVData and UVCal), with read/write support in beamfits
- Added `x_orientation` as an optional parameter in read_cst_beam and in cst settings yaml files.
- All str2num or num2str calls on UVData and UVBeam pass the object's x_orientation
- New `UVData.fast_concat` method to allow fast concatenation of UVData objects (or files) along a particular axis.
- Added preliminary `UVFlag` module from hera_qm to pyuvdata. Will eventually promote to `UVBase` object, but for now this is undocumented functionality.

### Deprecated
- Defined 'east' and 'north' as the allowed 'x_orientation' values in UVData and UVCal, Backwards compatiblity support exists for 'E' and 'N' values
- `UVData.order_pols` method in favor of `UVData.reorder_pols`.

### Fixed
- Building pyuvdata on macOS now targets minimum macOS 10.9 if run on macOS 10.9 or above
- Possible bug where `check_variables` dictionary can change size during `read_miriad` call


## [1.3.7] - 2019-04-02

### Added
- Added `add_to_history` kwarg to UVH5.write_uvh5_part
- `_healpix_interp_bilinear` as a new interpolation method in `UVBeam`
- `freq_interpolation_kind` added as an attribute to `UVBeam`
- `tol` added as keyword argument to `UVBeam._interp_freq` which allows for a fast return of `data_array` slice if nearest-neighbor frequencies are all within the distance tolerance.
- `polarizations` added as keyword argument to `UVBeam` interpolation methods.
- Support for a yaml settings file to collect and propagate metadata for CST beam files.

### Changed
- `UVBeam._interp_freq` returns both `interp_data` and `interp_bandpass`, instead of just the former.

### Fixed
- Combining overlapping data along multiple axes (most common when reading in multiple files) no longer errors.


## [1.3.6] - 2019-02-15

### Added
- `keep_all_metadata` keyword for optionally discarding unused metadata when performing a select operation.

### Changed
- Extends `run_acceptability_check` for UVH5 metadata in `check_header` function.

### Fixed
- Antenna numbering bug in redundancy methods. It wasn't using the correct antenna numbers to make baseline indices.
- Redundancy code returns one group if all baselines are redundant. Previously returned each baseline as a separate group
- Redundancy code finds unique baselines along baseline_array without assuming Nblts = Nbls \* Ntimes. Previously assumed Nblts = Nbls \* Ntimes and attempted to slice array.
- "inflate_by_redundancy" method errored when phase_type == phased, due to _set_u_positive using phased uvw coordinates. It now uses ENU frame uvw coordinates.

## [1.3.5] - 2018-12-20

## [1.3.4] - 2018-12-19
### Added
- Methods on UVData objects to compress/inflate data by redundant baselines.
- Convenience functions on UVData for finding redundant baselines (calling the corresponding utils functions)
- memo describing the UVH5 format
- read/write support for uvh5 files with integer datatypes for visibilities
- Option to only do the select on the metadata. This is useful for partially defined objects as in pyuvsim setup or after reading only the metadata from a file.
- support for python3.7

### Changed
- UVdata.get_ENU_antpos() now defaults to using the telescope_location as the center rather than the median antenna position.
- UVBeam.efield_to_pstokes() no longer restricted to healpix coordinates
- latitude and longitude in uvh5 files are written in degrees instead of radians.
- Fixes a bug in redundancy methods for when there are no redundant baselines.

### Fixed
- `_key2inds` now properly reorders polarization axis for conjugated visibilities. This also effects the `get_data` function.
- long strings are saved correctly in miriad files from python3

## [1.3.3] - 2018-11-01
### Added
- option to save splines for reuse in UVBeam.interp function

### Changed
- improve string handling for uvh5 files
- changed top-level import structure to exclude file-specific class (e.g. `UVFITS`, `CALFITS`) and base classes (`UVBase`, `UVParameter`) and to not import utility functions into the top-level namespace

### Deprecated
- Support for UVData objects without antenna_positions. Antenna positions will be required in a future version.

## [1.3.2] - 2018-09-27
### Added
- Utility functions to identify redundant baselines (either by baseline or antenna pair)
- Reading FHD layout files to get antenna positions
- Polarization dict constants and parsing functions in utils.py for mapping between polarization strings and numbers

### Changed
- LST array is now optional in uvh5 files
- polarization string capitalization was unified throughout: use lower case except for Stokes parameters
- integration_time is now a vector of length NBlts instead of a scalar

### Deprecated
- Support for FHD inputs without layout files (containing antenna positions).

### Fixed
- flags are always returned as a boolean array from `UVData.get_flags`
- integration_time, uvw_array and lst_array are now always checked for consistency when adding UVData objects
- consistency checks on baseline lengths now uses the uvw_array tolerances

## [1.3.1] - 2018-07-24

## [1.3] - 2018-07-22
### Added
- compatibility with python 3 (retaining python 2 compatibility)
- generic `UVData.read` method for all input file types, including select on read options
- partial write support for uvh5 file format
- partial read support for uvfits, miriad and uvh5 file formats (including only reading metadata and select on read)
- new uvh5 file format: an HDF5 file standard matched to UVData objects
- new method to calculate uvws from antenna positions
- `UVBeam.get_beam_area` and `UVBeam.get_beam_sq_area` functions to calculate beam integrals, including for pseudo-Stokes beams
- beam interpolation methods, to any set of points and to healpix pixel centers
- a script to renumber antennas for CASA compatiblity if there are fewer than 256 antennas but numbers higher than that
- memo describing the beam fits file format
- method to peak normalize UVBeam objects
- support for reading FHD calibrations into UVCal objects
- support for sky-based calibration metadata in UVCal and the calfits file format
- method to convert E-field beams to power beams
- `UVData.get_ENU_antpos` method to get ENU coordinates from antenna positions
- support for `extra_keywords` on UVCal objects

### Changed
- Major overhaul of phasing code, see the phasing memo in docs/references for more details
- Make all earth location coordinate conversions use same axes order (n_points, 3)
- replaced `ant_pair_nums` keyword in `UVData.select` and partial read methods with `bls` which supports lists like [(0,1,'xx'), (2,3,'yy')]
- extracted miriad wrappers from aipy, removing aipy as a dependency

### Fixed
- Fixed error when combining auto-correlation only and cross-correlation only UVData objects
- Fixed error with the `UVData.select` function using a single antenna name
- Fixed bugs with `UVData.get_data`, `UVData.get_flags`, and `UVData.get_nsamples` for conjugated baseline polarizations
- Fixed a bug that caused a memory error in `UVData.write_uvfits`
- Fixed bugs in interpreting the uvw direction convention for uvfits
- Fixed a bug in reading azimuth locations from CST beam files
- Fixed a bug reading in single frequency uvfits files
- Fixed a bug in reading MWA Cotter measurement sets
- Fixed units errors in cal fits files
- Fixed a serious bug where data was overwritten in the add functions if the axes were out of canonical order
- Fixed scrambled data ordering in add function

## [1.2.1] - 2017-11-09
### Fixed
- Fixed a bug in parsing frequencies from CST file names

## [1.2] - 2017-11-08
### Added
- support doctest in the tutorial
- utility functions for converting between Jones numbers and polarization strings
- support for `antdiam` variable in miriad files
- module for reading a CST file into a UVBeam object
- utility functions for converting to rotated ECEF coordinates from ECEF coordinates
- support reading in a list of calfits files
- include `extra_keywords` to miriad files
- `__add__` method for UVCal objects
- `antenna_diameters` value to UVData objects
- `parse_ants` method on UVData objects
- `x_orientation` value to UVData objects
- "smart slicing" functionality to UVData objects
- convenience methods on UVData objects for easily getting data and metadata
- UVBeam object
- in-place selection for UVData objects
- `total_quality_array` value on UVCal objects
- `__add__` method for UVData objects
- utility functions for converting to local ENU coordinates from ECEF coordinates
- `convert_to_gain` method on UVCal delay-type objects
- read-only support for CASA measurement sets into UVData objects
- `select` method on UVCal objects

### Changed
- antenna names in miriad saved as strings instead of arrays of ASCII hex values

### Fixed
- baseline-time axis mis-ordering in add function
- handling of antenna positions in miriad and uvfits files
- selecting autocorrelation data from UVData objects
- indexing of spectral windows in calfits files
- handling of `total_quality_array` in UVCal objects when selecting a subset of data

## [1.1] - 2017-04-14
(historical information needs to be filled in)
