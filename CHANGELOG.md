# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

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

## [1.2.1] - 2018-11-09
### Fixed
- Fixed a bug in parsing frequencies from CST file names

## [1.2] - 2018-11-08
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
