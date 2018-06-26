# pyuvdata

[![Build Status](https://travis-ci.org/HERA-Team/pyuvdata.svg?branch=master)](https://travis-ci.org/HERA-Team/pyuvdata)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/pyuvdata/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/pyuvdata?branch=master)

pyuvdata defines a pythonic interface to interferometric data sets. Currently pyuvdata supports reading and writing of miriad, uvfits, and uvh5 files and reading of CASA measurement sets and FHD ([Fast Holographic Deconvolution](https://github.com/EoRImaging/FHD)) visibility save files.


# Motivation
The three main goals are:

1. To provide a high quality, well documented path to convert between data formats
2. Support the direct use of datasets from python with minimal software
3. Provide precise data definition via both human readable code and high quality online documentation

# Package Details
pyuvdata has three major user classes:

* UVData: supports interferometric data (visibilities) and associated metadata
* UVCal: supports interferometric calibration solutions (antenna-based) and associated metadata (Note that this is a fairly new object, consider it to be a beta version)
* UVBeam: supports primary beams (E-field or power) and associated metadata (Note that this is a new object and is very experimental)

## UVData Tested File Paths
* uvfits -> miriad
* uvfits -> uvh5
* miriad -> uvfits
* miriad -> uvh5
* FHD -> uvfits
* FHD -> miriad
* CASA measurement sets -> miriad
* CASA measurement sets -> uvfits
* uvh5 -> uvfits
* uvh5 -> miriad

## UVData File standard notes
* miriad is supported for aipy-style analysis, further testing is required for use in the miriad package
* uvfits conforms to AIPS memo 117 (as of May 2015).  It is tested against FHD, CASA, and AIPS. However AIPS is limited to <80 antennas and CASA imaging does not seem to support >255 antennas.
* uvh5 is an HDF5-based file format defined by the HERA collaboration, and is defined by a memo (coming soon).
* FHD (read-only support, tested against MWA and PAPER data)
* CASA measurement sets (read-only support)

## UVCal file formats
* calfits: a new format defined in pyuvdata, a detailed memo is available here: [calfits_memo](https://github.com/HERA-Team/pyuvdata/blob/master/docs/references/calfits_memo.pdf). Note that this format was recently defined and may change in coming versions, based on user needs. Consider it to be in a beta version, but we will strive to make future versions backwards compatible with the current format.
* FHD calibration files (read-only support)

## UVBeam file formats
* regularly gridded fits for both E-field and power beams
* non-standard HEALPix fits for both E-field and power beams (in an ImageHDU rather than a binary table to support frequency, polarization and E-field vector axes)
* read support for CST beam text files

## Known Issues and Planned Improvements
* UVData: different multiple spectral windows or multiple sources are not currently supported
* UVData: testing against miriad package
* UVData: add support for writing CASA measurement sets
* UVData: phasing is tested to a part in 10^3, and assumes planar array. Improvements are tracked on Issue \#148.
* UVCal: expand support for calibration solutions: support other formats beyond FITS
* UVBeam: support reading HFSS beam model file formats
* UVBeam: support reading/writing/combining standard HEALPix FITs files (individual files per frequency, polarization and E-field vector)

For details see the [issue log](https://github.com/HERA-Team/pyuvdata/issues).

## Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/HERA-Team/pyuvdata/issues) are very welcome.
Please submit improvements as pull requests against the repo after verifying that
the existing tests pass and any new code is well covered by unit tests.

Bug reports or feature requests are also very welcome, please add them to the
issue log after verifying that the issue does not already exist.
Comments on existing issues are also welcome.

# Documentation
A tutorial with example usage and developer API documentation is hosted on [ReadTheDocs](https://pyuvdata.readthedocs.io).

# History
pyuvdata was originally developed in the low frequency 21cm community to support the development of calibration and foreground subtraction pipelines. Particular focus has been paid to supporting drift and phased array modes.

# Installation
For simple installation, the latest stable version is available via conda
(preferred: ```conda install -c conda-forge pyuvdata```) or pip (```pip install pyuvdata```)

## Optionally install the development version
First install the dependencies, see below for package recommendations and
extra dependencies for HEALPix beams and CASA measurement set functionalities.

Clone the repository using
```git clone https://github.com/HERA-Team/pyuvdata.git```

Navigate into the directory and run ```python setup.py install```.
Note that this will attempt to automatically install any missing dependencies. If you use anaconda or another package manager you might prefer to first install the dependencies as described below.

To install without dependencies, run
```python setup.py develop --no-deps``` or ```pip install --no-deps```

To compile the binary extension modules such that you can successfully run `import pyuvdata` from the top-level directory of your Git checkout, run:

```
python setup.py build_ext --inplace
```

## Dependencies
The numpy and astropy versions are important, so be sure to make sure these are up to date before you install.

For anaconda users, we suggest using conda to install astropy, numpy, scipy, and optionally h5py, and
conda-forge for optionally installing python-casacore and healpy (e.g. ```conda install -c conda-forge python-casacore```).

* numpy >= 1.10
* scipy
* astropy >= 1.2
* h5py (optional: for reading and writing uvh5 format)
* python-casacore (optional: for CASA measurement set reading functionality)
* healpy (optional: working with beams in HEALPix formats)

### For CASA measurement set functionality, install python-casacore
python-casacore is required in order to use the measurement set capabilities of pyuvdata. python-casacore requires the casacore c++ libraries. To install via conda,  run

```conda config --add channels conda-forge```

```conda install python-casacore``` (This will install both python-casacore and the casacore c++ libraries as a requirement)

If you do not want to use conda, the casacore c++ libraries are available for ubuntu through the [kern suite](http://kernsuite.info/). On OSX, casacore is available through the [ska-sa brew tap](https://github.com/ska-sa/homebrew-tap). The python-casacore library (with manual install instructions) is available at https://github.com/casacore/python-casacore

### For working with beams in HEALPix formats, install healpy
To install via conda,  run
```conda install -c conda-forge healpy```


## Tests
Requires installation of `nose` package.
From the source pyuvdata directory run ```nosetests pyuvdata```.


# API
The primary interface to data from python is via the UVData object. It provides import functionality from all supported file formats (UVFITS, Miriad, UVH5, FHD, CASA measurement sets) and export to UVFITS, Miriad, and UVH5 formats and can be interacted with directly. Similarly, the primary calibration and beam interfaces are via the UVCal and UVBeam objects. The attributes of the UVData, UVCal and UVBeam objects are
described in the uvdata_parameters, uvcal_parameters and uvbeam_parameters descriptions at https://pyuvdata.readthedocs.io or [here](https://github.com/HERA-Team/pyuvdata/blob/master/docs).
