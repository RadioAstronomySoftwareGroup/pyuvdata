# pyuvdata

[![CircleCI](https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata.svg?style=svg&branch=master)](https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata?branch=master)
![](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/workflows/Run%20Tests/badge.svg?branch=master)
[![Build Status](https://dev.azure.com/radioastronomysoftwaregroup/pyuvdata/_apis/build/status/RadioAstronomySoftwareGroup.pyuvdata?branchName=master)](https://dev.azure.com/radioastronomysoftwaregroup/pyuvdata/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata/badge.svg?branch=master)](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00140/status.svg)](https://doi.org/10.21105/joss.00140)

pyuvdata defines a pythonic interface to interferometric data sets.
Currently pyuvdata supports reading and writing of miriad, uvfits, and uvh5 files
and reading of CASA measurement sets and FHD
([Fast Holographic Deconvolution](https://github.com/EoRImaging/FHD)) visibility save files.


# Motivation
The three main goals are:

1. To provide a high quality, well documented path to convert between data formats
2. Support the direct use of datasets from python with minimal software
3. Provide precise data definition via both human readable code and high quality online documentation

# Package Details
pyuvdata has four major user classes:

* UVData: supports interferometric data (visibilities) and associated metadata
* UVCal: supports interferometric calibration solutions (antenna-based) and
associated metadata (Note that this is a fairly new object, consider it to be a beta version)
* UVBeam: supports primary beams (E-field or power) and associated metadata
(Note that this is a new object and is very experimental, consider it to be an alpha version)
* UVFlag: A class to handle the manipulation and combination of flags for data sets.
Also can convert raw data quality metrics into flags using thresholding.
(This object is very new and experimental. Consider it to be a beta version)

## UVData File standard notes
* miriad has been throughly tested with aipy-style miriad files and minimally
tested with ATCA files
* uvfits conforms to AIPS memo 117 (as of May 2015).  It is tested against
FHD, CASA, and AIPS. However AIPS is limited to <80 antennas and CASA uvfits
import does not seem to support >255 antennas.
* uvh5 is an HDF5-based file format defined by the HERA collaboration,
details in the [uvh5 memo](docs/references/uvh5_memo.pdf). Note that this is a
new format and is still under development, consider it to be a beta version,
but we will strive to make future versions backwards compatible with the current format.
It is probably not compatible with other interferometric HDF5 files defined by other groups.
* FHD (read-only support, tested against MWA and PAPER data)
* CASA measurement sets (read-only support)

## UVCal file formats
* calfits: a new format defined in pyuvdata, details in the [calfits_memo](docs/references/calfits_memo.pdf).
Note that this format was recently defined and may change in coming versions,
based on user needs. Consider it to be a beta version, but we will strive to
make future versions backwards compatible with the current format.
* FHD calibration files (read-only support)

## UVBeam file formats
* regularly gridded fits for both E-field and power beams
* non-standard HEALPix fits for both E-field and power beams (in an ImageHDU
rather than a binary table to support frequency, polarization and E-field vector axes)
* read support for CST beam text files, with a defined yaml file format for
metadata, details here: [cst settings file](docs/cst_settings_yaml.rst)

## Under Development
* UVData: uvh5 file format (beta version), note that this is probably not
compatible with other interferometric HDF5 files defined by other groups.
* UVCal: object and calfits file format (beta version)
* UVBeam: object and beamfits file format (alpha version)
* UVFlag: object, initialization, and type changing. (beta version)

## Known Issues and Planned Improvements
* UVData: phasing (and the accuracy on the uvw coordinates) is only known to be
good to 2cm on a 3km baseline (this is limited by the accuracy of the test file,
see the [phasing memo](docs/references/phasing.pdf) for more details).
* UVData: Multiple spectral windows or multiple sources are not currently supported
* UVData: add support for writing CASA measurement sets
* UVBeam: support phased-array antenna beams (e.g. MWA beams).
* UVCal/UVData: method to apply calibration to data.
* package version detection can cause issues with installation directly from the
repo for some users (see [issue #590](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/590))
* UVFlag: Adding requires a high level knowledge of individual objects. (see [issue #653](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues/653))

For details see the [issue log](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues).

## Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues)
are very welcome, as are bug reports and feature requests.
Please see our [guide on contributing](.github/CONTRIBUTING.md)

# Versioning
We use a `generation.major.minor` version number format. We use the `generation`
number for very significant improvements or major rewrites, the `major` number
to indicate substantial package changes (intended to be released every 3-4 months)
and the `minor` number to release smaller incremental updates (intended to be
released approximately monthly and which usually do not include breaking API
changes). We do our best to provide a significant period (usually 2 major
generations) of deprecation warnings for all breaking changes to the API.
We track all changes in our [changelog](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/master/CHANGELOG.md).

# Documentation
A tutorial with example usage and developer API documentation is hosted on
[ReadTheDocs](https://pyuvdata.readthedocs.io).

# History
pyuvdata was originally developed in the low frequency 21cm community to support
the development of and interchange of data between calibration and foreground
subtraction pipelines. Particular focus has been paid to supporting drift and
phased array modes.

# Citation
Please cite pyuvdata by citing our JOSS paper:

Hazelton et al, (2017), pyuvdata: an interface for astronomical interferometeric
datasets in python, Journal of Open Source Software, 2(10), 140, doi:10.21105/joss.00140

[ADS Link](https://ui.adsabs.harvard.edu/abs/2017JOSS....2..140H);
[Bibtex entry](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2017JOSS....2..140H&data_type=BIBTEX&db_key=GEN&nocookieset=1)

# Installation
For simple installation, the latest stable version is available via conda
(preferred: ```conda install -c conda-forge pyuvdata```) or pip (```pip install pyuvdata```).

There are some optional dependencies that are required for specific functionality,
which will not be installed automatically by conda or pip.
See [Dependencies](#dependencies) for details on installing optional dependencies.

## Optionally install the development version
Clone the repository using
```git clone https://github.com/RadioAstronomySoftwareGroup/pyuvdata.git```

Navigate into the pyuvdata directory and run `pip install .`
(note that `python setup.py install` does not work).
Note that this will attempt to automatically install any missing dependencies.
If you use anaconda or another package manager you might prefer to first install
the dependencies as described in [Dependencies](#dependencies).

To install without dependencies, run `pip install --no-deps`

To compile the binary extension modules such that you can successfully run
`import pyuvdata` from the top-level directory of your Git checkout, run:
```python setup.py build_ext --inplace```

## Dependencies

Required:

* numpy >= 1.15
* scipy
* astropy >= 2.0
* h5py
* six

Optional:

* python-casacore (for working with CASA measurement sets)
* astropy-healpix (for working with beams in HEALPix formats)
* pyyaml (for working with settings files for CST beam files)

The numpy and astropy versions are important, so make sure these are up to date.

We suggest using conda to install all the dependencies. If you want to install
python-casacore and astropy-healpix, you'll need to add conda-forge as a channel
(```conda config --add channels conda-forge```).

If you do not want to use conda, most of the packages are also available on PyPI
(except python-casacore, see details for that package below).

### Installing python-casacore
python-casacore requires the casacore c++ libraries. It can be installed easily
using conda (```python-casacore``` on conda-forge).

If you do not want to use conda, the casacore c++ libraries are available for
ubuntu through the [kern suite](http://kernsuite.info/). On OSX, casacore is
available through the [ska-sa brew tap](https://github.com/ska-sa/homebrew-tap).
The python-casacore library (with manual install instructions) is available at
https://github.com/casacore/python-casacore

## Tests
Uses the `pytest` package to execute test suite.
From the source pyuvdata directory run ```pytest``` or ```python -m pytest```.

Testing of `UVFlag` module requires the `pytest-cases` plug-in (available from pip; may require `setuptools_scm` for python 2 developers).

# API
The primary interface to data from python is via the UVData object. It provides
import functionality from all supported file formats (UVFITS, Miriad, UVH5, FHD,
CASA measurement sets) and export to UVFITS, Miriad, and UVH5 formats and can
be interacted with directly. Similarly, the primary calibration, beam, and flag
interfaces are via the UVCal, UVBeam, and UVflag objects. The attributes of the UVData,
UVCal, UVBeam, and UVFlag objects are described in the uvdata_parameters, uvcal_parameters,
 uvbeam_parameters and uvflag_parameters descriptions at https://pyuvdata.readthedocs.io or
[here](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/master/docs).


# Maintainers
pyuvdata is maintained by the RASG Managers, which currently include:
 - Adam Beardsley (Arizona State University)
 - Bryna Hazelton (University of Washington)
 - Daniel Jacobs (Arizona State University)
 - Paul La Plante (University of Pennsylvania)
 - Jonathan Pober (Brown University)

Please use the channels discussed in the [guide on contributing](.github/CONTRIBUTING.md)
for code-related discussions. You can contact us privately if needed at
[rasgmanagers@gmail.com](mailto:rasgmanagers@gmail.com).
