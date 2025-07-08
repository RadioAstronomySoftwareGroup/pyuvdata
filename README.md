# pyuvdata

[![CircleCI](https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata.svg?style=svg&branch=main)](https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata?branch=main)
[![testing](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/actions/workflows/macosx_windows_ci.yaml/badge.svg?branch=main)](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/actions)
[![codecov](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata/badge.svg?branch=main)](https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata)
[![](https://readthedocs.org/projects/pyuvdata/badge/?version=latest)](https://app.readthedocs.org/projects/pyuvdata/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07482/status.svg)](https://doi.org/10.21105/joss.07482)

pyuvdata defines a pythonic interface to interferometric data sets.
Currently pyuvdata supports reading and writing of miriad, uvfits, CASA measurement sets
and uvh5 files and reading of FHD
([Fast Holographic Deconvolution](https://github.com/EoRImaging/FHD)) visibility
save files, SMA Mir files and MWA correlator FITS files.

Documentation of the API, conventions used throughout the package, and a tutorial
is hosted on [ReadTheDocs](https://pyuvdata.readthedocs.io).

# Motivation
The main goals are:

1. To provide a high quality, well documented path to convert between file formats.
2. Support the direct use of interferometric datasets from python with minimal software.
3. Provide precise data definitions and convention details via both human
readable code and high quality documentation.

# Package Details
pyuvdata has four major user classes:

* UVData: supports interferometric data (visibilities) and associated metadata
* UVCal: supports interferometric calibration solutions (antenna-based) and
associated metadata.
* UVBeam: supports primary beams (E-field or power) and associated metadata. A
number of analytic beam objects are also available and the BeamInterface object
serves as a unified interface for UVBeam and analytic beam objects.
* UVFlag: A class to handle the manipulation and combination of flags for data sets.
Also can convert raw data quality metrics into flags using thresholding.
(This object is somewhat new and experimental. Consider it to be a beta version)

## UVData File standard notes
* Miriad has been thoroughly tested with aipy-style Miriad files and minimally
tested with ATCA files. Reading/writing Miriad files is not supported on Windows.
* UVFITS conforms to AIPS memo 117 (as of March 2020).  It is tested against
FHD, CASA, and AIPS. However AIPS is limited to <80 antennas and CASA's ``importuvfits``
task does not seem to support >255 antennas. Because of this and other limitations
to CASA's ``importuvfits`` task, we reccommend that users planning to work in CASA
avoid using ``importuvfits`` and use the measurement set writer instead.
* CASA measurement sets, primarily conforming to
[CASA Memo 229](https://casa.nrao.edu/Memos/229.html), with some elements taken
from the proposed v3.0 format documented in
[CASA Memo 264](https://casacore.github.io/casacore-notes/264.html). Measurement
sets are tested against datasets from VLA, MWA (filled via ``cotter``),
ALMA, and SMA (filled using the ``importuvfits`` task). Extensive loopback testing
has been done to verify that pyuvdata written measurement sets are compatible
with CASA.
* UVH5 is an HDF5-based file format defined by the HERA collaboration,
details in the [uvh5 memo](docs/references/uvh5_memo.pdf). Note that this is a
somewhat new format, so it may evolve a bit but we will strive to make future
versions backwards compatible with the current format.
It is probably not compatible with other interferometric HDF5 files defined by other groups.
* FHD (read-only support, tested against MWA and PAPER data)
* MIR (read-only support, though experimental write functions are available, tested against SMA data)
* MWA correlator FITS files (read-only support, tested against Cotter outputs and FHD)

## UVCal file formats
* CalH5: a new an HDF5-based file format defined in pyuvdata, details in the
[calh5 memo](docs/references/calh5.pdf).
Note that this format is a somewhat new format, so it may evolve a bit but we
will strive to make future versions backwards compatible with the current format.
* Measurement Set calibration files (read and write, gains/delay/bandpass supported,
beta version). Tested against a limited set of SMA, LWA, and VLA calibration
files generated in CASA.
* CalFITS: a custom format defined in pyuvdata, details in the
[calfits memo](docs/references/calfits_memo.pdf).
Note that this format does not support all possible UVCal objects, so we prefer
CalH5 which has full support for all variations of UVCal objects.
* FHD calibration files (read-only support)

## UVBeam file formats
* BeamFITS: a custom format defined in pyuvdata that supports both regularly
gridded beams and beams on a HEALPix grid for both E-field and power beams,
details in the [beamfits memo](docs/references/beamfits_memo.pdf).
* CST beam text files (read only support) with a defined yaml file format for
metadata, details here: [cst settings file](docs/cst_settings_yaml.rst)
* FEKO beam ffe files (read only support)
* MWA Beams (read only support)

## Known Issues and Planned Improvements
* Incorporating numba to alleviate bottlenecks and to replace some existing
cython extensions as appropriate.

See our [issue log](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues)
for a full list.

## Community Guidelines
Contributions to this package to add new file formats or address any of the
issues in the [issue log](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/issues)
are very welcome, as are bug reports and feature requests.
Please see our [guide on contributing](.github/CONTRIBUTING.md)

# Telescopes
pyuvdata has been used with data from the following telescopes. If you use it on
data from a telescope we don't have listed here please let us know how it went
via an issue! We would like to make pyuvdata generally useful to the community for
as many telescopes as possible.

* MWA
* HERA
* PAPER
* LWA
* ALMA
* VLA
* ATCA
* SMA
* ATA
* VLBA

# Versioning
We use a `generation.major.minor` version number format. We use the `generation`
number for very significant improvements or major rewrites, the `major` number
to indicate substantial package changes (intended to be released every 3-4 months)
and the `minor` number to release smaller incremental updates (intended to be
released approximately monthly and which usually do not include breaking API
changes). We do our best to provide a significant period (usually 2 major
generations) of deprecation warnings for all breaking changes to the API.
We track all changes in our [changelog](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/main/CHANGELOG.md).

# History
pyuvdata was originally developed in the low frequency 21cm community to support
the development of and interchange of data between calibration and foreground
subtraction pipelines. Particular focus has been paid to supporting drift and
phased array modes.

# Citation
Please cite both of our JOSS papers:

*Keating et al., (2025). pyuvdata v3: an interface for astronomical interferometric data sets in Python. Journal of Open Source Software, 10(109), 7482, https://doi.org/10.21105/joss.07482* [[ADS Link](https://ui.adsabs.harvard.edu/abs/2025JOSS...10.7482K)]

*Hazelton et al, (2017), pyuvdata: an interface for astronomical interferometeric datasets in python, Journal of Open Source Software, 2(10), 140, doi:10.21105/joss.00140* [[ADS Link](https://ui.adsabs.harvard.edu/abs/2017JOSS....2..140H)]

# Installation
Simple installation via conda is available for users, developers should follow
the directions under [Developer Installation](#developer-installation) below.

For simple installation, the latest stable version is available via conda
(preferred: ```conda install -c conda-forge pyuvdata```) or pip (```pip install pyuvdata```).

There are some optional dependencies that are required for specific functionality,
which will not be installed automatically by conda or pip.
See [Dependencies](#dependencies) for details on installing optional dependencies.

## Dependencies

Required:

* astropy >= 6.0
* docstring_parser>=0.15
* h5py >= 3.7
* numba >= 0.57.0
* numpy >= 1.23
* pyerfa >= 2.0.1.1
* python >= 3.11
* pyyaml >= 5.4.1
* scipy >= 1.9
* setuptools_scm >= 8.1

Optional:

* astropy-healpix >= 1.0.2 (for working with beams in HEALPix formats)
* astroquery >= 0.4.4 (for enabling phasing to ephemeris objects using JPL-Horizons)
* hdf5plugin >= 3.3.1 (for enabling bitshuffle and other hdf5 compression filters in uvh5 files)
* lunarsky >=0.2.5 (for working with simulated datasets for lunar telescopes)
* novas and novas_de405 (for using the NOVAS library for astrometry)
* python-casacore >= 3.5.2 (for working with CASA measurement sets)

The numpy and astropy versions are important, so make sure these are up to date.

We suggest using conda to install all the dependencies. If you want to install
python-casacore and astropy-healpix, you'll need to add conda-forge as a channel
(```conda config --add channels conda-forge```).

If you do not want to use conda, the packages are also available on PyPI
(python-casacore may require more effort, see details for that package below).
You can install the optional dependencies via pip by specifying an option
when you install pyuvdata, as in ```pip install pyuvdata[healpix]```
which will install all the required packages for using the HEALPix functionality
in pyuvdata. The options that can be passed in this way are:

- ``astroquery``
- ``casa``
- ``hdf5_compression``
- ``healpix``
- ``lunar``
- ``novas``
- ``all``
- ``test``
- ``doc``
- ``dev``

The first set (``astroquery``, ``casa``, ``hdf5_compression``, ``healpix``, ``lunar``,
and ``novas``) enable various specific functionality while ``all`` will install
all of the above to enable all functionality. The last three (``test``, ``doc``
and ``dev``) include everything installed with ``all`` plus packages for testing
and building the docs which may be useful for developers of pyuvdata.

### Installing python-casacore
python-casacore requires the casacore c++ libraries. It can be installed easily
using conda (```python-casacore``` on conda-forge).

If you do not want to use conda, the casacore c++ libraries are available for
ubuntu through the [kern suite](http://kernsuite.info/). On OSX, casacore is
available through the [ska-sa brew tap](https://github.com/ska-sa/homebrew-tap).
The python-casacore library (with manual install instructions) is available at
https://github.com/casacore/python-casacore

## Developer Installation
Clone the repository using
```git clone https://github.com/RadioAstronomySoftwareGroup/pyuvdata.git```

Navigate into the pyuvdata directory and run `pip install .`
(note that `python setup.py install` does not work).
Note that this will attempt to automatically install any missing dependencies.
If you use conda or another package manager you might prefer to first install
the dependencies as described in [Dependencies](#dependencies).

To install without dependencies, run `pip install --no-deps .`

To compile the binary extension modules such that you can successfully run
`import pyuvdata` from the top-level directory of your Git checkout, run:
```python setup.py build_ext --inplace```

If you want to do development on pyuvdata, in addition to the other dependencies
you will need the following packages:

* pytest >= 8.2
* pytest-cases >= 3.9.1
* pytest-cov
* cython == 3.0
* coverage
* pre-commit
* matplotlib
* sphinx
* pypandoc

One other package, pytest-xdist, is not required, but can be used to speed up running
the test suite by running tests in parallel. To use it call pytest with the
```-n auto``` option.

One way to ensure you have all the needed packages is to use the included
`environment.yaml` file to create a new environment that will
contain all the optional dependencies along with dependencies required for
testing and development (```conda env create -f environment.yaml```).
Alternatively, you can specify `test`, `doc`, or `dev` when installing pyuvdata
(as in `pip install pyuvdata[dev]`) to install the packages needed for testing
(including coverage and linting) and documentation development;
`dev` includes everything in `test` and `doc`.

To use pre-commit to prevent committing code that does not follow our style, you'll
need to run `pre-commit install` in the top level `pyuvdata` directory.

## Tests
Uses the `pytest` package to execute test suite.
From the source pyuvdata directory run ```pytest``` or ```python -m pytest```.

Testing of `UVFlag` module requires the `pytest-cases` plug-in.

# API
pyuvdata is organized around objects that contain all the data and metadata required
to represent and work with interferometric data, calibration solutions, flags,
antenna beams and telescope layouts. Each object has the data and metadata as
attributes along with many useful methods for importing and exporting files and
manipulating and transforming the data in useful ways. Please see our extensive
documentation on [ReadTheDocs](https://pyuvdata.readthedocs.io) for tutorials and
complete API details.

# Maintainers
pyuvdata is maintained by the RASG Managers, which currently include:

 - Adam Beardsley (Winona State University)
 - Bryna Hazelton (University of Washington)
 - Garrett "Karto" Keating (Smithsonian Astrophysical Observatory)
 - Daniel Jacobs (Arizona State University)
 - Matt Kolopanis (Arizona State University)
 - Paul La Plante (University of Nevada, Las Vegas)
 - Jonathan Pober (Brown University)

Please use the channels discussed in the [guide on contributing](.github/CONTRIBUTING.md)
for code-related discussions. You can contact us privately if needed at
[rasgmanagers@gmail.com](mailto:rasgmanagers@gmail.com).

# Acknowledgments

Support for pyuvdata was provided by NSF awards #1835421 and #1835120.
