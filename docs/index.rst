.. pyuvdata documentation master file, created by
   make_index.py on 2017-04-14




pyuvdata
========

.. image:: https://travis-ci.org/HERA-Team/pyuvdata.svg?branch=cal
    :target: https://travis-ci.org/HERA-Team/pyuvdata

.. image:: https://coveralls.io/repos/github/HERA-Team/pyuvdata/badge.svg?branch=master
    :target: https://coveralls.io/github/HERA-Team/pyuvdata?branch=master


pyuvdata defines a pythonic interface to interferometric data sets.
Currently pyuvdata supports reading and writing of miriad and uvfits
files and reading of FHD (`Fast Holographic
Deconvolution <https://github.com/EoRImaging/FHD>`__) visibility save
files.

Motivation
==========

The three main goals are:

1. To provide a high quality, well documented path to convert between
   data formats
2. Support the direct use of datasets from python with minimal software
3. Provide precise data definition via both human readable code and high
   quality online documentation

Package Details
===============

Tested File Paths for interferometer data
-----------------------------------------

-  uvfits -> miriad (aipy)
-  miriad (aipy) -> uvfits
-  FHD -> uvfits
-  FHD -> miriad (aipy)

File formats for calibration data
---------------------------------

-  calfits (a new format defined in pyuvdata, detailed memo coming
   soon). Note that this format was recently defined and may change in
   coming versions, based on user needs. Consider it to be in a beta
   version, but we will strive to make future versions backwards
   compatible with the current format.


File standards
--------------

-  miriad is supported for aipy-style analysis, further testing is
   required for use in the miriad package
-  uvfits conforms to AIPS memo 117 (as of May 2015). It is tested
   against FHD, CASA, and AIPS. However AIPS is limited to <80 antennas
   and CASA imaging does not seem to support >255 antennas.
-  FHD (read-only support, tested against MWA and PAPER data)

Known Issues and Planned Improvements
-------------------------------------

-  different multiple spectral windows or multiple sources are not
   currently supported
-  testing against miriad package
-  replacing AIPY and pyephem with astropy+NOVAS for time and phase
   calculations
-  support for direct reading and writing of Measurement Sets
-  expand support for calibration solutions: support other formats
   beyond FITS

For details see the `issue
log <https://github.com/HERA-Team/pyuvdata/issues>`__.

Community Guidelines
--------------------

Contributions to this package to add new file formats or address any of
the issues in the `issue
log <https://github.com/HERA-Team/pyuvdata/issues>`__ are very welcome.
Please submit improvements as pull requests against the repo after
verifying that the existing tests pass and any new code is well covered
by unit tests.

Bug reports or feature requests are also very welcome, please add them
to the issue log after verifying that the issue does not already exist.
Comments on existing issues are also welcome.

History
=======

pyuvdata was originally developed in the low frequency 21cm community to
support the development of calibration and foreground subtraction
pipelines. Particular focus has been paid to supporting drift and phased
array modes.

Installation
============

Dependencies
------------

First install dependencies. The numpy and astropy versions are
important, so be sure to make sure these are up to date before you
install.

-  numpy >= 1.10
-  scipy
-  astropy >= 1.2
-  aipy

Install pyuvdata
----------------

For simple installation, the latest stable version is available via pip
using ``pip install pyuvdata``


Optionally install the development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the development version, clone the repository using
``git clone https://github.com/HERA-Team/pyuvdata.git``

Navigate into the directory and run ``python setup.py install``. Note
that this will automatically install all dependencies. If you use
anaconda or another package manager you might prefer not to do this.

To install without dependencies, run
``python setup.py install --no-dependencies``

Tests
-----

Requires installation of ``nose`` package. From the source pyuvdata
directory run ``nosetests pyuvdata``.

API
===

The primary interface to data from python is via the UVData object. It
provides import and export functionality to all supported file formats
(UVFITS, Miriad, FHD) and can be interacted with directly. The primary
calibration interface is via the UVCal object. The attributes of the
UVData and UVCal objects are described in the uvdata\_parameters and
uvcal\_parameters descriptions.

Further Documentation
====================================
.. toctree::
   :maxdepth: 1

   tutorial
   uvdata_parameters
   uvdata
   uvcal_parameters

   uvcal

   developer_docs
