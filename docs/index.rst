.. pyuvdata documentation master file, created by
   make_index.py on 2016-11-29

pyuvdata
========

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
install. \* numpy >= 1.10 \* scipy \* astropy >= 1.2 \* pyephem \*
pyfits \* aipy

Install pyuvdata
----------------

Download the latest release here
github.com/HERA-Team/pyuvdata/release1.0

Navigate into the directory and run ``python setup.py install`` Note
that this will automatically install all dependencies. If you use
anaconda or another package manager you might prefer to do not do this.

To install without dependencies, run
``python setup.py install --no-dependencies``

Tests
-----

Requires installation of ``nose`` package. From the source pyuvdata
directory run ``nosetests``.

API
===

The primary interface to data from python is via the UVData object. It
provides import and export functionality to all supported file formats
(UVFITS, Miriad, FHD) and can be interacted with directly. The
attributes of the UVData object are described in the parameters description.

Further Documentation
====================================
.. toctree::
   :maxdepth: 1

   tutorial
   parameters
   developer_docs
