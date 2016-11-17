# pyuvdata
pyuvdata defines a pythonic interface to interferometric data sets. Currently pyuvdata supports reading and writing of miriad and uvfits files and reading of FHD ([Fast Holographic Deconvolution](https://github.com/EoRImaging/FHD)) visibility save files.   


# Motivation
The three main goals are:

1. To provide a high quality, well documented path to convert between data formats
2. Support the direct use of datasets from python with minimal software
3. Provide precise data definition via both human readable code and high quality online documentation

# History
pyuvdata was originally developed in the low frequency 21cm community to support the development of calibration and foreground subtraction pipelines. Particular focus has been paid to supporting drift and phased array modes.




# Installation
## Dependencies
First install dependencies. The numpy and astropy versions are important, so be sure to make sure these are up to date before you install.
* numpy >= 1.10
* scipy
* astropy >= 1.2
* pyephem
* pyfits
* aipy

## Install pyuvdata
Download the latest release here github.com/HERA-Team/pyuvdata/release1.0

Navigate into the directory and run 
`python setup.py install` Note that this will automatically install all dependencies.  If you use anaconda or another package manager you might prefer to do not do this.

To install without dependencies, run
`python setup.py install --no-dependencies`

## Tests
Requires installation of `nose` package.
From the source pyuvdata directory run `nosetests`. 


# API
The primary interface to data from python is via the UVData object. This is described in detail at https://pyuvdata.readthedocs.io or [here](https://github.com/HERA-Team/pyuvdata/blob/master/docs/parameters.rst).
