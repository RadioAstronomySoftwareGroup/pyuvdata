# pyuvdata
pyuvdata defines a pythonic interface to interferometric data sets. Currently pyuvdata supports reading and writing of miriad and uvfits.   


# Motivation
The two main goals are
1. to provide a high quality, well documented path to convert between datasets
2. support the direct use of datasets from python with minimal software
3. provide precise data definition via both human readable code and high quality online documentation

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

Navigate into the directory and run.
`python setup.py install  #note that this will automatically` install all dependencies.  If you use anaconda or another package manager you might prefer to do not do this.

To install without dependencies, run
`python setup.py install --no-dependencies`

## Tests
** requires installation of `nose` package.
From the source pyuvdata directory run `nosetests`. 


# API
One interfaces to data from python via the UVData object. This is described at pyuvdata.readthedocs.io
