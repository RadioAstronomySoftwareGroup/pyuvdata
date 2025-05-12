---
title: 'pyuvdata v3: an interface for astronomical interferometric data sets in Python'
tags:
 - radio astronomy
 - UVFITS
 - MIRIAD
authors:
 - name: Garrett K. Keating
   orcid: 0000-0002-3490-146X
   affiliation: 1
 - name: Bryna J. Hazelton
   orcid: 0000-0001-7532-645X
   affiliation: 2, 3
 - name: Matthew Kolopanis
   orcid: 0000-0002-2950-2974
   affiliation: 4
 - name: Steven Murray
   orcid: 0000-0003-3059-3823
   affiliation: 4
 - name: Adam P. Beardsley
   orcid: 0000-0001-9428-8233
   affiliation: 5
 - name: Daniel C. Jacobs
   orcid: 0000-0002-0917-2269
   affiliation: 3
 - name: Nicholas Kern
   affiliation: 6
   orcid: 0000-0002-8211-1892
 - name: Adam Lanman
   affiliation: 7
   orcid: 0000-0003-2116-3573
 - given-names: Paul
   surname: La Plante
   affiliation: 8
   orcid: 0000-0002-4693-0102
 - name: Jonathan C. Pober
   orcid: 0000-0002-3492-0433
   affiliation: 9
 - name: Pyxie Star
   affiliation: 3
affiliations:
 - name: Center for Astrophysics | Harvard & Smithsonian, USA
   index: 1
 - name: eScience Institute, University of Washington, USA
   index: 2
 - name: Physics Department, University of Washington, USA
   index: 3
 - name: Scuola Normale Superiore, Italy
   index: 4
 - name: Physics Department, Winona State University, USA
   index: 5
 - name: Physics Department, Massachusetts Institute of Technology, USA
   index: 6
 - name:  Kavli Institute for Astrophysics and Space Research, Massachusetts Institute of Technology, USA
   index: 7
 - name: Department of Computer Science, University of Nevada, Las Vegas, USA
   index: 8
 - name: Physics Department, Brown University, USA
   index: 9
date: 2 July 2024
bibliography: paper.bib
---

# Summary
pyuvdata is an open-source software package that seeks to provide a well-documented,
feature-rich interface for many of the different data formats that exist within radio
interferometry, including support for reading and writing the following formats:
UVH5 [@uvh5], UVFITS [@uvfits], MIRIAD [@miriad], and measurement set [@ms] visibility
files. It offers read-only support for fast holographic deconvolution [FHD, @fhd] and MIR [@mir] visibility save files.
Additionally, pyuvdata supports reading/writing measurement set, CalFITS [@calfits], and
CalH5 [@calh5] calibration solutions; and reading of FHD calibration solutions. pyuvdata
also provides interfaces for and handling of models of antenna primary beams, including
BeamFITS [read and write, @beamfits], CST (read-only) and MWA beam formats (read-only).
It also provides interfaces for handling of data flags.

# Statement of Need
There are several standard formats for astronomical interferometric data, but
translating between them in a robust and well-understood way has historically been
challenging.  This is partially due to conflicting assumptions and standards, giving
rise to significant (though sometimes subtle) differences between formats.
Interfacing with different data formats---like one does when they convert from one
format to another---thus requires careful accounting for the complex mathematical
relationships between both data and metadata to ensure proper data fidelity. This is
required both for leveraging existing community-favored tools that are typically built
to interface with a specific data format, as well as analyses requiring bespoke tools
for specialized types of analyses and simulations leveraging data in a variety of
formats.

pyuvdata has been designed to facilitate interoperability between different instruments
and codes by providing high-quality, well-documented conversion routines as well as an
interface to interact with interferometric data and simulations directly in Python.
Originally motivated to support new low frequency instruments
(e.g.~[MWA](http://www.mwatelescope.org/), [PAPER](http://eor.berkeley.edu/), [HERA](http://reionization.org/)),
the capabilities of pyuvdata have been steadily expanded
to support handling of data from several telescopes, ranging from meter to submillimeter
wavelengths (including [SMA](https://cfa.harvard.edu/sma), ALMA, VLA, ATCA, CARMA,
LWA, among others).

# Major updates in this version
In the time since it was initially published [@pyuvdata_v1], pyuvdata has undergone a
significant expansion in capabilities. In addition to general performance improvements
and restructuring, the newest version of pyuvdata includes several new major features,
including:

- The addition of the `UVCal` class, which provides a container for handling calibration
solutions (bandpass, delays, and gains) for interferometric data. Supported data formats
include MS, FHD, CalFITS, and CalH5.
- The addition of the `UVBeam` class, which provides a container for handling models
of the primary beam for antennas within an interferometric array. Supported data formats
include BeamFITS, MWA, and CST.
- The addition of the `UVFlag` class, which provides a container for handling flags/masking
of bad data for visibility data.
- Drastically improved handling of astrometry.
- Increased speed and accuracy of algorithms used to ``phase-up'' data (i.e., change
the sky position upon which the interferometer is centered).
- Support for several new visibility data formats, including MIR, MS, and MWA/MWAX.
- Support for data sets containing multiple spectral windows.
- Support for data sets containing observations of multiple sources/phase centers.
- Many new convenience methods for working with interferometric data, including
splitting and combining data sets, averaging in time and frequency, and applying
calibration solutions and flags.

# Acknowledgements
This work was supported by the National Science Foundation (AST-1835421); and by the
Submillimeter Array, a joint project between the Smithsonian Astrophysical Observatory
and the Academia Sinica Institute of Astronomy and Astrophysics, and funded by the
Smithsonian Institution and the Academia Sinica.

# References
