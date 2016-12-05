---
title: 'pyuvdata: an interface for astronomical interferometeric datasets in python'
tags:
 - radio astronomy
 - uvfits
 - miriad
authors:
 - name: Bryna J. Hazelton
   orcid: 0000-0001-7532-645X
   affiliation: 1, 2
 - name: Daniel C. Jacobs
   orcid: 0000-0002-0917-2269
   affiliation: 3
 - name: Jonathan C. Pober
   orcid: 0000-0002-3492-0433
   affiliation: 4
 - name: Adam P. Beardsley
   orcid: 0000-0001-9428-8233
   affiliation: 3
affiliations:
 - name: University of Washington, eScience Institute
   index: 1
 - name: University of Washington, Physics Department
   index: 2
 - name: Arizona State University, School of Earth and Space Exploration
   index: 3
 - name: Brown University, Physics Department
   index: 4
date: 22 November 2016
bibliography: paper.bib
---

# Summary

There are several standard formats for astronomical interferometric data, but
converting between them in a stable and repeatable way has historically been
very challenging. This is partly because of subtle assumptions in the
implementations of the formats and the complexity of the mathematical
relationships between the different formats (e.g. drift mode vs phased data)
and partly because data analysis for individual telescopes
typically used just one of the standards along with the associated analysis
code. New low frequency instruments (e.g. MWA (http://www.mwatelescope.org/),
PAPER (http://eor.berkeley.edu/), HERA (http://reionization.org/)),
have required custom analysis and simulation software that rely on a range of
different file formats. pyuvdata was designed to facilitate interoperability
between these instruments and codes by providing high quality, well documented
conversion routines as well as an interface to interact with interferometric
data and simulations directly in python.

pyuvdata currently supports reading and writing uvfits [@uvfits] and
miriad [@miriad] files and reading FHD [@fhd] visibility save files.

# References
