---
title: 'pyuvdata v3: an interface for astronomical interferometeric datasets in python'
tags:
 - radio astronomy
 - uvfits
 - miriad
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
 - name: Paul La Plante
   affiliation: 8
   orcid: 0000-0002-4693-0102
 - name: Jonathan C. Pober
   orcid: 0000-0002-3492-0433
   affiliation: 9
 - name: Pyxie Star
   orcid: XXXX-XXXX-XXXX-XXXX
   affiliation: 2
affiliations:
 - name: Center for Astrophysics | Harvard & Smithsonian
   index: 1
 - name: University of Washington, eScience Institute
   index: 2
 - name: University of Washington, Physics Department
   index: 3
 - name: Arizona State University, School of Earth and Space Exploration
   index: 4
 - name: Winona State University, Physics Department
   index: 5
 - name: Massachusetts Institute of Technology, Physics Department
   index: 6
 - name: Kavli Institute of Astrophysics and Space Research
   index: 7
 - name: University of Nevada, Las Vegas, Department of Computer Science
   index: 8
 - name: Brown University, Physics Department
   index: 9
date: 2 July 2024
bibliography: paper.bib
---

# Summary
pyuvdata is an open-source software package that seeks to provide a well-documented,
feature-rich interface for many of the different data formats that exist within radio
interferometry, including support for reading and writing UVH5 [@uvh5], UVFITS
[@uvfits], MIRIAD [@miriad], and measurement set [@ms] visibility files; and reading of
FHD [@fhd] and MIR [@mir] visibility save files. Additionally, pyuvdata supports reading
and writing measurement set, CalFITS [@calfits], and CalH5 (Hazelton et al., _in prep_)
calibration tables; and reading of FHD calibration tables. pyuvdata also provides
interfaces for and handling of models of antenna primary beams, including reading and
writing of BeamFITS [@beamfits] and MWA-formatted beam models, as well as for flags
tables.

# Statement of Need
There are several standard formats for astronomical interferometric data, but
converting between them in a stable, repeatable way has historically been
challenging.  This is partially due to conflicting assumptions and standards, giving
rise to significant (though sometimes subtle) differences between formats.
Interfacing with different data formats -- like one does when they convert from one
format to another -- thus requires careful accounting for the complex mathematical
relationships between both data and metadata to ensure proper data fidelity. This is
required both for leveraging existing community-favored tools that typically built
to interface with a specific data format, as well as analyses requiring bespoke tools
for specialized types of analyses and simulations leveraging data in a variety of
formats.

pyuvdata has been designed to facilitate interoperability between different instruments
and codes by providing high quality, well documented conversion routines as well as an
interface to interact with interferometric data and simulations directly in Python.
Originally motivated to support new low frequency instruments (e.g. MWA
(http://www.mwatelescope.org/), PAPER (http://eor.berkeley.edu/), HERA
(http://reionization.org/)), the capabilities of pyuvdata have been steadily expanded
to support handling of data from several telescopes, ranging from meter to submillimeter
wavelengths.

# References
