Conventions
===========
Documentation for various conventions that are used or otherwise assumed within
pyuvdata objects and methods.


Baseline conjugation and *uvw*-direction
----------------------------------------
Because of the Hermitian nature of the *uv*-plane (arising from the fact that images of
the sky should contain no imaginary components), a single visibility at a given
(*u*, *v*, *w*) coordinate is simply the complex conjugate of the visibility at
(-*u*, -*v*, -*w*). Because the two values are highly redundant, most file formats and
software packages expect that only one of these values to be recorded, saving a factor
of two in memory usage and disk space in the processes. But which member of the pair
is stored depends on the file-format and/or software package.

There are two two standard conventions for this. The first is *ant2-ant1*, where a
given visibility :math:`\mathcal{V}` is calculated by taking the complex conjugate of
the data from the first antenna multiplied against the data from the second (i.e.,
:math:`\mathcal{V}_{12}=\langle V^{*}_1 V_{2} \rangle`), in which case the
*uvw*-coordinate is calculated by taking the position of the second antenna minus
the position of the first (after which various rotations are applied). The second is
*ant1-ant2*, where :math:`\mathcal{V}_{12}=\langle V_{1} V_{2}^{*} \rangle`, and the
*uvw*-coordinate is calculated by taking the position of the first antenna minus
the second. The pyuvdata software package uses the *ant2-ant1* convention, as does
MIRIAD; with UVFITS, FHD, and Mir formats use the opposite convention. MeasurementSet
format (used within CASA) appears to support both conventions, with the convention
selected based on the number of the antennas in the pair: when the antenna number of
*ant1* is greater than that of *ant2*, then the *ant2-ant1* convention is used,
otherwise the *ant1-ant2* convention is used (e.g., the 3-7 baseline would use the
*ant1-ant2* convention within CASA, whereas the 7-3 baseline would use the *ant2-ant1*
convention).

Phase Center Types
------------------
Several different pyuvdata classes have support for (and in the case of ``UVData``,
in fact require) handling information about the "phase center" of the telescope, which
records the position in the sky for which geometric delays of the interferometer have
been compensated for (and, if imaging, where the center of the field-of-view will
appear). There is support for four different "phase types", each of one which is
discussed further below:

  - ``"sidereal"``: A sidereal phase center is one whose position remains (relatively)
    static in a celestial frame, such as ICRS or FK5/J2000. Sidereal sources are
    permitted to have both proper motion and parallax (the former of which is
    calculated relative to the epoch date provided for the phase center), but otherwise
    is assumed that it can be described by a single coordinate pair.
  - ``"ephem"``: An ephem phase center is one whose position varies with time within
    a celestial frame, and whose exact position requires an ephemeris (which can be
    interpolated to provide exact position information for phasing).
  - ``"driftscan"``: A driftscan phase center is similar to a sidereal one, except that
    the phase center is assumed to be static within the observer frame, and therefore
    desribed by a single pair of horizontal coordinates (i..e, azimuth and elevation).
  - ``"unprojected"``: An unprojected phase center is a bit of an oxymoron, in that it
    denotes when no phasing has been applied (and therefore no phase center exists). In
    this context, geometric delays are *not* compensated for, but *uvw*-coordinates
    are calculated with respect to zenith for a given telescope.

Polarization Normalization
--------------------------
For an unpolarized source, a single-polarization receiver will typically detect half of
the Stokes I emission (e.g., an "xx"-polarization feed will physically detect 0.5 Jy of
a 1 Jy unpolarized point source). However, several software packages (including CASA and
MIRIAD) normalize to account for thus and double this quantity, such that a
single-polarization baseline is expressed as a Stokes I equivalent (e.g., an "xx"
baseline will show 1 Jy of flux for a 1 Jy unpolarized point source).

As mixing the two conventions can result in data having a factor of two error in
amplitude, both ``UVCal`` and ``UVData`` objects have the parameter ``pol_convention``,
which denotes the normalization behavior. The two supported options are ``"sum"``, where
Stokes I is the sum of fluxes measured on the XX and YY baselines for a linear
polarization system (i.e., I = XX + YY), or RR and LL baselines for a circular system;
and ``"mean"``, where Stokes I is the mean of the XX/YY or RR/LL baselines
(i.e, I = (XX + YY) / 2).

Feed angles and x-orientation
-----------------------------
As part of the ``Telescope`` object, one can specify ``Telescope.feed_type`` to specify
the handedness that a single-polarization receiver/input sees, which includes options
for both linear polarization receivers (``"x"`` and ``"y"``) and circular polarization
receivers (``"l"`` and ``"r"``). However, how individual feeds are oriented can impact
which polarization of light they see. A case in point: "x" and "y" polarizations through
a simple 90-degree rotation of the detector can either see vertically or horizontally
polarized light from the perspective of the observer. While this ambiguity is apparent
with linear feeds, it impacts measurements from circular feeds as well -- what is
ascribed to Stokes parameters Q and U (and correspondingly, the polarization angle
of a given source) can be modulated based on the orientation of the feeds.

To account for this potential ambiguity, there also exists a ``Telescope.feed_angle``
parameter, which is recorded in units of radians and preserves the orientation
information of the feeds. How this maps to polarization on sky depends on a few
different parameters (see the discussion about ``Telescope.mount_type`` below), but
to provide a few examples:

- For a Cassegrain antenna on an Alt-Az mount (like VLA), a 0 degree feed angle is in
  the direction of zenith.

- For a "fixed" antenna that points at zenith (like HERA), a 0 degree feed angle points
  in the direction of north.

In the absence of any information about the antenna mount, within pyuvdata a 0 degree
feed angle is one where the polarization response is aligned with the direction of
zenith for steerable antennas, otherwise toward north for stationary antennas. This is
equivalent to setting the now-defunct parameter ``Telescope.x_orientation`` to
``"north"`` when handling the x-polarization or ``"east"`` when handling the
y-polarization.

Feed angles are provided on a per-polarization basis, even if a given feed may be
dual-polarization. Nominally, feed angles for linearly polarized feeds should be
separated by 90 degrees (i.e., if ``Telescope.feed_array`` is ``[["x", "y"]]``, then
one might expect ``Telescope.feed_angle`` to be ``[[np.pi / 2, 0]]`` where the
"x-orientation" is pointing toward the east).

The ``UVBeam`` and ``AnalyticBeam`` objects also have ``feed_array``,
``feed_angle`` and ``mount_type`` attributes with the same meanings as on the
``Telescope`` object, except that they are for a single antenna beam so do not
have the ``Nants`` length axis that they have on the ``Telescope`` object.

Azimuth and Pixel Coordinate Systems in ``UVBeam``
--------------------------------------------------
For describing antenna beams, the ``UVBeam`` object provides several different choices
for the coordinate system in which the beam data are laid out, as recorded in
``UVBeam.pixel_coordinate_systems``. However, across all conventions, local east is
ascribed as the zero-point for the coordinate system, increasing as one moves to north
(90 degrees). It is important to note that this is different than some other parameters,
such as ``UVBeam.feed_angle``, where for an antenna pointed toward zenith, 0 corresponds
to local north, and 90 degrees points to local east.

Note that ``UVBeam.feed_angle`` specifies the angles of the feeds as discussed
above and therefore which polarization each feed is sensitive to. This should not
be confused with the ``UVBeam.axis1_array`` and ``UVBeam.axis2_array`` attributes
(available on beams represented in regularly gridded coordinate systems) which
are the coordinates for the directions on the sky that the beam values are provided.
These coordinates are the same for all feeds and polarizations represented in
the beam.

Telescope Mount Types
---------------------
As part of the ``Telescope`` object, one can specify ``Telescope.mount_type``, which
records information about the optics of the antenna. The telescope optics can impact
data in multiple ways, though we focus our discussion here on its impact for polarimetric
measurements, since a "static" single-polarization receiver can be sensitive to
differing polarizations of light (depending on the optics type and where the source is
in the sky from the persepctive of the observer).


One good reference that covers several of the mount types listed below is
`Dodson and Rioja (2022) <https://arxiv.org/abs/2210.13381>`_, as well as the
`Wikipedia page on telescope mounts <https://en.wikipedia.org/wiki/Telescope_mount>`_
(as well as the page on `Nasmyth telescopes <https://en.wikipedia.org/wiki/Nasmyth_telescope>`_).
We limit our focus here to those mount types presently supported in pyuvdata, which
include:

  - ``"alt-az"``: Arguably the most common form of antenna mount, where the two axes of
    rotation rotate in azimuth and elevation (i.e., "altitude"), and "up" from the
    perspective of the receiver points toward zenith. For a mount of this type, a feed
    angle of 0 degrees is aligned with the parallactic angle at the given position of
    the sky.
  - ``"equatorial"``: Similar to "alt-az", but where one axis of rotation is aligned
    to the celestial equatorial plane (i.e., where the declination is zero), such that
    the two axes of rotation are aligned to the right ascension/hour angle and
    declination axes. For a mount of this type, a feed angle of 0 degrees is aligned
    with north on the (apparent) celestial sphere.
  - ``"orbiting"``: Denotes that the antenna is in orbit, such that the orientation of
    the antenna changes with time. This mode appears to have been added to UVFITS to
    support the `VLBI Space Observatory Programme <https://en.wikipedia.org/wiki/HALCA>`_.
    pyuvdata does not currently support additional orbital parameters.
  - ``"x-y"``: A mount type that is somewhat prevalent for tracking of LEO and MEO
    satellites due to its ability to `track more easily through zenith than "alt-az"
    mounts <https://ntrs.nasa.gov/api/citations/19650021134/downloads/19650021134.pdf>`_,
    this has one rotation axis aligned to local north-south, and the other lies in the
    plane of the great circle connecting local east-west through zenith. For this mount
    type, a feed angle of 0 degrees varies with sky position, aligned to
    :math:`\arctan(\cos(\textrm{HA}) / (\sin(\textrm{HA}) * \sin(\delta)))`,
    where :math:`\delta` is the declination of the source and HA the hour angle.
  - ``"alt-az+nasmyth-r"``: Similar to "alt-az", with the addition of a flat tertiary
    mirror that allows the detector to sit at a fixed elevation while the antenna primary
    moves up and down in elevation. For this "right-handed" Nasmyth variant, a feed angle
    of 0 degrees will be aligned to sum of the parallactic and elevation angles.
  - ``"alt-az+nasmyth-l"``: The "left-handed" variant of the Nasmyth mount, where a
    feed angle of 0 degrees is aigned to the **difference** of the parallactic and
    elevation angles.
  - ``"phased"``: Denotes an instrument where the an individual antenna input is
    a phased array of detectors that are "beamformed" into a single voltage stream
    (otherwise sometimes referred to as "electronically steered"). For this mount, a
    0-degree feed angle is aligned/parallel to the direction of local north. Note
    while supported in UVFITS, it is a later addition to the format, and may be grouped
    into "other" in some software packages.
  - ``"fixed"``: Similar to "phased", except that there is neither mechanical or
    electical steering of the antenna, and thus the beam remains fixed in the
    azimuth-elevation frame. In this frame, a feed angle of 0 degrees is aligned/
    parallel to the direction of local north. Note that this is a pyuvdata-defined mount
    type, and does not necessarily have a corresponding entry in, for example, UVFITS.
  - ``"other"``  While nominally a pyuvdata-defined mount type, UVFITS and CASA both
    allow for the designation of "bizarre" mount-types, which for all intents and purposes
    denotes the same lack of knowledge of underlying optics behavior.
