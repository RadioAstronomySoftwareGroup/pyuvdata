------
UVBeam
------

UVBeam objects hold all of the metadata and data required to work with primary beam
models for radio telescopes. UVBeam supports both E-field and power beams
(labelled by the ``beam_type`` attribute) in a few different coordinate and gridding
systems (e.g. regular azimuth/zenith angle grids, zenith-based HEALPix grids, labelled
by the ``pixel_coordinate_system`` attribute).


UVBeam: Reading/writing
-----------------------
Reading and writing beam files using UVBeam.

The text files saved out of CST beam simulations do not have much of the
critical metadata needed for UVBeam objects. When reading in CST files, you
can either provide the required metadata using keywords to the :meth:`pyuvdata.UVBeam.read` method
and pass the raw CST files, or you can pass a settings yaml file which lists
the raw files and provides the required metadata to the :meth:`pyuvdata.UVBeam.read` method. Both
options are shown in the examples below. More details on creating a new yaml
settings files can be found in :doc:`cst_settings_yaml`.

UVBeam (like UVData) has a generic :meth:`pyuvdata.UVBeam.read` method and a
:meth:`pyuvdata.UVBeam.from_file` class method in addition to the file-type specific
methods.

a) Reading a CST power beam file
********************************
.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> beam = UVBeam()

  # you can pass several filenames and the objects from each file will be
  # combined across the appropriate axis -- in this case frequency
  >>> filenames = [os.path.join(DATA_PATH, f) for f
  ...              in ['NicCSTbeams/HERA_NicCST_150MHz.txt',
  ...                  'NicCSTbeams/HERA_NicCST_123MHz.txt']]

  # You have to specify the telescope_name, feed_name, feed_version, model_name
  # and model_version because they are not included in the raw CST files.
  # You should also specify the polarization that the file represents and you can
  # set rotate_pol to generate the other polarization by rotating by 90 degrees.
  # The feed_pol defaults to 'x' and rotate_pol defaults to True.
  >>> beam.read(
  ...   filenames, beam_type='power', frequency=[150e6, 123e6],
  ...   feed_pol='x', rotate_pol=True, telescope_name='HERA',
  ...   feed_name='PAPER_dipole', feed_version='0.1',
  ...   model_name='E-field pattern - Rigging height 4.9m',
  ...   model_version='1.0'
  ... )
  >>> print(beam.beam_type)
  power
  >>> print(beam.pixel_coordinate_system)
  az_za
  >>> print(beam.data_normalization)
  physical

  >>> # You can also use a yaml settings file.
  >>> # Note that using a yaml file requires that pyyaml is installed.
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='power')
  >>> print(beam.beam_type)
  power
  >>> print(beam.pixel_coordinate_system)
  az_za
  >>> print(beam.data_normalization)
  physical

  >>> # number of beam polarizations and polarization type.
  >>> print((beam.Npols, beam.polarization_array))
  (2, array([-5, -6]))
  >>> print(beam.Nfreqs)
  2
  >>> print(beam.data_array.shape)
  (1, 1, 2, 2, 181, 360)

  >>> # plot zenith angle cut through beam
  >>> plt.plot(beam.axis2_array, beam.data_array[0, 0, 0, 0, :, 0]) # doctest: +SKIP
  >>> plt.xscale('log') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Power') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

b) Reading a CST E-field beam file
**********************************
.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import numpy as np
  >>> beam = UVBeam()

  >>> # the same interface as for power beams, just specify beam_type = 'efield'
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='efield')
  >>> print(beam.beam_type)
  efield

  >>> # UVBeam also has a `from_file` class method we can call directly.
  >>> beam3 = UVBeam.from_file(settings_file, beam_type="efield")
  >>> beam == beam3
  True

c) Reading in the MWA full embedded element beam
************************************************
.. code-block:: python

  >>> # To get all the frequencies available for the MWA full embedded element beam
  >>> # you need to download the output simulation file via
  >>> # `wget http://cerberus.mwa128t.org/mwa_full_embedded_element_pattern.h5`
  >>> # For this tutorial we use the file saved in the test data which only
  >>> # contains a few frequencies.
  >>> # The `read_mwa_beam` method takes delay and amplitude arrays to generate beams
  >>> # pointed any where or with varying gains per dipole. Set a delay to 32
  >>> # to get a beam where that dipole is turned off (or set the amplitude to zero).
  >>> # The native format of the beam is spherical harmonic modes, so there is also
  >>> # an option `pixels_per_deg` to set the output beam resolution
  >>> # (default is 5 pixels per degree).

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> beam = UVBeam()

  >>> mwa_beam_file = os.path.join(DATA_PATH, 'mwa_full_EE_test.h5')
  >>> beam.read(mwa_beam_file)
  >>> print(beam.beam_type)
  efield

  >>> delays = np.zeros((2, 16), dtype='int')
  >>> delays[:, 0] = 32
  >>> beam.read(mwa_beam_file, pixels_per_deg=1, delays=delays)


d) Writing a regularly gridded beam FITS file
**********************************************
.. code-block:: python

  >>> import os
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='power')
  >>> write_file = os.path.join('.', 'tutorial.fits')
  >>> beam.write_beamfits(write_file, clobber=True)

e) Writing a HEALPix beam FITS file
***********************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='power')

  >>> # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'

  >>> # note that the `to_healpix` method requires astropy_healpix to be installed
  >>> # this beam file is very large. Let's cut down the size to ease the computation
  >>> # More on the `select` method is covered in the following section
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> beam.to_healpix()
  >>> write_file = os.path.join('.', 'tutorial.fits')
  >>> beam.write_beamfits(write_file, clobber=True)

UVBeam: Selecting data
----------------------
The :meth:`pyuvdata.UVBeam.select` method lets you select specific image axis indices
(or pixels if pixel_coordinate_system is HEALPix), frequencies and feeds
(or polarizations if beam_type is power) to keep in the object while removing others.

a) Selecting a range of Zenith Angles
*************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='power')
  >>> new_beam = beam.select(axis2_inds=np.arange(0, 20), inplace=False)

  >>> # plot zenith angle cut through beams
  >>> plt.plot(beam.axis2_array, beam.data_array[0, 0, 0, 0, :, 0], # doctest: +SKIP
  ...         new_beam.axis2_array, new_beam.data_array[0, 0, 0, 0, :, 0], 'r')
  >>> plt.xscale('log') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Power') # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

a) Selecting Feeds or Polarizations
***********************************
Selecting feeds on E-field beams can be done using the feed name (e.g. "x" or "y"). If
``x_orientation`` is set on the object, strings represting the physical orientation of
the feed can also be used (e.g. "n" or "e).

Selecting polarizations on power beams can be done either using the polarization
numbers or the polarization strings (e.g. "xx" or "yy" for linear polarizations or
"rr" or "ll" for circular polarizations). If ``x_orientation`` is set on the object,
strings represting the physical orientation of the dipole can also be used (e.g. "nn"
or "ee).

.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> import pyuvdata.utils as uvutils
  >>> uvb = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> uvb.read(settings_file, beam_type='efield')

  >>> # The feeds names can be found in the feed_array
  >>> print(uvb.feed_array)
  ['x' 'y']

  >>> # make a copy and select a feed
  >>> uvb2 = uvb.copy()
  >>> uvb2.select(feeds=["y"])
  >>> print(uvb2.feed_array)
  ['y']

  >>> # check the x_orientation
  >>> print(uvb.x_orientation)
  east

  >>> # make a copy and select a feed by phyiscal orientation
  >>> uvb2 = uvb.copy()
  >>> uvb2.select(feeds=["n"])
  >>> print(uvb2.feed_array)
  ['y']

  >>> # convert to a power beam for selecting on polarizations
  >>> uvb.efield_to_power()
  >>> # polarization numbers can be found in the polarization_array
  >>> print(uvb.polarization_array)
  [-5 -6 -7 -8]

  >>> # polarization numbers can be converted to strings using a utility function
  >>> print(uvutils.polnum2str(uvb.polarization_array))
  ['xx', 'yy', 'xy', 'yx']

  >>> # select polarizations using the polarization numbers
  >>> uvb.select(polarizations=[-5, -6, -7])

  >>> # print polarization numbers and strings after select
  >>> print(uvb.polarization_array)
  [-5 -6 -7]
  >>> print(uvutils.polnum2str(uvb.polarization_array))
  ['xx', 'yy', 'xy']

  >>> # select polarizations using the polarization strings
  >>> uvb.select(polarizations=["xx", "yy"])

  >>> # print polarization numbers and strings after select
  >>> print(uvb.polarization_array)
  [-5 -6]
  >>> print(uvutils.polnum2str(uvb.polarization_array))
  ['xx', 'yy']

  >>> # print x_orientation
  >>> print(uvb.x_orientation)
  east

  >>> # select polarizations using the physical orientation strings
  >>> uvb.select(polarizations=["ee"])

  >>> # print polarization numbers and strings after select
  >>> print(uvb.polarization_array)
  [-5]
  >>> print(uvutils.polnum2str(uvb.polarization_array))
  ['xx']


UVBeam: Interpolating to HEALPix
--------------------------------
Note that interpolating from one gridding format to another incurs interpolation
errors. If the beam is going to be interpolated (e.g. to source locations) in
downstream code we urge the user use the beam in the original format to avoid incurring
extra interpolation errors.

.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from astropy_healpix import HEALPix
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from matplotlib.colors import LogNorm # doctest: +SKIP
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='power')

  >>> # have to specify which interpolation function to use
  >>> beam.interpolation_function = 'az_za_simple'

  >>> # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> hpx_beam = beam.to_healpix(inplace=False)
  >>> hpx_obj = HEALPix(nside=hpx_beam.nside, order=hpx_beam.ordering)
  >>> lon, lat = hpx_obj.healpix_to_lonlat(hpx_beam.pixel_array)
  >>> plt.scatter(lon, lat, c=hpx_beam.data_array[0,0,0,0,:], norm=LogNorm()) # doctest: +SKIP


UVBeam: Converting from E-Field beams to Power Beams
----------------------------------------------------

a) Convert a regularly gridded efield beam to a power beam (leaving original intact).
*************************************************************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='efield')
  >>> new_beam = beam.efield_to_power(inplace=False)

  >>> # plot zenith angle cut through the beams
  >>> plt.plot(beam.axis2_array, beam.data_array[1, 0, 0, 0, :, 0].real, label='E-field real') # doctest: +SKIP
  >>> plt.plot(beam.axis2_array, beam.data_array[1, 0, 0, 0, :, 0].imag, 'r', label='E-field imaginary') # doctest: +SKIP
  >>> plt.plot(new_beam.axis2_array, np.sqrt(new_beam.data_array[0, 0, 0, 0, :, 0]), 'black', label='sqrt Power') # doctest: +SKIP
  >>> plt.xlabel('Zenith Angle (radians)') # doctest: +SKIP
  >>> plt.ylabel('Magnitude') # doctest: +SKIP
  >>> plt.legend() # doctest: +SKIP
  >>> plt.show() # doctest: +SKIP

b) Generating pseudo Stokes ('pI', 'pQ', 'pU', 'pV') beams
**********************************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt # doctest: +SKIP
  >>> from matplotlib.colors import LogNorm # doctest: +SKIP
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> from pyuvdata import utils as uvutils
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='efield')
  >>> beam.interpolation_function = 'az_za_simple'

  >>> # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> pstokes_beam = beam.efield_to_pstokes(inplace=False)

  >>> # plotting pseudo-stokes I
  >>> pol_array = pstokes_beam.polarization_array
  >>> pstokes = uvutils.polstr2num('pI')
  >>> pstokes_ind = np.where(np.isin(pol_array, pstokes))[0][0]
  >>> azimuth, za = np.meshgrid(pstokes_beam.axis1_array, pstokes_beam.axis2_array)
  >>> plt.scatter(azimuth, 1-za, c=np.abs(pstokes_beam.data_array[0, 0, pstokes_ind, 0, :]), norm=LogNorm()) # doctest: +SKIP


UVBeam: Calculating beam areas
------------------------------
Calculations of the beam area and beam squared area are frequently required inputs for
Epoch of Reionization power spectrum calculations. These areas can be calculated for
either instrumental or pseudo Stokes beams using the :meth:`pyuvdata.UVBeam.get_beam_area`
and :meth:`pyuvdata.UVBeam.get_beam_sq_area` methods. Currently these methods do require
that the beams are in Healpix coordinates in order to take advantage of equal pixel areas.
They can be interpolated to HEALPix using the :meth:`pyuvdata.UVBeam.to_healpix` method.

a) Calculating pseudo Stokes ('pI', 'pQ', 'pU', 'pV') beam area and beam squared area
*************************************************************************************
.. code-block:: python

  >>> import os
  >>> import numpy as np
  >>> from pyuvdata import UVBeam
  >>> from pyuvdata.data import DATA_PATH
  >>> beam = UVBeam()
  >>> settings_file = os.path.join(DATA_PATH, 'NicCSTbeams/NicCSTbeams.yaml')
  >>> beam.read(settings_file, beam_type='efield')
  >>> beam.interpolation_function = 'az_za_simple'

  >>> # note that the `to_healpix` method requires astropy_healpix to be installed
  >>> # this beam file is very large. Let's cut down the size to ease the computation
  >>> za_max = np.deg2rad(10.0)
  >>> za_inds_use = np.nonzero(beam.axis2_array <= za_max)[0]
  >>> beam.select(axis2_inds=za_inds_use)

  >>> pstokes_beam = beam.to_healpix(inplace=False)
  >>> pstokes_beam.efield_to_pstokes()
  >>> pstokes_beam.peak_normalize()

  >>> # calculating beam area
  >>> freqs = pstokes_beam.freq_array
  >>> pI_area = pstokes_beam.get_beam_area('pI')
  >>> pQ_area = pstokes_beam.get_beam_area('pQ')
  >>> pU_area = pstokes_beam.get_beam_area('pU')
  >>> pV_area = pstokes_beam.get_beam_area('pV')
  >>> pI_area1, pI_area2 = round(pI_area[0].real, 5), round(pI_area[1].real, 5)
  >>> pQ_area1, pQ_area2 = round(pQ_area[0].real, 5), round(pQ_area[1].real, 5)
  >>> pU_area1, pU_area2 = round(pU_area[0].real, 5), round(pU_area[1].real, 5)
  >>> pV_area1, pV_area2 = round(pV_area[0].real, 5), round(pV_area[1].real, 5)

  >>> print (f'Beam area at {freqs[0][0]*1e-6} MHz for pseudo-stokes\nI: {pI_area1}\nQ: {pQ_area1}\nU: {pU_area1}\nV: {pV_area1}')
  Beam area at 123.0 MHz for pseudo-stokes
  I: 0.04674
  Q: 0.02904
  U: 0.02879
  V: 0.0464


  >>> print (f'Beam area at {freqs[0][1]*1e-6} MHz for pseudo-stokes\nI: {pI_area2}\nQ: {pQ_area2}\nU: {pU_area2}\nV: {pV_area2}')
  Beam area at 150.0 MHz for pseudo-stokes
  I: 0.03237
  Q: 0.01995
  U: 0.01956
  V: 0.03186


  >>> # calculating beam squared area
  >>> freqs = pstokes_beam.freq_array
  >>> pI_sq_area = pstokes_beam.get_beam_sq_area('pI')
  >>> pQ_sq_area = pstokes_beam.get_beam_sq_area('pQ')
  >>> pU_sq_area = pstokes_beam.get_beam_sq_area('pU')
  >>> pV_sq_area = pstokes_beam.get_beam_sq_area('pV')
  >>> pI_sq_area1, pI_sq_area2 = round(pI_sq_area[0].real, 5), round(pI_sq_area[1].real, 5)
  >>> pQ_sq_area1, pQ_sq_area2 = round(pQ_sq_area[0].real, 5), round(pQ_sq_area[1].real, 5)
  >>> pU_sq_area1, pU_sq_area2 = round(pU_sq_area[0].real, 5), round(pU_sq_area[1].real, 5)
  >>> pV_sq_area1, pV_sq_area2 = round(pV_sq_area[0].real, 5), round(pV_sq_area[1].real, 5)

  >>> print (f'Beam squared area at {freqs[0][0]*1e-6} MHz for pseudo-stokes\nI: {pI_sq_area1}\nQ: {pQ_sq_area1}\nU: {pU_sq_area1}\nV: {pV_sq_area1}')
  Beam squared area at 123.0 MHz for pseudo-stokes
  I: 0.02474
  Q: 0.01186
  U: 0.01179
  V: 0.0246


  >>> print (f'Beam squared area at {freqs[0][1]*1e-6} MHz for pseudo-stokes\nI: {pI_sq_area2}\nQ: {pQ_sq_area2}\nU: {pU_sq_area2}\nV: {pV_sq_area2}')
  Beam squared area at 150.0 MHz for pseudo-stokes
  I: 0.01696
  Q: 0.00798
  U: 0.00792
  V: 0.01686
