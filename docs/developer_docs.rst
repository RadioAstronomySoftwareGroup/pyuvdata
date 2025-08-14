Developer Docs
==============

Test data management
--------------------
All our test data is hosted in the
`RASG datasets repo <https://github.com/RadioAstronomySoftwareGroup/rasg-datasets/>`__.
We use pooch via the :func:`pyuvdata.datasets.fetch_data` function to download and
cache the test data in our tests. If you need to add new test data the steps are:

- Make a PR on the rasg-datasets repo adding the test data. Make sure to update
  the readme in the folder to give some provenance information for the test data
  that you are adding. Note that datasets that contain multiple files should be
  tarred and gzipped into one file.

- Once the PR is merged, we need to make a new release of that repo in order to
  be able to use it in tests. Update ``pyuvdata.datasets.py`` with the new
  version number for the rasg-datasets repo.

- Add the new test file information to the ``test_data_registry.txt`` and
  ``test_data.yaml`` files in the ``src/pyuvdata/data`` folder. The registry file
  requires a checksum which can be calculated with ``sha256sum <your_new_file>``.
  The yaml maps the test file to a nickname used in our tests. The nickname
  should include the telescope name and descriptive information about the dataset.

- Call :func:`pyuvdata.datasets.fetch_data` with the nickname in any test
  requiring your new dataset to get the path to the dataset on disk. We strongly
  suggest setting up fixtures for new datasets, particularly if there are multiple
  tests that use them. Please avoid calling ``fetch_data`` in pytest.parametrize
  decorators to avoid invoking it during test setup, instead call it inside a
  fixture or test function.

Under-the-hood classes and functions
------------------------------------

Documentation for all the under-the-hood classes and functions that most users
won't need to interact with.

Base Classes
************
These classes are the under-the-hood classes that provide much of the
infrastructure for pyuvdata's user classes.

``UVParameter`` is the base class for the attributes on pyuvdata's user classes,
allowing the attributes to carry information like expected type and shape,
acceptable values or ranges and tolerances. We also have some type specific
parameter objects (e.g. for angles and earth locations) with extra functionality.

``UVBase`` is the base class for pyuvdata's user classes which has the common
infrastructure to turn the UVParameter attributes into properties and check the
attribute shapes and values.

.. autoclass:: pyuvdata.parameter.UVParameter
  :members:

.. autoclass:: pyuvdata.parameter.AngleParameter
  :members:

.. autoclass:: pyuvdata.parameter.LocationParameter
  :members:

.. autoclass:: pyuvdata.parameter.SkyCoordParameter
  :members:

.. autoclass:: pyuvdata.uvbase.UVBase
  :members:


File Type Specific Classes
**************************
These classes inherit from pyuvdata's user classes and hold the file type
specific code. The read and write methods on the user classes convert between
the user classes and the file-specific classes automatically as needed, so users
generally do not need to interact with these classes, but developers may need to.


UVData Classes
~~~~~~~~~~~~~~

.. autoclass:: pyuvdata.uvdata.fhd.FHD
  :members:

.. autoclass:: pyuvdata.uvdata.mir.Mir
  :members:

.. autoclass:: pyuvdata.uvdata.miriad.Miriad
  :members:

.. autoclass:: pyuvdata.uvdata.ms.MS
  :members:

.. autoclass:: pyuvdata.uvdata.mwa_corr_fits.MWACorrFITS
  :members:

.. autoclass:: pyuvdata.uvdata.uvfits.UVFITS
  :members:

.. autoclass:: pyuvdata.uvdata.uvh5.UVH5
  :members:


UVCal Classes
~~~~~~~~~~~~~

.. autoclass:: pyuvdata.uvcal.calfits.CALFITS
  :members:

.. autoclass:: pyuvdata.uvcal.calh5.CalH5
  :members:

.. autoclass:: pyuvdata.uvcal.fhd_cal.FHDCal
  :members:

UVBeam Classes
~~~~~~~~~~~~~~

.. autoclass:: pyuvdata.uvbeam.beamfits.BeamFITS
  :members:

.. autoclass:: pyuvdata.uvbeam.cst_beam.CSTBeam
  :members:

.. autoclass:: pyuvdata.uvbeam.mwa_beam.MWABeam
  :members:


Other Modules and Functions
***************************

aipy extracts
~~~~~~~~~~~~~

.. automodule:: pyuvdata.uvdata.aipy_extracts
  :members:

datasets
~~~~~~~~

.. automodule:: pyuvdata.datasets
  :members:


MIR parser
~~~~~~~~~~

.. automodule:: pyuvdata.uvdata.mir_parser
  :members:

MIR metadata
~~~~~~~~~~~~

.. automodule:: pyuvdata.uvdata.mir_meta_data
  :members:


UVFlag Functions
~~~~~~~~~~~~~~~~
Some useful flag handling functions.

.. autofunction:: pyuvdata.uvflag.uvflag.and_rows_cols

.. autofunction:: pyuvdata.uvflag.uvflag.flags2waterfall
