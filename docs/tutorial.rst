Tutorial
========

By default, following the tutorial will write any files to the current working directory.
Alternatively you can change the location the output files are saved to
by changing the arguments to the ``os.path.join`` calls.
When running the tutorial during test suite execution,
output files are written to a temporary directory created by pytest.

Tutorials are available for each major user class:

.. toctree::
    :maxdepth: 1
    :titlesonly:

    uvdata_tutorial.rst
    uvcal_tutorial.rst
    uvbeam_tutorial.rst
