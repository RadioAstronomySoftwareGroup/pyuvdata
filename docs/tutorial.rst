Tutorial
========

By default, following the tutorial will write any files to the current working directory.
Alternatively you can change the location the output files are saved to
by changing the arguments to the ``os.path.join`` calls.
When running the tutorial during test suite execution,
output files are written to a temporary directory created by pytest.

Tutorials are available for each major user class:

:doc:`uvdata_tutorial`

:doc:`uvcal_tutorial`

:doc:`uvbeam_tutorial`
