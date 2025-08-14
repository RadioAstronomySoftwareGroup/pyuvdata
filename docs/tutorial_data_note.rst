.. note::
  Our tutorial uses small data files for examples. The data files are hosted in
  the the `RASG datasets repo <https://github.com/RadioAstronomySoftwareGroup/rasg-datasets/>`__,
  organized by data type and telescope. In the tutorials this data is downloaded
  and cached using the pooch package via the ``pyuvdata.datasets.fetch_data``
  function. To run those commands you'll need to have pooch installed (you can
  install it yourself or use ``pip install pyuvdata[tutorial]``). Note that pooch
  will download the file the first time you ask for it and save it in a cache
  folder, subsequent calls to fetch that data will not re-download it.
