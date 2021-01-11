Known Telescopes
================

Known Telescope Data
--------------------
pyuvdata uses `Astropy sites <https://docs.astropy.org/en/stable/api/astropy.coordinates.EarthLocation.html#astropy.coordinates.EarthLocation.get_site_names>`_
for telescope location information, in addition to the following telescope information
that is tracked within pyuvdata:

.. exec::
    import json
    from pyuvdata.telescopes import KNOWN_TELESCOPES
    json_obj = json.dumps(KNOWN_TELESCOPES, sort_keys=True, indent=4)
    json_obj = json_obj[:-1] + " }"
    print('.. code-block:: JavaScript\n\n {json_str}\n\n'.format(json_str=json_obj))

Related class and functions
---------------------------

.. automodule:: pyuvdata.telescopes
  :members:
