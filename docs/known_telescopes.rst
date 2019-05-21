Known Telescopes
================

Known Telescope Data
--------------------
pyuvdata has the following known telescopes:

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
