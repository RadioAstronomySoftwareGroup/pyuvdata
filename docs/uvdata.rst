UVdata Class
=============

Code layout
-----------
UVBase defines common structures
UVData is the main user class, provides import and export functionality to all supported file formats
Data structures from each file format is defined by subclassing UVData and then adding read and write functions back to UVData.


.. automodule:: uvdata
   :members: UVData

.. automodule:: miriad
   :members: Miriad

.. automodule:: uvfits
   :members: UVFITS

.. automodule:: fhd
   :members: FHD
