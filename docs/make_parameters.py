"""
Format the UVData object parameters into a sphinx rst file.
"""
from pyuvdata import UVData
import numpy as np
from astropy.time import Time

UV = UVData()
out = 'Parameters\n==============\n'
out += ("These are the standard attributes of UVData objects.\n\nUnder the hood "
        "they are actually properties based on UVParameter objects.\n\nAngle type "
        "attributes also have convenience properties named the same thing \nwith "
        "'_degrees' appended through which you can get or set the value in "
        "degrees.\n\nSimilarly location type attributes (which are given in "
        "topocentric xyz coordinates) \nhave convenience properties named the "
        "same thing with '_lat_lon_alt' and \n'_lat_lon_alt_degrees' appended "
        "through which you can get or set the values using \nlatitude, longitude and "
        "altitude values in radians or degrees and meters.\n\n")
out += 'Required\n----------------\n'
out += ('These parameters are required to have a sensible UVData object and \n'
        'are required for most kinds of uv data files.')
out += "\n\n"
for thing in UV.required():
    obj = getattr(UV, thing)
    out += '**{name}**\n'.format(name=obj.name)
    out += '     {desc}\n'.format(desc=obj.description)
    out += "\n"


out += 'Optional\n----------------\n'
out += ('These parameters are defined by one or more file standard but are not '
        'always required.\nSome of them are required depending on the '
        'phase_type (as noted below).')
out += "\n\n"
for thing in UV.extra():
    obj = getattr(UV, thing)
    out += '**{name}**\n'.format(name=obj.name)
    out += '     {desc}\n'.format(desc=obj.description)
    out += "\n"
t = Time.now()
t.out_subfmt = 'date'
out += "last updated: {date}".format(date=t.iso)
F = open('parameters.rst', 'w')
F.write(out)
print "wrote parameters.rst"
