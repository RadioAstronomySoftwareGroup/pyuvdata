"""
Format the UVCal object parameters into a sphinx rst file.
"""
from pyuvdata import UVCal
import numpy as np
from astropy.time import Time

cal = UVCal()
out = 'UVCal Parameters\n==============\n'
out += ("These are the standard attributes of UVCal objects.\n\nUnder the hood "
        "they are actually properties based on UVParameter objects.\n\n")
out += 'Required\n----------------\n'
out += ('These parameters are required to have a sensible UVCal object and \n'
        'are required for most kinds of uv cal files.')
out += "\n\n"
for thing in cal.required():
    obj = getattr(cal, thing)
    out += '**{name}**\n'.format(name=obj.name)
    out += '     {desc}\n'.format(desc=obj.description)
    out += "\n"


out += 'Optional\n----------------\n'
out += ('These parameters are defined by one or more file standard but are not '
        'always required.\nSome of them are required depending on the '
        'cal_type (as noted below).')
out += "\n\n"
for thing in cal.extra():
    obj = getattr(cal, thing)
    out += '**{name}**\n'.format(name=obj.name)
    out += '     {desc}\n'.format(desc=obj.description)
    out += "\n"
t = Time.now()
t.out_subfmt = 'date'
out += "last updated: {date}".format(date=t.iso)
F = open('uvcal_parameters.rst', 'w')
F.write(out)
print "wrote uvcal_parameters.rst"
