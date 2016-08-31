"""
Format the UVData object parameters into a sphinx rst file.
"""
from uvdata import UVData
import numpy as np
from astropy.time import Time

UV = UVData()
out = 'Parameters\n==============\n'
out += 'Required\n----------------\n'
out += """These parameters are required to make most kinds of uv data files.  In many cases there are default values available which let you sidestep the requirement (see spoof values).

            Parameters are implimented using the UVParameter object. See the uvbase documentation for more

            """
out += "\n\n"
for thing in UV.required():
    obj = getattr(UV, thing)
    out += '*UVData.*\\ **{name}**\n'.format(name=obj.name)
    out += '     {desc}\n'.format(desc=obj.description)
    out += "\n"


out += 'Not required\n----------------\n'
out += 'These parameters are defined by one or more file standard but are not officially required.\n'
out += "\n\n"
for thing in UV.extra():
    obj = getattr(UV, thing)
    out += '*UVData.*\\ **{name}**\n'.format(name=obj.name)
    out += '     {desc}\n'.format(desc=obj.description)
    out += "\n"
t = Time.now()
t.out_subfmt = 'date'
out += "last updated: {date}".format(date=t.iso)
F = open('parameters.rst', 'w')
F.write(out)
print "wrote parameters.rst"
