# -*- coding: utf-8 -*-

"""
Format the UVBeam object parameters into a sphinx rst file.

"""
from __future__ import absolute_import, division, print_function

import os
import inspect
from pyuvdata import UVBeam
import numpy as np
from astropy.time import Time


def write_beamparams_rst(write_file=None):
    beam = UVBeam()
    out = 'UVBeam Parameters\n======================================\n'
    out += ("These are the standard attributes of UVBeam objects.\n\nUnder the hood "
            "they are actually properties based on UVParameter objects.\n\n")
    out += 'Required\n----------------\n'
    out += ('These parameters are required to have a sensible UVBeam object and \n'
            'are required for most kinds of beam files.')
    out += "\n\n"
    for thing in beam.required():
        obj = getattr(beam, thing)
        out += '**{name}**\n'.format(name=obj.name)
        out += '     {desc}\n'.format(desc=obj.description)
        out += "\n"

    out += 'Optional\n----------------\n'
    out += ('These parameters are defined by one or more file standard but are not '
            'always required.\nSome of them are required depending on the '
            'beam_type, antenna_type and pixel_coordinate_systems (as noted below).')
    out += "\n\n"
    for thing in beam.extra():
        obj = getattr(beam, thing)
        out += '**{name}**\n'.format(name=obj.name)
        out += '     {desc}\n'.format(desc=obj.description)
        out += "\n"
    t = Time.now()
    t.out_subfmt = 'date'
    out += "last updated: {date}".format(date=t.iso)
    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, 'uvbeam_parameters.rst')
    F = open(write_file, 'w')
    F.write(out)
    print("wrote " + write_file)
