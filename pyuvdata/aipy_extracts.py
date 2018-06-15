# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the 2-clause BSD License

"""This module extracts some Python code from AIPY used in our MIRIAD I/O
routines. It was copied from AIPY commit
6cb5a70876f33dccdd68d4063b076f8d42d9edae, then reformatted. The only routine
actually used in pyuvdata is ``uv_selector``.

"""
from __future__ import absolute_import, division, print_function

__all__ = [
    'uv_selector',
]

import re

str2pol = {
    'I' :  1, # Stokes Paremeters
    'Q' :  2,
    'U' :  3,
    'V' :  4,
    'rr': -1, # Circular Polarizations
    'll': -2,
    'rl': -3,
    'lr': -4,
    'xx': -5, # Linear Polarizations
    'yy': -6,
    'xy': -7,
    'yx': -8,
}


def bl2ij(bl):
    bl = int(bl)

    if bl > 65536:
        bl -= 65536
        mant = 2048
    else:
        mant = 256

    return (bl // mant - 1, bl % mant - 1)


def ij2bl(i, j):
    if i > j:
        i, j = j, i

    if j + 1 < 256:
        return 256 * (i + 1) + (j + 1)

    return 2048 * (i + 1) + (j + 1) + 65536


ant_re = r'(\(((-?\d+[xy]?,?)+)\)|-?\d+[xy]?)'
bl_re = '(^(%s_%s|%s),?)' % (ant_re, ant_re, ant_re)


def parse_ants(ant_str, nants):
    """Generate list of (baseline, include, pol) tuples based on parsing of the
    string associated with the 'ants' command-line option.

    """
    rv, cnt = [], 0

    while cnt < len(ant_str):
        m = re.search(bl_re, ant_str[cnt:])

        if m is None:
            if ant_str[cnt:].startswith('all'):
                rv = []
            elif ant_str[cnt:].startswith('auto'):
                rv.append(('auto',1,-1))
            elif ant_str[cnt:].startswith('cross'):
                rv.append(('auto',0,-1))
            else:
                raise ValueError('Unparsable ant argument "%s"' % ant_str)

            c = ant_str[cnt:].find(',')

            if c >= 0:
                cnt += c + 1
            else:
                cnt = len(ant_str)
        else:
            m = m.groups()
            cnt += len(m[0])

            if m[2] is None:
                ais = [m[8]]
                ajs = range(nants)
            else:
                if m[3] is None:
                    ais = [m[2]]
                else:
                    ais = m[3].split(',')

                if m[6] is None:
                    ajs = [m[5]]
                else:
                    ajs = m[6].split(',')

            for i in ais:
                if type(i) == str and i.startswith('-'):
                    i = i[1:] # nibble the - off the string
                    include_i = 0
                else:
                    include_i = 1

                for j in ajs:
                    include = None

                    if type(j) == str and j.startswith('-'):
                        j = j[1:]
                        include_j = 0
                    else:
                        include_j = 1

                    include = int(include_i and include_j)
                    pol = None
                    i, j = str(i), str(j)

                    if not i.isdigit():
                        ai = re.search(r'(\d+)([x,y])',i).groups()
                    if not j.isdigit():
                        aj = re.search(r'(\d+)([x,y])',j).groups()

                    if i.isdigit() and not j.isdigit():
                        pol = ['x' + aj[1], 'y' + aj[1]]
                        ai = [i, '']
                    elif not i.isdigit() and j.isdigit():
                        pol = [ai[1] + 'x', ai[1] + 'y']
                        aj = [j, '']
                    elif not i.isdigit() and not j.isdigit():
                        pol = [ai[1] + aj[1]]

                    if not pol is None:
                        bl = ij2bl(abs(int(ai[0])), abs(int(aj[0])))
                        for p in pol:
                            rv.append((bl, include, p))
                    else:
                        bl = ij2bl(abs(int(i)), abs(int(j)))
                        rv.append((bl, include, -1))
    return rv


def uv_selector(uv, ants=-1, pol_str=-1):
    """Call uv.select with appropriate options based on string argument for
    antennas (can be 'all', 'auto', 'cross', '0,1,2', or '0_1,0_2') and string
    for polarization ('xx','yy','xy','yx').

    """
    if ants != -1:
        if type(ants) == str:
            ants = parse_ants(ants, uv['nants'])

        for cnt, (bl, include, pol) in enumerate(ants):
            if cnt > 0:
                if include:
                    uv.select('or', -1, -1)
                else:
                    uv.select('and', -1, -1)

            if pol == -1:
                pol = pol_str # default to explicit pol parameter

            if bl == 'auto':
                uv.select('auto', 0, 0, include=include)
            else:
                i, j = bl2ij(bl)
                uv.select('antennae', i, j, include=include)

            if pol != -1:
                for p in pol.split(','):
                    polopt = str2pol[p]
                    uv.select('polarization', polopt, 0)
    elif pol_str != -1:
        for p in pol_str.split(','):
            polopt = str2pol[p]
            uv.select('polarization', polopt, 0)
