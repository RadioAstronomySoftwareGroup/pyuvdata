# -*- coding: utf-8 -*-

"""
Format the readme.md file into the sphinx index.rst file.

"""
from __future__ import absolute_import, division, print_function

import os
import inspect
import re
import pypandoc
from astropy.time import Time

from pyuvdata.version import construct_version_info


def write_index_rst(readme_file=None, write_file=None):
    t = Time.now()
    t.out_subfmt = 'date'
    out = ('.. pyuvdata documentation master file, created by\n'
           '   make_index.py on {date}\n\n').format(date=t.iso)

    if readme_file is None:
        main_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.stack()[0][1])))
        readme_file = os.path.join(main_path, 'README.md')

    readme_md = pypandoc.convert_file(readme_file, 'md')

    readme_text = pypandoc.convert_file(readme_file, 'rst')

    title_badge_text = (
        'pyuvdata\n========\n\n'
        '.. image:: https://travis-ci.org/RadioAstronomySoftwareGroup/pyuvdata.svg?branch=master\n'
        '    :target: https://travis-ci.org/RadioAstronomySoftwareGroup/pyuvdata\n\n'
        '.. image:: https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata.svg?style=svg\n'
        '    :target: https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata\n\n'
        '.. image:: https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata/branch/master/graph/badge.svg\n'
        '  :target: https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata\n\n')

    begin_desc = 'pyuvdata defines a pythonic interface'
    start_desc = str.find(readme_text, begin_desc)

    readme_text = readme_text[start_desc:]

    # convert relative links in readme to explicit links
    version_info = construct_version_info()
    branch = version_info['git_branch']

    first_docs_loc = readme_text.find('docs/')

    readme_text = readme_text.replace(
        '<docs/', '<https://github.com/RadioAstronomySoftwareGroup/pyuvdata/tree/'
        + branch + '/docs/')

    readme_text = readme_text.replace(
        '<.github/', '<https://github.com/RadioAstronomySoftwareGroup/pyuvdata/tree/'
        + branch + '/.github/')

    readme_text = title_badge_text + readme_text

    end_text = 'parameters descriptions'
    regex = re.compile(end_text.replace(' ', r'\s+'))
    loc = re.search(regex, readme_text).start()

    out += readme_text[0:loc] + end_text + '.'
    out += ('\n\nFurther Documentation\n====================================\n'
            '.. toctree::\n'
            '   :maxdepth: 1\n\n'
            '   tutorial\n'
            '   uvdata_parameters\n'
            '   uvdata\n'
            '   uvcal_parameters\n'
            '   uvcal\n'
            '   uvbeam_parameters\n'
            '   uvbeam\n'
            '   cst_settings_yaml\n'
            '   utility_functions\n'
            '   known_telescopes\n'
            '   developer_docs\n')

    out.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\xa0", " ")

    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, 'index.rst')
    F = open(write_file, 'w')
    F.write(out)
    print("wrote " + write_file)
