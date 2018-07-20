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


def write_index_rst(readme_file=None, write_file=None):
    t = Time.now()
    t.out_subfmt = 'date'
    out = ('.. pyuvdata documentation master file, created by\n'
           '   make_index.py on {date}\n\n').format(date=t.iso)

    if readme_file is None:
        main_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.stack()[0][1])))
        readme_file = os.path.join(main_path, 'README.md')

    readme_md = pypandoc.convert_file(readme_file, 'md')

    travis_str = 'https://travis-ci.org/RadioAstronomySoftwareGroup/pyuvdata.svg'
    regex_travis = re.compile(travis_str)
    loc_travis_start = re.search(regex_travis, readme_md).start()
    loc_travis_end = re.search(regex_travis, readme_md).end()
    end_branch_str = r'\)\]'
    regex_end = re.compile(end_branch_str)
    loc_branch_end = re.search(regex_end, readme_md).start()
    branch_str = readme_md[loc_travis_end:loc_branch_end]

    cover_str = 'https://coveralls.io/repos/github/RadioAstronomySoftwareGroup/pyuvdata/badge.svg'
    regex_cover = re.compile(cover_str)
    loc_cover_start = re.search(regex_cover, readme_md).start()
    loc_cover_end = re.search(regex_cover, readme_md).end()

    readme_text = pypandoc.convert_file(readme_file, 'rst')

    rst_status_badge = '.. image:: ' + travis_str + branch_str + '\n    :target: https://travis-ci.org/RadioAstronomySoftwareGroup/pyuvdata'
    status_badge_text = '|Build Status|'
    readme_text = readme_text.replace(status_badge_text, rst_status_badge + '\n\n')

    rst_status_badge = '.. image:: ' + cover_str + branch_str + '\n    :target: https://coveralls.io/github/RadioAstronomySoftwareGroup/pyuvdata' + branch_str
    status_badge_text = '|Coverage Status|'
    readme_text = readme_text.replace(status_badge_text, rst_status_badge)

    readme_text = readme_text.replace(' ' + rst_status_badge, rst_status_badge)

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
            '   developer_docs\n')

    out.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\xa0", " ")

    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, 'index.rst')
    F = open(write_file, 'w')
    F.write(out)
    print("wrote " + write_file)
