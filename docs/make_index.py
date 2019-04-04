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

    # find parts of Travis badge
    travis_str = 'https://travis-ci.org/RadioAstronomySoftwareGroup/pyuvdata.svg'
    regex_travis = re.compile(travis_str)
    loc_travis_start = re.search(regex_travis, readme_md).start()
    loc_travis_end = re.search(regex_travis, readme_md).end()
    end_branch_str = r'\)\]'
    regex_end = re.compile(end_branch_str)
    loc_branch_end = re.search(regex_end, readme_md).start()
    branch_str = readme_md[loc_travis_end:loc_branch_end]

    start_link_str = r'\('
    regex_start_link = re.compile(start_link_str)
    end_link_str = r'\)\n'
    regex_end_link = re.compile(end_link_str)
    loc_link_start = re.search(regex_start_link, readme_md[loc_travis_end:]).start() + loc_travis_end
    loc_link_end = re.search(regex_end_link, readme_md[loc_link_start:]).start() + loc_link_start
    travis_link = readme_md[loc_link_start + 1:loc_link_end]

    # find parts of Circleci badge
    circleci_str = 'https://circleci.com/gh/RadioAstronomySoftwareGroup/pyuvdata.svg'
    regex_circleci = re.compile(circleci_str)
    loc_circleci_start = re.search(regex_circleci, readme_md).start()
    loc_circleci_end = re.search(regex_circleci, readme_md).end()

    loc_link_start = re.search(regex_start_link, readme_md[loc_circleci_end:]).start() + loc_circleci_end
    loc_link_end = re.search(regex_end_link, readme_md[loc_link_start:]).start() + loc_link_start
    circleci_link = readme_md[loc_link_start + 1:loc_link_end]

    # find parts of Coveralls badge
    cover_str = 'https://coveralls.io/repos/github/RadioAstronomySoftwareGroup/pyuvdata/badge.svg'
    regex_cover = re.compile(cover_str)
    loc_cover_start = re.search(regex_cover, readme_md).start()
    loc_cover_end = re.search(regex_cover, readme_md).end()

    loc_link_start = re.search(regex_start_link, readme_md[loc_cover_end:]).start() + loc_cover_end
    loc_link_end = re.search(regex_end_link, readme_md[loc_link_start:]).start() + loc_link_start
    cover_link = readme_md[loc_link_start + 1:loc_link_end]

    # find parts of Codecov badge
    codecov_str = 'https://codecov.io/gh/RadioAstronomySoftwareGroup/pyuvdata/badge.svg'
    regex_codecov = re.compile(codecov_str)
    loc_codecov_start = re.search(regex_codecov, readme_md).start()
    loc_codecov_end = re.search(regex_codecov, readme_md).end()

    loc_link_start = re.search(regex_start_link, readme_md[loc_codecov_end:]).start() + loc_codecov_end
    loc_link_end = re.search(regex_end_link, readme_md[loc_link_start:]).start() + loc_link_start
    codecov_link = readme_md[loc_link_start + 1:loc_link_end]

    readme_text = pypandoc.convert_file(readme_file, 'rst')

    # replace Travis badge
    rst_status_badge = '.. image:: ' + travis_str + branch_str + '\n    :target: ' + travis_link
    status_badge_text = '`Build\nStatus <' + travis_link + '>`__'
    readme_text = readme_text.replace(status_badge_text, rst_status_badge + '\n\n')

    # replace Circleci badge
    rst_status_badge = '.. image:: ' + circleci_str + branch_str + '&style=svg' + '\n    :target: ' + circleci_link
    status_badge_text = '`CircleCI <' + circleci_link + '>`__'
    readme_text = readme_text.replace(status_badge_text, rst_status_badge + '\n\n')

    # replace Coveralls badge
    rst_status_badge = '.. image:: ' + cover_str + branch_str + '\n    :target: ' + cover_link
    status_badge_text = '`Coverage\nStatus <' + cover_link + '>`__'
    readme_text = readme_text.replace(status_badge_text, rst_status_badge + '\n\n')

    # replace Codecov badge
    rst_status_badge = '.. image:: ' + codecov_str + branch_str + '\n    :target: ' + codecov_link
    status_badge_text = '`codecov <' + codecov_link + '>`__'
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
            '   cst_settings_yaml\n'
            '   developer_docs\n')

    out.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\xa0", " ")

    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, 'index.rst')
    F = open(write_file, 'w')
    F.write(out)
    print("wrote " + write_file)
