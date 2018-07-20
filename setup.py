# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension
import glob
import os
from os import listdir
import io
import numpy as np
import json

# When setting up, the binary extension modules haven't yet been built, so
# without a workaround we can't use the pyuvdata code to get the version.
os.environ['PYUVDATA_IGNORE_EXTMOD_IMPORT_FAIL'] = '1'
from pyuvdata import version  # noqa (pycodestyle complains about import below code)

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('pyuvdata', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()


global_c_macros = [
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

setup_args = {
    'name': 'pyuvdata',
    'author': 'Radio Astronomy Software Group',
    'url': 'https://github.com/RadioAstronomySoftwareGroup/pyuvdata',
    'license': 'BSD',
    'description': 'an interface for astronomical interferometeric datasets in python',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'pyuvdata': 'pyuvdata'},
    'packages': ['pyuvdata', 'pyuvdata.tests'],
    'ext_modules': [
        Extension(
            'pyuvdata._miriad',
            sources=[
                'pyuvdata/src/miriad_wrap.cpp',
                'pyuvdata/src/uvio.c',
                'pyuvdata/src/hio.c',
                'pyuvdata/src/pack.c',
                'pyuvdata/src/bug.c',
                'pyuvdata/src/dio.c',
                'pyuvdata/src/headio.c',
                'pyuvdata/src/maskio.c',
            ],
            define_macros=global_c_macros,
            include_dirs=[
                np.get_include(),
                'pyuvdata/src',
            ]
        )
    ],
    'scripts': glob.glob('scripts/*'),
    'version': version.version,
    'include_package_data': True,
    'setup_requires': ['numpy>=1.14', 'six>=1.10'],
    'install_requires': ['numpy>=1.14', 'six>=1.10', 'scipy', 'astropy>=2.0'],
    'classifiers': ['Development Status :: 5 - Production/Stable',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 2.7',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering :: Astronomy'],
    'keywords': 'radio astronomy interferometry'
}

if __name__ == '__main__':
    setup(**setup_args)
