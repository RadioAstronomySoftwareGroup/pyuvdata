from setuptools import setup, Extension
import glob
import os
import os.path as op
from os import listdir
import numpy as np
import json

# When setting up, the binary extension modules haven't yet been built, so
# without a workaround we can't use the pyuvdata code to get the version.
os.environ['PYUVDATA_IGNORE_EXTMOD_IMPORT_FAIL'] = '1'
from pyuvdata import version  # noqa (pycodestyle complains about import below code)

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(op.join('pyuvdata', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

global_c_macros = [
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

setup_args = {
    'name': 'pyuvdata',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/pyuvdata',
    'license': 'BSD',
    'description': 'an interface for astronomical interferometeric datasets in python',
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
    'setup_requires': ['numpy>=1.10'],
    'install_requires': ['numpy>=1.10', 'scipy', 'astropy>=1.2', 'pyephem'],
    'classifiers': ['Development Status :: 5 - Production/Stable',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 2.7',
                    'Topic :: Scientific/Engineering :: Astronomy'],
    'keywords': 'radio astronomy interferometry'
}

if __name__ == '__main__':
    setup(**setup_args)
