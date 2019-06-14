# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import glob
import os
import io
import sys
import platform
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
import json


# When setting up, the binary extension modules haven't yet been built, so
# without a workaround we can't use the pyuvdata code to get the version.
os.environ['PYUVDATA_IGNORE_EXTMOD_IMPORT_FAIL'] = '1'


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


# construct version info. Has to be here because otherwise the package can't be
# built if the dependencies are not already installed.
pyuvdata_dir = os.path.dirname(os.path.realpath(__file__))
print(pyuvdata_dir)


def py_major_version():
    version_info = sys.version_info
    return version_info[0]


def _get_git_output(args, capture_stderr=False):
    """Get output from Git, ensuring that it is of the ``str`` type,
    not bytes."""

    argv = ['git', '-C', pyuvdata_dir] + args

    if capture_stderr:
        data = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    else:
        data = subprocess.check_output(argv)

    data = data.strip()

    if py_major_version() == 2:
        return data
    return data.decode('utf8')


def _get_gitinfo_file(git_file=None):
    """Get saved info from GIT_INFO file that was created when installing package"""
    if git_file is None:
        git_file = os.path.join(pyuvdata_dir, 'GIT_INFO')

    with open(git_file) as data_file:
        data = [_unicode_to_str(x) for x in json.loads(data_file.read().strip())]
        git_origin = data[0]
        git_hash = data[1]
        git_description = data[2]
        git_branch = data[3]

    return {'git_origin': git_origin, 'git_hash': git_hash,
            'git_description': git_description, 'git_branch': git_branch}


def _unicode_to_str(u):
    if py_major_version() == 2:
        return u.encode('utf8')
    return u


def construct_version_info():
    """
    Get full version information, including git details

    Returns
    -------
    dict
        dictionary giving full version information
    """
    version_file = os.path.join(pyuvdata_dir, 'pyuvdata', 'VERSION')
    with open(version_file) as f:
        version = f.read().strip()

    git_origin = ''
    git_hash = ''
    git_description = ''
    git_branch = ''

    version_info = {'version': version, 'git_origin': '', 'git_hash': '',
                    'git_description': '', 'git_branch': ''}

    try:
        git_origin = _get_git_output(['config', '--get', 'remote.origin.url'], capture_stderr=True)
        if git_origin.split('/')[-1] != 'pyuvdata.git':  # pragma: no cover
            # this is version info for a non-pyuvdata repo, don't use it
            raise ValueError('This is not a pyuvdata repo')

        version_info['git_origin'] = git_origin
        version_info['git_hash'] = _get_git_output(['rev-parse', 'HEAD'], capture_stderr=True)
        version_info['git_description'] = _get_git_output(['describe', '--dirty', '--tag', '--always'])
        version_info['git_branch'] = _get_git_output(['rev-parse', '--abbrev-ref', 'HEAD'], capture_stderr=True)
    except (subprocess.CalledProcessError, ValueError, OSError):  # pragma: no cover
        try:
            # Check if a GIT_INFO file was created when installing package
            version_info.update(_get_gitinfo_file())
        except (IOError, OSError):
            pass

    return version_info


version_info = construct_version_info()
version = version_info['version']
git_origin = version_info['git_origin']
git_hash = version_info['git_hash']
git_description = version_info['git_description']
git_branch = version_info['git_branch']


data = [git_origin, git_hash, git_description, git_branch]
with open(os.path.join('pyuvdata', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()


def is_platform_mac():
    return sys.platform == 'darwin'


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
# implementation based on pandas, see https://github.com/pandas-dev/pandas/issues/23424
if is_platform_mac():
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

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
    'cmdclass': {'build_ext': CustomBuildExtCommand},
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
            include_dirs=['pyuvdata/src']
        )
    ],
    'scripts': glob.glob('scripts/*'),
    'version': version,
    'include_package_data': True,
    'setup_requires': ['pytest-runner', 'numpy>=1.15'],
    'install_requires': ['numpy>=1.15', 'six>=1.10', 'scipy', 'astropy>=2.0'],
    'tests_require': ['pytest'],
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
