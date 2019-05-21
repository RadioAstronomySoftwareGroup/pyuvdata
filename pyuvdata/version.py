# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import os
import six
import subprocess
import json

pyuvdata_dir = os.path.dirname(os.path.realpath(__file__))


def _get_git_output(args, capture_stderr=False):
    """Get output from Git, ensuring that it is of the ``str`` type,
    not bytes."""

    argv = ['git', '-C', pyuvdata_dir] + args

    if capture_stderr:
        data = subprocess.check_output(argv, stderr=subprocess.STDOUT)
    else:
        data = subprocess.check_output(argv)

    data = data.strip()

    if six.PY2:
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
    if six.PY2:
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
    version_file = os.path.join(pyuvdata_dir, 'VERSION')
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


def main():
    print('Version = {0}'.format(version))
    print('git origin = {0}'.format(git_origin))
    print('git branch = {0}'.format(git_branch))
    print('git description = {0}'.format(git_description))


if __name__ == '__main__':  # pragma: no cover
    main()
