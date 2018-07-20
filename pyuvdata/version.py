# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

from __future__ import absolute_import, division, print_function

import os
import six
import subprocess
import json


def construct_version_info():
    pyuvdata_dir = os.path.dirname(os.path.realpath(__file__))

    def get_git_output(args, capture_stderr=False):
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

    def unicode_to_str(u):
        if six.PY2:
            return u.encode('utf8')
        return u

    version_file = os.path.join(pyuvdata_dir, 'VERSION')
    with open(version_file) as f:
        version = f.read().strip()

    try:
        git_origin = get_git_output(['config', '--get', 'remote.origin.url'], capture_stderr=True)
        git_hash = get_git_output(['rev-parse', 'HEAD'], capture_stderr=True)
        git_description = get_git_output(['describe', '--dirty', '--tag', '--always'])
        git_branch = get_git_output(['rev-parse', '--abbrev-ref', 'HEAD'], capture_stderr=True)
        git_version = get_git_output(['describe', '--tags', '--abbrev=0'])
    except subprocess.CalledProcessError:
        try:
            # Check if a GIT_INFO file was created when installing package
            git_file = os.path.join(pyuvdata_dir, 'GIT_INFO')
            with open(git_file) as data_file:
                data = [unicode_to_str(x) for x in json.loads(data_file.read().strip())]
                git_origin = data[0]
                git_hash = data[1]
                git_description = data[2]
                git_branch = data[3]
        except (IOError, OSError):
            git_origin = ''
            git_hash = ''
            git_description = ''
            git_branch = ''

    version_info = {'version': version, 'git_origin': git_origin,
                    'git_hash': git_hash, 'git_description': git_description,
                    'git_branch': git_branch}
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


if __name__ == '__main__':
    main()
