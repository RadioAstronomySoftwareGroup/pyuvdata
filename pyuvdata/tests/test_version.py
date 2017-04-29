"""Tests for version.py."""
import nose.tools as nt
import sys
import os
from StringIO import StringIO
import subprocess
import pyuvdata


def test_construct_version_info():
    # this test is a bit silly because it uses the nearly the same code as the original,
    # but it will detect accidental changes that could cause problems.
    # It does test that the __version__ attribute is set on pyuvdata.
    # I can't figure out how to test the except clause in construct_version_info.
    git_origin = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'],
                                         stderr=subprocess.STDOUT).strip()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.STDOUT).strip()
    git_description = subprocess.check_output(['git', 'describe', '--dirty', '--tags', '--always']).strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                         stderr=subprocess.STDOUT).strip()
    git_version = subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0']).strip()

    test_version_info = {'version': pyuvdata.__version__, 'git_origin': git_origin,
                         'git_hash': git_hash, 'git_description': git_description,
                         'git_branch': git_branch}

    nt.assert_equal(pyuvdata.version.construct_version_info(), test_version_info)


def test_main():
    version_info = pyuvdata.version.construct_version_info()

    saved_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        pyuvdata.version.main()
        output = out.getvalue()
        nt.assert_equal(output, 'Version = {v}\ngit origin = {o}\n'
                        'git branch = {b}\ngit description = {d}\n'
                        .format(v=version_info['version'],
                                o=version_info['git_origin'],
                                b=version_info['git_branch'],
                                d=version_info['git_description']))
    finally:
        sys.stdout = saved_stdout
