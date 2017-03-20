from setuptools import setup
import glob
import os.path as op
from os import listdir
import subprocess

print "Generating pyuvdata/__version__.py: ",
try:
    __version__ = open('VERSION').read().strip()
except:
    __version__ = ''
try:
    git_origin = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'],
                                         stderr=subprocess.STDOUT).strip()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.STDOUT).strip()
    git_description = subprocess.check_output(['git', 'describe', '--dirty']).strip()
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                         stderr=subprocess.STDOUT).strip()
    git_version = subprocess.check_output(['git', 'describe', '--abbrev=0']).strip()
except:
    git_origin = ''
    git_hash = ''
    git_description = ''
    git_branch = ''

print('Version = {0}'.format(__version__))
print('git origin = {0}'.format(git_origin))
print('git description = {0}'.format(git_description))
version_text = ('version = "{0}"\ngit_origin = "{1}"\ngit_hash = "{2}"\n' +
                'git_description = "{3}"\ngit_branch = "{4}"'
                ).format(__version__, git_origin, git_hash, git_description, git_branch)
open('pyuvdata/version.py', 'w').write(version_text)

setup_args = {
    'name': 'pyuvdata',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/pyuvdata',
    'license': 'BSD',
    'description': 'an interface for astronomical interferometeric datasets in python',
    'package_dir': {'pyuvdata': 'pyuvdata', 'uvdata': 'uvdata'},
    'packages': ['pyuvdata', 'uvdata'],
    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    'package_data': {'pyuvdata': [f for f in listdir('./pyuvdata/data') if op.isfile(op.join('./pyuvdata/data', f))]},
    'install_requires': ['numpy>=1.10', 'scipy', 'astropy>=1.2', 'pyephem', 'aipy'],
    'classifiers': ['Development Status :: 5 - Production/Stable',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: BSD License',
                    'Programming Language :: Python :: 2.7',
                    'Topic :: Scientific/Engineering :: Astronomy'],
    'keywords': 'radio astronomy interferometry'
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
