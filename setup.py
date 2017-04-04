from setuptools import setup
import glob
import os.path as op
from os import listdir
from pyuvdata import version

version_text = ('"git_origin","{0}"\n"git_hash","{1}"\n' +
                '"git_description","{2}"\n"git_branch","{3}"'
                ).format(version.git_origin, version.git_hash,
                         version.git_description, version.git_branch)
open(op.join('pyuvdata', 'GIT_INFO'), 'w').write(version_text)

setup_args = {
    'name': 'pyuvdata',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/pyuvdata',
    'license': 'BSD',
    'description': 'an interface for astronomical interferometeric datasets in python',
    'package_dir': {'pyuvdata': 'pyuvdata', 'uvdata': 'uvdata'},
    'packages': ['pyuvdata', 'uvdata'],
    'scripts': glob.glob('scripts/*'),
    'version': open('VERSION').read().strip(),
    'include_package_data': True,
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
