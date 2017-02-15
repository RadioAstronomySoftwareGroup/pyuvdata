from setuptools import setup
import glob
import os.path as op
from os import listdir

__version__ = '1.0'

setup_args = {
    'name': 'pyuvdata',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/pyuvdata',
    'license': 'BSD',
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
