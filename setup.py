from setuptools import setup
import glob
import os.path as op
from os import listdir

__version__ = '1.0'

setup_args = {
    'name': 'uvdata',
    'author': 'HERA Team',
    'license': 'BSD',
    'package_dir': {'uvdata': 'uvdata'},
    'packages': ['uvdata'],
    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    'package_data': {'uvdata': [f for f in listdir('./uvdata/data') if op.isfile(op.join('./uvdata/data', f))]},
    'install_requires': ['numpy>=1.10', 'scipy', 'astropy>=1.2', 'pyephem', 'aipy']
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
