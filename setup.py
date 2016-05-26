from distutils.core import setup
import glob
import os.path as op

__version__ = '0.0.1'

setup_args = {
    'name': 'uvdata',
    'author': 'HERA Team',
    'license': 'BSD',
    'package_dir': {'uvdata': 'uvdata'},
    'packages': ['uvdata'],
    'scripts': glob.glob('scripts/*'),
    'version': __version__,
    'package_data': {'uvdata': [op.join('data', '*')]}
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
