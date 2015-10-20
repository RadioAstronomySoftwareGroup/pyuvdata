from distutils.core import setup
import glob

__version__ = '0.0.1'

setup_args = {
    'name': 'uvdata',
    'author': 'HERA Team',
    'license': 'BSD',
    'package_dir': {'uvdata':'src'},
    'packages': ['uvdata'],
    'scripts': glob.glob('scripts/*'),
    'version': __version__
}

if __name__== '__main__':
    apply(setup, (), setup_args)
    
