import os
import subprocess
from data import DATA_PATH


def construct_version_info():
    version_file = os.path.join(os.path.dirname(os.path.dirname(DATA_PATH)), 'VERSION')
    print(os.path.dirname(__file__))
    version = open(version_file).read().strip()

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

if __name__ == '__main__':
    print('Version = {0}'.format(version))
    print('git origin = {0}'.format(git_origin))
    print('git branch = {0}'.format(git_branch))
    print('git description = {0}'.format(git_description))
