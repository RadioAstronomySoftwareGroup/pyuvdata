import os
import subprocess
import csv


def construct_version_info():
    pyuvdata_dir = os.path.dirname(os.path.realpath(__file__))
    version_file = os.path.join(pyuvdata_dir, 'VERSION')
    version = open(version_file).read().strip()

    try:
        git_origin = subprocess.check_output(['git', '-C', pyuvdata_dir, 'config',
                                              '--get', 'remote.origin.url'],
                                             stderr=subprocess.STDOUT).strip()
        git_hash = subprocess.check_output(['git', '-C', pyuvdata_dir, 'rev-parse', 'HEAD'],
                                           stderr=subprocess.STDOUT).strip()
        git_description = subprocess.check_output(['git', '-C', pyuvdata_dir,
                                                   'describe', '--dirty']).strip()
        git_branch = subprocess.check_output(['git', '-C', pyuvdata_dir, 'rev-parse',
                                              '--abbrev-ref', 'HEAD'],
                                             stderr=subprocess.STDOUT).strip()
        git_version = subprocess.check_output(['git', '-C', pyuvdata_dir, 'describe',
                                               '--abbrev=0']).strip()
    except:
        try:
            # Check if a GIT_INFO file was created when installing package
            git_file = os.path.join(pyuvdata_dir, 'GIT_INFO')
            csvfile = csv.reader(open(git_file))
            git_info = dict(csvfile)
            git_origin = git_info['git_origin']
            git_hash = git_info['git_hash']
            git_description = git_info['git_description']
            git_branch = git_info['git_branch']
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


def main():
    print('Version = {0}'.format(version))
    print('git origin = {0}'.format(git_origin))
    print('git branch = {0}'.format(git_branch))
    print('git description = {0}'.format(git_description))

if __name__ == '__main__':
    main()
