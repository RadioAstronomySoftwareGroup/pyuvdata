import os


def setup_package():
    testdir = '../data/test/'
    if not os.path.exists(testdir):
        print('making test directory')
        os.mkdir(testdir)
