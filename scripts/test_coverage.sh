#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

python setup.py install

cd pyuvdata/tests
nosetests --with-coverage --cover-erase --cover-package=pyuvdata --cover-html "$@"
