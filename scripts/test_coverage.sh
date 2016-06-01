#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

python setup.py install

cd uvdata/tests
nosetests --with-coverage --cover-erase --cover-package=uvdata --cover-html "$@"
