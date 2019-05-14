#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

python setup.py install

cd pyuvdata/tests
python -m pytest --cov=pyuvdata --cov-config=../../.coveragerc\
       --cov-report term --cov-report html:cover \
       "$@"
