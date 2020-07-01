#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

pip install .[test]

cd pyuvdata
python -m pytest --cov=pyuvdata --cov-config=../.coveragerc\
       --cov-report term --cov-report html:tests/cover \
       "$@"
