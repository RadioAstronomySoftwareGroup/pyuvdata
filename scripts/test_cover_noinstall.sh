#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

# We want to make sure we build the Cython extensions now.
CFLAGS="-DCYTHON_TRACE=1 -DCYTHON_TRACE_NOGIL=1" pip install -e . --no-deps

cd pyuvdata
python -m pytest --cov=pyuvdata --cov-config=../.coveragerc\
       --cov-report term --cov-report html:tests/cover \
       "$@"
