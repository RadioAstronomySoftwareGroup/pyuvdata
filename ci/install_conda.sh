#!/bin/bash -l
set -xe

micromamba info
# need these to add gxx and gcc to build novas and cython
micromamba create --name=${ENV_NAME}  python=$PYTHON gxx gcc -f ci/${ENV_NAME}.yml -yq
micromamba activate ${ENV_NAME}
micromamba list -n ${ENV_NAME}
# check that the python version matches the desired one; exit immediately if not
PYVER=`python -c "import sys; print('{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor))"`
if [[ $PYVER != $PYTHON ]]; then
  exit 1;
fi
