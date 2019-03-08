set -xe

apt-get update; apt-get install -y gcc g++
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda config --add channels conda-forge
conda info -a
conda create --name=${ENV_NAME}  python=$PYTHON --quiet
conda env update -f ci/${ENV_NAME}.yml
source activate ${ENV_NAME}
conda list -n ${ENV_NAME}
# check that the python version matches the desired one; exit immediately if not
PYVER=`python -c "from __future__ import print_function; import sys; print('{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor))"`
if [[ $PYVER != $PYTHON ]]; then
  exit 1;
fi
