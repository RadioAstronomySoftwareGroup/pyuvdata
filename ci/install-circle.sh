set -xe

apt-get update; apt-get install -y gcc g++
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda config --add channels conda-forge
conda info -a
conda env create -f ci/${ENV_NAME}.yml --name=${ENV_NAME}  python=$PYTHON --quiet
source activate ${ENV_NAME}
conda list -n ${ENV_NAME}
