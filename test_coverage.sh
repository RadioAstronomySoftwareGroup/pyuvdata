#! /bin/bash

python setup.py install

cd uvdata/tests
nosetests --with-coverage --cover-erase --cover-package=uvdata --cover-html
