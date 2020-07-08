#!/bin/bash
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# This function is make_dev_version.sh
#
# Usage:
#   ./ci/make_dev_version.sh
#
# This function *must* be called from the main package folder where the
# `setup.py` file lives. It echoes a modified version of the package which
# includes the standard setuptools_scm "number of commits since last release"
# counter, *plus* a numericized version of the git hash. This allows for uploads
# to the PyPI test server that do not collide for branches that are both the
# same number of commits since the last release, but have unique git hashes.
#
# This uses the "main" part of the version and the local git hash to generate a
# version number that is acceptable to the testpypi server. According to PEP
# 440, only numeric values are allowed in the ".dev" field, and testpypi does
# not allow for "local" information (such as the git version). To get around
# these constraints, we substitute the hex characters in the git hash for the
# corresponding numeric values (e.g., a -> 10, b -> 11, etc.). We also prepend
# numbers with a leading "0" (e.g., 1 -> 01) to avoid hash collisions.
#
# As an added benefit, this approach also handles tagged release versions (such
# as "2.0.3" with no trailing "dev" information), and does not mangle the
# version information.

# get version from python
ver=$(python setup.py --version | tail -1)

# get the first part of the version identifier
main_ver=$(awk -F+ '{print $1}' <<< $ver)

# extract just the git hash
git_hash=$(awk -F. '{print $1}' <<< $(awk -F+ '{print $2}' <<< $ver))

# add leading zeros to numbers
git_hash_zeros=$(echo $git_hash | \
    sed -E 's/0/00/g' | \
    sed -E 's/1/01/g' | \
    sed -E 's/2/02/g' | \
    sed -E 's/3/03/g' | \
    sed -E 's/4/04/g' | \
    sed -E 's/5/05/g' | \
    sed -E 's/6/06/g' | \
    sed -E 's/7/07/g' | \
    sed -E 's/8/08/g' | \
    sed -E 's/9/09/g')

# only numbers are allowed in dev version; substitute hex values for numeric ones
git_hash_numeric=$(echo $git_hash_zeros | \
    sed -E 's/a/10/g' | \
    sed -E 's/b/11/g' | \
    sed -E 's/c/12/g' | \
    sed -E 's/d/13/g' | \
    sed -E 's/e/14/g' | \
    sed -E 's/f/15/g')

# strip leading "g" from git hash
echo ${main_ver}${git_hash_numeric:1}
