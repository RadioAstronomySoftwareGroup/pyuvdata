# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import glob
import io
import os
import platform
import sys
from sysconfig import get_config_var

import numpy
from Cython.Build import cythonize
from packaging.version import parse
from setuptools import Extension, find_namespace_packages, setup

# add pyuvdata to our path in order to use the branch_scheme function
sys.path.append("pyuvdata")
from branch_scheme import branch_scheme  # noqa

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


def is_platform_mac():
    return sys.platform == "darwin"


def is_platform_windows():
    return sys.platform == "win32"


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
# implementation based on pandas, see https://github.com/pandas-dev/pandas/issues/23424
if is_platform_mac():
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = parse(platform.mac_ver()[0])
        python_target = parse(get_config_var("MACOSX_DEPLOYMENT_TARGET"))
        if python_target < parse("10.9") and current_system >= parse("10.9"):
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

# define the cython compile args, depending on platform
if is_platform_windows():
    extra_compile_args = ["/Ox", "/openmp"]
    extra_link_args = ["/openmp"]
elif is_platform_mac():
    extra_compile_args = ["-O3"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

global_c_macros = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
]

miriad_extension = Extension(
    "pyuvdata._miriad",
    sources=[
        "pyuvdata/uvdata/src/miriad_wrap.pyx",
        "pyuvdata/uvdata/src/uvio.c",
        "pyuvdata/uvdata/src/hio.c",
        "pyuvdata/uvdata/src/pack.c",
        "pyuvdata/uvdata/src/bug.c",
        "pyuvdata/uvdata/src/dio.c",
        "pyuvdata/uvdata/src/headio.c",
        "pyuvdata/uvdata/src/maskio.c",
    ],
    define_macros=global_c_macros,
    include_dirs=["pyuvdata/uvdata/src/", numpy.get_include()],
)

corr_fits_extension = Extension(
    "pyuvdata._corr_fits",
    sources=["pyuvdata/uvdata/corr_fits.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

utils_extension = Extension(
    "pyuvdata._utils",
    sources=["pyuvdata/utils.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

uvbeam_extension = Extension(
    "pyuvdata._uvbeam",
    sources=["pyuvdata/uvbeam/uvbeam.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

extensions = [corr_fits_extension, utils_extension, uvbeam_extension]

# don't build miriad on windows
if not is_platform_windows():
    extensions.append(miriad_extension)

astroquery_reqs = ["astroquery>=0.4.4"]
casa_reqs = ["python-casacore>=3.3.1"]
cst_reqs = ["pyyaml>=5.1"]
hdf5_compression_reqs = ["hdf5plugin>=3.1.0"]
healpix_reqs = ["astropy_healpix>=0.6"]
lunar_reqs = ["lunarsky>=0.1.2"]
novas_reqs = ["novas", "novas_de405"]
all_optional_reqs = (
    astroquery_reqs
    + casa_reqs
    + cst_reqs
    + hdf5_compression_reqs
    + healpix_reqs
    + lunar_reqs
    + novas_reqs
)
test_reqs = all_optional_reqs + [
    "pytest>=6.2",
    "pytest-xdist",
    "pytest-cases>=3.6.9",
    "pytest-cov",
    "cython",
    "coverage",
    "pre-commit",
]
doc_reqs = ["sphinx", "pypandoc"]

setup_args = {
    "name": "pyuvdata",
    "author": "Radio Astronomy Software Group",
    "url": "https://github.com/RadioAstronomySoftwareGroup/pyuvdata",
    "license": "BSD",
    "description": "an interface for astronomical interferometeric datasets in python",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"pyuvdata": "pyuvdata"},
    "packages": find_namespace_packages(),
    "ext_modules": cythonize(extensions, language_level=3),
    "scripts": [fl for fl in glob.glob("scripts/*") if not os.path.isdir(fl)],
    "use_scm_version": {"local_scheme": branch_scheme},
    "include_package_data": True,
    "install_requires": [
        "astropy>=5.0.4",
        "h5py>=3.0",
        "numpy>=1.19",
        "pyerfa>=2.0",
        "scipy>=1.3",
        "setuptools_scm!=7.0.0,!=7.0.1,!=7.0.2",
    ],
    "extras_require": {
        "astroquery": astroquery_reqs,
        "casa": casa_reqs,
        "cst": cst_reqs,
        "hdf5_compression": hdf5_compression_reqs,
        "healpix": healpix_reqs,
        "lunar": lunar_reqs,
        "novas": novas_reqs,
        "all": all_optional_reqs,
        "test": test_reqs,
        "doc": doc_reqs,
        "dev": test_reqs + doc_reqs,
    },
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    "keywords": "radio astronomy interferometry",
}

if __name__ == "__main__":
    setup(**setup_args)
