# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

import os
import platform
import sys
from sysconfig import get_config_var

import numpy
from Cython.Build import cythonize
from packaging.version import parse
from setuptools import Extension, setup


# define the branch scheme. Have to do it here so we don't have to modify the path
def branch_scheme(version):
    """
    Local version scheme that adds the branch name for absolute reproducibility.

    If and when this is added to setuptools_scm this function can be removed.
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch == "main":
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            return version.format_choice("+{node}.{branch}", "+{node}.{branch}.dirty")


def is_platform_mac():
    return sys.platform == "darwin"


def is_platform_windows():
    return sys.platform == "win32"


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
# implementation based on pandas, see https://github.com/pandas-dev/pandas/issues/23424
if is_platform_mac() and "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
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

global_c_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

miriad_extension = Extension(
    "pyuvdata.uvdata._miriad",
    sources=[
        "src/pyuvdata/uvdata/src/miriad_wrap.pyx",
        "src/pyuvdata/uvdata/src/uvio.c",
        "src/pyuvdata/uvdata/src/hio.c",
        "src/pyuvdata/uvdata/src/pack.c",
        "src/pyuvdata/uvdata/src/bug.c",
        "src/pyuvdata/uvdata/src/dio.c",
        "src/pyuvdata/uvdata/src/headio.c",
        "src/pyuvdata/uvdata/src/maskio.c",
    ],
    define_macros=global_c_macros,
    include_dirs=["src/pyuvdata/uvdata/src/", numpy.get_include()],
)

corr_fits_extension = Extension(
    "pyuvdata.uvdata._corr_fits",
    sources=["src/pyuvdata/uvdata/corr_fits.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

coordinates_extension = Extension(
    "pyuvdata.utils._coordinates",
    sources=["src/pyuvdata/utils/coordinates.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

phasing_extension = Extension(
    "pyuvdata.utils._phasing",
    sources=["src/pyuvdata/utils/phasing.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

uvbeam_extension = Extension(
    "pyuvdata.uvbeam._uvbeam",
    sources=["src/pyuvdata/uvbeam/uvbeam.pyx"],
    define_macros=global_c_macros,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

extensions = [
    corr_fits_extension,
    coordinates_extension,
    phasing_extension,
    uvbeam_extension,
]

# don't build miriad on windows
if not is_platform_windows():
    extensions.append(miriad_extension)

setup(
    use_scm_version={"local_scheme": branch_scheme},
    ext_modules=cythonize(extensions, language_level=3),
)
