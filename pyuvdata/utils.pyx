# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True

# python imports
import warnings
# cython imports
cimport numpy
cimport cython
from libc.math cimport sin, cos, sqrt, atan2

# This initializes the numpy 1.7 c-api.
# cython 3.0 will do this by default.
# We may be able to just remove this then.
numpy.import_array()

# in order to not have circular dependencies
# define transformation parameters here
# parameters for transforming between xyz & lat/lon/alt
cdef numpy.float64_t _gps_a = 6378137
cdef numpy.float64_t _gps_b = 6356752.31424518
cdef numpy.float64_t _e2 = 6.69437999014e-3
cdef numpy.float64_t _ep2 = 6.73949674228e-3
# this one is useful in the xyz from lla calculation
cdef numpy.float64_t b_div_a2 = (_gps_b / _gps_a)**2

# expose up to python
gps_a = _gps_a
gps_b = _gps_b
e_squared = _e2
e_prime_squared = _ep2

ctypedef fused int_or_float:
    numpy.int64_t
    numpy.int32_t
    numpy.float64_t
    numpy.float32_t


cdef inline int_or_float max(int_or_float a, int_or_float b):
    return a if a > b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int_or_float arraymin(int_or_float[::1] array) nogil:
    cdef int_or_float minval = array[0]
    cdef Py_ssize_t i
    for i in range(array.shape[0]):
        if array[i] < minval:
            minval = array[i]
    return minval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int_or_float arraymax(int_or_float[::1] array) nogil:
    cdef int_or_float maxval = array[0]
    cdef Py_ssize_t i
    for i in range(array.shape[0]):
        if array[i] > maxval:
            maxval = array[i]
    return maxval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _bl_to_ant_256(
    numpy.int64_t[::1] _bl,
    numpy.int64_t[:, ::1] _ants,
    long nbls,
):
  cdef Py_ssize_t i

  for i in range(nbls):
    _ants[1, i] = (_bl[i]) % 256 - 1
    _ants[0, i] = (_bl[i] - (_ants[1, i] + 1)) // 256 - 1
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _bl_to_ant_2048(
    numpy.int64_t[::1] _bl,
    numpy.int64_t[:, ::1] _ants,
    int nbls
):
  cdef Py_ssize_t i
  for i in range(nbls):
    _ants[1, i] = (_bl[i] - 2 ** 16) % 2048 - 1
    _ants[0, i] = (_bl[i] - 2 ** 16 - (_ants[1, i] + 1)) // 2048 - 1
  return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.int64_t, ndim=2] baseline_to_antnums(
    numpy.int64_t[::1] _bl
):
  cdef int _min = arraymin(_bl)
  cdef bint use2048 = _min > 2 ** 16
  cdef long nbls = _bl.shape[0]
  cdef int ndim = 2
  cdef numpy.npy_intp * dims = [2, <numpy.npy_intp> nbls]
  cdef numpy.ndarray[ndim=2, dtype=numpy.int64_t] ants = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_INT64, 0)
  cdef numpy.int64_t[:, ::1] _ants = ants

  if use2048:
    _bl_to_ant_2048(_bl, _ants, nbls)
  else:
    _bl_to_ant_256(_bl, _ants,  nbls)
  return ants

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _antnum_to_bl_2048(
  numpy.int64_t[::1] ant1,
  numpy.int64_t[::1] ant2,
  numpy.int64_t[::1] baselines,
  int nbls,
):
  cdef Py_ssize_t i

  for i in range(nbls):
    baselines[i] = 2048 * (ant1[i] + 1) + (ant2[i] + 1) + 2 ** 16
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _antnum_to_bl_256(
  numpy.int64_t[::1] ant1,
  numpy.int64_t[::1] ant2,
  numpy.int64_t[::1] baselines,
  int nbls,
):
  cdef Py_ssize_t i
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  for i in range(nbls):
    baselines[i] = 256 * (ant1[i] + 1) + (ant2[i] + 1)
  return

cpdef numpy.ndarray[dtype=numpy.int64_t] antnums_to_baseline(
  numpy.int64_t[::1] ant1,
  numpy.int64_t[::1] ant2,
  bint attempt256=False
):
  cdef int ndim = 1
  cdef int nbls = ant1.shape[0]
  cdef numpy.npy_intp * dims = [<numpy.npy_intp>nbls]
  cdef numpy.ndarray[ndim=1, dtype=numpy.int64_t] baseline = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_INT64, 0)
  cdef numpy.int64_t[::1] _bl = baseline
  cdef bint less255

  if attempt256:
    less255 = max(
      arraymax(ant1),
      arraymax(ant2),
    ) < 255
    if less255:
      _antnum_to_bl_256(ant1, ant2, _bl, nbls)

    else:
      message = (
        "antnums_to_baseline: found antenna numbers > 256, using "
        "2048 baseline indexing. Beware compatibility "
        "with CASA etc"
      )
      warnings.warn(message)
      _antnum_to_bl_2048(ant1, ant2, _bl, nbls)

  else:
    _antnum_to_bl_2048(ant1, ant2, _bl, nbls)

  return baseline

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _lla_from_xyz(
  numpy.float64_t[:, ::1] xyz,
):
  cdef Py_ssize_t ind
  cdef int ndim = 2
  cdef int n_pts = xyz.shape[1]
  cdef numpy.npy_intp * dims = [3, <numpy.npy_intp>n_pts]

  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] lla = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_FLOAT64, 0)
  cdef numpy.float64_t[:, ::1] _lla = lla

  cdef numpy.float64_t gps_p, gps_theta

  # see wikipedia geodetic_datum and Datum transformations of
  # GPS positions PDF in docs/references folder
  for ind in range(n_pts):
    gps_p = sqrt(xyz[0, ind] ** 2 + xyz[1, ind] ** 2)
    gps_theta = atan2(xyz[2, ind] * _gps_a, gps_p * _gps_b)

    _lla[0, ind] = atan2(
      xyz[2, ind] + _ep2 * _gps_b * sin(gps_theta) ** 3,
      gps_p - _e2 * _gps_a * cos(gps_theta) ** 3,
    )

    _lla[1, ind] = atan2(xyz[1, ind], xyz[0, ind])

    _lla[2, ind] = (gps_p / cos(lla[0, ind])) - _gps_a / sqrt(1.0 - _e2 * sin(lla[0, ind]) ** 2)

  return lla

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _xyz_from_latlonalt(
  numpy.float64_t[::1] _lat,
  numpy.float64_t[::1] _lon,
  numpy.float64_t[::1] _alt,
):
  cdef Py_ssize_t i
  cdef int ndim = 2
  cdef int n_pts = _lat.shape[0]
  cdef numpy.npy_intp * dims = [3, <numpy.npy_intp>n_pts]

  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] xyz = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_FLOAT64, 0)
  cdef numpy.float64_t[:, ::1] _xyz = xyz

  cdef numpy.float64_t  sin_lat, cos_lat, sin_lon, cos_lon, gps_n

  for ind in range(n_pts):
    sin_lat = sin(_lat[ind])
    sin_lon = sin(_lon[ind])

    cos_lat = cos(_lat[ind])
    cos_lon = cos(_lon[ind])

    gps_n = _gps_a / sqrt(1.0 - _e2 * sin_lat ** 2)

    _xyz[0, ind] = (gps_n + _alt[ind]) * cos_lat * cos_lon
    _xyz[1, ind] = (gps_n + _alt[ind]) * cos_lat * sin_lon

    _xyz[2, ind] = (b_div_a2 * gps_n + _alt[ind]) * sin_lat
  return xyz

# this function takes memoryviews as inputs
# that is why _lat, _lon, and _alt are indexed below to get the 0th entry
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[numpy.float64_t, ndim=2] _ENU_from_ECEF(
  numpy.float64_t[:, ::1] xyz,
  numpy.float64_t[::1] _lat,
  numpy.float64_t[::1] _lon,
  numpy.float64_t[::1] _alt,
):
  cdef Py_ssize_t i
  cdef int ndim = 2
  cdef int nblts = xyz.shape[1]
  cdef numpy.npy_intp * dims =  [3, <numpy.npy_intp> nblts]
  cdef numpy.float64_t xyz_use[3]

  cdef numpy.float64_t sin_lat, cos_lat, sin_lon, cos_lon

  # we want a memoryview of the xyz of the center
  # this looks a little silly but we don't have to define 2 different things
  cdef numpy.float64_t[:] xyz_center = _xyz_from_latlonalt(_lat, _lon, _alt).T[0]

  cdef numpy.ndarray[numpy.float64_t, ndim=2] _enu = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_FLOAT64, 0)
  cdef numpy.float64_t[:, ::1] enu = _enu

  sin_lat = sin(_lat[0])
  cos_lat = cos(_lat[0])

  sin_lon = sin(_lon[0])
  cos_lon = cos(_lon[0])

  for i in range(nblts):
    xyz_use[0] = xyz[0, i] - xyz_center[0]
    xyz_use[1] = xyz[1, i] - xyz_center[1]
    xyz_use[2] = xyz[2, i] - xyz_center[2]

    enu[0, i] = -sin_lon * xyz_use[0] + cos_lon * xyz_use[1]
    enu[1, i] = (
      - sin_lat * cos_lon * xyz_use[0]
      - sin_lat * sin_lon * xyz_use[1]
      + cos_lat * xyz_use[2]
    )
    enu[2, i] = (
      cos_lat * cos_lon * xyz_use[0]
      + cos_lat * sin_lon * xyz_use[1]
      + sin_lat * xyz_use[2]
    )

  return _enu

# this function takes memoryviews as inputs
# that is why _lat, _lon, and _alt are indexed below to get the 0th entry
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t] _ECEF_from_ENU(
  numpy.float64_t[:, ::1] enu,
  numpy.float64_t[::1] _lat,
  numpy.float64_t[::1] _lon,
  numpy.float64_t[::1] _alt,
):
  cdef Py_ssize_t i
  cdef int ndim = 2
  cdef int nblts = enu.shape[1]
  cdef numpy.npy_intp * dims = [3, <numpy.npy_intp>nblts]
  cdef numpy.float64_t sin_lat, cos_lat, sin_lon, cos_lon

  # allocate memory then make memory view for faster access
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _xyz = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_FLOAT64, 0)
  cdef numpy.float64_t[:, ::1] xyz = _xyz

  # we want a memoryview of the xyz of the center
  # this looks a little silly but we don't have to define 2 different things
  cdef numpy.float64_t[:] xyz_center = _xyz_from_latlonalt(_lat, _lon, _alt).T[0]

  sin_lat = sin(_lat[0])
  cos_lat = cos(_lat[0])

  sin_lon = sin(_lon[0])
  cos_lon = cos(_lon[0])

  for i in range(nblts):
    xyz[0, i] = (
      - sin_lat * cos_lon * enu[1, i]
      - sin_lon * enu[0, i]
      + cos_lat * cos_lon * enu[2, i]
      + xyz_center[0]
    )
    xyz[1, i] = (
      - sin_lat * sin_lon * enu[1, i]
      + cos_lon * enu[0, i]
      + cos_lat * sin_lon * enu[2, i]
      + xyz_center[1]
    )
    xyz[2, i] = cos_lat * enu[1, i] + sin_lat * enu[2, i] + xyz_center[2]

  return _xyz

# inital_uvw is a memoryviewed array as an input
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _phase_uvw(
    numpy.float64_t ra,
    numpy.float64_t dec,
    numpy.float64_t[:, ::1] initial_uvw
):
  cdef int i
  cdef int ndim = 2
  cdef int nuvw = initial_uvw.shape[1]
  cdef numpy.npy_intp * dims = [3, <numpy.npy_intp>nuvw]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] uvw = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_FLOAT64, 0)

  # make a memoryview for the numpy array in c
  cdef numpy.float64_t[:, ::1] _uvw = uvw

  cdef numpy.float64_t sin_ra, cos_ra, sin_dec, cos_dec

  sin_ra = sin(ra)
  cos_ra = cos(ra)
  sin_dec = sin(dec)
  cos_dec = cos(dec)

  for i in range(nuvw):
    _uvw[0, i] = - sin_ra * initial_uvw[0, i] + cos_ra * initial_uvw[1, i]

    _uvw[1, i] = (
      - sin_dec * cos_ra * initial_uvw[0, i]
      - sin_dec * sin_ra * initial_uvw[1, i]
      + cos_dec * initial_uvw[2, i]
    )

    _uvw[2, i] = (
      cos_dec * cos_ra * initial_uvw[0, i]
      + cos_dec * sin_ra * initial_uvw[1, i]
      + sin_dec * initial_uvw[2, i]
    )
  return uvw

# uvw is a memoryviewed array as an input
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _unphase_uvw(
    numpy.float64_t ra,
    numpy.float64_t dec,
    numpy.float64_t[:, ::1] uvw
):
  cdef int i
  cdef int ndim = 2
  cdef int nuvw = uvw.shape[1]
  cdef numpy.npy_intp * dims = [3, <numpy.npy_intp>nuvw]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] unphased_uvw = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_FLOAT64, 0)

  # make a memoryview for the numpy array in c
  cdef numpy.float64_t[:, ::1] _u_uvw = unphased_uvw

  cdef numpy.float64_t sin_ra, cos_ra, sin_dec, cos_dec

  sin_ra = sin(ra)
  cos_ra = cos(ra)
  sin_dec = sin(dec)
  cos_dec = cos(dec)

  for i in range(nuvw):
    _u_uvw[0, i] = (
      - sin_ra * uvw[0, i]
      - sin_dec * cos_ra * uvw[1, i]
      + cos_dec * cos_ra * uvw[2, i]
    )

    _u_uvw[1, i] = (
      cos_ra * uvw[0, i]
      - sin_dec * sin_ra * uvw[1, i]
      + cos_dec * sin_ra * uvw[2, i]
    )

    _u_uvw[2, i] = cos_dec * uvw[1, i] + sin_dec * uvw[2, i]

  return unphased_uvw
