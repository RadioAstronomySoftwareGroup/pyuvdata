# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True

# python imports
import enum

# cython imports
cimport cython
cimport numpy
from libc.math cimport atan2, cos, sin, sqrt

numpy.import_array()


cdef class Ellipsoid:
  cdef readonly numpy.float64_t gps_a, gps_b, e_squared, e_prime_squared, b_div_a2

  @cython.cdivision
  def __init__(self, numpy.float64_t gps_a, numpy.float64_t gps_b):
    self.gps_a = gps_a
    self.gps_b = gps_b
    self.b_div_a2 = (self.gps_b / self.gps_a)**2
    self.e_squared = (1 - self.b_div_a2)
    self.e_prime_squared = (self.b_div_a2**-1 - 1)


# A python interface for different celestial bodies
cdef earth = Ellipsoid(6378137, 6356752.31424518)
Earth = earth

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _lla_from_xyz(
  numpy.float64_t[:, ::1] xyz,
  Ellipsoid body,
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
    gps_theta = atan2(xyz[2, ind] * body.gps_a, gps_p * body.gps_b)

    _lla[0, ind] = atan2(
      xyz[2, ind] + body.e_prime_squared * body.gps_b * sin(gps_theta) ** 3,
      gps_p - body.e_squared * body.gps_a * cos(gps_theta) ** 3,
    )

    _lla[1, ind] = atan2(xyz[1, ind], xyz[0, ind])

    _lla[2, ind] = (gps_p / cos(lla[0, ind])) - body.gps_a / sqrt(1.0 - body.e_squared * sin(lla[0, ind]) ** 2)

  return lla

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _xyz_from_latlonalt(
  numpy.float64_t[::1] _lat,
  numpy.float64_t[::1] _lon,
  numpy.float64_t[::1] _alt,
  Ellipsoid body,
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

    gps_n = body.gps_a / sqrt(1.0 - body.e_squared * sin_lat ** 2)

    _xyz[0, ind] = (gps_n + _alt[ind]) * cos_lat * cos_lon
    _xyz[1, ind] = (gps_n + _alt[ind]) * cos_lat * sin_lon

    _xyz[2, ind] = (body.b_div_a2 * gps_n + _alt[ind]) * sin_lat
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
  Ellipsoid body,
):
  cdef Py_ssize_t i
  cdef int ndim = 2
  cdef int nblts = xyz.shape[1]
  cdef numpy.npy_intp * dims =  [3, <numpy.npy_intp> nblts]
  cdef numpy.float64_t xyz_use[3]

  cdef numpy.float64_t sin_lat, cos_lat, sin_lon, cos_lon

  # we want a memoryview of the xyz of the center
  # this looks a little silly but we don't have to define 2 different things
  cdef numpy.float64_t[:] xyz_center = _xyz_from_latlonalt(_lat, _lon, _alt, body).T[0]

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
  Ellipsoid body,
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
  cdef numpy.float64_t[:] xyz_center = _xyz_from_latlonalt(_lat, _lon, _alt, body).T[0]

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
