# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# python imports
import numpy as np
import warnings
# cython imports
cimport numpy
cimport cython
from libc.math cimport sin, cos, sqrt, atan2

# in order to not have circular dependencies
# define transformation parameters here
# parameters for transforming between xyz & lat/lon/alt
gps_b = 6356752.31424518
gps_a = 6378137
e_squared = 6.69437999014e-3
e_prime_squared = 6.73949674228e-3

# make c-viewed versions of these variables
cdef numpy.float64_t _gps_a = gps_a
cdef numpy.float64_t _gps_b = gps_b
cdef numpy.float64_t _e2 = e_squared
cdef numpy.float64_t _ep2 = e_prime_squared
# this one is useful in the xyz from lla calculation
cdef numpy.float64_t b_div_a2 = (_gps_b / _gps_a)**2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple baseline_to_antnums(numpy.ndarray[ndim=1, dtype=numpy.int64_t] baseline):
  cdef unsigned long n = baseline.size
  cdef numpy.ndarray[ndim=1, dtype=numpy.int64_t] ant1 = np.empty(n, dtype=np.int64)
  cdef numpy.ndarray[ndim=1, dtype=numpy.int64_t] ant2 = np.empty(n, dtype=np.int64)
  cdef long _min = baseline.min()
  cdef int i
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  cdef numpy.int64_t[::1] _a1 = ant1
  cdef numpy.int64_t[::1] _a2 = ant2
  cdef numpy.int64_t[::1] _bl = baseline

  with nogil:
    for i in range(n):
      if _min > 2 ** 16:
        _a2[i] = (_bl[i] - 2 ** 16) % 2048 - 1
        _a1[i] = (_bl[i] - 2 ** 16 - (_a2[i] + 1)) // 2048 - 1
      else:
        _a2[i] = (_bl[i]) % 256 - 1
        _a1[i] = (_bl[i] - (_a2[i] + 1)) // 256 - 1
  return ant1, ant2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[dtype=numpy.int64_t] _antnum_to_bl_2048(
numpy.ndarray[ndim=1, dtype=numpy.int64_t] ant1,
numpy.ndarray[ndim=1, dtype=numpy.int64_t] ant2,
):
  cdef unsigned long n = ant1.size
  cdef unsigned int i
  cdef numpy.ndarray[ndim=1, dtype=numpy.int64_t] baselines = np.empty(n, dtype=np.int64)
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  cdef numpy.int64_t[::1] _bl = baselines
  cdef numpy.int64_t[::1] _a1 = ant1
  cdef numpy.int64_t[::1] _a2 = ant2

  with nogil:
    for i in range(n):
      _bl[i] = 2048 * (_a1[i] + 1) + (_a2[i] + 1) + 2 ** 16

  return baselines

@cython.boundscheck(False)
@cython.wraparound(False)
cdef numpy.ndarray[dtype=numpy.int64_t] _antnum_to_bl_256(
numpy.ndarray[ndim=1, dtype=numpy.int64_t] ant1,
numpy.ndarray[ndim=1, dtype=numpy.int64_t] ant2,
):
  cdef unsigned long n = ant1.size
  cdef unsigned int i
  cdef numpy.ndarray[dtype=numpy.int64_t, ndim=1] baselines = np.empty(n, dtype=np.int64)
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  cdef numpy.int64_t[::1] _bl = baselines
  cdef numpy.int64_t[::1] _a1 = ant1
  cdef numpy.int64_t[::1] _a2 = ant2

  with nogil:
    for i in range(n):
      _bl[i] = 256 * (_a1[i] + 1) + (_a2[i] + 1)
  return baselines

cpdef numpy.ndarray[dtype=numpy.int64_t] antnums_to_baseline(
  numpy.ndarray[dtype=numpy.int64_t, ndim=1] ant1,
  numpy.ndarray[dtype=numpy.int64_t, ndim=1] ant2,
  bint attempt256=False
):
  if attempt256 and np.max([ant1, ant2]) < 255:
    baseline = _antnum_to_bl_256(ant1, ant2)

  elif attempt256 and np.max([ant1, ant2]) >=  255:
    message = (
      "antnums_to_baseline: found > 256 antennas, using "
      "2048 baseline indexing. Beware compatibility "
      "with CASA etc"
    )
    warnings.warn(message)
    baseline = _antnum_to_bl_2048(ant1, ant2)

  else:
    baseline = _antnum_to_bl_2048(ant1, ant2)

  return baseline

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef _latlonalt_from_xyz(numpy.float64_t[:, ::1] xyz):
  cdef int n = xyz.shape[0]
  cdef int i

  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] latitude = np.empty(n, dtype=np.float64)
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] longitude = np.empty(n, dtype=np.float64)
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] altitude = np.empty(n, dtype=np.float64)
  # create some memoryviews
  cdef numpy.float64_t[::1] _lat = latitude
  cdef numpy.float64_t[::1] _lon = longitude
  cdef numpy.float64_t[::1] _alt = altitude

  # see wikipedia geodetic_datum and Datum transformations of
  # GPS positions PDF in docs/references folder
  cdef numpy.float64_t gps_p, gps_theta, gps_n
  for i in range(n):
    gps_p = sqrt(xyz[i, 0] ** 2 + xyz[i, 1] ** 2)
    gps_theta = atan2(xyz[i, 2] * _gps_a, gps_p * _gps_b)

    _lat[i] = atan2(
      xyz[i, 2] + _ep2 * _gps_b * sin(gps_theta) ** 3,
      gps_p - _e2 * _gps_a * cos(gps_theta) ** 3,
    )

    _lon[i] = atan2(xyz[i, 1], xyz[i, 0])

    gps_n = _gps_a / sqrt(1.0 - _e2 * sin(_lat[i]) ** 2)
    _alt[i] = (gps_p / cos(_lat[i])) - gps_n

  return latitude, longitude, altitude


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _xyz_from_latlonalt(
    numpy.float64_t[::1] _lat,
    numpy.float64_t[::1] _lon,
    numpy.float64_t[::1] _alt,
):
  cdef Py_ssize_t i
  cdef int n_pts = _lat.shape[0]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] xyz = np.empty((3, n_pts), dtype=np.float64)
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
cpdef numpy.ndarray[dtype=numpy.float64_t] _ENU_from_ECEF(
    numpy.float64_t[:, ::1] xyz,
    numpy.float64_t[::1] _lat,
    numpy.float64_t[::1] _lon,
    numpy.float64_t[::1] _alt,
):
  cdef int i
  cdef int nblts = xyz.shape[0]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] enu = np.empty((nblts, 3), dtype=np.float64)

  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] xyz_center = _xyz_from_latlonalt(_lat, _lon, _alt)
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] xyz_use = np.empty(3, dtype=np.float64)
  # make a memoryview for the numpy array in c
  cdef numpy.float64_t[:, ::1] _enu = enu

  with nogil:
    for i in range(nblts):
      xyz_use[0] = xyz[i, 0] - xyz_center[0]
      xyz_use[1] = xyz[i, 1] - xyz_center[1]
      xyz_use[2] = xyz[i, 2] - xyz_center[2]

      _enu[i, 0] = -sin(_lon[0]) * xyz_use[0] + cos(_lon[0]) * xyz_use[1]
      _enu[i, 1] = (
        - sin(_lat[0]) * cos(_lon[0]) * xyz_use[0]
        - sin(_lat[0]) * sin(_lon[0]) * xyz_use[1]
        + cos(_lat[0]) * xyz_use[2]
      )
      _enu[i, 2] = (
        cos(_lat[0]) * cos(_lon[0]) * xyz_use[0]
        + cos(_lat[0]) * sin(_lon[0]) * xyz_use[1]
        + sin(_lat[0]) * xyz_use[2]
      )

  return enu

# this function takes memoryviews as inputs
# that is why _lat, _lon, and _alt are indexed below to get the 0th entry
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t] _ECEF_FROM_ENU(
    numpy.float64_t[:, ::1] enu,
    numpy.float64_t[::1] _lat,
    numpy.float64_t[::1] _lon,
    numpy.float64_t[::1] _alt,
):
  cdef int i
  cdef int nblts = enu.shape[0]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] xyz = np.empty((nblts, 3), dtype=np.float64)
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] xyz_center = _xyz_from_latlonalt(_lat, _lon, _alt)

  # make a memoryview for the numpy array in c
  cdef numpy.float64_t[:, ::1] _xyz = xyz
  with nogil:
    for i in range(nblts):
      _xyz[i, 0] = (
        - sin(_lat[0]) * cos(_lon[0]) * enu[i, 1]
        - sin(_lon[0]) * enu[i, 0]
        + cos(_lat[0]) * cos(_lon[0]) * enu[i, 2]
        + xyz_center[0]
      )
      _xyz[i, 1] = (
        - sin(_lat[0]) * sin(_lon[0]) * enu[i, 1]
        + cos(_lon[0]) * enu[i, 0]
        + cos(_lat[0]) * sin(_lon[0]) * enu[i, 2]
        + xyz_center[1]
      )
      _xyz[i, 2] = cos(_lat[0]) * enu[i, 1] + sin(_lat[0]) * enu[i, 2] + xyz_center[2]

  return xyz

# inital_uvw is a memoryviewed array as an input
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t] _phase_uvw(
    numpy.float64_t ra,
    numpy.float64_t dec,
    numpy.float64_t[:, ::1] initial_uvw
):
  cdef int i
  cdef int nuvw = initial_uvw.shape[0]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] uvw = np.empty((nuvw, 3), dtype=np.float64)

  # make a memoryview for the numpy array in c
  cdef numpy.float64_t[:, ::1] _uvw = uvw
  with nogil:
    for i in range(nuvw):
      _uvw[i, 0] = - sin(ra) * initial_uvw[i, 0] + cos(ra) * initial_uvw[i, 1]
      _uvw[i, 1] = (
        - sin(dec) * cos(ra) * initial_uvw[i, 0]
        - sin(dec) * sin(ra) * initial_uvw[i, 1]
        + cos(dec) * initial_uvw[i, 2]
      )
      _uvw[i, 2] = (
        cos(dec) * cos(ra) * initial_uvw[i, 0]
        + cos(dec) * sin(ra) * initial_uvw[i, 1]
        + sin(dec) * initial_uvw[i, 2]
      )
  return uvw

# uvw is a memoryviewed array as an input
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t] _unphase_uvw(
    numpy.float64_t ra,
    numpy.float64_t dec,
    numpy.float64_t[:, ::1] uvw
):
  cdef int i
  cdef int nuvw = uvw.shape[0]
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] unphased_uvw = np.empty((nuvw, 3), dtype=np.float64)

  # make a memoryview for the numpy array in c
  cdef numpy.float64_t[:, ::1] _u_uvw = unphased_uvw
  with nogil:
    for i in range(nuvw):
      _u_uvw[i, 0] = (
        - sin(ra) * uvw[i, 0]
        - sin(dec) * cos(ra) * uvw[i, 1]
        + cos(dec) * cos(ra) * uvw[i, 2]
      )

      _u_uvw[i, 1] = (
        cos(ra) * uvw[i, 0]
        - sin(dec) * sin(ra) * uvw[i, 1]
        + cos(dec) * sin(ra) * uvw[i, 2]
      )

      _u_uvw[i, 2] = cos(dec) * uvw[i, 1] + sin(dec) * uvw[i, 2]

  return unphased_uvw
