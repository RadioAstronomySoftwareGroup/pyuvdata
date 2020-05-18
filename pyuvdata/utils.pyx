# distutils: language = c
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# python imports
import numpy as np
import warnings
from .utils import gps_a, gps_b, e_squared, e_prime_squared
# cython imports
cimport numpy
cimport cython
from libc.math cimport sin, cos, sqrt, atan2

# parameters for transforming between xyz & lat/lon/alt
# make c-viewed versions of these variables
cdef numpy.float64_t _gps_a = gps_a
cdef numpy.float64_t _gps_b = gps_b
cdef numpy.float64_t _e2 = e_squared
cdef numpy.float64_t _ep2 = e_prime_squared

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple baseline_to_antnums(numpy.ndarray[ndim=1, dtype=numpy.int_t] baseline):
  cdef unsigned long n = baseline.size
  cdef numpy.ndarray[ndim=1, dtype=numpy.int_t] ant1 = np.empty(n, dtype=np.int)
  cdef numpy.ndarray[ndim=1, dtype=numpy.int_t] ant2 = np.empty(n, dtype=np.int)
  cdef long _min = baseline.min()
  cdef int i
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  cdef numpy.int_t[::1] _a1 = ant1
  cdef numpy.int_t[::1] _a2 = ant2
  cdef numpy.int_t[::1] _bl = baseline

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
  cdef numpy.ndarray[ndim=1, dtype=numpy.int64_t] baselines = np.empty(n, dtype=np.int_)
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
  cdef numpy.ndarray[dtype=numpy.int64_t, ndim=1] baselines = np.empty(n, dtype=np.int_)
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  cdef numpy.int64_t[::1] _bl = baselines
  cdef numpy.int64_t[::1] _a1 = ant1
  cdef numpy.int64_t[::1] _a2 = ant2

  with nogil:
    for i in range(n):
      _bl[i] = 256 * (_a1[i] + 1) + (_a2[i] + 1)
  return baselines

def antnums_to_baseline(
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
cpdef numpy.ndarray[dtype=numpy.float64_t] _xyz_from_latlonalt(
    numpy.float64_t[::1] _lat,
    numpy.float64_t[::1] _lon,
    numpy.float64_t[::1] _alt,
):
    cdef int n_pts = len(_lat)
    cdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] xyz = np.empty((n_pts, 3))
    cdef numpy.float64_t _gps_a = gps_a
    cdef numpy.float64_t _gps_b = gps_b
    cdef numpy.float64_t _e2 = e_squared

    # create a memoryview
    cdef numpy.float64_t[:, ::1] _xyz = xyz

    cdef numpy.float64_t gps_n
    with nogil:
      for i in range(n_pts):
        gps_n = _gps_a / sqrt(1.0 - _e2 * sin(_lat[i]) ** 2)

        _xyz[i, 0] = (gps_n + _alt[i]) * cos(_lat[i]) * cos(_lon[i])
        _xyz[i, 1] = (gps_n + _alt[i]) * cos(_lat[i]) * sin(_lon[i])

        _xyz[i, 2] = (_gps_b ** 2 / _gps_a ** 2 * gps_n + _alt[i]) * sin(_lat[i])

    return xyz.squeeze()
