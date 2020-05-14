# distutils: language = c
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# python imports
import numpy as np
import warnings
# cython imports
cimport numpy
cimport cython

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
