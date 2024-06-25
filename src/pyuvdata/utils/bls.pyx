# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True

# python imports
import warnings

# cython imports

cimport cython
cimport numpy

numpy.import_array()

ctypedef fused int_or_float:
    numpy.uint64_t
    numpy.int64_t
    numpy.int32_t
    numpy.uint32_t
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
    numpy.uint64_t[::1] _bl,
    numpy.uint64_t[:, ::1] _ants,
    long nbls,
):
  cdef Py_ssize_t i

  for i in range(nbls):
    _ants[1, i] = (_bl[i]) % 256
    _ants[0, i] = (_bl[i] - (_ants[1, i])) // 256
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _bl_to_ant_2048(
    numpy.uint64_t[::1] _bl,
    numpy.uint64_t[:, ::1] _ants,
    int nbls
):
  cdef Py_ssize_t i
  for i in range(nbls):
    _ants[1, i] = (_bl[i] - 2 ** 16) % 2048
    _ants[0, i] = (_bl[i] - 2 ** 16 - (_ants[1, i])) // 2048
  return

# defining these constants helps cython not cast the large
# numbers as python ints
cdef numpy.uint64_t bl_large = 2 ** 16 + 2 ** 22
cdef numpy.uint64_t large_mod = 2147483648

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void _bl_to_ant_2147483648(
    numpy.uint64_t[::1] _bl,
    numpy.uint64_t[:, ::1] _ants,
    int nbls
):
  cdef Py_ssize_t i
  for i in range(nbls):
    _ants[1, i] = (_bl[i] - bl_large) % large_mod
    _ants[0, i] = (_bl[i] - bl_large - (_ants[1, i])) // large_mod
  return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.uint64_t, ndim=2] baseline_to_antnums(
    numpy.uint64_t[::1] _bl
):
  cdef numpy.uint64_t _min = arraymin(_bl)
  cdef long nbls = _bl.shape[0]
  cdef int ndim = 2
  cdef numpy.npy_intp * dims = [2, <numpy.npy_intp> nbls]
  cdef numpy.ndarray[ndim=2, dtype=numpy.uint64_t] ants = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_UINT64, 0)
  cdef numpy.uint64_t[:, ::1] _ants = ants

  if  _min >= (2 ** 16 + 2 ** 22):
    _bl_to_ant_2147483648(_bl, _ants, nbls)
  elif _min >= 2 ** 16:
    _bl_to_ant_2048(_bl, _ants, nbls)
  else:
    _bl_to_ant_256(_bl, _ants,  nbls)
  return ants

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _antnum_to_bl_2147483648(
  numpy.uint64_t[::1] ant1,
  numpy.uint64_t[::1] ant2,
  numpy.uint64_t[::1] baselines,
  int nbls,
):
  cdef Py_ssize_t i

  for i in range(nbls):
    baselines[i] = large_mod * (ant1[i]) + (ant2[i]) + bl_large
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _antnum_to_bl_2048(
  numpy.uint64_t[::1] ant1,
  numpy.uint64_t[::1] ant2,
  numpy.uint64_t[::1] baselines,
  int nbls,
):
  cdef Py_ssize_t i

  for i in range(nbls):
    baselines[i] = 2048 * (ant1[i]) + (ant2[i]) + 2 ** 16
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _antnum_to_bl_2048_miriad(
  numpy.uint64_t[::1] ant1,
  numpy.uint64_t[::1] ant2,
  numpy.uint64_t[::1] baselines,
  int nbls,
):
  cdef Py_ssize_t i

  for i in range(nbls):
    if ant2[i] > 255:      # MIRIAD uses 1-index antenna IDs
      baselines[i] = 2048 * (ant1[i]) + (ant2[i]) + 2 ** 16
    else:
      baselines[i] = 256 * (ant1[i]) + (ant2[i])
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _antnum_to_bl_256(
  numpy.uint64_t[::1] ant1,
  numpy.uint64_t[::1] ant2,
  numpy.uint64_t[::1] baselines,
  int nbls,
):
  cdef Py_ssize_t i
  # make views as c-contiguous arrays of a known dtype
  # effectivly turns the numpy array into a c-array
  for i in range(nbls):
    baselines[i] = 256 * (ant1[i]) + (ant2[i])
  return

cpdef numpy.ndarray[dtype=numpy.uint64_t] antnums_to_baseline(
  numpy.uint64_t[::1] ant1,
  numpy.uint64_t[::1] ant2,
  bint attempt256=False,
  bint nants_less2048=True,
  bint use_miriad_convention=False
):
  cdef int ndim = 1
  cdef int nbls = ant1.shape[0]
  cdef numpy.npy_intp * dims = [<numpy.npy_intp>nbls]
  cdef numpy.ndarray[ndim=1, dtype=numpy.uint64_t] baseline = numpy.PyArray_EMPTY(ndim, dims, numpy.NPY_UINT64, 0)
  cdef numpy.uint64_t[::1] _bl = baseline
  cdef bint less255
  cdef bint ants_less2048

  # to ensure baseline numbers are unambiguous,
  # use the 2048 calculation for antennas >= 256
  # and use the 2147483648 calculation for antennas >= 2048
  ants_less2048 = max(
    arraymax(ant1),
    arraymax(ant2),
  ) < 2048

  # Some UVFITS readers (e.g. MWA and AAVS) expect the
  # MIRIAD baseline convention.
  if use_miriad_convention:
      _antnum_to_bl_2048_miriad(ant1, ant2, _bl, nbls)

  elif attempt256:
    less256 = max(
      arraymax(ant1),
      arraymax(ant2),
    ) < 256

    if less256:
      _antnum_to_bl_256(ant1, ant2, _bl, nbls)

    elif ants_less2048 and nants_less2048:
        message = (
          "antnums_to_baseline: found antenna numbers > 255, using "
          "2048 baseline indexing."
        )
        warnings.warn(message)
        _antnum_to_bl_2048(ant1, ant2, _bl, nbls)
    else:
      message = (
        "antnums_to_baseline: found antenna numbers > 2047 or "
        "Nants_telescope > 2048, using 2147483648 baseline indexing."
      )
      warnings.warn(message)
      _antnum_to_bl_2147483648(ant1, ant2, _bl, nbls)

  elif ants_less2048 and nants_less2048:
    _antnum_to_bl_2048(ant1, ant2, _bl, nbls)

  else:
    _antnum_to_bl_2147483648(ant1, ant2, _bl, nbls)

  return baseline
