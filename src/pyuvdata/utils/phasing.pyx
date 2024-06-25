# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2024 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True

# cython imports

cimport cython
cimport numpy
from libc.math cimport cos, sin

numpy.import_array()


# inital_uvw is a memoryviewed array as an input
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _old_uvw_calc(
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
cpdef numpy.ndarray[dtype=numpy.float64_t, ndim=2] _undo_old_uvw_calc(
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
