# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True

# cython imports
cimport cython
cimport numpy
from libc.stdlib cimport strtod
from libc.string cimport strncmp, strtok, memcpy

from cython.parallel import prange, parallel
from libc.math cimport exp, pi, sqrt

# This initializes the numpy 1.7 c-api.
# cython 3.0 will do this by default.
# We may be able to just remove this then.
numpy.import_array()

ctypedef fused int_like:
  numpy.int_t
  int

cdef inline int_like pfb_mapper(int_like index):
  # the polyphase filter bank maps inputs to outputs, which the MWA
  # correlator then records as the antenna indices.
  # the following is taken from mwa_build_lfiles/mwac_utils.c
  # inputs are mapped to outputs via pfb_mapper as follows
  # (from mwa_build_lfiles/antenna_mapping.h):
  # floor(index/4) + index%4 * 16 = input
  # for the first 64 outputs, pfb_mapper[output] = input
  return index // 4  + index % 4 * 16

cpdef dict input_output_mapping():
  """Build a mapping dictionary from pfb input to output numbers."""
  cdef int p, i
  cdef dict pfb_inputs_to_outputs = {}
  # build a mapper for all 256 inputs
  for p in range(4):
    for i in range(64):
      pfb_inputs_to_outputs[pfb_mapper(i) + p * 64] = p * 64 + i

  return pfb_inputs_to_outputs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void generate_map(
  dict ants_to_pf,
  numpy.int32_t[::1] map_inds,
  numpy.npy_bool[::1] conj,
):
  """Compute the map between pfb inputs and antenna numbersself.

  This function operates on input `map_inds` and `conj` arrays inplace.

  Parameters
  ----------
  map_inds : 1D numpy array of type int32
    The array into which mapping indices will be populated.
  conj : 1D numpy array of type np.bool_
    The array into which indices of baselines to conjugate will be populated.

  """
  cdef int ant1, ant2, p1, p2, pol_ind, bls_ind, out_ant1, out_ant2
  cdef int out_p1, out_p2, ind1_1, ind1_2, ind2_1, ind2_2, data_index

  cdef dict in_to_out = input_output_mapping()

  for ant1 in range(128):
    for ant2 in range(ant1, 128):
      for p1 in range(2):
        for p2 in range(2):
          # generate the indices in self.data_array for this combination
          # baselines are ordered (0,0),(0,1),...,(0,127),(1,1),.....
          # polarizion of 0 (1) corresponds to y (x)
          pol_ind = int(2 * p1 + p2)
          bls_ind = int(128 * ant1 - ant1 * (ant1 + 1) / 2 + ant2)
          # find the pfb input indices for this combination
          ind1_1 = ants_to_pf[(ant1, p1)]
          ind1_2 = ants_to_pf[(ant2, p2)]

          # find the pfb output indices
          ind2_1 = in_to_out[ind1_1]
          ind2_2 = in_to_out[ind1_2]

          out_ant1 = int(ind2_1 / 2)
          out_ant2 = int(ind2_2 / 2)
          out_p1 = ind2_1 % 2
          out_p2 = ind2_2 % 2
          # the correlator has ind2_2 <= ind2_1 except for
          # redundant data. The redundant data is not perfectly
          # redundant; sometimes the values of redundant data
          # are off by one in the imaginary part.
          # For consistency, we are ignoring the redundant values
          # that have ind2_2 > ind2_1
          if ind2_2 > ind2_1:
            # get the index for the data
            data_index = int(
              2 * out_ant2 * (out_ant2 + 1)
              + 4 * out_ant1
              + 2 * out_p2
              + out_p1
            )
            # need to take the complex conjugate of the data
            map_inds[bls_ind * 4 + pol_ind] = data_index
            conj[bls_ind * 4 + pol_ind] = True
          else:
            data_index = int(
              2 * out_ant1 * (out_ant1 + 1)
              + 4 * out_ant2
              + 2 * out_p1
              + out_p2
            )
            map_inds[bls_ind * 4 + pol_ind] = data_index

  return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _make_length_array(
  const int max_length,
  char[:, ::1] cable_lens,
  numpy.float64_t[::1] cable_array
):
  cdef char * token
  cdef char clen[30]
  # "the velocity factor of electic fields in RG-6 like coax"
  # from MWA_Tools/CONV2UVFITS/convutils.h
  cdef float v_factor = 1.204
  cdef int n_cables = cable_lens.shape[0]

  # check if the cable length already has the velocity factor applied
  for i in range(n_cables):
    # copy the location in memory to our character array
    memcpy(clen, &cable_lens[i, 0], max_length)
    # attempt to split on the character "_"
    token = strtok(clen, b"_")

    if strncmp(token, b"EL", 2) == 0:
      # has already had the velocity factor applied
      # grab the next bit of the string after EL
      token = strtok(NULL, b"_")
      cable_array[i] = strtod(token, NULL)
    else:
      cable_array[i] = strtod(token, NULL) * v_factor
  return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] get_cable_len_diffs(
  numpy.int_t[::1] ant1_array,
  numpy.int_t[::1] ant2_array,
  char[:, ::1] cable_lens,
):
  """Computer the difference in cable lengths for each baseline.

  The inputs are one dimensional and will be both C and F contiguous but
  we want to declare the memory layout in a consistent way with the rest of the code.

  Parameters
  ----------
  ant1_array : numpy array of type int_t
    Array of antenna 1 numbers for each baseline.
  ant2_array : numpy array of type int_t
    Array of antenna 2 numbers for each baseline.
  cable_lens : numpy array
    Array of strings of the length of the cable for each antenna.
    However it is cast to uint8 and reshaped as (cable_lens.size, cable_lens.dytype.itemsize)
    see more about this approach here: https://stackoverflow.com/a/28777163

  Returns
  -------
  cable_diffs : numpy array of type float64
    Array of length Nblts with the difference of cable lengths for each baseline.

  """
  cdef Py_ssize_t i
  cdef int Nblts = ant1_array.shape[0]
  cdef int n_cables = cable_lens.shape[0]
  cdef int max_length = cable_lens.shape[1]

  cdef numpy.npy_intp * dims_cables = [n_cables]
  cdef numpy.npy_intp * dims_diffs = [Nblts]
  cdef numpy.float64_t[::1] cable_array = numpy.PyArray_ZEROS(1, dims_cables, numpy.NPY_FLOAT64, 0)
  cdef numpy.ndarray[dtype=numpy.float64_t, ndim=1] cable_diffs = numpy.PyArray_ZEROS(1, dims_diffs, numpy.NPY_FLOAT64, 0)
  cdef numpy.float64_t[::1] _cable_diffs = cable_diffs

  # fill out array of cable lengths
  _make_length_array(max_length, cable_lens, cable_array)

  for i in range(Nblts):
    _cable_diffs[i] = cable_array[ant2_array[i]] - cable_array[ant1_array[i]]

  return cable_diffs


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef numpy.ndarray[ndim=2, dtype=numpy.float64_t] _compute_khat(
  numpy.float64_t[:, ::1] x,
  numpy.float64_t[::1] sig1,
  numpy.float64_t[::1] sig2,
):
  cdef int ndim = 2
  cdef numpy.npy_intp * dims = [x.shape[0], x.shape[1]]
  cdef numpy.ndarray[ndim=2, dtype=numpy.float64_t] khat = numpy.PyArray_ZEROS(ndim, dims, numpy.NPY_FLOAT64, 0)

  cdef numpy.float64_t[:, ::1] _khat = khat

  ind1 = numpy.PyArray_Reshape(
    numpy.PyArray_Arange(0.5, 7.5, 1, numpy.NPY_FLOAT64),
    (7, 1),
  )
  cdef numpy.float64_t[:, ::1]  j_ind = ind1 / sig1
  cdef numpy.float64_t[:, ::1]  k_ind = ind1 / sig2

  cdef Py_ssize_t i, j, k, l

  if x.size > 800:  # pragma: nocover
    with nogil, parallel():
      for j in prange(x.shape[1]):
        for i in range(x.shape[0]):
          for l in range(k_ind.shape[0]):
            for k in range(j_ind.shape[0]):
              _khat[i, j] += (
                1./ (pi * sqrt( 1 - x[i, j] ** 2)) * (
                  exp(-1. / (2 * (1 - x[i,j] ** 2)) * (j_ind[k, j] ** 2 + k_ind[l, j] ** 2 - 2 * x[i, j] * j_ind[k, j] * k_ind[l,j]))
                  + exp(-1. / (2 * (1 - x[i,j] ** 2)) * (j_ind[k, j] ** 2 + k_ind[l, j] ** 2 + 2 * x[i, j] * j_ind[k, j] * k_ind[l,j]))
                )
              )
  else:
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for l in range(k_ind.shape[0]):
          for k in range(j_ind.shape[0]):
            _khat[i, j] += (
              1./ (pi * sqrt( 1 - x[i, j] ** 2)) * (
                exp(-1. / (2 * (1 - x[i,j] ** 2)) * (j_ind[k, j] ** 2 + k_ind[l, j] ** 2 - 2 * x[i, j] * j_ind[k, j] * k_ind[l,j]))
                + exp(-1. / (2 * (1 - x[i,j] ** 2)) * (j_ind[k, j] ** 2 + k_ind[l, j] ** 2 + 2 * x[i, j] * j_ind[k, j] * k_ind[l,j]))
              )
            )

  return khat


cpdef numpy.ndarray[dtype=numpy.float64_t] get_khat(rho, sig1, sig2):
  """Compute generalized k-hat matrix for van vleck correction.

  Generalized for 1 or two dimenional rho inputs.

  Parameters
  ----------
  rho : numpy array
    Array of rho inputs.
  sig1 : array_like
    Array of sigma inputs corresponding to antenna 1.
  sig2: array_like
    Array of sigma inputs corresponding to antenna 2.

  """
  # NPY_ARRAY_OUT_ARRAY is C-contiguous, writeable, aligned versions of the inputs
  rho = numpy.PyArray_FROMANY(rho, numpy.NPY_FLOAT64, 0, 2, numpy.NPY_ARRAY_OUT_ARRAY)
  sig1 = numpy.PyArray_FROMANY(sig1, numpy.NPY_FLOAT64, 1, 1, numpy.NPY_ARRAY_OUT_ARRAY)
  sig2 = numpy.PyArray_FROMANY(sig2, numpy.NPY_FLOAT64, 1, 1, numpy.NPY_ARRAY_OUT_ARRAY)

  cdef int ndim = numpy.PyArray_NDIM(rho)
  cdef bint squeeze = False
  if ndim == 1:
    squeeze = True
    rho = numpy.PyArray_Reshape(rho, (1, -1))

  cdef numpy.ndarray[ndim=2, dtype=numpy.float64_t] khat = _compute_khat(rho, sig1, sig2)

  if squeeze:
    return numpy.PyArray_Squeeze(khat)

  return khat


@cython.wraparound(False)
@cython.boundscheck(False)
# this function could be reworked to return a C array
cdef numpy.ndarray[ndim=2, dtype=numpy.float64_t[:, ::1]] _get_cheby_coeff(
  numpy.float64_t[:, :, ::1] rho_coeff,
  numpy.int64_t[::1] sv_inds_right1,
  numpy.int64_t[::1] sv_inds_right2,
  numpy.float64_t[::1] ds1,
  numpy.float64_t[::1] ds2
):
  """
  Perform a bilinear interpolation to get Chebyshev coefficients.

  Explicitly assumes the grid spacing is 0.01.

  Parameters
  ----------
  rho_coeff : numpy array of type float64_t
    Array of Chebyeshev coefficients to interpolate over.
  sv_inds_right1 : numpy array of type int64_t
    Array of right indices nearest to sigmas for antenna 1.
  sv_inds_right2 : numpy array of type int64_t
    Array of right indices nearest to sigmas for antenna 2.
  ds1 : numpy array of type float64_t
    Array of differences between sigmas for antenna 1 and nearest sigmas at
    sv_inds_right_1.
  ds1 : numpy array of type float64_t
    Array of differences between sigmas for antenna 2 and nearest sigmas at
    sv_inds_right_2.

  Returns
  -------
  t : numpy array of type float64_t
    Array of coefficients for the first three odd Chebyshev polynomials for each
    pair of sigmas.

  """
  cdef int i
  cdef int j
  cdef int n = ds1.shape[0]

  cdef int ndim = 2
  cdef numpy.npy_intp * dims = [n, 3]
  cdef numpy.ndarray[ndim=2, dtype=numpy.float64_t] t = numpy.PyArray_ZEROS(ndim, dims, numpy.NPY_FLOAT64, 0)


  for i in prange(n, nogil=True):
    for j in range(3):
      t[i, j] = 1e4 * (
        rho_coeff[(sv_inds_right1[i] - 1), (sv_inds_right2[i] - 1), j] * ds1[i] * ds2[i]
        + rho_coeff[(sv_inds_right1[i] - 1), sv_inds_right2[i], j] * ds1[i] * (0.01 - ds2[i])
        + rho_coeff[sv_inds_right1[i], (sv_inds_right2[i] - 1), j] * (0.01 - ds1[i]) * ds2[i]
        + rho_coeff[sv_inds_right1[i], sv_inds_right2[i], j] * (0.01 - ds1[i]) * (0.01 - ds2[i])
      )

  return t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void van_vleck_cheby(
  numpy.float64_t[:, ::1] kap,
  numpy.float64_t[:, :, ::1] rho_coeff,
  numpy.int64_t[::1] sv_inds_right1,
  numpy.int64_t[::1] sv_inds_right2,
  numpy.float64_t[::1] ds1,
  numpy.float64_t[::1] ds2
):
  """
  Compute Van Vleck corrected cross-correlations using Chebyshev polynomials.

  This function operates on input `kap` array inplace.

  Parameters
  ----------
  kap : numpy array of type float64_t
    Array of values to correct.
  rho_coeff : numpy array of type float64_t
    Array of Chebyeshev coefficients to interpolate over.
  sv_inds_right1 : numpy array of type int64_t
    Array of right indices nearest to sigmas for antenna 1.
  sv_inds_right2 : numpy array of type int64_t
    Array of right indices nearest to sigmas for antenna 2.
  ds1 : numpy array of type float64_t
    Array of differences between sigmas for antenna 1 and nearest sigmas at
    sv_inds_right_1.
  ds1 : numpy array of type float64_t
    Array of differences between sigmas for antenna 2 and nearest sigmas at
    sv_inds_right_2.

  """
  cdef numpy.float64_t[:, ::1] t = _get_cheby_coeff(rho_coeff, sv_inds_right1, sv_inds_right2, ds1, ds2)
  cdef int n = kap.shape[1]
  cdef int i
  cdef int j

  for i in prange(n, nogil=True):
    kap[0, i] = (
      kap[0, i] * (t[i, 0] - 3 * t[i, 1] + 5 * t[i, 2])
      + kap[0, i] ** 3 * (4 * t[i, 1] - 20 * t[i, 2])
      + kap[0, i] ** 5 * (16 * t[i, 2])
    )
    kap[1, i] = (
      kap[1, i] * (t[i, 0] - 3 * t[i, 1] + 5 * t[i, 2])
      + kap[1, i] ** 3 * (4 * t[i, 1] - 20 * t[i, 2])
      + kap[1, i] ** 5 * (16 * t[i, 2])
    )

  return
