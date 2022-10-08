# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2022 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

# distutils: language = c
# cython: linetrace=True

# python imports
import warnings

# cython imports

cimport cython
cimport numpy
from libc.math cimport fabs
from numpy.math cimport PI

# This initializes the numpy 1.7 c-api.
# cython 3.0 will do this by default.
# We may be able to just remove this then.
numpy.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.npy_bool] find_healpix_indices(
    numpy.float64_t[::1] theta_grid,
    numpy.float64_t[::1] phi_grid,
    numpy.float64_t[::1] theta_hpx,
    numpy.float64_t[::1] phi_hpx,
    numpy.float64_t pixel_resolution
):
  cdef Py_ssize_t itheta, iphi, ipix
  cdef numpy.float64_t theta_h, phi_h, theta_g, phi_g, dist, dist_test, dtheta, dphi, phi0
  cdef numpy.npy_bool found_pixel
  cdef int ndim = 1
  cdef int n_theta = theta_grid.shape[0]
  cdef int n_phi = phi_grid.shape[0]
  cdef int n_pix = theta_hpx.shape[0]
  cdef numpy.npy_intp *dims = [<numpy.npy_intp>n_pix]

  cdef numpy.ndarray[dtype=numpy.npy_bool, ndim=1] in_map = numpy.PyArray_ZEROS(
      ndim, dims, numpy.NPY_BOOL, 0
  )
  cdef numpy.npy_bool[::1] _in_map = in_map

  # pre-compute test distance
  dist_test = 4 * pixel_resolution**2

  for ipix in range(n_pix):
      theta_h = theta_hpx[ipix]
      phi_h = phi_hpx[ipix]

      found_pixel = False
      for itheta in range(n_theta):
          if found_pixel:
              break
          theta_g = theta_grid[itheta]

          # normalize theta_g to be between -pi and pi
          while theta_g < -PI:
              theta_g += 2 * PI
          while theta_g > PI:
              theta_g -= 2 * PI

          # Now, we need to fix up negative values of theta_g, adding a phase
          # flip in phi as necessary. This comes from the fact that (-theta,
          # phi) maps to (theta, phi + pi).
          if theta_g < 0:
              theta_g = fabs(theta_g)
              phi0 = PI
          else:
              phi0 = 0

          # compute difference in theta angle
          dtheta = fabs(theta_h - theta_g)

          # only look through phi if we're within the pixel resolution in theta
          dtheta *= dtheta

          if dtheta < dist_test:
              for iphi in range(n_phi):
                  phi_g = phi_grid[iphi] + phi0
                  # normalize phi_g to be between 0 and 2*pi
                  while phi_g < 0:
                      phi_g += 2 * PI
                  while phi_g > 2 * PI:
                      phi_g -= 2 * PI

                  # compute "actual" phi separation
                  # dphi_max = pi because of periodicity
                  dphi = fabs(phi_h - phi_g)
                  if dphi > PI:
                      dphi = fabs(2 * PI - dphi)

                  dist = dtheta + dphi ** 2
                  if dist < dist_test:
                      _in_map[ipix] = True
                      found_pixel = True
                      break

  return in_map
