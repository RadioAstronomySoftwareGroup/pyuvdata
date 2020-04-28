# distutils: language = c
# python imports
import numpy as np
# cython imports
cimport cython
cimport numpy


cpdef generate_map(
  dict ants_to_pf,
  dict in_to_out,
  numpy.ndarray[ndim=1, dtype=numpy.int32_t] map_inds,
  numpy.ndarray[ndim=1, dtype=numpy.int_t] conj
):
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
                (ind1_1, ind1_2) = (
                    ants_to_pf[(ant1, p1)],
                    ants_to_pf[(ant2, p2)],
                )
                # find the pfb output indices
                (ind2_1, ind2_2) = (
                    in_to_out[(ind1_1)],
                    in_to_out[(ind1_2)],
                )
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
  return map_inds, conj
