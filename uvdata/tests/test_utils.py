import nose.tools as nt
import astropy.time  # necessary for Jonnie's workflow help us all
import uvdata.utils as ut
import numpy as np


def test_XYZ_from_LatLonAlt():
    out_xyz = ut.XYZ_from_LatLonAlt(-26.7 * np.pi / 180.0,
                                    116.7 * np.pi / 180.0, 377.8)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)
    nt.assert_true(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))


def test_LatLonAlt_from_XYZ():
    out_latlonalt = ut.LatLonAlt_from_XYZ([-2562123.42683, 5094215.40141,
                                           -2848728.58869])
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
    nt.assert_true(np.allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3))
