"""Tests for common utility functions."""
import nose.tools as nt
import uvdata
import numpy as np


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to topocentric xyz with reference values."""
    out_xyz = uvdata.XYZ_from_LatLonAlt(-26.7 * np.pi / 180.0,
                                        116.7 * np.pi / 180.0, 377.8)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)
    nt.assert_true(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))


def test_LatLonAlt_from_XYZ():
    """Test conversion from topocentric xyz to lat/lon/alt with reference values."""
    out_latlonalt = uvdata.LatLonAlt_from_XYZ([-2562123.42683, 5094215.40141,
                                               -2848728.58869])
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
    nt.assert_true(np.allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3))
