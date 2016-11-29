"""Tests for common utility functions."""
import nose.tools as nt
import uvdata
import numpy as np

ref_latlonalt = (-26.7 * np.pi / 180.0, 116.7 * np.pi / 180.0, 377.8)
ref_xyz = (-2562123.42683, 5094215.40141, -2848728.58869)


def test_XYZ_from_LatLonAlt():
    """Test conversion from lat/lon/alt to topocentric xyz with reference values."""
    out_xyz = uvdata.XYZ_from_LatLonAlt(ref_latlonalt[0], ref_latlonalt[1],
                                        ref_latlonalt[2])
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    nt.assert_true(np.allclose(ref_xyz, out_xyz, rtol=0, atol=1e-3))


def test_LatLonAlt_from_XYZ():
    """Test conversion from topocentric xyz to lat/lon/alt with reference values."""
    out_latlonalt = uvdata.LatLonAlt_from_XYZ(ref_xyz)
    # Got reference by forcing http://www.oc.nps.edu/oc2902w/coord/llhxyz.htm
    # to give additional precision.
    nt.assert_true(np.allclose(ref_latlonalt, out_latlonalt, rtol=0, atol=1e-3))
    nt.assert_raises(ValueError, uvdata.LatLonAlt_from_XYZ, ref_latlonalt)
