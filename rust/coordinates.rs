use lazy_static::lazy_static;
use ndarray::{Array, ArrayView, Axis};
use numpy::{IntoPyArray, Ix2, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    pyclass, pymodule,
    sync::GILOnceCell,
    types::{PyDict, PyModule},
    PyResult, Python,
};
use std::{collections::HashMap, mem::MaybeUninit};

#[derive(Clone, Debug, Copy)]
struct Ellipsoid {
    gps_a: f64,

    gps_b: f64,

    e_squared: f64,

    e_prime_squared: f64,

    b_div_a2: f64,
}
impl Ellipsoid {
    fn new(gps_a: f64, gps_b: f64) -> Self {
        let b_div_a2 = (gps_b / gps_a).powi(2);

        Self {
            gps_a,
            gps_b,
            e_squared: 1.0 - b_div_a2,
            e_prime_squared: b_div_a2.powi(-1) - 1.0,
            b_div_a2,
        }
    }
}

static LIST_CELL: GILOnceCell<HashMap<String, Ellipsoid>> = GILOnceCell::new();

fn get_selenoid<'a>(py: Python<'_>, selenoid: &'a str) -> &'a Ellipsoid {
    LIST_CELL
        .get_or_try_init(py, || {
            let lunar_module = PyModule::import(py, "lunarsky")?;
            let lunar_moon = lunar_module.getattr("moon")?.downcast::<PyModule>()?;

            let selenoids: &PyDict = lunar_moon.getattr("SELENOIDS")?.downcast::<PyDict>()?;

            Ok::<_, pyo3::PyErr>(
                selenoids
                    .iter()
                    .map_while(|(key, selenoid)| {
                        Some((key.extract::<String>().ok()?, {
                            let radius = selenoid
                                .getattr("_equatorial_radius")
                                .ok()?
                                .call_method1("to_value", ("m",))
                                .ok()?
                                .extract::<f64>()
                                .ok()?;
                            let flattening = selenoid
                                .getattr("_flattening")
                                .ok()?
                                .extract::<f64>()
                                .ok()?;
                            Ellipsoid::new(radius, radius * (1.0 - flattening))
                        }))
                    })
                    .collect(),
            )
        })
        .expect("Failed to initialize lunarsky ellipsoids.")
        .get(selenoid)
        .expect("No lunary sky selenoid information on this build.")
}

lazy_static! {
    static ref EARTH: Ellipsoid = Ellipsoid::new(6378137_f64, 6356752.31424518_f64);
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
/// Celestial Ellipsoids used for Geodetic to Geocentric conversions.
enum Body {
    /// Earth Assumes a semi-major axis of 6378137m and a semi-minor axis of 6356752.31424518m.
    Earth,
    /// moon data taken from https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
    /// with radius from spice_utils
    Moon_sphere,
    Moon_gsfc,
    Moon_grail23,
    Moon_ce1lamgeo,
}
impl Body {
    fn get_body(&self) -> &Ellipsoid {
        match self {
            Body::Earth => &EARTH,
            Body::Moon_sphere => Python::with_gil(|py| get_selenoid(py, "SPHERE")),
            Body::Moon_gsfc => Python::with_gil(|py| get_selenoid(py, "GSFC")),
            Body::Moon_grail23 => Python::with_gil(|py| get_selenoid(py, "GRAIL23")),
            Body::Moon_ce1lamgeo => Python::with_gil(|py| get_selenoid(py, "CE-1-LAM-GEO")),
        }
    }
}

fn xyz_from_lla(lat: &[f64], lon: &[f64], alt: &[f64], body: Body) -> Array<f64, Ix2> {
    let mut xyz = Array::<f64, Ix2>::uninit((3, lat.len()));

    assert_eq!(lat.len(), lon.len());
    assert_eq!(lat.len(), alt.len());
    assert_eq!(xyz.shape()[1], lat.len());

    let ellipsoid = body.get_body();

    for (_lat, _lon, _alt, mut _xyz) in itertools::izip!(lat, lon, alt, xyz.axis_iter_mut(Axis(1)))
    {
        let sin_lat = _lat.sin();
        let cos_lat = _lat.cos();

        let sin_lon = _lon.sin();
        let cos_lon = _lon.cos();

        let gps_n = ellipsoid.gps_a / (1.0 - ellipsoid.e_squared * sin_lat.powi(2)).sqrt();

        _xyz[0] = MaybeUninit::new((gps_n + _alt) * cos_lat * cos_lon);
        _xyz[1] = MaybeUninit::new((gps_n + _alt) * cos_lat * sin_lon);
        _xyz[2] = MaybeUninit::new((ellipsoid.b_div_a2 * gps_n + _alt) * sin_lat);
    }

    unsafe { xyz.assume_init() }
}

fn lla_from_xyz(xyz: ArrayView<f64, Ix2>, body: Body) -> Array<f64, Ix2> {
    let mut lla = Array::<f64, Ix2>::uninit(xyz.raw_dim());
    let ellipsoid = body.get_body();

    for (_xyz, mut _lla) in xyz.axis_iter(Axis(1)).zip(lla.axis_iter_mut(Axis(1))) {
        let gps_p = (_xyz[0].powi(2) + _xyz[1].powi(2)).sqrt();
        let gps_theta = (_xyz[2] * ellipsoid.gps_a).atan2(gps_p * ellipsoid.gps_b);

        // create a temporary variable so we can reference it without needing unsafe pointer references.
        let lla0 = (_xyz[2]
            + ellipsoid.e_prime_squared * ellipsoid.gps_b * gps_theta.sin().powi(3))
        .atan2(gps_p - ellipsoid.e_squared * ellipsoid.gps_a * gps_theta.cos().powi(3));

        _lla[0] = MaybeUninit::new(lla0);

        _lla[1] = MaybeUninit::new(_xyz[1].atan2(_xyz[0]));

        _lla[2] = MaybeUninit::new(
            (gps_p / lla0.cos())
                - ellipsoid.gps_a / (1.0 - ellipsoid.e_squared * lla0.sin().powi(2)).sqrt(),
        );
    }
    unsafe { lla.assume_init() }
}

fn enu_from_ecef(
    xyz: ArrayView<f64, Ix2>,
    lat: f64,
    lon: f64,
    alt: f64,
    body: Body,
) -> Array<f64, Ix2> {
    let mut enu = Array::uninit(xyz.raw_dim());

    let xyz_center = xyz_from_lla(&[lat], &[lon], &[alt], body);

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();

    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    for (_xyz, mut enu) in xyz.axis_iter(Axis(1)).zip(enu.axis_iter_mut(Axis(1))) {
        let x = _xyz[0] - xyz_center[[0, 0]];
        let y = _xyz[1] - xyz_center[[1, 0]];
        let z = _xyz[2] - xyz_center[[2, 0]];

        enu[0] = MaybeUninit::new(-sin_lon * x + cos_lon * y);
        enu[1] = MaybeUninit::new(-sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z);
        enu[2] = MaybeUninit::new(cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z);
    }

    unsafe { enu.assume_init() }
}

fn ecef_from_enu(
    enu: ArrayView<f64, Ix2>,
    lat: f64,
    lon: f64,
    alt: f64,
    body: Body,
) -> Array<f64, Ix2> {
    let mut xyz = Array::uninit(enu.raw_dim());

    let xyz_center = xyz_from_lla(&[lat], &[lon], &[alt], body);

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();

    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    for (_enu, mut _xyz) in enu.axis_iter(Axis(1)).zip(xyz.axis_iter_mut(Axis(1))) {
        _xyz[0] = MaybeUninit::new(
            -sin_lat * cos_lon * _enu[1] - sin_lon * _enu[0]
                + cos_lat * cos_lon * _enu[2]
                + xyz_center[[0, 0]],
        );

        _xyz[1] = MaybeUninit::new(
            -sin_lat * sin_lon * _enu[1]
                + cos_lon * _enu[0]
                + cos_lat * sin_lon * _enu[2]
                + xyz_center[[1, 0]],
        );

        _xyz[2] = MaybeUninit::new(cos_lat * _enu[1] + sin_lat * _enu[2] + xyz_center[[2, 0]]);
    }

    unsafe { xyz.assume_init() }
}

#[pymodule]
pub(crate) fn _coordinates<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    m.add_class::<Body>()?;
    #[pyfn(m)]
    fn _xyz_from_latlonalt<'py>(
        py: Python<'py>,
        lat: PyReadonlyArray1<'py, f64>,
        lon: PyReadonlyArray1<'py, f64>,
        alt: PyReadonlyArray1<'py, f64>,
        body: Body,
    ) -> &'py PyArray2<f64> {
        // we're assuming lat, lon, and alt are all the same length
        // and have n_points > 1. This should be checked on the python side.
        xyz_from_lla(
            lat.as_slice().unwrap(),
            lon.as_slice().unwrap(),
            alt.as_slice().unwrap(),
            body,
        )
        .into_pyarray(py)
    }

    #[pyfn(m)]
    fn _lla_from_xyz<'py>(
        py: Python<'py>,
        xyz: PyReadonlyArray2<'py, f64>,
        body: Body,
    ) -> &'py PyArray2<f64> {
        lla_from_xyz(xyz.as_array(), body).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "_ENU_from_ECEF")]
    fn _enu_from_ecef<'py>(
        py: Python<'py>,
        xyz: PyReadonlyArray2<'py, f64>,
        lat: f64,
        lon: f64,
        alt: f64,
        body: Body,
    ) -> &'py PyArray2<f64> {
        enu_from_ecef(xyz.as_array(), lat, lon, alt, body).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "_ECEF_from_ENU")]
    fn _ecef_from_enu<'py>(
        py: Python<'py>,
        enu: PyReadonlyArray2<'py, f64>,
        lat: f64,
        lon: f64,
        alt: f64,
        body: Body,
    ) -> &'py PyArray2<f64> {
        ecef_from_enu(enu.as_array(), lat, lon, alt, body).into_pyarray(py)
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use ndarray::{stack, Array};

    use super::*;

    #[test]
    fn test_xyz_from_lla() {
        let ref_xyz =
            Array::from_shape_vec((3, 1), vec![-2562123.42683, 5094215.40141, -2848728.58869])
                .expect("Cannot make same shape.");
        let ref_latlonalt = (
            -26.7 * std::f64::consts::PI / 180.0,
            116.7 * std::f64::consts::PI / 180.0,
            377.8,
        );

        let xyz_out = xyz_from_lla(
            &[ref_latlonalt.0],
            &[ref_latlonalt.1],
            &[ref_latlonalt.2],
            Body::Earth,
        );

        approx::assert_abs_diff_eq!(ref_xyz, xyz_out, epsilon = 1e-5)
    }

    #[test]
    fn test_lla_from_xyz() {
        let ref_xyz =
            Array::from_shape_vec((3, 1), vec![-2562123.42683, 5094215.40141, -2848728.58869])
                .expect("Cannot make same shape.");
        let ref_latlonalt = Array::from_shape_vec(
            (3, 1),
            vec![
                -26.7 * std::f64::consts::PI / 180.0,
                116.7 * std::f64::consts::PI / 180.0,
                377.8,
            ],
        )
        .unwrap();

        let lla_out = lla_from_xyz(ref_xyz.view(), Body::Earth);

        approx::assert_abs_diff_eq!(ref_latlonalt, lla_out, epsilon = 1e-5)
    }

    #[test]
    fn test_enu_from_ecef() {
        let center_lat = -30.7215261207 * std::f64::consts::PI / 180.0;
        let center_lon = 21.4283038269 * std::f64::consts::PI / 180.0;
        let center_alt = 1051.7;

        let ref_enu = stack![
            Axis(0),
            [
                -97.87631659,
                -17.87126443,
                -15.17316938,
                -33.19049252,
                -137.60520964,
                84.67346748,
                -42.84049408,
                32.28083937,
                -76.1094745,
                63.40285935
            ],
            [
                -72.7437482,
                16.09066646,
                27.45724573,
                58.21544651,
                -8.02964511,
                -59.41961437,
                -24.39698388,
                -40.09891961,
                -34.70965816,
                58.18410876
            ],
            [
                0.54883333,
                -0.35004539,
                -0.50007736,
                -0.70035299,
                -0.25148791,
                0.33916067,
                -0.02019057,
                0.16979185,
                0.06945155,
                -0.64058124
            ]
        ];

        let xyz = stack![
            Axis(0),
            [
                5109327.46674067,
                5109339.76407785,
                5109344.06370947,
                5109365.11297147,
                5109372.115673,
                5109266.94314734,
                5109329.89620962,
                5109295.13656657,
                5109337.21810468,
                5109329.85680612
            ],
            [
                2005130.57953031,
                2005221.35184577,
                2005225.93775268,
                2005214.8436201,
                2005105.42364036,
                2005302.93158317,
                2005190.65566222,
                2005257.71335575,
                2005157.78980089,
                2005304.7729239
            ],
            [
                -3239991.24516348,
                -3239914.4185286,
                -3239904.57048431,
                -3239878.02656316,
                -3239935.20415493,
                -3239979.68381865,
                -3239949.39266985,
                -3239962.98805772,
                -3239958.30386264,
                -3239878.08403833
            ]
        ];

        let enu = enu_from_ecef(xyz.view(), center_lat, center_lon, center_alt, Body::Earth);

        approx::assert_abs_diff_eq!(ref_enu, enu, epsilon = 1e-5)
    }
}
