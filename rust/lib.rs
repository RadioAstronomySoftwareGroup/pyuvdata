use std::mem::MaybeUninit;

use ndarray::{s, Array, ArrayView, Axis, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pyclass, pymodule, types::PyModule, PyResult, Python};

const BLS_2_147_483_648: u64 = 2_u64.pow(16) + 2_u64.pow(22);
const BLS_2048: u64 = 2_u64.pow(16);

fn _baseline_to_antnums(bls_array: &[u64]) -> Array<u64, Ix2> {
    let nbls = bls_array.len();
    // unwrap is safe here because we require nbls >= 1 in python section
    let bls_min = *bls_array.iter().min().unwrap();

    let (offset, modulus) = if bls_min >= BLS_2_147_483_648 {
        (BLS_2_147_483_648, 2_147_483_648)
    } else if bls_min >= BLS_2048 {
        (BLS_2048, 2048)
    } else {
        (0, 256)
    };
    let mut ants_out = Array::<u64, Ix2>::uninit((2, nbls));

    // these asserts seem silly and obvious,
    // but they help the compiler optimize out some bounds checks
    assert_eq!(ants_out.shape()[1], nbls);
    assert_eq!(ants_out.shape()[0], 2);

    // Taking nbls slices for each antenna for ease of indexing.
    let (mut _ants1, mut _ants2) = ants_out.multi_slice_mut((s![0, ..], s![1, ..]));
    assert_eq!(_ants1.len(), nbls);
    assert_eq!(_ants2.len(), nbls);

    bls_array
        .iter()
        .map(|x| x - offset)
        .enumerate()
        .for_each(|(index, _bl)| {
            let a1 = _bl % modulus;
            _ants2[index] = MaybeUninit::new(a1);
            _ants1[index] = MaybeUninit::new((_bl - a1) / modulus);
        });

    // We have to tell the compiler that we have initialized all elements of the array.
    unsafe { ants_out.assume_init() }
}

fn _antnums_to_baseline(
    ant1: &[u64],
    ant2: &[u64],
    attempt_256: bool,
    nants_less_2048: bool,
    use_miriad_convention: bool,
) -> Array<u64, Ix1> {
    let nbls = ant1.len();

    // unwrap is safe here because we require nbls >= 1 in python section
    let ants_max = *ant1.iter().chain(ant2).max().unwrap();

    let (offset, modulus) = if ants_max < 256 && nants_less_2048 && attempt_256 {
        (0, 256)
    } else if ants_max < 2048 && nants_less_2048 {
        (BLS_2048, 2048)
    } else {
        (BLS_2_147_483_648, 2_147_483_648_u64)
    };

    let mut bls_out = Array::<u64, Ix1>::uninit(nbls);
    // these asserts seem silly and obvious,
    // but they help the compiler optimize out some bounds checks
    assert_eq!(ant1.len(), nbls);
    assert_eq!(ant2.len(), nbls);
    assert_eq!(bls_out.len(), nbls);

    ant1.iter()
        .zip(ant2)
        .enumerate()
        .for_each(|(ind, (a1, a2))| {
            if use_miriad_convention && a2 < &255 {
                bls_out[ind] = MaybeUninit::new(256 * a1 + a2);
            } else {
                bls_out[ind] = MaybeUninit::new(modulus * a1 + a2 + offset);
            }
        });

    // We have to tell the compiler that we have initialized all elements of the array.
    unsafe { bls_out.assume_init() }
}

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

#[pyclass]
#[derive(Debug, Clone, Copy)]
/// Celestial Ellipsoids used for Geodetic to Geocentric conversions.
enum Body {
    /// Earth Assumes a semi-major axis of 6378137m and a semi-minor axis of 6356752.31424518m.
    Earth,
    /// moon data taken from https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
    /// with radius from spice_utils
    Moon,
}
impl Body {
    fn get_body(&self) -> Ellipsoid {
        match self {
            Body::Earth => Ellipsoid::new(6378137_f64, 6356752.31424518_f64),
            Body::Moon => Ellipsoid::new(1737.1e3, 1737.1e3 * (1.0 - 0.0012)),
        }
    }
}

fn _xyz_from_latlonalt(lat: &[f64], lon: &[f64], alt: &[f64], body: Body) -> Array<f64, Ix2> {
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

fn _lla_from_xyz(xyz: ArrayView<f64, Ix2>, body: Body) -> Array<f64, Ix2> {
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

fn _enu_from_ecef(
    xyz: ArrayView<f64, Ix2>,
    lat: f64,
    lon: f64,
    alt: f64,
    body: Body,
) -> Array<f64, Ix2> {
    let mut enu = Array::uninit(xyz.raw_dim());

    let xyz_center = _xyz_from_latlonalt(&[lat], &[lon], &[alt], body);

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

fn _ecef_from_enu(
    enu: ArrayView<f64, Ix2>,
    lat: f64,
    lon: f64,
    alt: f64,
    body: Body,
) -> Array<f64, Ix2> {
    let mut xyz = Array::uninit(enu.raw_dim());

    let xyz_center = _xyz_from_latlonalt(&[lat], &[lon], &[alt], body);

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
fn _utils_rs<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    m.add_class::<Body>()?;

    #[pyfn(m)]
    fn xyz_from_lla<'py>(
        py: Python<'py>,
        lat: PyReadonlyArray1<'py, f64>,
        lon: PyReadonlyArray1<'py, f64>,
        alt: PyReadonlyArray1<'py, f64>,
        body: Body,
    ) -> &'py PyArray2<f64> {
        // we're assuming lat, lon, and alt are all the same length
        // and have n_points > 1. This should be checked on the python side.
        _xyz_from_latlonalt(
            lat.as_slice().unwrap(),
            lon.as_slice().unwrap(),
            alt.as_slice().unwrap(),
            body,
        )
        .into_pyarray(py)
    }

    #[pyfn(m)]
    fn lla_from_xyz<'py>(
        py: Python<'py>,
        xyz: PyReadonlyArray2<'py, f64>,
        body: Body,
    ) -> &'py PyArray2<f64> {
        _lla_from_xyz(xyz.as_array(), body).into_pyarray(py)
    }

    #[pyfn(m)]
    fn enu_from_ecef<'py>(
        py: Python<'py>,
        xyz: PyReadonlyArray2<'py, f64>,
        lat: f64,
        lon: f64,
        alt: f64,
        body: Body,
    ) -> &'py PyArray2<f64> {
        _enu_from_ecef(xyz.as_array(), lat, lon, alt, body).into_pyarray(py)
    }

    #[pyfn(m)]
    fn ecef_from_enu<'py>(
        py: Python<'py>,
        enu: PyReadonlyArray2<'py, f64>,
        lat: f64,
        lon: f64,
        alt: f64,
        body: Body,
    ) -> &'py PyArray2<f64> {
        _ecef_from_enu(enu.as_array(), lat, lon, alt, body).into_pyarray(py)
    }

    #[pyfn(m)]
    fn baseline_to_antnums<'py>(
        py: Python<'py>,
        bls: PyReadonlyArray1<'py, u64>,
    ) -> &'py PyArray2<u64> {
        // unwrap is safe here because he require nbls >= 1 in python section
        _baseline_to_antnums(bls.as_slice().unwrap()).into_pyarray(py)
    }

    #[pyfn(m)]
    fn antnums_to_baseline<'py>(
        py: Python<'py>,
        ant1: PyReadonlyArray1<'py, u64>,
        ant2: PyReadonlyArray1<'py, u64>,
        attempt256: bool,
        nants_less2048: bool,
        use_miriad_convention: bool,
    ) -> &'py PyArray1<u64> {
        // unwrap is safe here because he require nants >= 1 in python section
        _antnums_to_baseline(
            ant1.as_slice().unwrap(),
            ant2.as_slice().unwrap(),
            attempt256,
            nants_less2048,
            use_miriad_convention,
        )
        .into_pyarray(py)
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use ndarray::{stack, Array};

    use super::*;

    const ANTS: &[u64] = &[
        0_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ];

    #[test]
    fn bls_to_ants256() {
        let bls = Vec::from_iter(1..50_u64);
        let ants = Array::from_shape_vec((2, 49), ANTS.to_vec()).unwrap();

        let antnums = _baseline_to_antnums(bls.as_slice());

        assert_eq!(ants, antnums)
    }

    #[test]
    fn bls_to_ants2048() {
        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16)));
        let ants = Array::from_shape_vec((2, 49), ANTS.to_vec()).unwrap();

        let antnums = _baseline_to_antnums(bls.as_slice());

        assert_eq!(ants, antnums)
    }

    #[test]
    fn bls_to_antslarge() {
        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16) + 2_u64.pow(22)));
        let ants = Array::from_shape_vec((2, 49), ANTS.to_vec()).unwrap();

        let antnums = _baseline_to_antnums(bls.as_slice());

        assert_eq!(ants, antnums)
    }

    #[test]
    fn ants_to_bls() {
        let ant1 = &ANTS[..49];
        let ant2 = &ANTS[49..];

        let bls = Vec::from_iter(1..50_u64);

        let bls_out = _antnums_to_baseline(ant1, ant2, true, true, false);

        assert_eq!(Array::<u64, Ix1>::from(bls), bls_out)
    }

    #[test]
    fn ants_to_bls2048() {
        let ant1 = &ANTS[..49];
        let ant2 = &ANTS[49..];

        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16)));

        let bls_out = _antnums_to_baseline(ant1, ant2, false, true, false);

        assert_eq!(Array::<u64, Ix1>::from(bls), bls_out)
    }

    #[test]
    fn ants_to_blslarge() {
        let ant1 = &ANTS[..49];
        let ant2 = &ANTS[49..];

        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16) + 2_u64.pow(22)));

        let bls_out = _antnums_to_baseline(ant1, ant2, false, false, false);

        assert_eq!(Array::<u64, Ix1>::from(bls), bls_out)
    }

    #[test]
    fn xyz_from_lla() {
        let ref_xyz =
            Array::from_shape_vec((3, 1), vec![-2562123.42683, 5094215.40141, -2848728.58869])
                .expect("Cannot make same shape.");
        let ref_latlonalt = (
            -26.7 * std::f64::consts::PI / 180.0,
            116.7 * std::f64::consts::PI / 180.0,
            377.8,
        );

        let xyz_out = _xyz_from_latlonalt(
            &[ref_latlonalt.0],
            &[ref_latlonalt.1],
            &[ref_latlonalt.2],
            Body::Earth,
        );

        approx::assert_abs_diff_eq!(ref_xyz, xyz_out, epsilon = 1e-5)
    }

    #[test]
    fn lla_from_xyz() {
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

        let lla_out = _lla_from_xyz(ref_xyz.view(), Body::Earth);

        approx::assert_abs_diff_eq!(ref_latlonalt, lla_out, epsilon = 1e-5)
    }

    #[test]
    fn enu_from_ecef() {
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

        let enu = _enu_from_ecef(xyz.view(), center_lat, center_lon, center_alt, Body::Earth);

        approx::assert_abs_diff_eq!(ref_enu, enu, epsilon = 1e-5)
    }

    #[test]
    fn ecef_from_enu() {
        let center_lat = -30.7215261207 * std::f64::consts::PI / 180.0;
        let center_lon = 21.4283038269 * std::f64::consts::PI / 180.0;
        let center_alt = 1051.7;

        let ref_ecef = stack![
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

        let enu = stack![
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

        let ecef = _ecef_from_enu(enu.view(), center_lat, center_lon, center_alt, Body::Earth);
        approx::assert_abs_diff_eq!(ref_ecef, ecef, epsilon = 1e-5)
    }
}
