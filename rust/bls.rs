use std::mem::MaybeUninit;

use super::_warn;
use ndarray::{s, Array, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

const BLS_2_147_483_648: u64 = 2_u64.pow(16) + 2_u64.pow(22);
const BLS_2048: u64 = 2_u64.pow(16);

fn _baseline_to_antnums(bls_array: &[u64]) -> Array<u64, Ix2> {
    let nbls = bls_array.len();
    // we're okay with getting 0 if the bl array is empty
    let bls_min = *bls_array.iter().min().unwrap_or(&0);

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
            _ants2[index] = MaybeUninit::new(_bl % modulus);
            _ants1[index] = MaybeUninit::new(_bl / modulus);
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

    // we're okay with getting 0 if the ant arrays are empty.
    let ants_max = *ant1.iter().chain(ant2).max().unwrap_or(&0);

    let (offset, modulus) = if ants_max < 256 && nants_less_2048 && attempt_256 {
        (0, 256)
    } else if ants_max < 2048 && nants_less_2048 {
        if attempt_256 {
            _warn(
                "antnums_to_baseline: found antenna numbers > 255\
                    , using 2048 baseline indexing.",
            )
        }
        (BLS_2048, 2048)
    } else {
        if attempt_256 {
            _warn(
                "antnums_to_baseline: found antenna numbers > 2047 or \
                Nants_telescope > 2048, using 2147483648 baseline indexing.",
            );
        }
        (BLS_2_147_483_648, 2_147_483_648_u64)
    };

    let mut bls_out = Array::<u64, Ix1>::uninit(nbls);
    // these asserts seem silly and obvious,
    // but they help the compiler optimize out some bounds checks
    assert_eq!(ant1.len(), nbls);
    assert_eq!(ant2.len(), nbls);
    assert_eq!(bls_out.len(), nbls);

    if use_miriad_convention {
        ant1.iter()
            .zip(ant2)
            .enumerate()
            .for_each(|(ind, (a1, a2))| {
                if a2 <= &255 {
                    bls_out[ind] = MaybeUninit::new(256 * a1 + a2);
                } else {
                    bls_out[ind] = MaybeUninit::new(modulus * a1 + a2 + offset);
                }
            });
    } else {
        ant1.iter()
            .zip(ant2)
            .enumerate()
            .for_each(|(ind, (a1, a2))| {
                bls_out[ind] = MaybeUninit::new(modulus * a1 + a2 + offset);
            });
    }

    // We have to tell the compiler that we have initialized all elements of the array.
    unsafe { bls_out.assume_init() }
}

#[pymodule]
pub(crate) fn _bls(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn baseline_to_antnums<'py>(
        py: Python<'py>,
        bls: PyReadonlyArray1<'py, u64>,
    ) -> Bound<'py, PyArray2<u64>> {
        use numpy::IntoPyArray;
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
    ) -> Bound<'py, PyArray1<u64>> {
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
    use ndarray::Array;

    use super::*;

    const ANTS: &[u64] = &[
        0_u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ];

    #[test]
    fn bls_to_ants_256() {
        let bls = Vec::from_iter(1..50_u64);
        let ants = Array::from_shape_vec((2, 49), ANTS.to_vec()).unwrap();

        let antnums = _baseline_to_antnums(bls.as_slice());

        assert_eq!(ants, antnums)
    }

    #[test]
    fn bls_to_ants_2048() {
        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16)));
        let ants = Array::from_shape_vec((2, 49), ANTS.to_vec()).unwrap();

        let antnums = _baseline_to_antnums(bls.as_slice());

        assert_eq!(ants, antnums)
    }

    #[test]
    fn bls_to_ants_large() {
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
    fn ants_to_bls_2048() {
        let ant1 = &ANTS[..49];
        let ant2 = &ANTS[49..];

        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16)));

        let bls_out = _antnums_to_baseline(ant1, ant2, false, true, false);

        assert_eq!(Array::<u64, Ix1>::from(bls), bls_out)
    }

    #[test]
    fn ants_to_bls_large() {
        let ant1 = &ANTS[..49];
        let ant2 = &ANTS[49..];

        let bls = Vec::from_iter((1..50_u64).map(|x| x + 2_u64.pow(16) + 2_u64.pow(22)));

        let bls_out = _antnums_to_baseline(ant1, ant2, false, false, false);

        assert_eq!(Array::<u64, Ix1>::from(bls), bls_out)
    }
}
