use pyo3::{pymodule, types::PyModule, PyErr, PyResult, Python};

mod bls;

mod coordinates;

fn _warn(msg: &str) {
    Python::with_gil(|py| {
        let user_warning = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, user_warning, msg, 0)?;
        Ok::<_, PyErr>(())
    })
    .expect("Unable to issue python warning.");
}

#[pymodule]
fn _utils_rs<'py>(py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    let bls_mod = PyModule::new(py, "_utils_rs._bls")?;
    bls::_bls(py, bls_mod)?;
    m.add("_bls", bls_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("_utils_rs._bls", bls_mod)?;

    let coords = PyModule::new(py, "_utils_rs._coordinates")?;
    coordinates::_coordinates(py, coords)?;
    m.add("_coordinates", coords)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("_utils_rs._coordinates", coords)?;

    Ok(())
}
