use pyo3::{
    pymodule,
    types::{PyAnyMethods, PyModule, PyModuleMethods},
    Bound, PyErr, PyResult, Python,
};
use std::ffi::CString;
mod bls;

mod coordinates;

fn _warn(msg: &str) {
    Python::with_gil(|py| {
        let user_warning = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, &user_warning, &CString::new(msg).unwrap(), 0)?;
        Ok::<_, PyErr>(())
    })
    .expect("Unable to issue python warning.");
}

fn register_child_module(
    parent_module: &Bound<'_, PyModule>,
    child_name: &str,
    fn_mod: fn(m: &Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), child_name)?;
    fn_mod(&child_module)?;
    parent_module.add_submodule(&child_module)?;
    parent_module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item(child_name, &child_module)?;

    Ok(())
}

#[pymodule]
fn _utils_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_child_module(m, "_bls", bls::_bls)?;
    register_child_module(m, "_coordinates", coordinates::_coordinates)?;
    Ok(())
}
