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
        PyErr::warn(
            py,
            &user_warning,
            &CString::new(msg).expect("Warning message cannot be converted to CString."),
            0,
        )?;
        Ok::<_, PyErr>(())
    })
    .expect("Unable to issue python warning.");
}

// we have to do some oddly convolute things to make sure we can
// import he submodules as expected from the python side
fn register_child_module(
    parent_module: &Bound<'_, PyModule>,
    child_name: &str,
    fn_mod: fn(m: &Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    // initialize a submodule with the given name
    let child_module = PyModule::new(parent_module.py(), child_name)?;
    // init the actual submodule
    fn_mod(&child_module)?;
    // attach the submodule to the name defined above
    parent_module.add_submodule(&child_module)?;
    // add the correct entries into the __dict__ so it can be imported
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
