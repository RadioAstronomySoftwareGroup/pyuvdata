/* Helpers for compatibility in the C modules between Python 2 and Python 3. */

#ifndef __AIPY_COMPAT_H
#define __AIPY_COMPAT_H

#if PY_MAJOR_VERSION >= 3
# define PyCapsule_Type PyCObject_Type
# define PyInt_AsLong PyLong_AsLong
# define PyInt_FromLong PyLong_FromLong
# define PyInt_Check PyLong_Check
# define PyString_FromString PyUnicode_FromString
# define PyString_Check PyUnicode_Check
# define PyString_Size PyUnicode_GET_LENGTH
# define PyString_FromStringAndSize PyUnicode_FromStringAndSize
# define PyString_AsString PyUnicode_AsUTF8
#endif

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION >= 3
# define MOD_ERROR_VAL NULL
# define MOD_SUCCESS_VAL(val) val
# define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
# define MOD_DEF(ob, name, methods, doc) \
    static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
    ob = PyModule_Create(&moduledef);
#else
# define MOD_ERROR_VAL
# define MOD_SUCCESS_VAL(val)
# define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
# define MOD_DEF(ob, name, methods, doc) \
    ob = Py_InitModule3(name, methods, doc);
#endif

#endif
