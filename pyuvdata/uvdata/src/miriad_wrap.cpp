#include <Python.h>
#include "numpy/arrayobject.h"
#include <string>
#include "aipy_compat.h"
#include "miriad_wrap.h"

#define MAXVAR 32768

/*____                           _                    _
 / ___|_ __ ___  _   _ _ __   __| |_      _____  _ __| | __
| |  _| '__/ _ \| | | | '_ \ / _` \ \ /\ / / _ \| '__| |/ /
| |_| | | | (_) | |_| | | | | (_| |\ V  V / (_) | |  |   <
 \____|_|  \___/ \__,_|_| |_|\__,_| \_/\_/ \___/|_|  |_|\_\
*/

// Python object that holds handle to UV file
typedef struct {
    PyObject_HEAD
    int tno;
    long decimate;
    long decphase;
    long intcnt;
    double curtime;
} UVObject;

// Deallocate memory when Python object is deleted
static void UVObject_dealloc(UVObject *self) {
    if (self->tno != -1) uvclose_c(self->tno);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Allocate memory for Python object and Healpix_Base (__new__)
static PyObject *UVObject_new(PyTypeObject *type,
        PyObject *args, PyObject *kwds) {
    UVObject *self;
    self = (UVObject *) type->tp_alloc(type, 0);
    return (PyObject *) self;
}

// A simple error handler that we can use in bug.c
void error_handler(void) {
    throw MiriadError("Runtime error in MIRIAD");
}

// Initialize object (__init__)
static int UVObject_init(UVObject *self, PyObject *args, PyObject *kwds) {
    char *name=NULL, *status=NULL, *corrmode=NULL;
    self->tno = -1;
    self->decimate = 1;
    self->decphase = 0;
    self->intcnt = -1;
    self->curtime = -1;
    // Parse arguments and typecheck
    if (!PyArg_ParseTuple(args, "sss", &name, &status, &corrmode)) return -1;
    switch (corrmode[0]) {
        case 'r': case 'j': break;
        default:
            PyErr_Format(PyExc_ValueError, "UV corrmode must be 'r' or 'j' (got '%c')", corrmode[0]);
            return -1;
    }
    // Setup an error handler so MIRIAD doesn't just exit
    bugrecover_c(error_handler);
    try {
        uvopen_c(&self->tno, name, status);
        // Statically set the preamble format
        uvset_c(self->tno,"preamble","uvw/time/baseline",0,0.,0.,0.);
        uvset_c(self->tno,"corr",corrmode,0,0.,0.,0.);
    } catch (MiriadError &e) {
        self->tno = -1;
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return -1;
    }
    return 0;
}

/* ___  _     _           _     __  __      _   _               _
  / _ \| |__ (_) ___  ___| |_  |  \/  | ___| |_| |__   ___   __| |___
 | | | | '_ \| |/ _ \/ __| __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
 | |_| | |_) | |  __/ (__| |_  | |  | |  __/ |_| | | | (_) | (_| \__ \
  \___/|_.__// |\___|\___|\__| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
           |__/
*/

// Thin wrapper over uvrewind_c
PyObject * UVObject_rewind(UVObject *self) {
    uvrewind_c(self->tno);
    self->intcnt = -1;
    self->curtime = -1;
    Py_INCREF(Py_None);
    return Py_None;
}

/* Wrapper over uvread_c to deal with numpy arrays, conversion of baseline
 * and polarization codes, and returning a tuple of all results.
 */
PyObject * UVObject_read(UVObject *self, PyObject *args) {
    PyArrayObject *data, *flags, *uvw;
    PyObject *rv;
    int nread, n2read, i, j;
    double preamble[PREAMBLE_SIZE];
    if (!PyArg_ParseTuple(args, "i", &n2read)) return NULL;
    // Make numpy arrays to hold the results
    npy_intp data_dims[1] = {n2read};
    data = (PyArrayObject *) PyArray_SimpleNew(1, data_dims, NPY_CFLOAT);
    CHK_NULL(data);
    flags = (PyArrayObject *) PyArray_SimpleNew(1, data_dims, NPY_INT);
    CHK_NULL(flags);
    while (1) {
        // Here is the MIRIAD call
        try {
            uvread_c(self->tno, preamble,
                     (float *)PyArray_DATA(data), (int *)PyArray_DATA(flags), n2read, &nread);
        } catch (MiriadError &e) {
            PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
            Py_DECREF(data);
            Py_DECREF(flags);
            return NULL;
        }
        if (preamble[3] != self->curtime) {
            self->intcnt += 1;
            self->curtime = preamble[3];
        }
        if ((self->intcnt-self->decphase) % self->decimate == 0 || nread==0) {
            break;
        }
    }
    // Now we build a return value of ((uvw,t,(i,j)), data, flags, nread)
    npy_intp uvw_dims[1] = {3};
    uvw = (PyArrayObject *) PyArray_SimpleNew(1, uvw_dims, NPY_DOUBLE);
    CHK_NULL(uvw);
    IND1(uvw,0,double) = preamble[0];
    IND1(uvw,1,double) = preamble[1];
    IND1(uvw,2,double) = preamble[2];
    i = GETI(preamble[4]);
    j = GETJ(preamble[4]);
    rv = Py_BuildValue("((Od(ii))OOi)",
        (PyObject *)uvw, preamble[3], i, j,
        (PyObject *)data, (PyObject *)flags, nread);
    CHK_NULL(rv);
    Py_DECREF(uvw); Py_DECREF(data); Py_DECREF(flags);
    return rv;
}

/* Wrapper over uvwrite_c to deal with numpy arrays, conversion of baseline
 * codes, and accepts preamble as a tuple.
 */
PyObject * UVObject_write(UVObject *self, PyObject *args) {
    PyArrayObject *data=NULL, *flags=NULL, *uvw=NULL;
    int i, j;
    double preamble[PREAMBLE_SIZE], t;
    // Parse arguments and typecheck
    if (!PyArg_ParseTuple(args, "(O!d(ii))O!O!",
        &PyArray_Type, &uvw, &t, &i, &j,
        &PyArray_Type, &data, &PyArray_Type, &flags)) return NULL;
    if (RANK(uvw) != 1 || DIM(uvw,0) != 3) {
        PyErr_Format(PyExc_ValueError, "uvw must have shape (3,) %d", RANK(uvw));
        return NULL;
    } else if (RANK(data)!=1 || RANK(flags)!=1 || DIM(data,0)!=DIM(flags,0)) {
        PyErr_Format(PyExc_ValueError,
            "data and flags must be 1 dimensional and have the same shape");
        return NULL;
    }
    CHK_ARRAY_TYPE(uvw, NPY_DOUBLE);
    CHK_ARRAY_TYPE(data, NPY_CFLOAT);
    // Check for both int,long, b/c label of 32b number is platform dependent
    if (TYPE(flags) != NPY_INT && \
            (sizeof(int) == sizeof(long) && TYPE(flags) != NPY_LONG)) {
        PyErr_Format(PyExc_ValueError, "type(flags) != NPY_LONG or NPY_INT");
        return NULL;
    }
    // Fill up the preamble
    preamble[0] = IND1(uvw,0,double);
    preamble[1] = IND1(uvw,1,double);
    preamble[2] = IND1(uvw,2,double);
    preamble[3] = t;
    preamble[4] = MKBL(i,j);
    // Here is the MIRIAD call
    try {
        uvwrite_c(self->tno, preamble,
            (float *)PyArray_DATA(data), (int *)PyArray_DATA(flags), DIM(data,0));
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// A thin wrapper over uvcopyvr_c
PyObject * UVObject_copyvr(UVObject *self, PyObject *args) {
    UVObject *uv;
    if (!PyArg_ParseTuple(args, "O!", &UVType, &uv)) return NULL;
    try {
        uvcopyvr_c(uv->tno, self->tno);
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// A thin wrapper over uvtrack_c
PyObject * UVObject_trackvr(UVObject *self, PyObject *args) {
    char *name, *sw;
    if (!PyArg_ParseTuple(args, "ss", &name, &sw)) return NULL;
    try {
        uvtrack_c(self->tno, name, sw);
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

#define RET_IA(htype,pyconstructor,type1,type2,npy_type) \
    if (length == 1) { \
        uvgetvr_c(self->tno,htype,name,value,length); \
        return pyconstructor((type1) ((type2 *)value)[0]); } \
    rv = (PyArrayObject *) PyArray_SimpleNew(1,dims,npy_type); \
    CHK_NULL(rv); \
    uvgetvr_c(self->tno,htype,name,(char *)(PyArray_DATA(rv)),length);\
    return PyArray_Return(rv);

/* rdvr is responsible for reading variables of all types.  It wraps the
 * output of uvgetvr_c into various python structures, and returns
 * arrays as numpy arrays. */
PyObject * UVObject_rdvr(UVObject *self, PyObject *args) {
    char *name, *type, value[MAXVAR];
    int length, updated;
    npy_intp dims[1];
    PyArrayObject *rv;
    int elem_size;

    if (!PyArg_ParseTuple(args, "ss", &name, &type))
        return NULL;


    uvprobvr_c(self->tno, name, value, &length, &updated);

    switch (type[0]) {
    case 'a':
        elem_size = 1;
        break;
    case 'j':
        elem_size = 2;
        break;
    case 'i':
        elem_size = 4;
        break;
    case 'r':
        elem_size = 4;
        break;
    case 'd':
        elem_size = 8;
        break;
    case 'c':
        elem_size = 8;
        break;
    default:
        PyErr_Format(PyExc_ValueError, "unknown type of UV variable \"%s\": %c", name, type[0]);
        return NULL;
    }

    if (length * elem_size > MAXVAR) {
        PyErr_Format(PyExc_ValueError, "UV variable \"%s\" too big for pyuvdata's "
                     "internal buffers", name);
        return NULL;
    }

    dims[0] = length;
    try {
        switch (type[0]) {
            case 'a':
                uvgetvr_c(self->tno,H_BYTE,name,value,length+1);
                return PyString_FromStringAndSize(value, length);
            case 'j':
                uvgetvr_c(self->tno,H_INT2,name,value,length);
                if (length == 1)
                    return PyInt_FromLong((long) ((short *)value)[0]);
                rv = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT);
                CHK_NULL(rv);
                for (int i=0; i < length; i++)
                    IND1(rv,i,int) = ((short *)value)[i];
                return PyArray_Return(rv);
            case 'i':
                RET_IA(H_INT,PyInt_FromLong,long,int,NPY_INT);
            case 'r':
                RET_IA(H_REAL,PyFloat_FromDouble,double,float,NPY_FLOAT);
            case 'd':
                RET_IA(H_DBLE,PyFloat_FromDouble,double,double,NPY_DOUBLE);
            case 'c':
                if (length == 1) {
                    uvgetvr_c(self->tno,H_CMPLX,name,value,length);
                    return PyComplex_FromDoubles(((double *)value)[0],
                                                 ((double *)value)[1]);
                }
                rv = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_CDOUBLE);
                CHK_NULL(rv);
                uvgetvr_c(self->tno,H_CMPLX,name,(char *)PyArray_DATA(rv),length);
                return PyArray_Return(rv);
            default:
                return NULL; /* can't happen with previous switch */
        }
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

#define STORE_IA(npy_type,htype,chk_type,type,pyconverter) \
    if (PyArray_Check(wr_val)) { \
        CHK_ARRAY_TYPE(wr_arr,npy_type); \
        uvputvr_c(self->tno,htype,name,(char *) PyArray_DATA(wr_arr),DIM(wr_arr,0)); \
    } else { \
        chk_type(wr_val); \
        ((type *)value)[0] = (type) pyconverter(wr_val); \
        uvputvr_c(self->tno,htype,name,value,1); }

/* wrvr is the complement of rdvr that uses uvputvr_c to write variables of
 * various types.  Accepts numpy arrays for writing. */
PyObject * UVObject_wrvr(UVObject *self, PyObject *args) {
    char *name, *type, value[MAXVAR];
    char *st;
    PyObject *wr_val;
    PyArrayObject *wr_arr=NULL;
    if (!PyArg_ParseTuple(args, "ssO", &name, &type, &wr_val)) return NULL;
    if (PyArray_Check(wr_val)) {
        wr_arr = (PyArrayObject *) wr_val;
        CHK_ARRAY_RANK(wr_arr,1);
    }
    try {
        switch (type[0]) {
            case 'a':
                CHK_STRING(wr_val);
                st = const_cast<char*>(PyString_AsString(wr_val));
                uvputvr_c(self->tno,H_BYTE,name, st, PyString_Size(wr_val)+1);
                break;
            case 'j':
                STORE_IA(NPY_LONG,H_INT2,CHK_INT,short,PyInt_AsLong);
                break;
            case 'i':
                STORE_IA(NPY_LONG,H_INT,CHK_INT,int,PyInt_AsLong);
                break;
            case 'r':
                STORE_IA(NPY_FLOAT,H_REAL,CHK_FLOAT,float,PyFloat_AsDouble);
                break;
            case 'd':
                STORE_IA(NPY_DOUBLE,H_DBLE,CHK_FLOAT,double,PyFloat_AsDouble);
                break;
            case 'c':
                if (PyArray_Check(wr_val)) {
                    CHK_ARRAY_TYPE(wr_arr,NPY_CDOUBLE);
                    uvputvr_c(self->tno,H_CMPLX,name,(char *)PyArray_DATA(wr_arr),
                        DIM(wr_arr,0));
                } else {
                    CHK_COMPLEX(wr_val);
                    ((double *)value)[0] = PyComplex_RealAsDouble(wr_val);
                    ((double *)value)[1] = PyComplex_ImagAsDouble(wr_val);
                    uvputvr_c(self->tno,H_CMPLX,name,value,1);
                }
                break;
            default:
                PyErr_Format(PyExc_ValueError, "unknown UV variable type: %c", type[0]);
                return NULL;
        }
        Py_INCREF(Py_None);
        return Py_None;
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

// A thin wrapper over uvselect_c
PyObject * UVObject_select(UVObject *self, PyObject *args) {
    char *name;
    double n1, n2;
    int include;
    if (!PyArg_ParseTuple(args, "sddi", &name, &n1, &n2, &include)) return NULL;
    if (strncmp(name,"decimation",5) == 0) {
        self->decimate = (long) n1;
        self->decphase = (long) n2;
    } else {
        try {
            uvselect_c(self->tno, name, n1, n2, include);
        } catch (MiriadError &e) {
            PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
            return NULL;
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}

// A thin wrapper over haccess_c
PyObject * UVObject_haccess(UVObject *self, PyObject *args) {
    char *name, *mode;
    int item_hdl, iostat;
    if (!PyArg_ParseTuple(args, "ss", &name, &mode)) return NULL;
    try {
        haccess_c(self->tno, &item_hdl, name, mode, &iostat);
        CHK_IO(iostat);
        return PyInt_FromLong(item_hdl);
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

// A thin wrapper over hdaccess_c
PyObject * WRAP_hdaccess(UVObject *self, PyObject *args) {
    int item_hdl, iostat;
    if (!PyArg_ParseTuple(args, "i", &item_hdl)) return NULL;
    try {
        hdaccess_c(item_hdl, &iostat);
        Py_INCREF(Py_None);
        return Py_None;
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

#define INIT(type_item,size) \
    hwriteb_c(item_hdl,type_item,0,ITEM_HDR_SIZE,&iostat); \
    CHK_IO(iostat); \
    offset = mroundup(ITEM_HDR_SIZE,size);

/* hwrite_init encodes the type of a header item in the first few bytes */
PyObject * WRAP_hwrite_init(PyObject *self, PyObject *args) {
    int item_hdl, offset, iostat;
    char *type;
    if (!PyArg_ParseTuple(args, "is", &item_hdl, &type)) return NULL;
    try {
        switch(type[0]) {
            case 'a': INIT(char_item,H_BYTE_SIZE); break;
            case 'b': INIT(binary_item,ITEM_HDR_SIZE); break;
            case 'i': INIT(int_item,H_INT_SIZE); break;
            case 'j': INIT(int2_item,H_INT2_SIZE); break;
            case 'l': INIT(int8_item,H_INT8_SIZE); break;
            case 'r': INIT(real_item,H_REAL_SIZE); break;
            case 'd': INIT(dble_item,H_DBLE_SIZE); break;
            case 'c': INIT(cmplx_item,H_CMPLX_SIZE); break;
            default:
                PyErr_Format(PyExc_ValueError, "unknown item type: %c", type[0]);
                return NULL;
        }
        return PyInt_FromLong(offset);
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

#define FIRSTINT(s) ((int *)s)[0]
/* hread_init surmises the type of a header item from the first few bytes */
PyObject * WRAP_hread_init(PyObject *self, PyObject *args) {
    int item_hdl, offset, iostat, code;
    char s[ITEM_HDR_SIZE];
    if (!PyArg_ParseTuple(args, "i", &item_hdl)) return NULL;
    try {
        hreadb_c(item_hdl,s,0,ITEM_HDR_SIZE,&iostat);
        CHK_IO(iostat);
        code = FIRSTINT(s);
        if (code == FIRSTINT(char_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_BYTE_SIZE);
            return Py_BuildValue("si", "a", offset);
        } else if (code == FIRSTINT(binary_item)) {
            return Py_BuildValue("si", "b", ITEM_HDR_SIZE);
        } else if (code == FIRSTINT(int_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_INT_SIZE);
            return Py_BuildValue("si", "i", offset);
        } else if (code == FIRSTINT(int2_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_INT2_SIZE);
            return Py_BuildValue("si", "j", offset);
        } else if (code == FIRSTINT(int8_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_INT8_SIZE);
            return Py_BuildValue("si", "l", offset);
        } else if (code == FIRSTINT(real_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_REAL_SIZE);
            return Py_BuildValue("si", "r", offset);
        } else if (code == FIRSTINT(dble_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_DBLE_SIZE);
            return Py_BuildValue("si", "d", offset);
        } else if (code == FIRSTINT(cmplx_item)) {
            offset = mroundup(ITEM_HDR_SIZE,H_CMPLX_SIZE);
            return Py_BuildValue("si", "c", offset);
        }
        PyErr_Format(PyExc_RuntimeError, "unknown item type");
        return NULL;
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

/* hwrite supports writing to header items of all types, using the various
 * hwrite_c calls.  Writes one item per call. */
PyObject * WRAP_hwrite(PyObject *self, PyObject *args) {
    int item_hdl, offset, iostat;
    char *type;
    PyObject *val;
    int in; short sh; long lg; float fl; double db; float cx[2]; char *st;
    if (!PyArg_ParseTuple(args, "iiOs", &item_hdl, &offset, &val, &type))
        return NULL;
    try {
        switch (type[0]) {
            case 'a':
                CHK_STRING(val);
                st = const_cast<char*>(PyString_AsString(val));
                in = PyString_Size(val); // # bytes to write
                hwriteb_c(item_hdl, st, offset, in, &iostat);
                CHK_IO(iostat);
                offset = H_BYTE_SIZE * in;
                break;
            case 'i':
                CHK_INT(val);
                in = (int) PyInt_AsLong(val);
                hwritei_c(item_hdl, &in, offset, H_INT_SIZE, &iostat);
                CHK_IO(iostat);
                offset = H_INT_SIZE;
                break;
            case 'j':
                CHK_INT(val);
                sh = (short) PyInt_AsLong(val);
                hwritej_c(item_hdl, &sh, offset, H_INT2_SIZE, &iostat);
                CHK_IO(iostat);
                offset = H_INT2_SIZE;
                break;
            case 'l':
                CHK_LONG(val);
                lg = PyLong_AsLong(val);
                hwritel_c(item_hdl, &lg, offset, H_INT8_SIZE, &iostat);
                CHK_IO(iostat);
                offset = H_INT8_SIZE;
                break;
            case 'r':
                CHK_FLOAT(val);
                fl = (float) PyFloat_AsDouble(val);
                hwriter_c(item_hdl, &fl, offset, H_REAL_SIZE, &iostat);
                CHK_IO(iostat);
                offset = H_REAL_SIZE;
                break;
            case 'd':
                CHK_FLOAT(val);
                db = PyFloat_AsDouble(val);
                hwrited_c(item_hdl, &db, offset, H_DBLE_SIZE, &iostat);
                CHK_IO(iostat);
                offset = H_DBLE_SIZE;
                break;
            case 'c':
                CHK_COMPLEX(val);
                cx[0] = (float) PyComplex_RealAsDouble(val);
                cx[1] = (float) PyComplex_ImagAsDouble(val);
                hwritec_c(item_hdl, cx, offset, H_CMPLX_SIZE, &iostat);
                CHK_IO(iostat);
                offset = H_CMPLX_SIZE;
                break;
            default:
                PyErr_Format(PyExc_ValueError, "unknown item type: %c", type[0]);
                return NULL;
        }
        return PyInt_FromLong(offset);
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

/* hread supports reading of header items of all types using the various
 * hread_c calls.  Reads one item per call. */
PyObject * WRAP_hread(PyObject *self, PyObject *args) {
    int item_hdl, offset, iostat;
    char *type;
    PyObject *val, *rv;
    int in; short sh; long lg; float fl; double db; float cx[2]; char st[2];
    if (!PyArg_ParseTuple(args, "iis", &item_hdl, &offset, &type))
        return NULL;
    try {
        switch (type[0]) {
            case 'a':
                hreadb_c(item_hdl, st, offset, H_BYTE_SIZE, &iostat);
                CHK_IO(iostat);
#if PY_MAJOR_VERSION >= 3
                return Py_BuildValue("yi", st, H_BYTE_SIZE);
#else
                return Py_BuildValue("si", st, H_BYTE_SIZE);
#endif
            case 'i':
                hreadi_c(item_hdl, &in, offset, H_INT_SIZE, &iostat);
                CHK_IO(iostat);
                return Py_BuildValue("ii", in, H_INT_SIZE);
            case 'j':
                hreadj_c(item_hdl, &sh, offset, H_INT2_SIZE, &iostat);
                CHK_IO(iostat);
                return Py_BuildValue("ii", sh, H_INT2_SIZE);
            case 'l':
                hreadl_c(item_hdl, &lg, offset, H_INT8_SIZE, &iostat);
                CHK_IO(iostat);
                return Py_BuildValue("li", lg, H_INT8_SIZE);
            case 'r':
                hreadr_c(item_hdl, &fl, offset, H_REAL_SIZE, &iostat);
                CHK_IO(iostat);
                return Py_BuildValue("fi", fl, H_REAL_SIZE);
            case 'd':
                hreadd_c(item_hdl, &db, offset, H_DBLE_SIZE, &iostat);
                CHK_IO(iostat);
                return Py_BuildValue("fi", db, H_DBLE_SIZE);
            case 'c':
                hreadc_c(item_hdl, cx, offset, H_CMPLX_SIZE, &iostat);
                CHK_IO(iostat);
                val = PyComplex_FromDoubles((double) cx[0], (double) cx[1]);
                rv = Py_BuildValue("Oi", val, H_CMPLX_SIZE);
                Py_DECREF(val);
                return rv;
            default:
                PyErr_Format(PyExc_ValueError, "unknown item type: %c", type[0]);
                return NULL;
        }
    } catch (MiriadError &e) {
        PyErr_Format(PyExc_RuntimeError, "%s", e.get_message());
        return NULL;
    }
}

/*_        __                     _               _   _
 \ \      / / __ __ _ _ __  _ __ (_)_ __   __ _  | | | |_ __
  \ \ /\ / / '__/ _` | '_ \| '_ \| | '_ \ / _` | | | | | '_ \
   \ V  V /| | | (_| | |_) | |_) | | | | | (_| | | |_| | |_) |
    \_/\_/ |_|  \__,_| .__/| .__/|_|_| |_|\__, |  \___/| .__/
                     |_|   |_|            |___/        |_|
*/
// Bind methods to object
static PyMethodDef UVObject_methods[] = {
    {"rewind", (PyCFunction)UVObject_rewind, METH_NOARGS,
        "rewind()\nSeek to the beginning of a UV file."},
    {"raw_read", (PyCFunction)UVObject_read, METH_VARARGS,
        "_read(num)\nRead up to the specified number of channels from a spectrum.  Returns (preamble, data, flags) where preamble = (uvw,time,(ant_i,ant_j)), data = complex64 numpy array of data, flags = integer32 array of data valid where == 1.  Note that this definition of flags is the inverse of numpy's definition."},
    {"raw_write", (PyCFunction)UVObject_write, METH_VARARGS,
        "_write(preamble,data,flags)\nWrite the provided preamble, data, flags to file.  See _read() for definitions of preamble, data, flags."},
    {"copyvr", (PyCFunction)UVObject_copyvr, METH_VARARGS,
        "copyvr(uv)\nCopy any variables which changed during the last read into the provided uv interface."},
    {"trackvr", (PyCFunction)UVObject_trackvr, METH_VARARGS,
        "trackvr(name,code)\nIf code=='c', set variable to be copied by copyvr()."},
    {"_rdvr", (PyCFunction)UVObject_rdvr, METH_VARARGS,
        "_rdvr(name,type)\nReturn the current value of a variable of the provided Miriad type (a,j,i,r,d,c).  If variable has multiple values, an array (or string if pertinent) will be returned."},
    {"_wrvr", (PyCFunction)UVObject_wrvr, METH_VARARGS,
        "_wrvr(name,type,val)\nWrite a value to a variable of the provided Miriad type (see _rdvr()).  If val is an array, multiple values will be written."},
    {"_select", (PyCFunction)UVObject_select, METH_VARARGS,
        "_select(name,n1,n2,include)\nSelect which data is returned by _read().  See select() for more information."},
    {"haccess", (PyCFunction)UVObject_haccess, METH_VARARGS,
        "haccess(name,mode)\nOpen a header item in the given mode ('read','write').  Returns an integer handle."},
    {NULL}  /* Sentinel */
};

PyTypeObject UVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
//     0,                         /*ob_size*/
    "_miriad.UV", /*tp_name*/
    sizeof(UVObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)UVObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "This class provides the basic interfaces to a UV file at a slightly higher level than the raw Miriad function calls.  UV(filename,status)",       /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    UVObject_methods,             /* tp_methods */
    0,                     /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)UVObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    UVObject_new,       /* tp_new */
};

// Module methods
static PyMethodDef _miriad_methods[] = {
    {"hdaccess", (PyCFunction)WRAP_hdaccess, METH_VARARGS,
        "hdaccess(handle)\nCloses an open header item."},
    {"hwrite_init", (PyCFunction)WRAP_hwrite_init, METH_VARARGS,
        "hwrite_init(handle,type)\nInitialize an open header item for writing a given Miriad type (a,b,i,j,l,r,d,c).  Return the offset of end of the type tag."},
    {"hread_init", (PyCFunction)WRAP_hread_init, METH_VARARGS,
        "hread_init(handle)\nReturn the encoded type of an open header item, and the offset to the end of the type tag."},
    {"hwrite", (PyCFunction)WRAP_hwrite, METH_VARARGS,
        "hwrite(handle,offset,value,type)\nWrite a value at the provided offset to an open header item of the given type."},
    {"hread", (PyCFunction)WRAP_hread, METH_VARARGS,
        "hread(handle,offset,type)\nRead a value of the given type from an open header item at the provided offset."},
    {NULL}  /* Sentinel */
};

// Module init
MOD_INIT(_miriad) {
    PyObject* m;

    Py_Initialize();

    UVType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&UVType) < 0)
        return MOD_ERROR_VAL;

    // Module definitions and functions
    MOD_DEF(m, "_miriad", _miriad_methods, \
            "This is a hand-written Python wrapper (by Aaron Parsons) for MIRIAD.");
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    Py_INCREF(&UVType);
    PyModule_AddObject(m, "UV", (PyObject *)&UVType);
    PyModule_AddObject(m, "MAXCHAN", PyInt_FromLong(MAXCHAN));

    return MOD_SUCCESS_VAL(m);
}
