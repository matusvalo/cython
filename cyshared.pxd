from cython.cimports.cpython.ref import PyObject, PyTypeObject

cdef extern from *:
    ctypedef struct __Pyx_TypeInfo:
        pass
    ctypedef struct __Pyx_BufFmt_StackElem:
        pass
    ctypedef struct __Pyx_memviewslice:
        pass
    ctypedef struct __Pyx_BufFmt_Context:
        pass
    cdef struct __pyx_memoryview_obj:
        pass

cdef PyObject* __Pyx_PyNumber_LongWrongResultType(PyObject* result) #1
cdef long __Pyx__PyObject_Ord(PyObject* c) #2
cdef PyObject *__Pyx_PyLong_AbsNeg(PyObject *n) #3
cdef PyObject *__Pyx_PyObject_GetIndex(PyObject *obj, PyObject *index) #4
cdef PyObject *__Pyx_PyObject_GetItem_Slow(PyObject *obj, PyObject *key) #5
cdef PyObject *__Pyx_GetItemInt_Generic(PyObject *o, PyObject* j) #6
cdef PyObject *__Pyx_PyDict_GetItem(PyObject *d, PyObject* key) #7
cdef PyObject *__Pyx_Py3MetaclassPrepare(PyObject *metaclass, PyObject *bases, PyObject *name, PyObject *qualname,
                                           PyObject *mkw, PyObject *modname, PyObject *doc) #8
cdef PyObject *__Pyx_Py3ClassCreate(PyObject *metaclass, PyObject *name, PyObject *bases, PyObject *dict,
                                      PyObject *mkw, int calculate_metaclass, int allow_py2_metaclass) #9
cdef PyObject *__Pyx_CalculateMetaclass(PyTypeObject *metaclass, PyObject *bases) #10
cdef int __pyx_check_strides(Py_buffer *buf, int dim, int ndim, int spec) #11
cdef int __pyx_check_suboffsets(Py_buffer *buf, int dim, int ndim, int spec) #12
cdef int __pyx_verify_contig(Py_buffer *buf, int ndim, int c_or_f_flag) #13

cdef int __Pyx_ValidateAndInit_memviewslice(
                int *axes_specs,
                int c_or_f_flag,
                int buf_flags,
                int ndim,
                __Pyx_TypeInfo *dtype,
                __Pyx_BufFmt_StackElem stack[],
                __Pyx_memviewslice *memviewslice,
                PyObject *original_obj) #14

cdef int __pyx_typeinfo_cmp(__Pyx_TypeInfo *a, __Pyx_TypeInfo *b) #15

cdef  void __Pyx_BufFmt_Init(__Pyx_BufFmt_Context* ctx,
                              __Pyx_BufFmt_StackElem* stack,
                              __Pyx_TypeInfo* type) #16
cdef const char* __Pyx_BufFmt_CheckString(__Pyx_BufFmt_Context* ctx, const char* ts) #17

cdef void __Pyx_ZeroBuffer(Py_buffer* buf) #18
cdef  int __Pyx__GetBufferAndValidate(
        Py_buffer* buf, PyObject* obj,  __Pyx_TypeInfo* dtype, int flags,
        int nd, int cast, __Pyx_BufFmt_StackElem* stack) #19

cdef int __Pyx_init_memviewslice(
                __pyx_memoryview_obj *memview,
                int ndim,
                __Pyx_memviewslice *memviewslice,
                int memview_is_new_reference) #20

cdef __Pyx_memviewslice __pyx_memoryview_copy_new_contig(const __Pyx_memviewslice *from_mvs,
                                 const char *mode, int ndim,
                                 size_t sizeof_dtype, int contig_flag,
                                 int dtype_is_object) #21

cdef int __pyx_slices_overlap(__Pyx_memviewslice *slice1,
                                __Pyx_memviewslice *slice2,
                                int ndim, size_t itemsize) #22

cdef int __pyx_memviewslice_is_contig(const __Pyx_memviewslice mvs, char order, int ndim) #23

