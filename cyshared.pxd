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
    cdef struct __pyx_memoryview_obj '__pyx_memoryview':
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

cdef Py_ssize_t __pyx_memoryview_slice_get_size(__Pyx_memviewslice *src, int ndim) noexcept nogil #24

cdef Py_ssize_t __pyx_fill_contig_strides_array(
                Py_ssize_t *shape, Py_ssize_t *strides, Py_ssize_t stride,
                int ndim, char order) noexcept nogil #25

cdef void __pyx_memoryview__slice_assign_scalar(char *data, Py_ssize_t *shape,
                              Py_ssize_t *strides, int ndim,
                              size_t itemsize, void *item) noexcept nogil # 26

cdef bytes __pyx_format_from_typeinfo(__Pyx_TypeInfo *type) #27

cdef void __pyx_memoryview__slice_assign_scalar(char *data, Py_ssize_t *shape,
                              Py_ssize_t *strides, int ndim,
                              size_t itemsize, void *item) noexcept nogil #28

cdef void __pyx_memoryview_slice_assign_scalar(__Pyx_memviewslice *dst, int ndim,
                              size_t itemsize, void *item,
                              bint dtype_is_object) noexcept nogil #29

cdef void __pyx_memoryview_refcount_copying(__Pyx_memviewslice *dst, bint dtype_is_object, int ndim, bint inc) noexcept nogil #30

cdef void __pyx_memoryview_refcount_objects_in_slice_with_gil(char *data, Py_ssize_t *shape,
                                             Py_ssize_t *strides, int ndim,
                                             bint inc) noexcept with gil #31

cdef void __pyx_memoryview_refcount_objects_in_slice(char *data, Py_ssize_t *shape,
                                    Py_ssize_t *strides, int ndim, bint inc) noexcept #32

cdef int __pyx_memoryview_err_extents(int i, Py_ssize_t extent1,
                             Py_ssize_t extent2) except -1 with gil #33

cdef int __pyx_memoryview_err_dim(PyObject *error, str msg, int dim) except -1 with gil #34

cdef int __pyx_memoryview_err(PyObject *error, str msg) except -1 with gil #35

cdef int __pyx_memoryview_err_no_memory() except -1 with gil #36

cdef int __pyx_memoryview_copy_contents(__Pyx_memviewslice src,
                                  __Pyx_memviewslice dst,
                                  int src_ndim, int dst_ndim,
                                  bint dtype_is_object) except -1 nogil #37

cdef void __pyx_memoryview_broadcast_leading(__Pyx_memviewslice *mslice,
                            int ndim,
                            int ndim_other) noexcept nogil #38

cdef char __pyx_get_best_slice_order(__Pyx_memviewslice *mslice, int ndim) noexcept nogil #39

cdef __pyx_memoryview_fromslice(__Pyx_memviewslice memviewslice,
                          int ndim,
                          object (*to_object_func)(char *),
                          int (*to_dtype_func)(char *, object) except 0,
                          bint dtype_is_object) #40

cdef __Pyx_memviewslice *__pyx_memoryview_get_slice_from_memoryview(__pyx_memoryview_obj *memview,
                                                   __Pyx_memviewslice *mslice) except NULL #41

cdef void __pyx_memoryview_slice_copy(__pyx_memoryview_obj *memview, __Pyx_memviewslice *dst) noexcept #42

cdef __pyx_memoryview_copy_object(__pyx_memoryview_obj *memview) #43

cdef __pyx_memoryview_copy_object_from_slice(__pyx_memoryview_obj *memview, __Pyx_memviewslice *memviewslice) #44

cdef int __pyx_memoryview_slice_memviewslice(
        __Pyx_memviewslice *dst,
        Py_ssize_t shape, Py_ssize_t stride, Py_ssize_t suboffset,
        int dim, int new_ndim, int *suboffset_dim,
        Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step,
        int have_start, int have_stop, int have_step,
        bint is_slice) except -1 nogil

# cdef char *__pyx_pybuffer_index(Py_buffer *view, char *bufp, Py_ssize_t index,
#                           Py_ssize_t dim) except NULL

cdef int __pyx_memslice_transpose(__Pyx_memviewslice *memslice) except -1 nogil

cdef __pyx_memoryview_obj *__pyx_memview_slice(__pyx_memoryview_obj *memview, object indices)

cdef __pyx_memoryview_new(object o, int flags, bint dtype_is_object, __Pyx_TypeInfo *typeinfo)

cdef bint __pyx_memoryview_check(object o) noexcept
