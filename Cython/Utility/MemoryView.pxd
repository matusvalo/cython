# from cpython cimport ...

cimport cython
cdef extern from "Python.h":
    ctypedef struct PyObject
    ctypedef void *PyThread_type_lock
    PyObject *PyExc_IndexError
    PyObject *PyExc_ValueError

cdef extern from *:
    ctypedef struct {{memviewslice_name}}:
        Py_ssize_t shape[{{max_dims}}]
        Py_ssize_t strides[{{max_dims}}]
        Py_ssize_t suboffsets[{{max_dims}}]
        char *data

    ctypedef struct __pyx_buffer "Py_buffer":
        PyObject *obj

    ctypedef struct __Pyx_TypeInfo:
        pass

    cdef struct __pyx_memoryview "__pyx_memoryview_obj":
        Py_buffer view
        PyObject *obj
        const __Pyx_TypeInfo *typeinfo

    ctypedef int __pyx_atomic_int_type


@cname("__pyx_array")
cdef class array:

    cdef:
        char *data
        Py_ssize_t len
        char *format
        int ndim
        Py_ssize_t *_shape
        Py_ssize_t *_strides
        Py_ssize_t itemsize
        unicode mode  # FIXME: this should have been a simple 'char'
        bytes _format
        void (*callback_free_data)(void *data) noexcept
        # cdef object _memview
        cdef bint free_data
        cdef bint dtype_is_object

    @cname('get_memview')
    cdef get_memview(self)

@cname("__pyx_array_new")
cdef array array_cwrapper(tuple shape, Py_ssize_t itemsize, char *format, const char *c_mode, char *buf)

@cname('__pyx_memoryview')
cdef class memoryview:

    cdef object obj
    cdef object _size
    # This comes before acquisition_count so can't be removed without breaking ABI compatibility
    cdef void* _unused
    cdef PyThread_type_lock lock
    cdef __pyx_atomic_int_type acquisition_count
    cdef Py_buffer view
    cdef int flags
    cdef bint dtype_is_object
    cdef const __Pyx_TypeInfo *typeinfo

    cdef char *get_item_pointer(memoryview self, object index) except NULL
    cdef is_slice(self, obj)
    cdef setitem_slice_assignment(self, dst, src)
    cdef setitem_slice_assign_scalar(self, memoryview dst, value)
    cdef setitem_indexed(self, index, value)
    cdef convert_item_to_object(self, char *itemp)
    cdef assign_item_from_object(self, char *itemp, object value)
    cdef _get_base(self)

@cname('__pyx_memoryview_new')
cdef memoryview_cwrapper(object o, int flags, bint dtype_is_object, const __Pyx_TypeInfo *typeinfo)

@cname('__pyx_memoryview_check')
cdef inline bint memoryview_check(object o) noexcept:
    return isinstance(o, memoryview)

@cname('__pyx_memoryview_err_dim')
cdef inline int _err_dim(PyObject *error, str msg, int dim) except -1 with gil:
    raise <object>error, msg % dim

@cname('__pyx_memoryview_slice_memviewslice')
cdef inline int slice_memviewslice(
        {{memviewslice_name}} *dst,
        Py_ssize_t shape, Py_ssize_t stride, Py_ssize_t suboffset,
        int dim, int new_ndim, int *suboffset_dim,
        Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step,
        int have_start, int have_stop, int have_step,
        bint is_slice) except -1 nogil:
    cdef Py_ssize_t new_shape
    cdef bint negative_step

    if not is_slice:
        # index is a normal integer-like index
        if start < 0:
            start += shape
        if not 0 <= start < shape:
            _err_dim(PyExc_IndexError, "Index out of bounds (axis %d)", dim)
    else:
        # index is a slice
        if have_step:
            negative_step = step < 0
            if step == 0:
                _err_dim(PyExc_ValueError, "Step may not be zero (axis %d)", dim)
        else:
            negative_step = False
            step = 1

        # check our bounds and set defaults
        if have_start:
            if start < 0:
                start += shape
                if start < 0:
                    start = 0
            elif start >= shape:
                if negative_step:
                    start = shape - 1
                else:
                    start = shape
        else:
            if negative_step:
                start = shape - 1
            else:
                start = 0

        if have_stop:
            if stop < 0:
                stop += shape
                if stop < 0:
                    stop = 0
            elif stop > shape:
                stop = shape
        else:
            if negative_step:
                stop = -1
            else:
                stop = shape

        # len = ceil( (stop - start) / step )
        with cython.cdivision(True):
            new_shape = (stop - start) // step

            if (stop - start) - step * new_shape:
                new_shape += 1

        if new_shape < 0:
            new_shape = 0

        # shape/strides/suboffsets
        dst.strides[new_ndim] = stride * step
        dst.shape[new_ndim] = new_shape
        dst.suboffsets[new_ndim] = suboffset

    # Add the slicing or indexing offsets to the right suboffset or base data *
    if suboffset_dim[0] < 0:
        dst.data += start * stride
    else:
        dst.suboffsets[suboffset_dim[0]] += start * stride

    if suboffset >= 0:
        if not is_slice:
            if new_ndim == 0:
                dst.data = (<char **> dst.data)[0] + suboffset
            else:
                _err_dim(PyExc_IndexError, "All dimensions preceding dimension %d "
                                     "must be indexed and not sliced", dim)
        else:
            suboffset_dim[0] = new_ndim

    return 0



@cname('__pyx_memslice_transpose')
cdef int transpose_memslice({{memviewslice_name}} *memslice) except -1 nogil

@cname('__pyx_memoryview_fromslice')
cdef memoryview_fromslice({{memviewslice_name}} memviewslice,
                          int ndim,
                          object (*to_object_func)(char *),
                          int (*to_dtype_func)(char *, object) except 0,
                          bint dtype_is_object)

@cname('__pyx_memoryview_copy_contents')
cdef int memoryview_copy_contents({{memviewslice_name}} src,
                                  {{memviewslice_name}} dst,
                                  int src_ndim, int dst_ndim,
                                  bint dtype_is_object) except -1 nogil
