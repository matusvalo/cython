# tag: numpy

from numpy cimport int64_t

cpdef test_divmod():
    """
    >>> test_divmod()
    -420000000
    """
    cdef int64_t val = -420000000000
    us, remainder = divmod(val, 1000)
    if remainder >= 500:
        us += 1
    return us
