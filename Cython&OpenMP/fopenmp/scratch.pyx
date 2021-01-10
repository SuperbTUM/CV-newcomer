# distutils: language=c++
import numpy as np
cimport numpy as cnp
import cython
cimport cython
from libcpp.map cimport map
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void target_mean_v8_parallel(int[:] data_y, int[:] data_x, double[:] result) nogil:
    cdef int i, j
    cdef map[int, int] value_dict
    cdef map[int, int] count_dict
    for i in prange(data_y.shape[0], nogil=True, num_threads=4):
        if value_dict.count(data_x[i]):
            value_dict[data_x[i]] += data_y[i]
            count_dict[data_x[i]] += 1
        else:
            value_dict[data_x[i]] = data_y[i]
            count_dict[data_x[i]] = 1

    for j in prange(data_y.shape[0], nogil=True, num_threads=4):
        result[j] = (value_dict[data_x[j]] - data_y[j]) / (count_dict[data_x[j]] - 1)

cpdef target_mean_v8_parallel_test():
    data_y = np.random.randint(2, size=(5000,), dtype=np.int32)
    data_x = np.random.randint(10, size=(5000,), dtype=np.int32)
    data_result = np.random.randn(5000)
    target_mean_v8_parallel(data_y, data_x, data_result)
    print(data_result)