# cython: language_level=3
## distutils: define_macros=CYTHON_TRACE_NOGIL=1
## cython: linetrace=True
## cython: binding=True

# for profiling uncomment the 3 above, and compile with cythonize 


## better compile always with cythonize, but just in case this would be the line
# cython Overlap.pyx; gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -ffast-math  -I/home/fer/opt/miniconda3/envs/smiTE/include/python3.8/ -o Overlap.so Overlap.c

cimport cython
from cpython.array cimport array, clone
cdef extern from "Python.h":
     cdef Py_ssize_t PY_SSIZE_T_MAX
import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t



@cython.boundscheck(False)
@cython.wraparound(False)
def getIdx(DTYPE_t[:] p, DTYPE_t v):
    return _getIdx(p, v)


@cython.boundscheck(False)
@cython.wraparound(False)
def intersect(DTYPE_t[:] p1, DTYPE_t[:] p2):
    cdef Py_ssize_t i, s, v
    cdef DTYPE_t [:] res
    if p1.shape[0] < p2.shape[0]:
        p1, p2 = p2, p1
    s = 0
    res = clone(array('I'), p1.shape[0], False)
    for i in range(p2.shape[0]):
        v = p2[i]
        if v >= p1[0] and v <= p1[p1.shape[0]-1]:
            if _getIdx(p1, v) < PY_SSIZE_T_MAX:
                res[s] = v
                s += 1
    return res[:s]
            



@cython.boundscheck(False)
@cython.wraparound(False) 
cdef inline Py_ssize_t _getIdx(DTYPE_t[:] p, DTYPE_t v) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t size = p.shape[0]
    for i in range(size):
        if p[i] == v:
            return i
    else:
        return PY_SSIZE_T_MAX # use PY_SSIZE_T_MAX to denote v not being inside p


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t ol_length(DTYPE_t[:] p1, DTYPE_t[:] p2) nogil:
    # p1 is assumed to be smaller or equal to p2
    cdef Py_ssize_t res = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t size = p1.shape[0]
    for i in range(size):
        if p1[i]==p2[i]:
            res+=1
        else:
            break
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def overlap(DTYPE_t[:] p1, DTYPE_t[:] p2, bint subsets_as_overlaps = True, bint assume_nosign = False):
    
    # assume_nosign will assume that as long as one vertex of p1 is contained in p2 the overlap is perfect
    
    cdef bint goodJoin = False
    cdef DTYPE_t p1start, p2start, p1end, p2end
    cdef Py_ssize_t p1_p2start, p2_p1start
    cdef DTYPE_t [:] cand
    cdef DTYPE_t [:] res
    cdef Py_ssize_t ol
    p1start, p2start, p1end, p2end = p1[0], p2[0], p1[p1.shape[0]-1], p2[p2.shape[0]-1]
    if p1start > p2start: # 2 comes first (assuming vertex ids are in topological order)
        p1, p2 = p2, p1
        p1start, p2start, p1end, p2end = p2start, p1start, p2end, p1end
    if p1end < p2start: # no overlap possible (assuming vertex ids are in topological order)
        pass
    else:
        p1_p2start = _getIdx(p1, p2start)
        if p1_p2start < PY_SSIZE_T_MAX: # p2 start is inside p1, p1 might come before p2
            cand = p1[p1_p2start:]
            ol = min((cand.shape[0], p2.shape[0])) if assume_nosign else ol_length(cand, p2)
            if ol == cand.shape[0]: # we have overlap along the candidate region
                if ol < p2.shape[0]: # p2 exceeds the candidate region
                    if ol < p1.shape[0]: # p1 also exceeds it
                        goodJoin = True
                        res = clone(array('I'), p1_p2start+p2.shape[0], False) # this is faster than allocating an array using np.empty, and WAY faster than using np.concatenate
                        res[:p1_p2start] = p1[:p1_p2start]
                        res[p1_p2start:] = p2
                    elif subsets_as_overlaps: # p1 is contained in p2
                        goodJoin = True
                        res = p2
                elif subsets_as_overlaps: # p2 is contained in p1
                    goodJoin = True
                    res = p1
            elif subsets_as_overlaps and ol == p2.shape[0]: # p2 is contained in p1
                goodJoin = True
                res = p1
##        else: # uncomment if we are not sure that the sequences are in order
##            p2_p1start = _getIdx(p2, p1start)
##            if p2_p1start < PY_SSIZE_T_MAX: # p2 start is inside p1, p1 might come before p2
##                cand = p2[p2_p1start:]
##                ol = min((cand.shape[0], p1.shape[0])) if assume_nosign else ol_length(cand, p1)
##                if ol == cand.shape[0]: # we have overlap along the candidate region
##                    if ol < p1.shape[0]: # p1 exceeds the candidate region
##                        if ol < p2.shape[0]: # p2 also exceeds it
##                             goodJoin = True
##                             res = clone(array('I'), p2_p1start+p1.shape[0], False) # this is faster than allocating an array using np.empty, and WAY faster than using np.concatenate
##                             res[:p2_p1start] = p2[:p2_p1start] 
##                             res[p2_p1start:] = p1
##                        elif subsets_as_overlaps: # p2 is contained in p2
##                            goodJoin = True
##                    elif subsets_as_overlaps and ol == p1.shape[0]: # p1 is contained in p2
##                        goodJoin = True
##                        res = p2
##                elif subsets_as_overlaps and ol == p1.shape[0]: # p1 is contained in p2
##                    goodJoin = True
##                    res = p2
    if not goodJoin:
        res = p1[0:0]

    return res
