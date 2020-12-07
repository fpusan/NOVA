from Overlap import overlap, check_bimera
import numpy as np
B = np.array([0,1,2,10,11,12,40,50], dtype=np.uint32)
NB1 = np.array([0,1,2,10,11,13], dtype=np.uint32)
NB2 = np.array([1,2,10,11,12], dtype=np.uint32)
NB3 = np.array([10,11,12,30,60,80], dtype=np.uint32)
p1 = np.array([0,1,2,10,11,12,20,30], dtype=np.uint32)
p2 = np.array([8,9,10,11,12,40,50], dtype=np.uint32)


def is_bimera(B, p1, p2):
    p1_L = ol_length(B, p1)
    p2_L = ol_length(B, p2)
    p1_R = ol_length_R(B, p1)
    p2_R = ol_length_R(B, p2)

    if not p1_L and not p1_R:
        return ()
    elif p1_R == len(p1) and p2_R == len(p2):
        return ()
    else:
        if p1_L > p2_R or p2_L > p1_R:
            if p2_L > p1_R:
                p1, p2 = p2, p1
                p1_L, p1_R, p2_L, p2_R = p2_L, p2_R, p1_L, p1_R
            ol = overlap(p1[:p1_L], p2[p2_R-1:])
            if not ol.shape[0]:
                return ()
            else:
                if np.array_equal(ol, B):
                    return B[p2_R-2], B[p1_L]
        else: # No overlap
            return ()

def ol_length(p1, p2):
    res = 0
    for i in range(len(p1)):
        if p1[i] == p2[i]:
            res += 1
        else:
            break
    return res

def ol_length_R(p1, p2):
    res = len(p1)
    for i in range(len(p1)):
        if p1[len(p1)-i-1] == p2[len(p2)-i-1]:
            res -= 1
        else:
            break
    return res
    



##
##for i in range(1000000):
##    check_bimera(B, p1, p2)
