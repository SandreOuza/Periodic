import numpy as np

#circular convolution
def convmat(v):
    n = len(v)
    tmp = 1.0/n
    mat = np.zeros((n,n))
    for j in range(n):
        for i in range(n):
            mat[i,j] = v[(i-j+n)%n]*tmp
    return mat

def convmat2(v):
    n = len(v)
    tmp = 1.0/n
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mat[i,j] = v[(i-j+n)%n]*tmp
    return mat
    
