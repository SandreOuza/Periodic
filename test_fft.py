import numpy as np
import timeit
import fft

n = 4
x = np.arange(n)
print(x)
mat1 = fft.convmat(x)
mat2 = fft.convmat2(x)
print(mat1)
print(mat2)
print('First we need to test to see wwhich convmat is faster')


n = 2
x = np.arange(n)
print('For a matrix of size ({0:0.2E},{0:0.2E})'.format(n))
tic = timeit.default_timer()
mat1 = fft.convmat(x)
toc = timeit.default_timer() - tic
print('It takes {0:0.2E} seconds to calculate the first one'.format(toc))

tic = timeit.default_timer()
mat2 = fft.convmat2(x)
toc = timeit.default_timer() - tic
print('It takes {0:0.2E} seconds to calculate the seconds one'.format(toc))


sigma = 10
r = 28
b = 8.0/3.0
def dlorenzf(x,y,z,w):
    n = 3*len(x)
    m = n//3
    j=np.zeros((n,n))*1j
    #Block 1
    j[0:m,0:m] = -sigma*np.identity(m) -1j*np.diag(w)
    #Block 2
    j[0:m,0+m:m+m] = sigma*np.identity(m) 
    #Block 3,already zero
    #Block 4
    j[0+m:m+m,0:m] = r*np.identity(m) - fft.convmat(z)
    #Block 5
    j[0+m:m+m,0+m:m+m] = -np.identity(m) -1j*np.diag(w)
    #Block 6
    j[0+m:m+m,0+m+m:m+m+m] = -fft.convmat(x)
    #Block 7
    j[0+m+m:m+m+m,0:m] = fft.convmat(y)
    #Block 8
    j[0+m+m:m+m+m,0+m:m+m] = fft.convmat(x)
    #Block 9
    j[0+m+m:m+m+m,0+m+m:m+m+m] = -b*np.identity(m) -1j*np.diag(w)
    return j

def dlorenz(x):
    j=np.zeros((3,3))
    j[0] = [-sigma,sigma,0.0]
    j[1] = [r- x[2], -1.0, -x[0]]
    j[2] = [x[1],x[0],-b]
    return j

x = np.array([1])
xx = np.array([100,212,-3])

print(dlorenzf(np.array([xx[0]]),np.array([xx[1]]),np.array([xx[2]]),x*0.0)-dlorenz(xx))

n = 256*4
xx = np.ones(n)
tic = timeit.default_timer()
gg = dlorenzf(xx,xx,xx,xx)
toc = timeit.default_timer() - tic
print('It takes {0:0.2E} seconds to construct the jacobian matrix for lorenz for n={1:0.2E}'.format(toc,n))

xxx = np.ones(3*n)
tic = timeit.default_timer()
np.linalg.solve(gg,xxx)
toc = timeit.default_timer() - tic
print('solving ax = b takes {0:0.2E} seconds'.format(toc,n))

