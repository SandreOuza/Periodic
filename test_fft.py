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


n = 256*2*4
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
