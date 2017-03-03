import numpy as np
from scipy import linalg


def solve(resid, df, st, tol = 10**(-12.0), maxit=10):
    '''
    solve(resid, df, st, tol = 10**(-12.0), maxit=10)
    resid: a function, returns the residual
    df: a function, returns the jacobian
    st: a vector, an initial guess
    tol: a number, tolerance for newton solver
    maxit: a natural number, max number of iterations
    returns a [vector,vector, the solution to the newtons solver and the history of the norm of the residual
    '''
    xnew = st
    rhist = np.ones(maxit)
    j=0
    for i in range(maxit):
        xold = xnew
        dx = np.linalg.solve(df(xold),-1.0*resid(xold))
        xnew = xold + dx
        rval = np.linalg.norm(resid(xnew))
        rhist[i] = rval
        if  rval < tol:
            j = i
            break
        else:
            j = i
    return xnew, rhist[0:j+1]

def rnls(r,x,dx):
    '''
    residual norm line search
    r: a function, the residual
    x: a vector, a vector
    dx: a vector, an increment
    '''
    num = 10
    step = 0.1
    rlist = np.zeros(num)
    for i in range(num):
        rlist[i] = np.linalg.norm(r(x+step*i*dx))
    return rlist, step

def linesearch(r,x,dx):
    '''
    residual norm line search
    r: a function, the residual
    x: a vector, a vector
    dx: a vector, an increment
    returns: a vector,
    '''
    print('IN TRUST REGION')
    trustregion = 0.01
    reldelta = np.linalg.norm(dx) / np.linalg.norm(x)
    alpha = 1
    if reldelta > trustregion:
        alpha = trustregion/reldelta
    num = 10
    step = 0.1*alpha
    rlist = np.zeros(num)
    for i in range(num):
        rlist[i] = np.linalg.norm(r(x+step*i*dx))
    j = 0
    mm = min(rlist)
    for i in range(len(rlist)):
        if ll[i] == mm:
            j = i
    if j==0:
        print('STEP SIZES ARE TOO LARGE')
    return x + step*j*dx

def rob(resid, df, st, tol = 10**(-12.0), maxit=10):
    '''
    rob(resid, df, st, tol = 10**(-12.0), maxit=10)
    resid: a function, returns the residual
    df: a function, returns the jacobian
    st: a vector, an initial guess
    tol: a number, tolerance for newton solver
    maxit: a natural number, max number of iterations
    returns a [vector,vector, the solution to the newtons solver and the history of the norm of the residual
    '''
    xnew = st
    rhist = np.ones(maxit+1)
    j=0
    for i in range(maxit):
        xold = xnew
        rhist[i] = np.linalg.norm(resid(xold))
        dx = np.linalg.solve(df(xold),-1.0*resid(xold))
        rval = np.linalg.norm(resid(xold + dx))
        if rval < rhist[i]:
            xnew = xold + dx
        else:
            xnew = linesearch(resid,xold,dx)
        rval = np.linalg.norm(resid(xnew))
        rhist[i+1] = rval
        if  rval < tol:
            j = i
            break
        else:
            j = i
        return xnew, rhist[0:j+1]
