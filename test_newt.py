import newt
import numpy as np
import matplotlib.pyplot as plt

print('In this module we test our newton solver')
p1 = False

def f(x):
    return np.array([(x[0]-3.0)*(x[0]+3.0)])
def df(x):
    return np.array([[2.0*x[0]]])

def f2(x):
    return np.array([(x[0]-3.0)*(x[0]+3.0), (x[1]-5.0)*(x[1]+5.0)])

def df2(x):
    return np.array([[2.0*x[0], 0.0],[0.0,2.0*x[1]]])

print('Lets look at a 1D example')
xold = np.array([-8.0])
print(f(xold))
print(df(xold))

answ,hist = newt.solve(f, df, xold, tol = 10**(-12.0), maxit=10)
print(answ)
print(hist)
plt.plot(np.log(hist+10**(-15.0))/np.log(10.0))
plt.grid(True)
plt.title('resid history for 1D')
if p1==True: plt.show()

print('-------')
print('Lets look at a 2D example')
xold2 = np.array([-8.0,-8.0])

print(f2(xold2))
print(df2(xold2))

answ, hist = newt.solve(f2, df2, xold2, tol = 10**(-12.0), maxit=10)
print(hist)
print(answ)
print(hist[-1])
print(hist[0])
#plot the history
plt.plot(np.log(hist+10**(-16.0))/np.log(10.0))
plt.grid(True)
plt.title('resid history for 2D')
if p1==True: plt.show()
print('-------')


r= 28.0
b = 8.0/3.0
sigma = 10.0
def lorenz(x):
    res = np.zeros(3)
    res[0] = sigma*(-x[0] + x[1])
    res[1] = -x[1] - x[0]*x[2] + r*x[0]
    res[2] = -b*x[2] + x[1]*x[0]
    return res

def dlorenz(x):
    j=np.zeros((3,3))
    j[0] = [-sigma,sigma,0.0]
    j[1] = [r- x[2], -1.0, -x[0]]
    j[2] = [x[1],x[0],-b]
    return j



mit = 10
tole = 10**(-16.0)

print('First Lorenz System')
xold3 = np.array([0.1,0.1,0.1])
print(lorenz(xold3))
print(dlorenz(xold3))
answ, hist = newt.solve(lorenz, dlorenz, xold3, tol = tole, maxit=mit)
print('After {0:0.2E} iterations we get'.format(mit))
print(answ)
print('The last residual is {0:0.2E}'.format(hist[-1]))
#plot the history
plt.plot(np.log(hist)/np.log(10.0))
plt.grid(True)
plt.title('resid history for lorenz 1')
if p1==True: plt.show()

print('------')
print('Second Lorenz System')
xold3 = np.array([28.0,28.0,28.0])
#xold3 = np.array([-np.sqrt(b*(r-1)),-np.sqrt(b*(r-1)),r-1])+np.array([0.1,0.2,0.1])*0.00001
answ, hist = newt.solve(lorenz, dlorenz, xold3, tol = tole, maxit=mit)
print('After {0:0.2E} iterations we get'.format(mit))
print(answ)
#print(hist)
#plot the history
plt.plot(np.log(hist)/np.log(10.0))
plt.grid(True)
plt.title('resid history for lorenz 2')
if p1==True: plt.show()
print('The last residual is {0:0.2E}'.format(hist[-1]))
#true answ
truansw = np.array([np.sqrt(b*(r-1)),np.sqrt(b*(r-1)),r-1])
print('Should be ')
print(truansw)
print('The resid of the true answ is ')
print(lorenz(np.array(truansw)))

print('------')
print('Third Lorenz System')
xold3 = np.array([-28.0,-28.0,28.0])
answ, hist = newt.solve(lorenz, dlorenz, xold3, tol = tole, maxit=mit)
print('After {0:0.2E} iterations we get'.format(mit))
print(answ)
print('The last residual is {0:0.2E}'.format(hist[-1]))
#true answ
truansw = [-np.sqrt(b*(r-1)),-np.sqrt(b*(r-1)),r-1]
print('Should be ')
print(truansw)
print('The resid of the true answ is ')
print(lorenz(np.array(truansw)))
#plot the history
plt.plot(np.log(hist)/np.log(10.0))
plt.grid(True)
plt.title('resid history for lorenz 3')
if p1==True: plt.show()


perturb = np.array([1.0,1.0,1.0])*100
truansw = np.array([np.sqrt(b*(r-1)),np.sqrt(b*(r-1)),r-1])
#truansw = np.array([0.0,0.0,0.0])
xold3 = truansw + perturb
answ, hist = newt.rob(lorenz, dlorenz, xold3, tol = tole, maxit=mit)
print('=----=')
print('manual testing')
#print(xold3)
#print(lorenz(xold3))
#print(dlorenz(xold3))
#plt.show()
for i in range(0):
    print('Starting with {0:0.2E}'.format(i))
    print(np.linalg.norm(lorenz(xold3)))
    dx = np.linalg.solve(dlorenz(xold3),-lorenz(xold3))
    print(np.linalg.norm(lorenz(xold3+dx)))
    print(np.linalg.norm(dx))
    print(np.linalg.norm(xold3))
    print(np.linalg.norm(dx)/np.linalg.norm(xold3))
    ll,step = newt.rnls(lorenz,xold3,dx)
    #print(ll)
    j = 0
    mm = min(ll)
    for i in range(len(ll)):
        if ll[i] == mm:
            j = i
    print(j)
    print(j*step)
    print(j*step*np.linalg.norm(dx)/np.linalg.norm(xold3))
    print(np.linalg.norm(lorenz(dx)))
    #xold3 += step*j*dx
    xold3 += dx
    plt.plot(ll)
    plt.grid(True)
    plt.title('line search error')
    #plt.show()
    plt.plot(ll[j-4:j+4])
    plt.grid(True)
    plt.title('line search error zoom')
    #plt.show()
    print('------')
    

'''
    alpha = 1.0
    aa = np.linalg.norm(lorenz(xold3))
    bb = np.linalg.norm(lorenz(xold3+dx))
    cc = np.linalg.norm(lorenz(xold3+2.0*dx))
    print(xold3)
    print(aa)
    print(bb)
    print(cc)
    print(xold3)
    alpha = (1.5*aa - 2.0*bb + cc/2.0 )/(aa - 2.0*bb + cc)
    print(xold3)
    print(dx)
    print(np.linalg.norm(lorenz(xold3-dx)))
    print(xold3)
    print(np.linalg.norm(lorenz(xold3)))
    print(xold3)
    print(np.linalg.norm(lorenz(xold3+0.5*dx)))
    print(np.linalg.norm(lorenz(xold3+dx)))
    print(np.linalg.norm(lorenz(xold3+2.0*dx)))
    print(np.linalg.norm(lorenz(xold3+3.0*dx)))
    print(np.linalg.norm(lorenz(xold3+10.0*dx)))
    print(np.linalg.norm(lorenz(xold3+(r-1)*dx)))
    print(np.linalg.norm(lorenz(xold3+r*dx)))
    print(np.linalg.norm(lorenz(xold3+100.0*dx)))
    print(alpha)
    print(np.linalg.norm(lorenz(xold3+alpha*dx)))
    print('-----------')
'''
