import numpy as np
import scipy.linalg as la
from matrix_pencil_der import *

m = 2
n = 4
A = np.random.random((2,4))
U, s, VT = la.svd(A)
V = VT.T

# Approximation
h = 1e-6
pert = np.random.random((2,4))
Apos = A + h*pert
Aneg = A - h*pert
Upos, spos, VTpos = la.svd(Apos)
Uneg, sneg, VTneg = la.svd(Aneg)

approx = 0.5*(VTpos - VTneg)/h
print approx

# Attempting to get analytic
B = A.T.dot(A)
Sigma = np.zeros(B.shape)
for i in range(m):
    Sigma[i,i] = 1.0/s[i]**2
Binv = V.dot(Sigma).dot(V.T)

def dBdATrans(pert, A):
    m = A.shape[0]
    n = A.shape[1]
    dB = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            dB[i,j] = np.sum(A[:,j]*pert[:,i] + A[:,i]*pert[:,j])

    return dB

dU, ds, dVTdA = SVDDerivative(U, s, VT)
dVT = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        dVT[i,j] += np.sum(dVTdA[i,j,:,:]*pert)

print dVT

dB = dBdATrans(pert, A)
analytic = -Binv.dot(dB).dot(V[:,2:])
analytic = analytic.T
print analytic

# Attempt another time
def generate_transpose_matrix(m, n):
    T = np.zeros((m*n,m*n))
    for i in range(m):
        for j in range(n):
            T[j+n*i,i+m*j] = 1.0

    return T

Vhat = V[:,:n-m]
T = generate_transpose_matrix(4,2)
testvec = np.arange(8)
print testvec.flatten(4,2, order='F') 
print T.dot(testvec).reshape(4,2, order='C')
A = np.kron(np.eye(n-m), B)
C = np.kron(np.eye(n-m), Vhat)
B = np.kron(Vhat.T, np.eye(n-m)).dot(T) + np.kron(np.eye(n-m), Vhat.T)
d = -dB.dot(Vhat).flatten()
print A.shape
print B.shape
print C.shape
print d.shape
