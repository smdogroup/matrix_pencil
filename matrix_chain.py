"""
Matrix pencil derivatives, but now in chain rule form
"""
import numpy as np
from scipy import linalg as la

def DalphaDlamTrans(dcda, lam, dt):
    """
    Apply action of [d(alpha)/d(lam)]^{T} to the vector of derivatives
    [d(c)/d(alpha)]^{T} to obtain the derivatives d(c)/d(lam)

    Parameters
    ----------
    dcda : numpy.ndarray
        vector of derivatives d(c)/d(alpha)
    lam : numpy.ndarray
        eigenvalues of A matrix
    dt : float
        time step

    Returns
    -------
    numpy.ndarray
        vector of derivatives d(c)/d(lam)

    """
    dadl_real = (1.0/dt)*lam.real/np.real(np.conj(lam)*lam)
    dadl_imag = (1.0/dt)*lam.imag/np.real(np.conj(lam)*lam)

    return dcda*dadl_real + 1j*dcda*dadl_imag

def DlamDATrans(dcdl, W, V):
    """
    Apply action of [d(lam)/d(A)]^{T} to the vector of derivatives
    [d(c)/d(lam)]^{T} to obtain the derivatives d(c)/d(A)

    Parameters
    ----------
    dcdl : numpy.ndarray
        vector of derivatives d(c)/d(lam)
    W : numpy.ndarray
        left eigenvectors of matrix A
    V : numpy.ndarray
        right eigenvectors of matrix A

    Returns
    -------
    dcdA : numpy.ndarray
        vector of derivatives d(c)/d(A)

    """
    WH = W.conj().T
    m = len(dcdl)
    dcdA = np.zeros((m, m))
    for i in range(m):
        w = WH[i,:]
        v = V[:,i]
        norm = w.dot(v)
        dldA = np.outer(w,v)/norm
        dcdA += dcdl[i].real*dldA.real + dcdl[i].imag*dldA.imag

    return dcdA

def dAdV2hatTrans(dcdA, V1inv):
    """
    Apply action of [d(A)/d(V2hat^{T})]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(A)/d(V2hat^{T})

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    V1inv : numpy.ndarray
        generalized inverse of the tranpose of the V1hat matrix 

    Returns
    -------
    dcdV2 : numpy.ndarray
        vector of derivatives d(c)/d(V2hat^{T})

    """
    m = V1inv.shape[0]
    n = V1inv.shape[1]

    dcdV2 = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            dcdV2 = np.sum(dcdA[:,i]*V1inv[:,j])

    return dcdV2
