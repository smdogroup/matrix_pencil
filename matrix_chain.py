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

def dAdUbarTrans(dcdA, B, V2T):
    """
    Apply action of [d(A)/d(Ubar)]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(Ubar)

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    B : numpy.ndarray
        product of Vbar and Siginv
    V2T : numpy.ndarray
        filtered right singular vectors of Y1 matrix

    Returns
    -------
    dcdUbar : numpy.ndarray
        array of derivatives d(c)/d(Ubar)

    """
    m = B.shape[1]
    dcdUbar = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            dcdUbar[i,j] = np.sum(dcdA*np.outer(B[:,j], V2T[i,:]))

    return dcdUbar

def dAdsbarTrans(dcdA, VTbar, Siginv, D):
    """
    Apply action of [d(A)/d(sbar)]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(sbar)

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    VTbar : numpy.ndarray
        right singular vectors of V1T matrix
    Siginv : numpy.ndarray
        reciprocal singular values
    D : numpy.ndarray
        product of tranpose Ubar and V2T

    Returns
    -------
    dcdUbar : numpy.ndarray
        array of derivatives d(c)/d(Ubar)

    """
    sinv = np.diag(Siginv)
    n = len(sinv)

    dcds = np.zeros(n)

    for i in range(n):
        dcds[i] = -VTbar[i,:].dot(dcdA).dot(D[i,:])*sinv[i]**2

    return dcds

def dAdVTbarTrans(dcdA, C):
    """
    Apply action of [d(A)/d(VTbar)]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(VTbar)

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    C : numpy.ndarray
        product of Siginv, tranpose Ubar, and V2hat

    Returns
    -------
    dcdVTbar : numpy.ndarray
        array of derivatives d(c)/d(VTbar)

    """
    n = dcdA.shape[0]
    dcdVTbar = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            dcdVTbar[i,j] = np.sum(dcdA[j,:]*C[i,:])

    return dcdVTbar

def dAdV2Trans(dcdA, V1inv):
    """
    Apply action of [d(A)/d(V2^{T})]^{T} to the array of derivatives
    [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(V2^{T})

    Parameters
    ----------
    dcdA : numpy.ndarray
        array of derivatives d(c)/d(A)
    V1inv : numpy.ndarray
        generalized inverse of the tranpose of the V1hat matrix 

    Returns
    -------
    dcdV2T : numpy.ndarray
        vector of derivatives d(c)/d(V2^{T})

    """
    m = V1inv.shape[1]
    n = V1inv.shape[0]

    dcdV2T = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            dcdV2T[i,j] = np.sum(dcdA[:,j]*V1inv[:,i])

    return dcdV2T
