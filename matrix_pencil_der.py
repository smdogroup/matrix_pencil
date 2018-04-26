import numpy as np
from scipy import linalg as la

def DcDalpha(alphas, rho):
    """
    Derivative of KS function with respect to the input array of exponents
    
    Parameters
    ----------
    alphas : numpy.ndarray
        real part of exponents computed from matrix pencil
    rho : float
        KS parameter

    Returns
    -------
    numpy.ndarray
        array of derivatives

    """
    m = alphas.max()
    a = np.sum(np.exp(rho*(alphas - m)))

    return np.exp(rho*(alphas - m))/a

def DalphaDlam(lam, dt):
    """
    Derivative of exponents with respect to eigenvalues

    Parameters
    ----------
    lam : numpy.ndarray
        eigenvalues
    dt : float
        time step

    Returns
    -------
    numpy.ndarray
        derivatives

    """
    real_part = (1.0/dt)*lam.real/np.real(np.conj(lam)*lam)
    imag_part = (1.0/dt)*lam.imag/np.real(np.conj(lam)*lam)

    return real_part + 1j*imag_part

def DlamDA(A):
    """
    Derivatives of each eigenvalue with respect to originating matrix

    Parameters
    ----------
    A : numpy.ndarray
        matrix

    Returns
    -------
    dlam : numpy.ndarray
        matrix of derivatives

    """
    lam, W, V = la.eig(A, left=True, right=True)
    WH = W.conj().T
    m = len(lam)
    dlam = np.zeros((m, m, m), dtype=lam.dtype)
    for i in range(m):
        w = WH[i,:]
        v = V[:,i]
        norm = w.dot(v)
        dlam[i,:,:] = np.outer(w,v)/norm

    return dlam

def SVDDerivative(U, s, VT):
    """
    Derivatives of SVD of rectangular matrix of size m x n

    Parameters
    ----------
    U : numpy.ndarray
        left singular vectors
    s : numpy.ndarray
        singular values
    VT : numpy.ndarray
        right singular vectors

    Returns
    -------
    dU : numpy.ndarray
        derivatives dU[i,j]/dA[k,l]
    ds : numpy.ndarray
        derivatives ds[i]/dA[k,l]
    dVT : numpy.ndarray
        derivatives dVT[i,j]/dA[k,l]

    Notes 
    -----
    This function does not address the case of degenerate SVDs. It expects that
    no two singular values will be identical

    You can find an explanation for the algorithm here at:
    http://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

    """
    m = U.shape[0]
    n = VT.shape[1]

    assert(m <= n), "Input matrix should have dimensions m <= n. Use transpose?"

    # Allocate output arrays
    dU = np.zeros((m,m,m,n))
    ds = np.zeros((m,m,n))
    dVT = np.zeros((n,n,m,n))

    # Square matrix of singular values
    S1 = np.diag(s)

    for k in range(m):
        for l in range(n):
            dP = np.outer(U[k,:], VT[:,l])
            dP1 = dP[:,:m]

            # Extract diagonal for ds
            ds[:,k,l] = np.diag(dP1)

            # Form skew-symmetric F matrix
            F = np.zeros((m,m))
            for i in range(m):
                for j in range(i+1,m):
                    F[i,j] = 1.0/(s[j]**2 - s[i]**2)
                    F[j,i] = 1.0/(s[i]**2 - s[j]**2)

            # Compute dC matrix and skew-symmetric part of the dD matrix
            dC = F*(dP1.dot(S1) + S1.dot(dP1.T))
            dD1 = F*(S1.dot(dP1) + dP1.T.dot(S1))

            # Assemble dD matrix
            if m < n:
                dP2 = dP[:,m:]
                S1inv = np.diag(1.0/s)
                dD2 = S1inv.dot(dP2)

                dDT = np.zeros((n,n))
                dDT[:m,:m] = -dD1
                dDT[:m,m:] = dD2
                dDT[m:,:m] = -dD2.T

            else:
                dDT = -dD1

            # Compute dU and dVT sensitivities from dC and dD
            dU[:,:,k,l] = U.dot(dC)
            dVT[:,:,k,l] = dDT.dot(VT)

    return dU, ds, dVT
