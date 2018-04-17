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

def SVDDerivative(A):
    """
    Derivatives of SVD of rectangular matrix of size m x n

    Parameters
    ----------
    A : numpy.ndarray
        matrix

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
    I'm not addressing the case of degenerate SVDs. I expect that no two
    singular values will be identical

    """
    m = A.shape[0]
    n = A.shape[1]
    ns = min(m, n) # number of singular values
    U, s, VT = la.svd(A)

    # Allocate output arrays
    dU = np.zeros((m,m,m,n))
    ds = np.zeros((ns,m,n))
    dVT = np.zeros((n,n,m,n))

    # Allocate skew-symmetric matrices
    WU = np.zeros((m,m))
    WV = np.zeros((n,n))

    for k in range(m):
        for l in range(n):
            # Form right-hand side for derivatives w.r.t. A[k,l]
            B = np.outer(U[k,:], VT[:,l])

            # Compute derivatives of singular values
            ds[:,k,l] = np.diag(B)

            # Evaluate skew-symmetric matrices
            for i in range(max(m,n)):
                for j in range(i+1,max(m,n)):
                    # Form 2x2 system
                    a11 = 0.0 if j+1 > ns else s[j]
                    a12 = 0.0 if i+1 > ns else s[i]
                    a21 = a12 
                    a22 = a11
                    b1 = 0.0 if i+1 > m or j+1 > n else B[i,j]
                    b2 = 0.0 if j+1 > m or i+1 > n else -B[j,i]

                    # Solve by Cramer's rule
                    wu = (b1*a22 - b2*a12)/(a11*a22 - a21*a12) 
                    wv = (a11*b2 - a21*b1)/(a11*a22 - a21*a12) 

                    # Store results in skew-symmetric matrices
                    if i < m and j < m:
                        WU[i,j] = wu
                        WU[j,i] = -wu
                    if i < n and j < n:
                        WV[i,j] = wv
                        WV[j,i] = -wv

            # Compute derivatives of singular vectors
            dU[:,:,k,l] = U.dot(WU)
            dVT[:,:,k,l] = WV.dot(VT)

    return dU, ds, dVT
