import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

def c_ks(alphas, rho):
    """
    Kreisselmeier-Steinhauser (KS) function to approximate maximum of input
    exponents

    Parameters
    ----------
    alphas : numpy.ndarray
        real part of exponents computed from matrix pencil
    rho : float
        KS parameter

    Returns
    -------
    float
        approximate maximum of input array

    """
    m = alphas.max()
    
    return m + np.log(np.sum(np.exp(rho*(alphas - m))))/rho

def MatrixPencil(N, X, DT):
    """
    Complex exponential fit of a sampled waveform by the Matrix Pencil Method

    Parameters
    ----------
    N : int
        number of data samples
    X : numpy.ndarray
        array containing sampled waveform
    DT : float
        sample interval

    Returns
    -------
    R : numpy.ndarray
        array containing residues (amplitudes)
    S : numpy.ndarray
        array containing exponents (poles)

    """
    # Set the pencil parameter L
    L = N/2 - 1
    print "L = ", L

    # Assemble the Hankel matrix Y
    Y = np.empty((N-L, L+1), dtype=X.dtype)
    for i in range(N-L):
        for j in range(L+1):
            Y[i,j] = X[i+j]

    # Compute the SVD of the Hankel matrix
    U, s, VT = la.svd(Y)

    # Estimate the modal order M based on the singular values
    M = EstimateModelOrder(s, L)

    # Filter the right singular vectors of the Hankel matrix based on M
    VT = VT[:M,:]
    V1T = VT[:,:-1]
    V2T = VT[:,1:]

    # Compute the SVD of V1hat in order to form generalized inverse
    Ubar, sbar, VTbar = la.svd(V1T)
    Siginv = np.vstack((np.diag(1.0/sbar), np.zeros((L-M, M))))
    V1inv = VTbar.T.dot(Siginv).dot(Ubar.T)

    # Form A matrix from V1T and V2T to reduce to eigenvalue problem
    A = V1inv.dot(V2T)
    np.savetxt("a.dat", A)

    # Solve eigenvalue problem to obtain poles
    lam = la.eig(A, left=False, right=False)

    # Compute the residues
    B = np.zeros((N, M), dtype=lam.dtype)
    for i in range(N):
        for j in range(M):
            B[i,j] = np.abs(lam[j])**i*np.exp(1.0j*i*np.angle(lam[j]))

    r, _, _, _ = la.lstsq(B, X)

    # Compute poles
    s = np.log(lam[:M])/DT

    return r, s

def EstimateModelOrder(s, L):
    """
    Return estimate of model order based on input singular values

    Parameters
    ----------
    s : numpy.ndarray
        singular values
    L : int
        pencil parameter

    Returns
    -------
    M : int
        estimate of model order

    """
    # Normalize singular values by maximum
    snorm = s/s.max()

    # Attempt to determine where singular values "bottom out" using the point
    # where the difference between adjacent singular values dips below some
    # tolerance
    tol = 1.0e-5
    sdiff = snorm[:-1] - snorm[1:]
    bottom_out_ind = np.argmax(sdiff[sdiff < tol])

    M = min(L-1, bottom_out_ind)
    print "M = ", M
    plt.figure(figsize=(8, 6))
    plt.scatter(bottom_out_ind, snorm[bottom_out_ind], s=30, c='r')
    plt.semilogy(np.arange(L+1), snorm)

    plt.figure(figsize=(8, 6))
    plt.semilogy(np.arange(L), snorm[:-1] - snorm[1:])
    #plt.show()

    return M

def ExtractDamping(lam, dt):
    return np.log(lam).real/dt

def ReconstructSignal(t, R, S):
    """
    Given the residues and exponents, attempt to reconstruct the signal that
    was decomposed using the matrix pencil method
    """
    a = np.abs(R)    # amplitude
    p = np.angle(R)  # phase
    x = S.real       # exponent
    w = S.imag       # frequency

    X = np.zeros(t.shape)
    for i in range(len(R)):
        X += a[i]*np.exp(x[i]*t)*np.cos(w[i]*t + p[i])

    return X
