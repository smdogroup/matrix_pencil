import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

def pencil(N, X, DT):
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

    # Fill the Hankel matrix
    Y = np.empty((N-L, L+1), dtype=X.dtype)
    for i in range(N-L):
        for j in range(L+1):
            Y[i,j] = X[i+j]

    # Compute SVD of Hankel matrix
    U, s, Vt = la.svd(Y, full_matrices=False)

    # Determine modal order
    snorm = s/s.max()
    sdiff = snorm[:-1] - snorm[1:]
    bottom_out_ind = np.argmax(sdiff[sdiff < 10.0**-5])

    M = min(L-1, bottom_out_ind)
    print "M = ", M
    plt.figure(figsize=(8, 6))
    plt.scatter(bottom_out_ind, snorm[bottom_out_ind], s=30, c='r')
    plt.semilogy(np.arange(L+1), snorm)

    plt.figure(figsize=(8, 6))
    plt.semilogy(np.arange(L), snorm[:-1] - snorm[1:])
    plt.show()

    # Form generalized eigenvalue problem
    Sigma = np.diag(s)[:,:M]
    Vt = Vt[:M,:] # take M singular vectors
    V1t = Vt[:,:-1]
    V2t = Vt[:,1:]
    Y1 = U.dot(Sigma).dot(V1t)
    Y2 = U.dot(Sigma).dot(V2t)
    A = V1t.T.dot(V2t)

    # Solve generalized eigenvalue problem
    tuple = la.eig(A)
    z = tuple[0]

    # Compute the residues
    B = np.zeros((N, M), dtype=z.dtype)
    for i in range(N):
        for j in range(M):
            B[i,j] = np.abs(z[j])**i*np.exp(1.0j*i*np.angle(z[j]))

    R, _, _, _ = la.lstsq(B, X)

    # Compute poles
    S = np.log(z)/DT

    return R, S
