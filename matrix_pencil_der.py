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

def EigenDerivative():
    """
    Derivatives of each eigenvalue with respect to originating matrix
    """

def SVDDerivative():
    """
    Derivatives of SVD of rectangular matrix of size m x n
    """
