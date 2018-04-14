import numpy as np
from scipy import linalg as la
from matrix_pencil import *
from matrix_pencil_der import *

np.set_printoptions(precision=5)

def TestDcDalpha():
    """
    Test DcDalpha function using random data and complex step approximation
    """
    # Create random exponent values
    M = 10
    alphas = np.random.random(M).astype(np.complex)

    # Add random complex perturbation
    h = 1.0e-30
    pert = np.random.random(M)
    alphas += 1j*h*pert

    # Obtain derivative by complex step approximation 
    rho = 100.0
    approx = c_ks(alphas, rho).imag/h
    print "Approximation: ", approx

    # Obtain derivative analytically
    analytic = pert.dot(DcDalpha(alphas.real, rho))
    print "Analytic:      ", analytic

    # Compute relative error
    rel_error = (analytic - approx)/approx
    print "Rel. error:    ", rel_error

    return

def TestDalphaDlam():
    """
    Test DcDalpha function using random data and finite difference
    approximation
    """
    # Create random complex eigenvalues
    M = 10
    dt = 0.1
    alphas = -1.0 + 2.0*np.random.random(M) 
    omegas = -1.0 + 2.0*np.random.random(M) 
    s = alphas + 1j*omegas
    lam = np.exp(s*dt)

    # Perturb the real and imaginary parts of the eigenvalues 
    h = 1.0e-6
    pert = np.ones(M)
    lam_pos_real = lam + h*pert
    lam_neg_real = lam - h*pert
    lam_pos_imag = lam + 1j*h*pert
    lam_neg_imag = lam - 1j*h*pert
    
    f = lambda x: np.log(x).real/dt 

    # Compute finite difference approximatation to real and imaginary
    # part derivatives
    approx_real = 0.5*(f(lam_pos_real) - f(lam_neg_real))/h
    approx_imag = 0.5*(f(lam_pos_imag) - f(lam_neg_imag))/h

    approx = approx_real + 1j*approx_imag
    print "Approximation: ", approx

    # Obtain derivatives analytically
    analytic = DalphaDlam(lam, dt)
    print "Analytic:      ", analytic

    # Compute relative error
    rel_error_real = (analytic.real - approx.real)/approx.real
    rel_error_imag = (analytic.imag - approx.imag)/approx.imag
    print "Rel. error r:  ", rel_error_real
    print "Rel. error i:  ", rel_error_imag

    return

def TestDlamDA():
    """
    Test DlamDA function using random data and finite difference
    approximation
    """
    # Create random matrix and obtain eigenvalues
    m = 3
    A = np.random.random((m,m))
    A = np.loadtxt("a.dat")
    m = A.shape[0]
    lam = la.eig(A, left=False, right=False)

    # Perturb matrix and obtain eigenvalues
    h = 1e-6
    pert = np.random.random((m,m))
    lam_pert = la.eig(A + h*pert, left=False, right=False)

    # Compute finite difference approximation to derivative
    approx = (lam_pert - lam)/h
    print "Approximation: ", approx

    # Obtain derivatives analytically
    dlam = DlamDA(A)
    analytic = np.zeros(lam.shape, dtype=np.complex)
    for k in range(len(lam)):
        analytic[k] = np.sum(dlam[:,:,k]*pert)
    print "Analytic:      ", analytic

    # Compute relative error
    rel_error_real = (analytic.real - approx.real)/approx.real
    rel_error_imag = (analytic.imag - approx.imag)/approx.imag
    print "Rel. error r:  ", rel_error_real
    print "Rel. error i:  ", rel_error_imag

if __name__ == "__main__":
    print "Testing derivative of KS function"
    print "---------------------------------"
    TestDcDalpha()
    print
    print "Testing derivative of exponent extraction"
    print "-----------------------------------------"
    TestDalphaDlam()
    print
    print "Testing derivative of eigenvalue problem"
    print "----------------------------------------"
    TestDlamDA()
