import numpy as np
from scipy import linalg as la
from matrix_pencil import *
from matrix_pencil_der import *

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

    # Perturb the eigenvalues 
    h = 1.0e-6
    pert = np.ones(M) + 1j*np.ones(M)
    lam_pos = lam + h*pert
    lam_neg = lam - h*pert

    # Compute the perturbed alphas
    alpha_pos = np.log(lam_pos).real/dt
    alpha_neg = np.log(lam_neg).real/dt

    # Compute finite difference approximatation to derivatives
    approx = 0.5*(alpha_pos - alpha_neg)/h
    print "Approximation: ", approx

    # Obtain derivatives analytically
    analytic = DalphaDlam(lam, dt)
    print "Analytic:      ", analytic

    # Compute relative error
    rel_error = (analytic - approx)/approx
    print "Rel. error:    ", rel_error

if __name__ == "__main__":
    print "Testing derivative of KS function..."
    TestDcDalpha()
    print "Testing derivative of exponent extraction..."
    TestDalphaDlam()

