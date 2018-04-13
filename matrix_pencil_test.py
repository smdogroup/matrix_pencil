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



if __name__ == "__main__":
    print "Testing derivative of KS function..."
    TestDcDalpha()

