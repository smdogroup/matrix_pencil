import numpy as np
from scipy import linalg as la
from matrix_pencil import *
from matrix_pencil_der import *
import matplotlib.pyplot as plt

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
    m = 10
    A = np.random.random((m,m))
    #A = np.loadtxt("a.dat")
    #m = A.shape[0]
    lam = la.eig(A, left=False, right=False)

    # Perturb matrix and obtain eigenvalues
    h = 1e-7
    pert = np.random.random((m,m))
    lam_pos = la.eig(A + h*pert, left=False, right=False)
    lam_neg = la.eig(A - h*pert, left=False, right=False)

    # Compute finite difference approximation to derivative
    approx = 0.5*(lam_pos - lam_neg)/h
    print "Approximation: ", approx

    # Obtain derivatives analytically
    dlam = DlamDA(A)
    analytic = np.zeros(lam.shape, dtype=np.complex)
    for i in range(len(lam)):
        analytic[i] = np.sum(dlam[i,:,:]*pert)
    print "Analytic:      ", analytic

    # Compute relative error
    rel_error_real = (analytic.real - approx.real)/approx.real
    rel_error_imag = (analytic.imag - approx.imag)/approx.imag
    print "Rel. error r:  ", rel_error_real
    print "Rel. error i:  ", rel_error_imag

    return

def TestSVDDerivative():
    # Create random rectangular matrix
    m = 4
    n = 2
    ns = min(m,n) # number of singular values
    A = np.random.random((m,n))
    U, s, VT = la.svd(A)

    # Perturb matrix and compute SVD
    h = 1.0e-8
    pert = np.random.random((m,n))
    Upos, spos, VTpos = la.svd(A + h*pert)
    Uneg, sneg, VTneg = la.svd(A - h*pert)

    # Compute finite difference approximations to derivatives
    Uapprox = 0.5*(Upos - Uneg)/h
    sapprox = 0.5*(spos - sneg)/h
    VTapprox = 0.5*(VTpos - VTneg)/h

    # Obtain derivatives analytically
    dU, ds, dVT = SVDDerivative(U, s, VT)

    Uanalytic = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            Uanalytic[i,j] = np.sum(dU[i,j,:,:]*pert)

    sanalytic = np.zeros(ns)
    for i in range(ns):
        sanalytic[i] = np.sum(ds[i,:,:]*pert)

    VTanalytic = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            VTanalytic[i,j] = np.sum(dVT[i,j,:,:]*pert)

    # Compute relative error
    U_rel_error = (Uanalytic - Uapprox)/Uapprox
    print "Approx., U:    " 
    print Uapprox
    print "Analytic, U:   " 
    print Uanalytic
    print "Rel. error, U: " 
    print U_rel_error
    print

    s_rel_error = (sanalytic - sapprox)/sapprox
    print "Approx., s:    ", sapprox
    print "Analytic, s:   ", sanalytic
    print "Rel. error, s: ", s_rel_error
    print

    VT_rel_error = (VTanalytic - VTapprox)/VTapprox
    print "Approx., VT  : " 
    print VTapprox
    print "Analytic, VT:  " 
    print VTanalytic
    print "Rel. error, VT:" 
    print VT_rel_error
    print

    return

def TestPseudoinverseDer():
    """
    Test derivative of pseudoinverse w.r.t. original matrix

    """
    m = 3
    n = 5

    A = np.random.random((m,n))
    _, _, VT = la.svd(A)
    A = VT
    Ainv = la.pinv(A)

    dAinv = PseudoinverseDerivative(A, Ainv)
    
    # Approximate by finite differences
    h = 1.0e-6
    Apert = np.random.random(A.shape)

    Apos = A + h*Apert
    Aneg = A - h*Apert

    Aposinv = la.pinv(Apos)
    Aneginv = la.pinv(Aneg)

    approx = 0.5*(Aposinv - Aneginv)/h
    print "Approx., Ainv: " 
    print approx

    # Obtain analytic derivative
    analytic = np.zeros(Ainv.shape)
    for i in range(n):
        for j in range(m):
            analytic[i,j] = np.sum(dAinv[i,j,:,:]*Apert)
    print "Analytic, Ainv:" 
    print analytic

    # Compute relative error
    rel_error = (analytic - approx)/approx
    print "Error, Ainv:   " 
    print rel_error

    return
    
def TestChain():
    """
    Test derivative of output of KS function w.r.t. something

    """
    N = 20
    L = N/2 - 1
    M = 3
    h = 1.0e-8
    dt = 0.1
    rho = 100

    # Create random Vhat matrix
    Y = np.random.random((N-L,L+1))
    U, s, VT = la.svd(Y)
    #VT = np.random.random((L+1,L+1))
    Vhat = VT[:M,:]
    V1T = Vhat[:,:-1]
    V2T = Vhat[:,1:]
    Ubar, sbar, VTbar = la.svd(V1T)

    # Create random matrix for eigenvalue problem and obtain damping
    Siginv = np.vstack((np.diag(1.0/sbar), np.zeros((L-M,M))))
    A = VTbar.T.dot(Siginv).dot(Ubar.T).dot(V2T)
    lam, W, V = la.eig(A, left=True, right=True)
    alphas = np.log(lam).real/dt

    # Perturb the matrix and obtain eigenvalues
    VTpert = np.random.random((L+1,L+1))
    VTpos = VT + h*VTpert
    VTneg = VT - h*VTpert

    #_, _, VTpos = la.svd(Ypos)
    Vhatpos = VTpos[:M,:]
    V1Tpos = Vhatpos[:,:-1]
    V2Tpos = Vhatpos[:,1:]
    Upos, spos, VTpos = la.svd(V1Tpos)
    Sigpos = np.vstack((np.diag(1.0/spos), np.zeros((L-M,M))))
    Apos = VTpos.T.dot(Sigpos).dot(Upos.T).dot(V2Tpos)

    #_, _, VTneg = la.svd(Yneg)
    Vhatneg = VTneg[:M,:]
    V1Tneg = Vhatneg[:,:-1]
    V2Tneg = Vhatneg[:,1:]
    Uneg, sneg, VTneg = la.svd(V1Tneg)
    Signeg = np.vstack((np.diag(1.0/sneg), np.zeros((L-M,M))))
    Aneg = VTneg.T.dot(Signeg).dot(Uneg.T).dot(V2Tneg)

    # Compute the perturbed eigenvalues
    lampos = la.eig(Apos, left=False, right=False)
    lamneg = la.eig(Aneg, left=False, right=False)

    # Compute the pertubed damping from the eigenvalues
    alphapos = np.log(lampos).real/dt
    alphaneg = np.log(lamneg).real/dt

    # Compute the perturbed outputs of the KS_function
    cpos = c_ks(alphapos, rho)
    cneg = c_ks(alphaneg, rho)

    # Compute the finite difference approximation to the derivative
    approx = 0.5*(cpos - cneg)/h
    print "Approximation: ", approx

    # Compute the analytic derivative
    V1inv = VTbar.T.dot(Siginv).dot(Ubar.T)

    dcda = DcDalpha(alphas, rho)
    dcdl = DalphaDlamTrans(dcda, lam, dt)
    dcdA = DlamDATrans(dcdl, W, V)
    dcdV1T = dAdV1Trans(dcdA, Ubar, sbar, VTbar, V2T)
    dcdV2T = dAdV2Trans(dcdA, V1inv)
    dcdVhat = dV12dVhatTrans(dcdV1T, dcdV2T)
    dcdVT = dVTdVhatTrans(dcdVhat)
    #dcdY = dVTdYTrans(dcdVT, U, s, VT)

    analytic = np.sum(dcdVT*VTpert)
    print "Analytic:      ", analytic

    # Compute relative error
    rel_error = (analytic - approx)/approx
    print "Rel. error:    ", rel_error

    return

def TestFullMatrixPencilDer():
    t = np.linspace(0.0, 1.0, 101)
    X = np.exp(-t)*np.sin(2.0*np.pi*t) + np.sin(4.0*np.pi*t)
    dt = 0.1

    cutoff = 1.0e-10
    pencil = MatrixPencil(X, dt, True, tol=cutoff)
    pencil.ComputeDampingAndFrequency()
    pencil.ComputeAmplitudeAndPhase()
    Xre = pencil.ReconstructSignal(t)
    plt.figure()
    plt.plot(t, X, 'orange')
    plt.plot(t, Xre, '--b')
    plt.show()
    c = pencil.AggregateDamping()
    cder = pencil.AggregateDampingDer()
    
    # Perturb the intial data
    h = 1.0e-6
    Xpert = np.random.random(X.shape)
    Xpos = X + h*Xpert
    Xneg = X - h*Xpert

    # Approximate the derivative using finite differences
    pencilpos = MatrixPencil(Xpos, dt, tol=cutoff)
    pencilpos.ComputeDampingAndFrequency()
    cpos = pencilpos.AggregateDamping()

    pencilneg = MatrixPencil(Xneg, dt, tol=cutoff)
    pencilneg.ComputeDampingAndFrequency()
    cneg = pencilneg.AggregateDamping()

    approx = 0.5*(cpos - cneg)/h
    print approx

    # Compute analytic derivative
    analytic = np.sum(cder*Xpert)
    print analytic

    # Compute relative error
    rel_error = (analytic - approx)/approx
    print rel_error

    return

if __name__ == "__main__":
    #print "======================"
    #print "Basic derivative tests"
    #print "======================"
    #print
    #print "Testing derivative of KS function"
    #print "---------------------------------"
    #TestDcDalpha()
    #print
    #print "Testing derivative of exponent extraction"
    #print "-----------------------------------------"
    #TestDalphaDlam()
    #print
    #print "Testing derivative of eigenvalue problem"
    #print "----------------------------------------"
    #TestDlamDA()
    #print
    #print "Testing derivative of SVD"
    #print "-------------------------"
    #TestSVDDerivative()
    #print
    print "Testing derivative of Pseudoinverse"
    print "-----------------------------------"
    TestPseudoinverseDer()
    print
    #print "========================"
    #print "Chained derivative tests"
    #print "========================"
    #print
    #print "Testing dc/dY"
    #print "-------------"
    #TestChain()
    #print
    #print "Testing dc/dX"
    #print "-------------"
    #TestFullMatrixPencilDer()

