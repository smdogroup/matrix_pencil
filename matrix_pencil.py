import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import matplotlib.pyplot as plt
from matrix_pencil_der import *

class MatrixPencil(object):
    def __init__(self, X, dt, output=False, rho=100):
        """
        Provide the capability to:
        1) decompose input signal into a series of complex exponentials by the
        Matrix Pencil Method; and 
        2) compute the derivative of the KS (Kreisselmeier-Steinhauser)
        aggregation of the real exponents (damping) with respect to the input
        data.

        Parameters
        ----------
        X : numpy.ndarray
            array containing sampled waveform
        dt : float
            sample interval
        output : bool
            choose whether or not to provide information
        rho : float
            KS parameter

        """
        self.X = X
        self.dt = dt
        self.output = output
        self.rho = rho

        # Set the pencil parameter L
        self.N = X.shape[0]
        self.L = self.N/2 - 1

        if self.output:
            print "Initializing Matrix Pencil method..."
            print "Number of samples, N = ", self.N
            print "Pencil parameter, L = ", self.L

        # Store model order
        self.M = None

        # Save SVD of Hankel matrix
        self.U = None
        self.s = None
        self.VT = None

        # Save the filtered right singular vectors V1 and the pseudoinverse
        self.V1T = None
        self.V1inv = None

        # Save right singular vectors V2
        self.V2T = None

        # Save eigenvalues and eigenvectors of A matrix
        self.lam = None
        self.W = None
        self.V = None

        # Save amplitudes, damping, frequencies, and phases
        self.amps = None
        self.damp = None
        self.freq = None
        self.faze = None

        # Save input KS parameter

    def ComputeDampingAndFrequency(self):
        """
        Compute the damping and frequencies of the complex exponentials 

        """
        # Assemble the Hankel matrix Y from the samples X
        Y = np.empty((self.N-self.L, self.L+1), dtype=self.X.dtype)
        for i in range(self.N-self.L):
            for j in range(self.L+1):
                Y[i,j] = self.X[i+j]
                
        # Compute the SVD of the Hankel matrix
        self.U, self.s, self.VT = la.svd(Y)

        # Estimate the modal order M based on the singular values
        self.EstimateModelOrder()

        # Filter the right singular vectors of the Hankel matrix based on M
        Vhat = self.VT[:self.M,:]
        self.V1T = Vhat[:,:-1]
        self.V2T = Vhat[:,1:]

        # Compute the pseudoinverse of V1T
        self.V1inv = la.pinv(self.V1T)

        # Form A matrix from V1T and V2T to reduce to eigenvalue problem
        A = self.V1inv.dot(self.V2T)

        # Solve eigenvalue problem to obtain poles
        self.lam, self.W, self.V = la.eig(A, left=True, right=True)

        # Compute damping and freqency
        s = np.log(self.lam[:self.M])/self.dt
        self.damp = s.real
        self.freq = s.imag

        return

    def EstimateModelOrder(self):
        """
        Store estimate of model order based on input singular values

        Notes
        -----
        This is a pretty crude method for estimating the model order.
        Development of a more sophisticated method would probably yield greater
        robustness

        """
        tol = 1.0e-6
        n_above_tol = len(self.s[self.s > tol])

        w = [-1.0, 1.0]
        diff = sig.convolve(self.s, w, 'valid')
        diffdiff = sig.convolve(diff, w, 'valid')

        tol = 1.0e-3
        n_bottom_out = 2 + len(diffdiff[diffdiff > tol])

        self.M = min(min(n_above_tol, n_bottom_out), self.L)

        if self.output:
            print "Model order, M = ", self.M
            plt.figure(figsize=(8, 6))
            plt.scatter(self.M-1, self.s[self.M-1], s=30, c='r')
            plt.semilogy(np.arange(len(self.s)), self.s)
            plt.title('Singular values', fontsize=16)

            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(len(diffdiff)), diffdiff)
            plt.title('Approx. 2nd Derivative of Singular Values', fontsize=16)

        return

    def AggregateDamping(self):
        """
        Kreisselmeier-Steinhauser (KS) function to approximate maximum real
        part of exponent

        Returns
        -------
        float
            approximate maximum of real part of exponents

        """
        m = self.damp.max()
        
        return m + np.log(np.sum(np.exp(self.rho*(self.damp - m))))/self.rho

    def AggregateDampingDer(self):
        """
        Use the data saved prior to computing the aggregate damping to compute
        the derivative of the damping with respect to the input data

        """
        dcda = DcDalpha(self.damp, self.rho)
        dcdl = DalphaDlamTrans(dcda, self.lam, self.dt)
        dcdA = DlamDATrans(dcdl, self.W, self.V)
        dcdV1T = dAdV1Trans(dcdA, self.V1T, self.V1inv, self.V2T)
        dcdV2T = dAdV2Trans(dcdA, self.V1inv)
        dcdVhat = dV12dVhatTrans(dcdV1T, dcdV2T)
        dcdY = dVhatdYTrans(dcdVhat, self.U, self.s, self.VT)
        dcdX = dYdXTrans(dcdY)

        return dcdX

    def ComputeAmplitudeAndPhase(self):
        """
        Compute the amplitudes and phases of the complex exponentials

        """
        # Compute the residues
        B = np.zeros((self.N, self.M), dtype=self.lam.dtype)
        for i in range(self.N):
            for j in range(self.M):
                B[i,j] = np.abs(self.lam[j])**i*np.exp(1.0j*i*np.angle(self.lam[j]))

        r, _, _, _ = la.lstsq(B, self.X)

        # Extract amplitudes and phases from residues
        self.amps = np.abs(r)
        self.faze = np.angle(r)

        return

    def ReconstructSignal(self, t):
        """
        Having computed the amplitudes, damping, frequencies, and phases, can
        produce signal based on sum of series of complex exponentials

        Parameters
        ----------
        t : numpy.ndarray
            array of times

        Returns
        -------
        X : numpy.ndarray
            reconstructed signal
        
        """
        X = np.zeros(t.shape)
        for i in range(self.M):
            a = self.amps[i]
            x = self.damp[i]
            w = self.freq[i]
            p = self.faze[i]

            X += a*np.exp(x*t)*np.cos(w*t + p)

        return X
