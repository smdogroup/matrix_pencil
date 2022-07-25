import numpy as np
import scipy.linalg as la


class MatrixPencil:
    def __init__(
        self, num_subsamples=-1, output_level=0, svd_tolerance=0.01, ks_rho=100.0
    ):
        """
        Provide the capability to decompose input signal into a series of
        complex exponentials by the matrix pencil method

        Parameters
        ----------
        num_subsamples:
            Number of point reduce sample to. The original signal is linearly interpolated to
            ``num_subsamples`` evenly space over the original time range. If ``num_subsamples=-1``,
            the full signal is used. Subsampling reduces the cost of the SVD.

        output_level:
            | choose different output levels
            | 0 -> no printing
            | 1 -> print summary of decomposition to screen
            | 2 -> write file for singular values
            | 3 -> write file for reconstructed signal
            | 4 -> write file for signal components

        svd_tolerance:
            cutoff tolerance for SVD (noise) filtering. SVs are kept if SV[i]/max(SV) > svd_tolerance
            if svd_tolerance < 0, the Gavish optimal threshold is used

        ks_rho:
            aggregate damping KS parameter
        """
        self.output_level = output_level

        # subsampled signal
        self.num_subsamples = num_subsamples
        self.X = np.zeros(0)
        self.T = np.zeros(0)
        self.t0 = 0.0

        # model order
        self.model_order: int = -1
        self.svd_tolerance = svd_tolerance

        # SVD of Hankel matrix
        self.normalized_singular_values = np.zeros(0)

        # Decomposed signal amplitudes, exponents, frequencies, and phases
        self.amplitudes = np.zeros(0)
        self.alpha = np.zeros(0)
        self.frequencies = np.zeros(0)
        self.phases = np.zeros(0)
        self.damping_ratios = np.zeros(0)

        # KS information
        self.ks_rho = ks_rho

    def compute(self, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the signal decomposition

        Parameters
        ----------
        t:
            time array (must be evenly spaced)
        x:
            array containing sampled waveform

        Returns
        -------
        damping:
            vector of damping ratio values
        frequency:
            vector of frequency values [rad/s]
        """

        self.t0 = t[0]
        self.t = t
        self.x = x

        # check that the step size is uniform
        if self.num_subsamples == -1:
            self.num_subsamples = x.shape[0]
            self.H = np.eye(self.num_subsamples)
            T = t[:]
            X = x[:]
            self.dt = T[1] - T[0]
            flag = 0
            tol = 1e-6
            for i in range(T.size - 1):
                if abs(self.dt - (T[i + 1] - T[i])) > tol:
                    flag = 1
            if flag > 0:
                print("Warning: Matrix pencil time step size is not uniform")
                print("         Please provide data with a uniform step size")
        else:

            T = np.linspace(t[0], t[-1], self.num_subsamples)
            X = self._linear_interp(T, t, x)
            self.dt = T[1] - T[0]

        self.T = T
        self.X = X

        # Set the pencil parameter L
        self.L = int(self.num_subsamples / 2) - 1

        if self.output_level > 0:
            print("Matrix Pencil: Number of samples, N = ", self.num_subsamples)
            print("Matrix Pencil: Pencil parameter, L = ", self.L)

        # Assemble the Hankel matrix Y from the samples x
        y = np.empty((self.num_subsamples - self.L, self.L + 1), dtype=self.X.dtype)
        for i in range(self.num_subsamples - self.L):
            for j in range(self.L + 1):
                y[i, j] = self.X[i + j]

        # Compute the SVD of the Hankel matrix
        self.U, self.sigma, self.VT = la.svd(y)

        # Estimate the modal order M based on the singular values
        self._estimate_model_order(self.sigma)

        # Filter the right singular vectors of the Hankel matrix based on M
        Vhat = self.VT[: self.model_order, :]
        self.V1T = Vhat[:, :-1]
        self.V2T = Vhat[:, 1:]

        # Compute the pseudoinverse of V1T
        self.V1inv = la.pinv(self.V1T)

        # Form A matrix from V1T and V2T to reduce to eigenvalue problem
        A = self.V2T.dot(self.V1inv)

        # Solve eigenvalue problem to obtain poles
        self.lam, self.W, self.V = la.eig(A, left=True, right=True)

        # Compute damping and freqency
        s = np.log(self.lam[: self.model_order]) / self.dt
        self.alpha = s.real
        self.frequencies = s.imag

        # force the damping of zero frequency mode to nonsense
        self.damping_ratios = np.zeros(self.frequencies.size)
        for i in range(self.alpha.size):
            if abs(self.frequencies[i]) < 1e-7:
                self.alpha[i] = -99999.0
                self.damping_ratios[i] = 0.0
            else:
                self.damping_ratios[i] = -self.alpha[i] / abs(self.frequencies[i])

        self._compute_amplitude_and_phase(self.X, self.lam)

        if self.output_level >= 1:
            print("Matrix Pencil Summary:")
            print(
                "  comp |       amplitude       |    damping ratio      |          freq         |       phase"
            )
            for i in range(self.model_order):
                print(
                    "--> %2i | % 10.14e | % 10.14e | % 10.14e | % 10.14e"
                    % (
                        i,
                        self.amplitudes[i],
                        self.damping_ratios[i],
                        self.frequencies[i],
                        self.phases[i],
                    )
                )

        if self.output_level >= 3:
            print("Matrix Pencil: reconstructing signal")
            self.x_reconstruct = self.reconstruct_signal(t)
            matrix = np.zeros((self.x_reconstruct.size, 2))
            matrix[:, 0] = t
            matrix[:, 1] = self.x_reconstruct
            np.savetxt("matrix_pencil_reconstruction.dat", matrix)
            print(
                "Matrix Pencil: reconstruction written to 'matrix_pencil_reconstruction.dat'"
            )

        if self.output_level >= 4:
            print("Matrix Pencil: constructing signal components")
            self.x_comps = self.construct_components(t)
            matrix = np.zeros((t.size, self.model_order + 1))
            matrix[:, 0] = t
            matrix[:, 1:] = self.x_comps
            np.savetxt("matrix_pencil_components.dat", matrix)
            print("Matrix Pencil: components written to 'matrix_pencil_components.dat'")

        return self.damping_ratios, self.frequencies

    def compute_aggregate_damping(self) -> float:
        """
        Kreisselmeier-Steinhauser (KS) function to approximate maximum real
        part of exponent

        Returns
        -------
        ks_value
            approximate maximum of real part of exponents

        """
        alpha = self.alpha[self.frequencies > 1e-7]

        m = alpha.max()
        c = m + np.log(np.sum(np.exp(self.ks_rho * (alpha - m)))) / self.ks_rho

        if self._is_complex():
            dcdx = self.compute_aggregate_damping_derivative()
            return -c - 1j * dcdx.dot(self.x.imag)
        else:
            return c

    def compute_aggregate_damping_derivative(self):
        """
        Use the data saved prior to computing the aggregate damping to compute
        the derivative of the damping with respect to the input data

        """
        dcda = self._compute_daggregate_dalpha(self.alpha, self.ks_rho)
        dc_dlam = self._apply_dalpha_dlambda(dcda, self.lam, self.dt)
        dcdA = self._apply_dlambda_dA(dc_dlam, self.W, self.V)
        dcdV1T = self._apply_dA_dV1T(dcdA, self.V1T, self.V1inv, self.V2T)
        dcdV2T = self._apply_dA_dV2T(dcdA, self.V1inv)
        dcdVhat = self._apply_dV1_and_dV2_dVhat(dcdV1T, dcdV2T)
        dcdY = self._apply_dVhat_dY(dcdVhat, self.U, self.sigma, self.VT)
        dcdX = self._apply_dY_dX(dcdY)
        dcdx = self.H.T.dot(dcdX)
        return -dcdx

    def reconstruct_signal(self, time: np.ndarray) -> np.ndarray:
        """
        Having computed the amplitudes, damping, frequencies, and phases, can
        produce signal based on sum of series of complex exponentials

        Parameters
        ----------
        time:
            time vector

        Returns
        -------
        X:
            reconstructed signal

        """
        reconstructed_signal = np.zeros(time.size)
        for i in range(self.model_order):
            a = self.amplitudes[i]
            x = self.alpha[i]
            w = self.frequencies[i]
            p = self.phases[i]

            if abs(w) < 1e-7:
                x = 0.0

            reconstructed_signal += (
                a * np.exp(x * (time - self.t0)) * np.cos(w * (time - self.t0) + p)
            )

        return reconstructed_signal

    def construct_components(self, time: np.ndarray) -> np.ndarray:
        """
        Having computed the amplitudes, damping, frequencies, and phases, time signals of
        the individual components of the decomposition can be produced

        Parameters
        ----------
        t:
            time vector

        Returns
        -------
        X:
            signal components. First index is the time instance. Second index is the component number

        """
        X = np.zeros((time.size, self.model_order))
        for i in range(self.model_order):
            a = self.amplitudes[i]
            x = self.alpha[i]
            w = self.frequencies[i]
            p = self.phases[i]

            if abs(w) < 1e-7:
                x = 0.0

            X[:, i] = (
                a * np.exp(x * (time - self.t0)) * np.cos(w * (time - self.t0) + p)
            )

        return X

    def _estimate_model_order(self, sigma: np.ndarray):
        """
        Store estimate of model order based on input singular values

        Notes
        -----
        This is a pretty crude method for estimating the model order.
        Development of a more sophisticated method would probably yield greater
        robustness

        """
        # Normalize the singular values by the maximum and cut out modes
        # corresponding to singular values below a specified tolerance
        normalized_sigma = sigma / sigma.max()
        if self.svd_tolerance < 0.0:
            self.svd_tolerance = self._compute_gavish_optimal_cutoff(normalized_sigma)
        n_above_tol = len(sigma[normalized_sigma > self.svd_tolerance])

        # Estimate the number of modes (model order) to have at least two but
        # otherwise informed by the cuts made above
        self.model_order = min(max(2, n_above_tol), self.L)

        # Report the model order
        if self.output_level >= 1:
            print("Matrix Pencil: Model order, M = ", self.model_order)
        if self.output_level >= 2:
            print(
                "Matrix Pencil: writing normalized singular values to 'matrix_pencil_singular_values.dat'"
            )
            np.savetxt("matrix_pencil_singular_values.dat", normalized_sigma)
        self.normalized_singular_values = normalized_sigma

    def _compute_gavish_optimal_cutoff(self, normalized_sigma: np.ndarray) -> float:
        n = normalized_sigma.size
        sigma = 2.858 * np.median(normalized_sigma)
        return 4.0 / np.sqrt(3) * np.sqrt(n) * sigma

    def _compute_amplitude_and_phase(self, X: np.ndarray, lam: np.ndarray):
        """
        Compute the amplitudes and phases of the complex exponentials

        """
        # Compute the residues
        B = np.zeros((self.num_subsamples, self.model_order), dtype=lam.dtype)
        for i in range(self.num_subsamples):
            for j in range(self.model_order):
                B[i, j] = np.abs(lam[j]) ** i * np.exp(1.0j * i * np.angle(lam[j]))

        r, _, _, _ = la.lstsq(B, X)

        # Extract amplitudes and phases from residues
        self.amplitudes = np.abs(r)
        self.phases = np.angle(r)

    def _linear_interp(self, T_interpolated, t_orig, x_orig):
        """
        Generate matrix that linearly interpolates the data (t, x) and returns X
        corresponding to T

        """
        n_interpolated = T_interpolated.shape[0]
        n_orig = t_orig.shape[0]

        self.H = np.zeros((n_interpolated, n_orig))

        if self.output_level >= 1:
            print(
                f"Matrix Pencil: Downsampling from {n_orig} to {n_interpolated} points..."
            )

        for i in range(n_interpolated):
            # Evaluate first and last basis functions
            j0 = np.sum(T_interpolated[i] > t_orig[:-1]) - 1
            j1 = n_orig - np.sum(T_interpolated[i] <= t_orig[1:])

            dt = t_orig[j1] - t_orig[j0]

            self.H[i, j0] += (t_orig[j1] - T_interpolated[i]) / dt
            self.H[i, j1] += (T_interpolated[i] - t_orig[j0]) / dt

        return self.H.dot(x_orig)

    def c_ks(self, alphas, rho):
        """
        Kreisselmeier-Steinhauser (KS) function to approximate maximum real
        part of exponent

        Returns
        -------
        float
            approximate maximum of real part of exponents

        """
        m = alphas.min()

        return -m + np.log(np.sum(np.exp(-rho * (alphas - m)))) / rho

    def _compute_daggregate_dalpha(self, alphas, rho):
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
        m = -alphas.min()
        a = np.sum(np.exp(rho * (-alphas - m)))

        return -np.exp(rho * (-alphas - m)) / a

    def DalphaDlam(self, lam, dt):
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
        real_part = (1.0 / dt) * lam.real / np.real(np.conj(lam) * lam)
        imag_part = (1.0 / dt) * lam.imag / np.real(np.conj(lam) * lam)

        return real_part + 1j * imag_part

    def DlamDA(self, A):
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
            w = WH[i, :]
            v = V[:, i]
            norm = w.dot(v)
            dlam[i, :, :] = np.outer(w, v) / norm

        return dlam

    def SVDDerivative(self, U, s, VT):
        """
        Derivatives of SVD of full-rank rectangular matrix of size m x n

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
        ns = len(s)

        # Allocate output arrays
        dU = np.zeros((m, m, m, n))
        ds = np.zeros((ns, m, n))
        dVT = np.zeros((n, n, m, n))

        # Square matrix of singular values
        S1 = np.diag(s)
        S1inv = np.diag(1.0 / s)

        # Form skew-symmetric F matrix
        F = np.zeros((ns, ns))
        for i in range(ns):
            for j in range(i + 1, ns):
                F[i, j] = 1.0 / (s[j] ** 2 - s[i] ** 2)
                F[j, i] = 1.0 / (s[i] ** 2 - s[j] ** 2)

        for k in range(m):
            for l in range(n):
                dP = np.outer(U[k, :], VT[:, l])

                # Extract diagonal for ds
                ds[:, k, l] = np.diag(dP)

                # Compute dC and dD matrices for various cases
                if m > n:
                    dP1 = dP[:n, :]
                    dP2 = dP[n:, :]

                    dC1 = F * (dP1.dot(S1) + S1.dot(dP1.T))
                    dDT = -F * (S1.dot(dP1) + dP1.T.dot(S1))

                    dC2T = dP2.dot(S1inv)

                    dC = np.zeros((m, m))
                    dC[:n, :n] = dC1
                    dC[:n, n:] = -dC2T.T
                    dC[n:, :n] = dC2T

                else:
                    dP1 = dP[:, :m]
                    dP2 = dP[:, m:]

                    dC = F * (dP1.dot(S1) + S1.dot(dP1.T))
                    dD1 = F * (S1.dot(dP1) + dP1.T.dot(S1))

                    dD2 = S1inv.dot(dP2)

                    if m == n:
                        dDT = -dD1
                    else:
                        dDT = np.zeros((n, n))
                        dDT[:m, :m] = -dD1
                        dDT[:m, m:] = dD2
                        dDT[m:, :m] = -dD2.T

                # Compute dU and dVT sensitivities from dC and dD
                dU[:, :, k, l] = U.dot(dC)
                dVT[:, :, k, l] = dDT.dot(VT)

        return dU, ds, dVT

    def PseudoinverseDerivative(self, A, Ainv):
        """
        Derivatives of pseudoinverse with respect to its generating matrix

        Parameters
        ----------
        A : numpy.ndarray
            input matrix
        Ainv : numpy.ndarray
            Pseudoinverse of A matrix

        Returns
        -------
        dAinv : numpy.ndarray
            derivatives dAinv[i,j]/dA[k,l]

        """
        m = A.shape[0]
        n = A.shape[1]

        # Allocate array for output
        dAinv = np.zeros((n, m, m, n))

        for k in range(m):
            for l in range(n):
                ek = np.zeros(m)
                ek[k] += 1.0
                el = np.zeros(n)
                el[l] += 1.0

                dA = np.outer(ek, el)

                dAinv[:, :, k, l] = (
                    -Ainv.dot(dA).dot(Ainv)
                    + Ainv.dot(Ainv.T).dot(dA.T).dot(np.eye(m) - A.dot(Ainv))
                    + (np.eye(n) - Ainv.dot(A)).dot(dA.T).dot(Ainv.T).dot(Ainv)
                )

        return dAinv

    def _apply_dalpha_dlambda(self, dcda, lam, dt):
        """
        Apply action of [d(alpha)/d(lam)]^{T} to the vector of derivatives
        [d(c)/d(alpha)]^{T} to obtain the derivatives d(c)/d(lam)

        Parameters
        ----------
        dcda : numpy.ndarray
            vector of derivatives d(c)/d(alpha)
        lam : numpy.ndarray
            eigenvalues of A matrix
        dt : float
            time step

        Returns
        -------
        numpy.ndarray
            vector of derivatives d(c)/d(lam)

        """
        M = dcda.shape[0]
        L = lam.shape[0]

        # Pad the dcda derivative with zeros
        if M < L:
            dcda = np.hstack((dcda, np.zeros(L - M)))

        dadl_real = (1.0 / dt) * lam.real / np.real(np.conj(lam) * lam)
        dadl_imag = (1.0 / dt) * lam.imag / np.real(np.conj(lam) * lam)

        return dcda * dadl_real + 1j * dcda * dadl_imag

    def _apply_dlambda_dA(self, dcdl, W, V):
        """
        Apply action of [d(lam)/d(A)]^{T} to the vector of derivatives
        [d(c)/d(lam)]^{T} to obtain the derivatives d(c)/d(A)

        Parameters
        ----------
        dcdl : numpy.ndarray
            vector of derivatives d(c)/d(lam)
        W : numpy.ndarray
            left eigenvectors of matrix A
        V : numpy.ndarray
            right eigenvectors of matrix A

        Returns
        -------
        dcdA : numpy.ndarray
            vector of derivatives d(c)/d(A)

        """
        WH = W.conj().T
        m = len(dcdl)
        dcdA = np.zeros((m, m))
        for i in range(m):
            w = WH[i, :]
            v = V[:, i]
            norm = w.dot(v)
            dldA = np.outer(w, v) / norm
            dcdA += dcdl[i].real * dldA.real + dcdl[i].imag * dldA.imag

        return dcdA

    def _apply_dA_dV1T(self, dcdA, V1T, V1inv, V2T):
        """
        Apply action of [d(A)/d(V1^{T})]^{T} to the array of derivatives
        [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(V1^{T})

        Parameters
        ----------
        dcdA : numpy.ndarray
            array of derivatives d(c)/d(A)
        V1T : numpy.ndarray
            filtered right singular vectors of Y1 matrix
        Vinv : numpy.ndarray
            pseudoinverse of V1T matrix
        V2T : numpy.ndarray
            filtered right singular vectors of Y2 matrix

        Returns
        -------
        dcdV1T : numpy.ndarray
            vector of derivatives d(c)/d(V1^{T})

        """
        L = V1inv.shape[0]
        M = V1inv.shape[1]

        # Compute pseudoinverse derivative
        dV1inv = self.PseudoinverseDerivative(V1T, V1inv)

        # Compute dcdV1inv derivative
        dcdV1inv = V2T.T @ dcdA

        dcdV1T = np.zeros((M, L))
        for i in range(M):
            for j in range(L):
                dcdV1T[i, j] = np.sum(dcdV1inv * dV1inv[:, :, i, j])

        return dcdV1T

    def _apply_dA_dV2T(self, dcdA, V1inv):
        """
        Apply action of [d(A)/d(V2^{T})]^{T} to the array of derivatives
        [d(c)/d(A)]^{T} to obtain the derivatives d(c)/d(V2^{T})

        Parameters
        ----------
        dcdA : numpy.ndarray
            array of derivatives d(c)/d(A)
        V1inv : numpy.ndarray
            generalized inverse of the tranpose of the V1hat matrix

        Returns
        -------
        dcdV2T : numpy.ndarray
            vector of derivatives d(c)/d(V2^{T})

        """
        m = V1inv.shape[1]
        n = V1inv.shape[0]

        dcdV2T = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                dcdV2T[i, j] = np.sum(dcdA[i, :] * V1inv[j, :])

        return dcdV2T

    def _apply_dV1_and_dV2_dVhat(self, dcdV1T, dcdV2T):
        """
        Pad the d(c)/d(V1^{T}) derivatives  with zeros, pad the d(c)/d(V2^{T})
        derivatives with zeros, and combine to obtain the derivatives
        d(c)/d(Vhat^{T})

        Parameters
        ----------
        dcdV1T : numpy.ndarray
            vector of derivatives d(c)/d(V1^{T})
        dcdV2T : numpy.ndarray
            vector of derivatives d(c)/d(V2^{T})

        Returns
        -------
        dcdVhat : numpy.ndarray
            vector of derivatives d(c)/d(Vhat^{T})

        """
        M = dcdV1T.shape[0]
        dcdVhat = np.hstack((dcdV1T, np.zeros(M).reshape((M, 1)))) + np.hstack(
            (np.zeros(M).reshape((M, 1)), dcdV2T)
        )

        return dcdVhat

    def _apply_dVhat_dY(self, dcdVhat, U, s, VT):
        """
        Apply action of [d(Vhat^{T})/d(Y)]^{T} to the array of derivatives
        [d(c)/d(Vhat^{T})]^{T} to obtain the derivatives d(c)/d(Y)

        Parameters
        ----------
        dcdVhat : numpy.ndarray
            array of derivatives d(c)/d(Vhat)
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

        Returns
        -------
        dcdY : numpy.ndarray
            array of derivatives d(c)/d(Y)

        """
        m = U.shape[0]
        n = VT.shape[1]
        M = dcdVhat.shape[0]

        dVhat = np.zeros((M, n, m, n))

        # If Y is full-rank, then use the standard SVD derivative
        if all(s[M:] > 1e-6):
            if self.output_level > 0:
                print("Computing exact SVD derivative...")
            _, _, dVT = self.SVDDerivative(U, s, VT)
            dVhat = dVT[:M, :, :, :]

        # If Y is numerically low-rank, then approximate the derivative
        else:
            if self.output_level > 0:
                print("Computing approximate SVD derivative...")
            ns = max(M, len(s[s > 1e-6]))

            # Square matrix of singular values
            S1 = np.diag(s[:ns])
            S1inv = np.diag(1.0 / s[:ns])

            # Form skew-symmetric F matrix
            F = np.zeros((ns, ns))
            for i in range(ns):
                for j in range(i + 1, ns):
                    F[i, j] = 1.0 / (s[j] ** 2 - s[i] ** 2)
                    F[j, i] = 1.0 / (s[i] ** 2 - s[j] ** 2)

            for k in range(m):
                for l in range(n):
                    dP = np.outer(U[k, :], VT[:, l])

                    dP1 = dP[:ns, :ns]
                    dD1 = F * (S1.dot(dP1) + dP1.T.dot(S1))

                    dP2 = dP[:ns, ns:]
                    dD2 = S1inv.dot(dP2)

                    dDT = np.zeros((ns, n))
                    dDT[:, :ns] = -dD1
                    dDT[:, ns:] = dD2

                    dVhat[:, :, k, l] = dDT[:M, :].dot(VT)

        # Apply chain rule to get
        dcdY = np.empty((m, n))

        for i in range(m):
            for j in range(n):
                dcdY[i, j] = np.sum(dcdVhat * dVhat[:, :, i, j])

        return dcdY

    def _apply_dY_dX(self, dcdY):
        """
        Apply action of [d(Y)/d(X)]^{T} to the array of derivatives [d(c)/d(Y)]^{T}
        to obtain the derivatives d(c)/d(X)

        Parameters
        ----------
        dcdY : numpy.ndarray
            array of derivatives d(c)/d(Y)

        Returns
        -------
        dcdX : numpy.ndarray
            array of derivatives d(c)/d(X)

        """
        L = dcdY.shape[1] - 1
        N = dcdY.shape[0] + L

        dcdX = np.zeros(N)

        # Sum the anti-diagonals into dcdX
        for i in range(N - L):
            for j in range(L + 1):
                dcdX[i + j] += dcdY[i, j]

        return dcdX

    def _is_complex(self):
        return self.x.dtype == np.complex128 or self.x.dtype == complex
