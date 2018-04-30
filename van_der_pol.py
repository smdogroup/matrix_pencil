"""
Test matrix pencil technique on Van der Pol oscillator
"""
import numpy as np
import scipy.integrate
from matrix_pencil import *

# Define damping parameter here as global variable
mu = 3.0

def van_der_pol(t, y):
    return np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])

# Instantiate integrator
t0 = 0.0
y0 = np.array([4.0, 0.0])
t_bound = 40.0
max_step = 0.1
integrator = scipy.integrate.BDF(van_der_pol, t0, y0, t_bound, max_step)

# Integrate to obtain time signal
t = np.array([t0])
y = np.array([y0]).reshape((2,1))
while integrator.t < t_bound:
    integrator.step()
    t_step = integrator.t
    t = np.hstack((t, t_step))
    y = np.hstack((y, integrator.y.reshape((2, 1))))

# Decompose signal using matrix pencil method
n = len(t)
print "n = ", n
N = 200
print "N = ", N
T = np.linspace(t[0], t[-1], N)
X = np.interp(T, t, y[0,:])
DT = T[1] - T[0]

pencil = MatrixPencil(X, DT, True)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print pencil.damp[np.argmax(pencil.amps)]
print pencil.AggregateDamping()

# Plot response
t_recon = np.linspace(t[0], t[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
smd = plt.figure(figsize=(8, 6))
plt.plot(T, X, label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()
smd.savefig('van_der_pol.png')
plt.show()
