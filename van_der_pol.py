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
N = 200
output_level = 1
pencil = MatrixPencil(t, y[0,:], N, output_level)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print "damping for largest mode = ", pencil.damp[np.argmax(pencil.amps)]
print "ks = ", pencil.AggregateDamping()

# Plot response
t_recon = np.linspace(t[0], t[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
smd = plt.figure(figsize=(8, 6))
plt.plot(t, y[0,:], label='data')
plt.plot(t_recon, x_recon, 'b--', label='reconstruction')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()
#plt.show()

# Output original and fit for plotting purposes
tx = np.empty((len(t), 2))
tx[:,0] = t
tx[:,1] = y[0,:]
np.savetxt('tx.dat', tx)

TX = np.empty((len(t_recon), 2))
TX[:,0] = t_recon
TX[:,1] = x_recon
np.savetxt('txrecon.dat', TX)

# Print RMS error of fit
print "RMS of fit: ", np.sqrt(np.mean(np.square(pencil.ReconstructSignal(t) - y[0,:])))
