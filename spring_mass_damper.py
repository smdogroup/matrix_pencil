"""
Test matrix pencil technique on simple spring-mass-damper system
"""
from __future__ import print_function
import numpy as np
from scipy import integrate
from matrix_pencil import *

# Spring-mass-damper system
zeta = -0.125
wn = 16.0

print "exact frequency: ", wn
print "exact damping:   ", zeta*wn

A = np.array([[0.0, 1.0], [-wn**2, -2.0*zeta*wn]])

def ydot(t, y):
    return A.dot(y)

# Integrate to get obtain response
t0 = 0.0
t_bound = 1.5
y0 = np.array([1.0, 0.0])
max_step = 0.001
rtol = 1.0e-8
atol = 1.0e-8
integrator = integrate.RK45(ydot, t0, y0, t_bound, max_step, rtol, atol)
t_step = t0
t = np.array([t_step])
y = np.array([y0]).reshape((2, 1))
while t_step < t_bound:
    integrator.step()
    t_step = integrator.t
    t = np.hstack((t, t_step))
    y = np.hstack((y, integrator.y.reshape((2, 1))))

# Decompose signal using matrix pencil method
N = 250
output_level = 1
rho = 1e6
pencil = MatrixPencil(t, y[0,:], N, output_level, rho)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print("damping of all modes = ", pencil.damp)
print("damping for largest mode = ", pencil.damp[np.argmax(pencil.amps)])
c = pencil.AggregateDamping()
print("ks = ", c)

# Plot response
t_recon = np.linspace(t[0], t[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
fig = plt.figure(figsize=(10, 8))
plt.plot(t, y[0,:], 'orange', lw=3, label='data')
plt.plot(t_recon, x_recon, 'b--', lw=3, label='reconstruction')
plt.xlabel(r'$t$', fontsize=30)
plt.ylabel(r'$x$', fontsize=30)
plt.legend(fontsize=20)
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

