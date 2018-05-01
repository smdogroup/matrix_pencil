"""
Test matrix pencil technique on simple spring-mass-damper system
"""
import numpy as np
from scipy import integrate
from matrix_pencil import *

# Spring-mass-damper system
m = 100.0  # kg
c = -200.0  # N*s/m
k = 7000.0  # N/m

wn = np.sqrt(k/m)
print "exact frequency: ", np.sqrt(k/m)
print "exact damping:   ", -0.5*c/m

A = np.array([[0, 1], [-k / m, -c / m]])

def ydot(t, y):
    return A.dot(y)

# Integrate to get obtain response
t0 = 0.0
t_bound = 3.0
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
output_level = 3
pencil = MatrixPencil(t, y[0,:], N, output_level)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print "damping for largest mode = ", pencil.damp[np.argmax(pencil.amps)]
print "ks = ", pencil.AggregateDamping()

# Plot response
t_recon = np.linspace(t[0], t[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(t, y[0,:], 'orange', label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()
plt.show()
