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
n = len(t)
print "n = ", n
N = 400
print "N = ", N
T = np.linspace(0.0, t[-1], N)
X = np.interp(T, t, y[0,:])
DT = T[1] - T[0]

# Plot response
plt.figure(figsize=(8, 6))
plt.plot(T, X)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.title(r'Plot of displacement of single DOF system')
plt.show()

R, S = pencil(N, X, DT)
print S
print R

print S[np.argmax(R.real)]
