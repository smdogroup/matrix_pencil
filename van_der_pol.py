"""
Test matrix pencil technique on Van der Pol oscillator
"""
from __future__ import print_function
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
t_bound = 80.0
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
N = 300
output_level = 1
pencil = MatrixPencil(t, y[0,:], N, output_level)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print("damping for largest mode = ", -pencil.damp[np.argmax(pencil.amps)])
print("ks = ", pencil.AggregateDamping())

# Plot response
t_recon = np.linspace(t[0], t[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
smd = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Set the fontsize
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot the full reconstruction
plt.plot(t, y[0,:], linewidth=2, label='data')
plt.plot(t_recon, x_recon, linewidth=2, label='full recon.')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)

# Plot the reconstruction from 
offset = 0
while y[0,offset] > 0.0:
    offset += 1

pencil = MatrixPencil(t[offset:], y[0,offset:], N, output_level)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print("damping for largest mode = ", pencil.damp[np.argmax(pencil.amps)])
print("ks = ", pencil.AggregateDamping())

# Plot response
t_recon = np.linspace(t[offset], t[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
# smd = plt.figure(figsize=(8, 6))
# plt.plot(t[offset:], y[0,offset:], label='data')
plt.plot(t_recon, x_recon, linewidth=2, label='partial recon.')
plt.legend()

smd = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Set the fontsize
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot the singular values
s = sorted(pencil.s)[::-1]
plt.semilogy(range(1, 1+len(pencil.s)), s, 
             marker='o', linewidth=0, label='singular value')
plt.semilogy([pencil.M+1], s[pencil.M], marker='o', c='r', linewidth=0)
plt.xlabel(r'index', fontsize=16)
plt.ylabel(r'singular value', fontsize=16)



# Plot the damping ratio
smd = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Set the fontsize
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

rho = 50
m = pencil.damp.min()
c0 = m - np.log(np.sum(np.exp(-rho*(pencil.damp - m))))/rho

# Plot the singular values
d = sorted(pencil.damp)[::-1]
plt.semilogy(range(1, 1+len(pencil.damp)), d-c0, 
             marker='o', linewidth=0, label='singular value')
plt.xlabel(r'index', fontsize=16)
plt.ylabel(r'singular value', fontsize=16)

for rho in [80, 100]:
    m = pencil.damp.min()
    c = m - np.log(np.sum(np.exp(-rho*(pencil.damp - m))))/rho

    plt.semilogy([1, 1+len(pencil.damp)], [c-c0, c-c0], linewidth=2, 
                 label=r'$\rho = %d$'%(rho))

plt.legend()
plt.show()



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
print("RMS of fit: ", 
    np.sqrt(np.mean(np.square(pencil.ReconstructSignal(t) - y[0,:]))))
