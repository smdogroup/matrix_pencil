"""
Test matrix pencil technique on signals coming from 2D aeroelastic model
"""
import numpy as np
from matrix_pencil import *

# Extract time histories
ts = []
hs = []
alphas = []

f = open('time_hist.dat', 'r')
for i, line in enumerate(f):
    data = line.split()
    ts.append(float(data[0]))
    hs.append(float(data[1]))
    alphas.append(float(data[2]))
f.close()

ts = np.array(ts)
hs = np.array(hs)
alphas = np.array(alphas)

plt.figure()
plt.plot(ts, alphas, ts, hs)

# Decompose plunge signal using matrix pencil method
print "For plunge:"
after_forcing_period = ts > 0.0
ts = ts[after_forcing_period]
hs = hs[after_forcing_period]
n = len(ts)
print "n = ", n
N = 150
print "N = ", N
T = np.linspace(ts[0], ts[-1], N)
X = np.interp(T, ts, hs)
DT = T[1] - T[0]

pencil = MatrixPencil(X, DT, True)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print pencil.damp[np.argmax(pencil.amps)]
print pencil.AggregateDamping()

# Plot plunge response
t_recon = np.linspace(ts[0], ts[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(T, X, label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()

# Decompose pitch signal using matrix pencil method
print "For pitch:"
alphas = alphas[after_forcing_period]
n = len(ts)
print "n = ", n
N = 150
print "N = ", N
T = np.linspace(ts[0], ts[-1], N)
X = np.interp(T, ts, alphas)
DT = T[1] - T[0]

pencil = MatrixPencil(X, DT, True)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print pencil.damp[np.argmax(pencil.amps)]
print pencil.AggregateDamping()

# Plot plunge response
t_recon = np.linspace(ts[0], ts[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(T, X, label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()

plt.show()
