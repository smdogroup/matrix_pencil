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
print
after_forcing_period = ts > 0.0
ts = ts[after_forcing_period]
hs = hs[after_forcing_period]
N = 150

pencil = MatrixPencil(ts, hs, N, True)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print "damping for largest mode = ", pencil.damp[np.argmax(pencil.amps)]
print "ks = ", pencil.AggregateDamping()
print

# Plot plunge response
t_recon = np.linspace(ts[0], ts[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(ts, hs, label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()

# Decompose pitch signal using matrix pencil method
print "For pitch:"
print
alphas = alphas[after_forcing_period]

pencil = MatrixPencil(ts, alphas, N, output_level=1)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()

print "damping for largest mode = ", pencil.damp[np.argmax(pencil.amps)]
print "ks = ", pencil.AggregateDamping()

h = 1.0e-8
alphapert = np.random.random(alphas.shape)
alphapos = alphas + h*alphapert
alphaneg = alphas - h*alphapert

pencilpos = MatrixPencil(ts, alphapos, N)
pencilpos.ComputeDampingAndFrequency()
cpos = pencilpos.AggregateDamping()

pencilneg = MatrixPencil(ts, alphaneg, N)
pencilneg.ComputeDampingAndFrequency()
cneg = pencilneg.AggregateDamping()

approx = 0.5*(cpos - cneg)/h

# Compute analytic derivative
cder = pencil.AggregateDampingDer()
analytic = np.sum(cder*alphapert)

# Compute relative error
rel_error = (analytic - approx)/approx
print "Approximation: ", approx
print "Analytic:      ", analytic
print "Rel. error:    ", rel_error

# Plot plunge response
t_recon = np.linspace(ts[0], ts[-1], 1000)
x_recon = pencil.ReconstructSignal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(ts, alphas, label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()

plt.show()
