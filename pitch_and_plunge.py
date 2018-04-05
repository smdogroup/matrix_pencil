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

# Decompose plunge signal using matrix pencil method
after_forcing_period = ts > 20.0
ts = ts[after_forcing_period]
hs = hs[after_forcing_period]
alphas = alphas[after_forcing_period]
n = len(ts)
print "n = ", n
N = 200
print "N = ", N
T = np.linspace(0.0, ts[-1], N)
X = np.interp(T, ts, hs)
DT = T[1] - T[0]

# Plot plunge response
plt.figure(figsize=(8, 6))
plt.plot(T, X)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$h$', fontsize=16)
plt.title(r'Plot of plunge response of system')
plt.show()

R, S = pencil(N, X, DT)
print S
print R

print S[np.argmax(np.abs(R.real))]
