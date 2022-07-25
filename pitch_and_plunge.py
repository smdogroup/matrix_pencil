"""
Test matrix pencil technique on signals coming from 2D aeroelastic model
"""
import numpy as np
from matrix_pencil import MatrixPencil
import matplotlib.pylab as plt

# Extract time histories
ts = []
hs = []
alphas = []

f = open("time_hist.dat", "r")
for i, line in enumerate(f):
    data = line.split()
    ts.append(float(data[0]))
    hs.append(float(data[1]))
    alphas.append(float(data[2]))
f.close()

ts = np.array(ts)
hs = np.array(hs)
alphas = np.array(alphas)

plt.figure(figsize=(8, 6))
plt.plot(ts, hs, label="plunge")
plt.plot(ts, alphas, label="pitch")
plt.xlabel(r"$t$", fontsize=16)
plt.legend()

# Decompose plunge signal using matrix pencil method
print("For plunge:\n")
offset = 0.5
after_forcing_period = ts > offset
ts = ts[after_forcing_period]
hs = hs[after_forcing_period]
N = 252

output_level = 5
pencil = MatrixPencil(output_level=output_level)
damp, freq = pencil.compute(ts, hs)

print("ks = ", pencil.compute_aggregate_damping())
print("")

# Plot plunge response
t_recon = np.linspace(ts[0], ts[-1], 1000)
x_recon = pencil.reconstruct_signal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(ts, hs, label="original")
plt.plot(t_recon, x_recon, "--", label="reconstructed")
plt.xlabel(r"$t$", fontsize=16)
plt.ylabel("plunge", fontsize=16)
plt.legend()

# Decompose pitch signal using matrix pencil method
print("For pitch:\n")
pert_ind = 43
alphas = alphas[after_forcing_period]

pencil = MatrixPencil(output_level=output_level)
damp, freq = pencil.compute(ts, alphas)

print("ks = ", pencil.compute_aggregate_damping())

# Plot pitch response
t_recon = np.linspace(ts[0], ts[-1], 1000)
x_recon = pencil.reconstruct_signal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(ts, alphas, label="original")
plt.plot(t_recon, x_recon, "--", label="reconstructed")
plt.xlabel(r"$t$", fontsize=16)
plt.ylabel("pitch", fontsize=16)
plt.legend()

plt.show()
