"""
Test matrix pencil technique on Van der Pol oscillator
"""

import numpy as np
import scipy.integrate
from matrix_pencil import MatrixPencil
import matplotlib.pylab as plt

# Define damping parameter here as global variable
mu = 3.0


def van_der_pol(t, y):
    return np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])


# Instantiate integrator
t0 = 0.0
y0 = np.array([4.0, 0.0])
t_bound = 80.0
max_step = 0.1
integrator = scipy.integrate.BDF(van_der_pol, t0, y0, t_bound, max_step)

# Integrate to obtain time signal
t = np.array([t0])
y = np.array([y0]).reshape((2, 1))
while integrator.t < t_bound:
    integrator.step()
    t_step = integrator.t
    t = np.hstack((t, t_step))
    y = np.hstack((y, integrator.y.reshape((2, 1))))

# Decompose signal using matrix pencil method
N = 300
output_level = 1
pencil = MatrixPencil(num_subsamples=N, output_level=output_level)
pencil.compute(t, y[0, :])

# Plot response
t_recon = np.linspace(t[0], t[-1], 1000)
x_recon = pencil.reconstruct_signal(t_recon)
smd = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

# Set the fontsize
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot the full reconstruction
plt.plot(t, y[0, :], linewidth=2, label="data")
plt.plot(t_recon, x_recon, linewidth=2, label="full recon.")
plt.xlabel(r"$t$", fontsize=16)
plt.ylabel(r"$x$", fontsize=16)

# Plot the reconstruction from
offset = 0
while y[0, offset] > 0.0:
    offset += 1

pencil = MatrixPencil(num_subsamples=N, output_level=output_level)

pencil.compute(t[offset:], y[0, offset:])

# Plot response
t_recon = np.linspace(t[offset], t[-1], 1000)
x_recon = pencil.reconstruct_signal(t_recon)

plt.plot(t_recon, x_recon, linewidth=2, label="partial recon.")
plt.legend()

smd = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

# Set the fontsize
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot the singular values
s = sorted(pencil.normalized_singular_values)[::-1]
plt.semilogy(range(1, 1 + len(s)), s, marker="o", linewidth=0, label="singular values")
plt.xlabel(r"index", fontsize=16)
plt.ylabel(r"singular value", fontsize=16)
plt.legend()

# Print RMS error of fit
print(
    "RMS of fit: ",
    np.sqrt(np.mean(np.square(pencil.reconstruct_signal(t[offset:]) - y[0, offset:]))),
)

plt.show()
