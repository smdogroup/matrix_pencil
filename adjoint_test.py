"""
"""
from __future__ import print_function
from mpi4py import MPI
from tacs import TACS, elements, functions
import numpy as np
from matrix_pencil import *


# Define an element in TACS using the pyElement feature
class SpringMassDamper(elements.pyElement):
    def __init__(self, num_nodes, num_disps, wn, zeta):
        super(SpringMassDamper, self).__init__(num_disps, num_nodes)
        self.wn = wn
        self.zeta = zeta

    def getInitConditions(self, v, dv, ddv, xpts):
        '''Define the initial conditions'''
        v[0] = 1.0

        return

    def addResidual(self, time, res, X, v, dv, ddv):
        '''Add the residual of the governing equations'''
        res[0] += ddv[0] + 2.0*self.zeta*self.wn*dv[0] + self.wn**2*v[0]

        return    

    def addJacobian(self, time, J, alpha, beta, gamma, X, v, dv, ddv):
        '''Add the Jacobian of the governing equations'''
        J[0] += alpha*self.wn**2 + beta*2.0*self.zeta*self.wn + gamma

        return

    def addAdjResProduct(self, time, scale, dvSens, psi, X, v, dv, ddv):
        '''
        Add the derivative of the product of the adjoint variables and the
        residuals to the design variable sensitivities
        '''
        dvSens += scale*psi[0]*2.0*self.wn*dv[0]

    def updateStiffness(self, k):
        self.k = k

        return

# Create instance of user-defined element
num_nodes = 1
num_disps = 1
m = 100.0
c = -200.0
k = 7000.0
wn = np.sqrt(k/m)
zeta = 0.5*c/np.sqrt(m*k)
spr = SpringMassDamper(num_nodes, num_disps, wn, zeta)     

# Add user-defined element to TACS
comm = MPI.COMM_WORLD
assembler = TACS.Assembler.create(comm, 1, 1, 1)

conn = np.array([0], dtype=np.intc)
ptr = np.array([0, 1], dtype=np.intc)
assembler.setElementConnectivity(conn, ptr)

assembler.setElements([spr])

assembler.initialize()

# Create instance of integrator
t0 = 0.0
dt = 0.001
num_steps = 3000
tf = num_steps*dt
order = 2
bdf = TACS.BDFIntegrator(assembler, t0, tf, num_steps, order)
bdf.setPrintLevel(0) # turn off printing

# Integrate governing equations and store time and states
bdf.iterate(0)
time, uvec, _, _ =  bdf.getStates(0)
T = np.zeros(num_steps+1)
U = np.zeros(num_steps+1)
T[0] = time
U[0] = uvec.getArray().copy()

for step in range(1,num_steps+1):
    bdf.iterate(step)
    time, uvec, _, _ = bdf.getStates(step)
    T[step] = time
    U[step] = uvec.getArray().copy()

# Approximate damping and get derivatives using matrix pencil 
N = 300
output_levels = 1
pencil = MatrixPencil(T, U, N, output_levels)
pencil.ComputeDampingAndFrequency()
pencil.ComputeAmplitudeAndPhase()
c = pencil.AggregateDamping()
cder = pencil.AggregateDampingDer()

print("exact damping:     ", zeta*wn)
print("estimated damping: ", c)

# Plot signal reconstructed by matrix pencil
t_recon = np.linspace(T[0], T[-1], 500)
x_recon = pencil.ReconstructSignal(t_recon)
plt.figure(figsize=(8, 6))
plt.plot(T, U, 'orange', label='original')
plt.plot(t_recon, x_recon, 'b--', label='reconstructed')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$x$', fontsize=16)
plt.legend()

# Specify the number of design variables and the function to the integrator
# (use Structural Mass as a dummy function)
num_dvs = 1
funclist = [functions.StructuralMass(assembler)]
bdf.setFunctions(funclist, num_dvs)

# Solve the adjoint equations and compute the gradient
dfdu_vec = assembler.createVec()
for step in range(num_steps, -1, -1):
    bdf.initAdjoint(step)
    dfdu = dfdu_vec.getArray()
    dfdu[0] = -cder[step]
    bdf.iterateAdjoint(step, [dfdu_vec])
    bdf.postAdjoint(step)

dcdx = np.array([0.0])
bdf.getGradient(dcdx)

print("exact derivative:     ", wn)
print("estimated derivative: ", dcdx[0])



plt.show()
