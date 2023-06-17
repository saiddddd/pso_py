import numpy as np
import matplotlib.pyplot as plt

# Define the details of the table design problem
nVar = 10
ub = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]) #n(nVar)
lb = np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10]) #n(nVar)

# Define the objective function
def ObjectiveFunction(x):
    # Implement your objective function here
    return sum(x**2)

# Define the PSO's parameters
noP = 30
maxIter = 500
wMax = 0.9
wMin = 0.2
c1 = 2
c2 = 2
vMax = (ub - lb) * 0.2
vMin = -vMax

# Define the Particle class
class Particle:
    def __init__(self):
        self.X = (ub - lb) * np.random.rand(nVar) + lb
        self.V = np.zeros(nVar)
        self.PBEST = ParticleBest()
        
class ParticleBest:
    def __init__(self):
        self.X = np.zeros(nVar)
        self.O = np.inf

# Initialize the particles
Swarm = []
for k in range(noP):
    particle = Particle()
    Swarm.append(particle)
    
GBEST = ParticleBest()

# Main loop
cgCurve = np.zeros(maxIter)
for t in range(maxIter):
    # Calculate the objective value
    for k in range(noP):
        currentX = Swarm[k].X
        Swarm[k].O = ObjectiveFunction(currentX)
        
        # Update the PBEST
        if Swarm[k].O < Swarm[k].PBEST.O:
            Swarm[k].PBEST.X = currentX
            Swarm[k].PBEST.O = Swarm[k].O
        
        # Update the GBEST
        if Swarm[k].O < GBEST.O:
            GBEST.X = currentX
            GBEST.O = Swarm[k].O
    
    # Update the X and V vectors
    w = wMax - t * ((wMax - wMin) / maxIter)
    
    #maintain the position and the velocity of each particle
    for k in range(noP):
        Swarm[k].V = w * Swarm[k].V + c1 * np.random.rand(nVar) * (Swarm[k].PBEST.X - Swarm[k].X) \
                                      + c2 * np.random.rand(nVar) * (GBEST.X - Swarm[k].X)
        
        # Check velocities
        index1 = np.where(Swarm[k].V > vMax)
        index2 = np.where(Swarm[k].V < vMin)
        
        Swarm[k].V[index1] = vMax[index1]
        Swarm[k].V[index2] = vMin[index2]
        
        Swarm[k].X = Swarm[k].X + Swarm[k].V
        
        # Check positions
        index1 = np.where(Swarm[k].X > ub)
        index2 = np.where(Swarm[k].X < lb)
        
        Swarm[k].X[index1] = ub[index1]
        Swarm[k].X[index2] = lb[index2]
    
    outmsg = 'Iteration# ' + str(t) + ' Swarm.GBEST.O = ' + str(GBEST.O)
    print(outmsg)
    
    cgCurve[t] = GBEST.O

plt.semilogy(cgCurve)
plt.xlabel('Iteration#')
plt.ylabel('Weight')
plt.show()