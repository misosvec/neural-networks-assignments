import numpy as np

class HopfieldNetwork:

    def __init__(self, size):
        self.size = size
        self.W = np.zeros((self.size, self.size))

    def train(self, patterns):
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    val = 0
                    for p in patterns:
                        val += p[i] * p[j]
                    self.W[i,j] = 1/self.size * val

    def energy(self, s):
        e = 0
        for j in range(self.size):
            for i in range(self.size):
                if i != j:
                    e += self.W[i][j]*s[i]*s[j]
        return -1/2 * e

    def run_sync(self, s, eps = 20):
        states = [s.copy()]
        energies = [self.energy(s)]
        
        for ep in range(eps):
            s = np.dot(self.W, s)
            s = np.where(s >= 0, 1, -1)

            states.append(s.copy()) 
            energies.append(self.energy(s))
        return states, energies
