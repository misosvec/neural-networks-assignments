import numpy as np

class HopfieldNetwork:

    def __init__(self, size):
        self.size = size
        self.W = np.zero(self.size)

    def train(self, patterns):
        for i in range(self.size):
            for j in range(self.size):
                val = 0
                for p in patterns:
                    val += p[i] * p[j]
                self.W[i,j] = 1/self.size * val

    def energy(self, s):
        '''
        Compute energy for given state
        '''
        e = 0
        for j in range(self.dim):
            for i in range(self.dim):
                if i != j:
                    e += self.W[i][j]*s[i]*s[j]
        return -1/2 * e

    def run_sync(self, s, eps = 20):
        
        for ep in range(eps):
            net = np.dot(self.W, s)
            net[net >= 0] = 1
            net[net < 0] = -1
