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

    def run_sync(self, s):
        states = [s.copy()]
        energies = [self.energy(s)]
        is_fixed_point = False
        while True:
            s = np.dot(self.W, s)
            s = np.where(s >= 0, 1, -1)

            states.append(s.copy()) 
            energies.append(self.energy(s))
            if self._is_fixed_point(states):
                is_fixed_point = True
                break
            if self._is_limit_cycle(states):
                break
        return states, energies, is_fixed_point
    
    def _is_fixed_point(self, states):
        if len(states) < 2:
            return False
        return np.array_equal(states[-1], states[-2])
    
    def _is_limit_cycle(self, states):
        if len(states) < 3:
            return False
        last_state = states[-1]
        for k in range(len(states) - 2):
            if np.array_equal(last_state, states[k]):
                return True
        return False
