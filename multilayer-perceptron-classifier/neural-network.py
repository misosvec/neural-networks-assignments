from typing import List
import numpy as np

class ActivationFunction:
    
    @classmethod
    def sigmoid(x):
        return 1 / (1+ np.exp(-x))
    
    def d_sigmoid(x):
        pass


class Layer:
    
    def __init__(self, afunc: ActivationFunction, num_neurons: int):
        self.afunc = afunc
        

        
        

class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        for i in range(1, len(self.layers)):
            self.layers[0].W = np.random.random(())
            self.layers[i].W = np.random.random(())
            # Wx
            # (n1,dim) x (dim,ns)     x (n2, n1) x(n1, ns)
            #      (n1, ns)                 (n2,ns) 
    
        



    def forward(self, x):
        for l in self.layers:
            l.forward.prop()

    def train(self, data, epochs:int):
        for ep in range(epochs):
            for x in data[:np.random.permutation(data.shape[1])]:
                





        

    





