from typing import List
import numpy as np
from utils import onehot_encode

class ActivationFunction:
    
    @classmethod
    def sigmoid(cls,x):
        return 1 / (1+ np.exp(-x))
    
    @classmethod
    def d_sigmoid(cls, x):
        sig = cls.sigmoid(x)
        return sig * (1 - sig)
    
    @classmethod
    def relu(cls, x):
        return np.maximum(0,x)
    
    @classmethod
    def d_relu(cls, x):
        return 0 if x <= 0 else 1
    
    @classmethod
    def softmax(cls, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
class Layer:
    
    def __init__(self, afunc: ActivationFunction, num_neurons: int):
        self.afunc = afunc
        self.num_neurons = num_neurons
        
class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        for i, l in enumerate(self.layers[1:]):
            l.W = np.random.rand(l.num_neurons, self.layers[i].num_neurons)
    
    def forward(self, x):
        o = x
        for l in self.layers[1:]:
            o = l.afunc(np.dot(l.W, o))
        return o
    
    def loss(self, x, target):
        print("loss x shape is ", x.shape)
        print("loss target shape is ", target.shape)
        x = x.T  
        return -np.sum(target * np.log(x)) / x.shape[1]

    def backprop(self, x):
        pass
    

    def train(self, data, target,  epochs:int):
        for ep in range(epochs):
            if len(data.shape) > 1:
                data = data[:, np.random.permutation(data.shape[1])]
            else:  
                data = np.random.permutation(data)
            res = self.forward(data)
            print(f"res after epoch {ep} is {res}")
            loss = self.loss(res, target)
            print(f"loss is {loss}")



if __name__ == "__main__":
    nn = NeuralNetwork(
        layers=[
            Layer(ActivationFunction.sigmoid, 2),
            Layer(ActivationFunction.sigmoid, 10),
            Layer(ActivationFunction.sigmoid, 8),
            Layer(ActivationFunction.softmax, 3),
        ]
    )

    train_data = np.loadtxt(
        'dataset/small.dat', 
        skiprows=1,
        dtype=[('x', float), ('y', float), ('target', 'U1')]
    )

    
    features = np.column_stack((train_data['x'], train_data['y']))
    labels = onehot_encode(train_data['target'])
    
    print(features)
    print(labels)
    # x = np.random.random((2,2))
    # print(x.shape)
    # nn.forward(x)
    nn.train(features.T, labels, 1)
    # print(f"x is {x}")