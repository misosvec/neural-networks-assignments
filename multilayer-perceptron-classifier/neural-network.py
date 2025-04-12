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
        return np.where(x > 0, 1, 0)
    
    @classmethod
    def softmax(cls, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
class Layer:
    
    def __init__(self, afunc: ActivationFunction, num_neurons: int):
        self.afunc = afunc
        self.num_neurons = num_neurons

    def forward(self, x):
        z = np.dot(self.W, x)
        return self.afunc(z), z

        
class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        for i, l in enumerate(self.layers[1:]):
            l.W = np.random.rand(l.num_neurons, self.layers[i].num_neurons)
    
    def forward(self, x, save:bool):
        al = x
        for l in self.layers[1:]:
            al, zl = l.forward(al)
            if save:
                l.a = al
                l.z = zl
        return al, zl
    
    def loss(self, x, target):
        x = x.T  
        return -np.sum(target * np.log(x)) / x.shape[1]

    def backprop(self, x):
        pass
    

    def train(self, data, target, lr: float, epochs:int):
        for ep in range(epochs):
            if len(data.shape) > 1:
                data = data[:, np.random.permutation(data.shape[1])]
            else:  
                data = np.random.permutation(data)
            al, zl = self.forward(data,True)

            loss = self.loss(al, target)
            print(f" epoch {ep} loss is ", loss)
            
            dz3 = (al - target.T)
            dW3 = dz3 @ self.layers[2].a.T

            da2 = self.layers[3].W.T @ dz3  
            dz2 = da2 * ActivationFunction.d_relu(self.layers[2].z)
            dW2 = dz2 @ self.layers[1].a.T

            da1 = self.layers[2].W.T @ dz2
            dz1 = da1 * ActivationFunction.d_relu(self.layers[1].z)
            dW1 = dz1 @ data.T

            self.layers[3].W = self.layers[3].W - lr * dW3
            self.layers[2].W = self.layers[2].W - lr * dW2
            self.layers[1].W = self.layers[1].W - lr * dW1


    def predict(self, x):
        print("x predict shape is ", x.shape)
        a, z = self.forward(x, False)
        predicted_class = np.argmax(a, axis=0)
        one_hot = np.zeros_like(a)
        one_hot[predicted_class] = 1
        return one_hot



if __name__ == "__main__":
    nn = NeuralNetwork(
        layers=[
            Layer(ActivationFunction.relu, 2),
            Layer(ActivationFunction.relu, 10),
            Layer(ActivationFunction.relu, 8),
            Layer(ActivationFunction.softmax, 3),
        ]
    )

    train_data = np.loadtxt(
        'dataset/2d.trn.dat', 
        skiprows=1,
        dtype=[('x', float), ('y', float), ('target', 'U1')]
    )

    test_data = np.loadtxt(
        'dataset/2d.tst.dat', 
        skiprows=1,
        dtype=[('x', float), ('y', float), ('target', 'U1')]
    )
    


    X_train = np.column_stack((train_data['x'], train_data['y']))
    y_train = onehot_encode(train_data['target'])

    X_test = np.column_stack((test_data['x'], test_data['y']))    
    y_test = onehot_encode(test_data['target'])

    def z_score_standardize(X):
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_standardized = (X - X_mean) / X_std
        return X_standardized


    X_train_standardized = z_score_standardize(X_train)
    X_test_standardized = z_score_standardize(X_test)

    print(f"X_train shape {X_train.shape}")
    print(f"y_train shape {y_train.shape}")

    print(f"X_test shape {X_test.shape}")
    print(f"y_test shape {y_test.shape}")

    def compute_accuracy(true, pred):
        correct = 0
        if np.sum(true.T - pred) == 0:
            correct += 1
        return correct/len(true)

    nn.train(X_train_standardized.T, y_train, lr=0.005, epochs=50)
    pred = nn.predict(X_test_standardized.T)
    print(compute_accuracy(y_test, pred))
    # print(f"x is {x}")