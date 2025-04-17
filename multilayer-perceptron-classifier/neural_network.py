from typing import List
import numpy as np
from utils import onehot_encode, add_bias, compute_accuracy

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
    
    def __init__(self, afunc: ActivationFunction, num_neurons: int, input_dim = None):
        self.afunc = afunc
        self.num_neurons = num_neurons
        self.input_dim = input_dim

    def forward(self, x):
        x_prime = add_bias(x)
        z = np.dot(self.W, x_prime) 
        a = self.afunc(z)
        return a, z

        
class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        for i, l in enumerate(self.layers):
            if i == 0:
                l.W = np.random.randn(l.num_neurons, l.input_dim + 1)  # input layer + bias
            else:
                l.W = np.random.randn(l.num_neurons, self.layers[i-1].num_neurons + 1)
    
    def forward(self, x, save:bool):
        al = x
        for l in self.layers:
            al, zl = l.forward(al)
            if save:
                l.a = al
                l.z = zl
        return al, zl
    
    def loss(self, y_true, y_pred):
        # print(f"loss pred shape is {pred.shape}") 
        # print(f"loss target shape is {target.shape}")

        # print(f"loss pred {pred[:20]}")
        # print(f"loss target {target[:20]}")

        y_pred = np.clip(y_pred, 1e-10, None) # prevent issues with log(0)

        return -np.sum(y_true * np.log(y_pred)) 

    
    def train(self, X_train, y_train, lr: float, epochs:int, X_val=None, y_val=None):
        print(f"Training with {X_train.shape[0]} samples, {X_train.shape[1]} features, {y_train.shape[0]} classes")
        print(f"Validation with {X_val.shape[0]} samples, {X_val.shape[1]} features, {y_val.shape[0]} classes") 
        m = X_train.shape[1]
        train_losses = []
        val_losses = []
        for ep in range(epochs):
            pred_train, zl = self.forward(X_train, True)

            loss_train = self.loss(y_train, pred_train)/m
            train_losses.append(loss_train)
            print(f"Epoch {ep} Loss: {loss_train:.6f}")
            if X_val is not None and y_val is not None:
                pred_val, zl = self.forward(X_val, save=False)
                val_loss = self.loss(y_val, pred_val)/X_val.shape[1]
                val_losses.append(val_loss)

            dz2 = pred_train - y_train
            dW2 = dz2 @ add_bias(self.layers[1].a).T / m 

            da1 = self.layers[2].W.T @ dz2
            dz1 = da1[:-1, :] * ActivationFunction.d_relu(self.layers[1].z)  
            dW1 = dz1 @ add_bias(self.layers[0].a).T / m

            da0 = self.layers[1].W.T @ dz1
            dz0 = da0[:-1, :] * ActivationFunction.d_relu(self.layers[0].z)
            dW0 = dz0 @ add_bias(X_train).T / m

            self.layers[2].W -= lr * dW2
            self.layers[1].W -= lr * dW1
            self.layers[0].W -= lr * dW0
        
        return train_losses, val_losses


    def predict(self, x):
        # print(f"predict x shape is {x.shape}")
        a, z = self.forward(x, save=False)
        # print(f"predict pred shape is {a.shape}")
        # print(f"predict pred samples are {a.T[10:20]}")

        predicted_class = np.argmax(a, axis=0)
        # print(f"predict predicted class shape {predicted_class.shape}")
        # print(f"predict predicted class data {predicted_class}")

        one_hot = np.zeros_like(a)
        one_hot[predicted_class, np.arange(a.shape[1])] = 1


        return one_hot

if __name__ == "__main__":
    pass
