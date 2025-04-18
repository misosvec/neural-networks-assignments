from typing import List, Literal
import numpy as np
from utils import add_bias

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

    
    def __init__(self, layers: List[Layer], weights_init: Literal['he-uniform', 'he-normal'] = 'he-uniform'):
        self.layers = layers
        for i, layer in enumerate(self.layers):
            fan_in = layer.input_dim if i == 0 else self.layers[i - 1].num_neurons
            fan_out = layer.num_neurons

            if weights_init == 'he-uniform':
                limit = np.sqrt(6 / (fan_in + fan_out))
                W = np.random.uniform(-limit, limit, size=(fan_out, fan_in))
            else:
                std = np.sqrt(2 / (fan_in + fan_out))
                W = np.random.normal(loc=0.0, scale=std, size=(fan_out, fan_in))

            layer.W = np.hstack([W, np.zeros((fan_out, 1))])
    
    def _forward(self, x, save:bool):
        al = x
        for l in self.layers:
            al, zl = l.forward(al)
            if save:
                l.a = al
                l.z = zl
        return al, zl
    
    def _loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, None) # prevent issues with log(0)
        return -np.sum(y_true * np.log(y_pred)) 

    
    def train(self, X_train, y_train, lr: float, epochs: int, batch_size: int = 32, X_val=None, y_val=None, verbose=False):
        m = X_train.shape[1]
        train_losses = []
        val_losses = []
        for ep in range(epochs):
            perm = np.random.permutation(m)
            total_train_loss = 0
        
            for i in range(0, m, batch_size):
                indices = perm[i:i + batch_size]
                X_batch = X_train[:, indices]
                y_batch = y_train[:, indices]
                batch_size_current = X_batch.shape[1]
                
                pred_batch, zl = self._forward(X_batch, True)
                mini_batch_loss = self._loss(y_batch, pred_batch)
                total_train_loss += mini_batch_loss
        
                dz = pred_batch - y_batch
                for j in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[j]
                    if j == 0:
                        previous_a = X_batch
                    else:
                        previous_a = self.layers[j - 1].a
                    dW_sum = dz @ add_bias(previous_a).T
                    dW = dW_sum / batch_size_current
                    layer.W -= lr * dW
                    if j > 0:
                        dz = (layer.W.T @ dz)[:-1, :] * ActivationFunction.d_relu(self.layers[j - 1].z)
            
            average_train_loss = total_train_loss / m
            train_losses.append(average_train_loss)
            
            if X_val is not None and y_val is not None:
                pred_val, zl = self._forward(X_val, False)
                val_loss = self._loss(y_val, pred_val) / X_val.shape[1]
                val_losses.append(val_loss)
            
            if verbose:
                print(f"Epoch {ep} Loss: {average_train_loss:.6f}", end="")
                if X_val is not None and y_val is not None:
                    print(f" Val Loss: {val_loss:.6f}")
                else:
                    print()
        
        return train_losses, val_losses

    def predict(self, x):
        a, z = self._forward(x, save=False)
        predicted_class = np.argmax(a, axis=0)
        one_hot = np.zeros_like(a)
        one_hot[predicted_class, np.arange(a.shape[1])] = 1
        return one_hot