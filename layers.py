from __future__ import division
import numpy as np

# Define two activations here

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sig_diff(x):
    return sigmoid(x)*(1-sigmoid(x))
def tanh(x):
    return np.tanh(x)
# Define the FClayer

class FClayer:
    def __init__(self, sizeIn, sizeOut, batch_size):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.batch_size = batch_size
        self.w = 2 * np.random.random([self.sizeIn, self.sizeOut]) - 1 # shift w to be between -1 and 1
        self.b = np.random.random([1,self.sizeOut])
        self.out = None
        self.In = None
        self.lr = 0.0001 # learning rate

    def forward(self, x):
        self.In = x
        self.out = np.dot(x, self.w) + self.b # here self.out = wx+b
        #print "The w is: ", self.w
        #print "The b is: ", self.b
        #print "The x is: ", x
        #print "the wx+b is: ", self.out
        #print "the sigmoid is: ", sigmoid(self.out)
        return sigmoid(self.out)
        
    def backward(self, d):
        delta = d * sig_diff(self.out)
        new_d = np.dot(delta, self.w.T)

        # update w & b
        dw = np.dot(self.In.T, delta)
        self.w = self.w - self.lr*dw
        db = np.dot(np.ones([1, delta.shape[0]]), delta)
        self.b = self.b - self.lr*db
   
        return new_d, self.w, self.b

class cross_entropy:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def forward(self, fx, y):
        loss = 0;
        #print "Fx is: ", fx
        #print "y is: ", y
        for i in range(self.batch_size): 
            loss = loss - np.dot(y[i,:], np.log(fx[i,:].T))
        #print "loss is: ", loss
        return loss
    def backward(self, fx, y):
        delta = np.zeros(fx.shape)
        
        for i in range(self.batch_size):
            delta[i, :] = - np.divide(y[i, :], np.maximum(fx[i, :], np.ones_like(fx[i,:]) * 1e-100))
        delta = delta / self.batch_size
        
        #print "delta shape is: ", delta.shape
        #delta = abs(fx-y)
        #print "new delta shape is: ", delta.shape
        return delta
