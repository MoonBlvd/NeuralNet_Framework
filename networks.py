from __future__ import division
import numpy as np
from layers import FClayer
from layers import cross_entropy

class FCnets:

    def __init__(self, numIter, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.numIter = numIter
        self.FClayer1 = FClayer(784,32, self.train_batch_size) 
        self.FClayer2 = FClayer(32,2, self.train_batch_size)
        self.trainXEnt = cross_entropy(self.train_batch_size)
        self.testXEnt = cross_entropy(self.test_batch_size)
        self.w_FC1 = None
        self.b_FC1 = None
        self.w_FC2 = None
        self.b_FC2 = None
    def train(self, xTrain, yTrain, labelTrain):
        nTrain = xTrain.shape[0] # number of data in the whole training set
        for i in range (1,self.numIter):
            # generate the training batch with batch_size
            randIdx = np.random.choice(nTrain,self.train_batch_size, replace = False)
            X = xTrain[randIdx,:] # training batch
            y = yTrain[randIdx,:]
            label = labelTrain[randIdx]

            #forward propagation
            out1 = self.FClayer1.forward(X)
            out2 = self.FClayer2.forward(out1)
            #print out1[1]
            #print out2[1]
            loss = self.trainXEnt.forward(out2, y)
            prediction = np.argmax(out2, axis = 1)
            numTrue = 0
            for j in range (0, prediction.shape[0]):
                numTrue += int(prediction[j]) == int(label[j]) 
            accuracy = numTrue/prediction.shape[0]
            print "The # of iters: ", i
            print "The training loss is: ", loss
            print "The training accuracy is: ", accuracy

            #backward propagation
            delta1 = self.trainXEnt.backward(out2, y)
            delta2, self.w_FC2, self.d_FC2 = self.FClayer2.backward(delta1)
            delta3, self.w_FC1, self.d_FC1 = self.FClayer1.backward(delta2)
        return self.w_FC1, self.b_FC1, self.w_FC2, self.b_FC2
    def test(self, xTest, yTest):
        out1 = self.FClayer1.forward(xTest)
        out2 = self.FClayer2.forward(out1)
        loss = self.testXEnt.forward(out2, yTest)

        # find the classes with the max probability    
        prediction = out2.argmax(axis = 1)
        
        return prediction, loss
        
