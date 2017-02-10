from __future__ import division # force the division to keep float
import os
import numpy as np
from scipy import misc
from networks import FCnets

pathTrain0 = "train/0/"
pathTrain1 = "train/1/"
pathTest0 = "test/0/"
pathTest1 = "test/1/"
def imread(path):
    imgSet = None
    imgList = os.listdir(path)
    for imgName in imgList:
        img =  misc.imread(path + imgName)
        #imgSet = np.con
        w,h = img.shape 
        img = np.reshape(img, (1,w*h))
        if imgSet is None:
            imgSet = img
        else:
            imgSet = np.concatenate((imgSet,img),axis = 0)
            
    return imgSet
        

if __name__ == "__main__":

    # read data
    trainSet0 = imread(pathTrain0)
    trainSet1 = imread(pathTrain1)
    testSet0 = imread(pathTrain0)
    testSet1 = imread(pathTest1)
     
    # make up the trainig set
    xTrain = np.concatenate((trainSet0, trainSet1),axis = 0) / 255 # normalize to 0-1
    yTrain = np.concatenate((np.zeros([trainSet0.shape[0],1]), np.ones([trainSet1.shape[0],1])), axis = 0) # labels
    tmp = np.concatenate((np.ones([trainSet0.shape[0],1]), np.zeros([trainSet1.shape[0],1])), axis = 0)  # tmp vector
    yTrain = np.concatenate((tmp, yTrain), axis = 1)
 
    # make up the testing set
    xTest = np.concatenate((testSet0, testSet1),axis = 0) / 255 # normalize to 0-1
    yTest = np.concatenate((np.zeros([testSet0.shape[0],1]), np.ones([testSet1.shape[0],1])), axis = 0)
    tmp = np.concatenate((np.ones([testSet0.shape[0],1]), np.zeros([testSet1.shape[0],1])), axis = 0)  # tmp vector
    yTest = np.concatenate((tmp, yTest), axis = 1)
    
    print xTrain.shape
    print yTrain.shape
    print xTest.shape
    print yTest.shape
   
    
    # hyperparameters
    train_batch_size = 20
    test_batch_size = xTest.shape[0]
    numIter = 5000
    # Start to train
   
    net = FCnets(numIter, train_batch_size, test_batch_size)
    w_FC1, b_FC1, w_FC2, b_FC2 = net.train(xTrain, yTrain)
#    prediction = net.test(xTest, yTest)

