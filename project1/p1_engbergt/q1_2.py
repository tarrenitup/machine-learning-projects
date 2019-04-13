import numpy as np
import sys

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

# Project 1 - Tarren Engberg (engbergt), sole group member.

testVector = np.array(  [1,2,3] , np.float32 )
testMatrix = np.array( [ [1,2,3], [4,5,6], [7,8,9] ], np.float32 )

##############
# Functions
##############

def fileTo2dFloatArr(path):
    return np.loadtxt(path)

def removeLabelsfromMatrix(matrix):
    cpMatrix = np.copy(matrix)
    return np.delete(cpMatrix, -1, 1) 

def getLabels(matrix):
    cpMatrix = np.copy(matrix)
    cpMatrixWidth = np.size(cpMatrix, 0)
    vector = np.empty(cpMatrixWidth)
    for i in range(0, cpMatrixWidth):
        vector[i] = cpMatrix[i][(np.size(cpMatrix[i])-1):][0] 
    return vector
    
def costMSE(guess, actual):
    return np.sum((guess - actual)**2) / actual.size



##############
# Procedures
##############

# 1) import Boston housing training data.
training2dArr = fileTo2dFloatArr('data/housing_train.txt')
# train2dArrNp = np.asarray(training2dArr, dtype=np.float32)

#######
# load training data into corresponding X (features) & Y (desired outputs) matrices.
#######

# Matrix (2D array) of features. Created by removing the last element (the label) of each row.
X = removeLabelsfromMatrix(training2dArr)

# Matirx (..vector?) 1D array of desired outputs.
Y = getLabels(training2dArr)



#######
# add as first column of X the dummy variable of '1'. Xd (X with dummy)
#######




#######
# compute optimal weight vector w (may use package such as numpy).
#######

#initialize a weight array of size Y with a guess of 1.


# Ymean = np.mean(Y)
# Xmeans = np.mean(X, axis=0) 

print(Y)

#######
# print the learned weight vector (w).
#######




#######
# 2) import Boston housing testing data.
#######




#######
# using w, compute the ASE of training and testing data.
#######




#######
# print out trainging ASE and testing ASE and which is largest.
#######
