import numpy as np

np.set_printoptions(precision=3)
#######
# Project 1 - Tarren Engberg (engbergt), sole group member.
#######



#######
# Functions
#######

def fileTo2dFloatArr(path):
    return map(lambda line : map(lambda str: float(str), line.split()), open(path, 'r').read().splitlines())

def removeLast(lst):
    newLst = []
    i = 0
    while i < len(lst) - 1:
        newLst.append(lst[i])
        i += 1
    return newLst


def costMSE(guess, actual):
    return np.sum((guess - actual)**2) / actual.size



# 1) import Boston housing training data.
training2dArr = fileTo2dFloatArr('data/housing_train.txt')
train2dArrNp = np.asarray(training2dArr, dtype=np.float32)

#######
# load training data into corresponding X (features) & Y (desired outputs) matrices.
#######

# Matrix (2D array) of features.
X = map(removeLast, training2dArr)

# Matirx (..vector? 1D array) of desired outputs.
Y = map(lambda line : line[len(line) - 1], training2dArr)

#######
# add as first column of X the dummy variable of '1'. Xd (X with dummy)
#######
Xd = map(lambda line : [1] + line, X)



#######
# compute optimal weight vector w (may use package such as numpy).
#######

#initialize a weight array of size Y with a guess of 1.
w = map(lambda num : 1, Y)

Ymean = np.mean(Y)
Xmeans = np.mean(X, axis=0)

print(train2dArrNp)


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
