import numpy as np
import sys

# Project 1 - Tarren Engberg (engbergt), sole group member.
# Procedures are performed on simple fake data first to ensure functions run properly before using real data.

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

fakeData = np.array( [ [1, 3, 10], [2, 4, 11], [3, 2, 12], [4, 4, 13], [5, 5, 14] ], np.float32 )

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

def dummify(matrix):
    cpMatrix = np.copy(matrix)
    n,m = cpMatrix.shape
    ones = np.ones((n,1))
    cpMatrixNew = np.hstack((ones, cpMatrix))
    return cpMatrixNew

def getWeights(xss, ys, xbars, ybar):
    xssHeight = np.size(xss, 0)
    w = np.empty(xssHeight)
    for row in range(xssHeight):
        m = 0
        for i in range(np.size(xss[row], 0)):
            x = xss[row][i]
            y = ys[row]
            xb = xbars[row]
            yb = ybar
            m += ((x - xb) * (y - yb) / ((x - xb)**2))
        w[row] = m
    return w

def hyp(xs, ws):
    tot = 0
    for x in range(xs):
        tot += xs[x] * ws[x]

def costJ(m, xxs, ys):
   return 5 



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
Xf = removeLabelsfromMatrix(fakeData)
X = removeLabelsfromMatrix(training2dArr)

# Matirx (..vector?) 1D array of desired outputs.
Yf = getLabels(fakeData)
Y = getLabels(training2dArr)



#######
# add as first column of X the dummy variable of '1'. Xd (X with dummy)
#######
Xdf = dummify(Xf)
Xd = dummify(X)



#######
# compute optimal weight vector w (may use package such as numpy).
#######

#initialize a weight array of size Y with a guess of 1.
wf = np.ones((np.size(fakeData, 0),1))
w = np.ones((np.size(training2dArr, 0),1))



# calculate means
Yfmean = np.mean(Yf)
Xfmeans = np.mean(Xf, axis=1)
print(Xfmeans)

Ymean = np.mean(Y)
Xmeans = np.mean(X, axis=0)

# print(getWeights(Xd, Y, Xmeans, Ymean))
# getWeights(Xdf, Yf, Xfmeans, Yfmean)


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
