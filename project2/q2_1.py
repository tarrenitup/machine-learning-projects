# Project 2, Question 2_1 - Tarren Engberg (engbergt): sole group member.

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

def fileTo2dFloatArr(path):
    return np.loadtxt(open(path, "rb"), delimiter=",")

trainFilePath = 'data/knn_train.csv' #sys.argv[1]
testFilePath = 'data/knn_test.csv' #sys.argv[2]

# load data
trainData = fileTo2dFloatArr(trainFilePath)
# testData = fileTo2dFloatArr(testFilePath)
# testData = pd.read_csv(testFilePath, header=None)

# separate into labels and features

dummyFeatures = np.array([[1,19.59,25,127.7,1191],[1,24.25,20.2,166.2,1761],[-1,14.8,17.66,95.88,674.8],[-1,11.6,12.84,74.34,412.6],[-1,11.08,14.71,70.21,372.7]])

# trainLabels = knnTrainData.iloc[:, :1].values
# trainFeatures = knnTrainData.iloc[:, 1:].values 

# testLabels = knnTestData.iloc[:, :1].values
# testFeatures = knnTestData.iloc[:, 1:].values

def getPlusCount(data):
    pluses = 0
    for row in data:
        if(row[0] > 0):
            pluses += 1
    return pluses

def getMinusCount(data):
    minuses = 0
    for row in data:
        if(row[0] < 0):
            minuses += 1
    return minuses

def entropy(data):
    plusCount = getPlusCount(data)
    minusCount = getMinusCount(data)
    totalCount = len(data)

    if (plusCount == 0 or minusCount == 0):
        print('no entropy!')
        return 0

    pLeft = plusCount/totalCount
    pRight = minusCount/totalCount

    return -1 * (pLeft * math.log2(pLeft) + pRight * math.log2(pRight))

def sortDataByCol(data, col):
    return data[data[:,col].argsort(kind='mergesort')]

def benefit(data, subDataLeft, subDataRight):
    return entropy(data) - ( entropy(subDataLeft) + entropy(subDataRight) )

# def splitDataAt(data, splitIndex):
#     width = len(data[0])
#     arrLeft = np.empty([1, width])
#     arrRight = np.empty([1, width])

#     index = 0
#     for el in data:
#         if(index < splitIndex):
#             arrLeft = np.append(arrLeft, el, axis=0)
#         else:
#             arrRight = np.append(arrRight, el, axis=0)
#         index += 1
    
#     arrLeft = np.delete(arrLeft, 0, axis=0) # get around numpy stupidity
#     arrRight = np.delete(arrLeft, 0, axis=0)

#     return np.array([arrLeft, arrRight])

def bestValueAndAttributeIndex(arr):
    index = 0
    bestEl = arr[0]
    bestInd = 0
    for el in arr:
        if el > bestEl:
            bestEl = el
            bestInd = index
        index += 1
    return [bestInd, bestEl]

def bestInfoGainIndAttr(data): # returns a tuple w/ best attr index and best value

    i = 0
    for example in data:

        j = 0
        bestValues = np.array([]) # the index of the highest value is the index of the best attribute
        for attribute in example:
            if(j == 0): # skip first attribute (label)
                j += 1
                continue

            dataSortedByX = sortDataByCol(data, j)
            bestValue = 0
            k = 0
            for attr in dataSortedByX:
                left = np.split(dataSortedByX, k)[0]
                right = np.split(dataSortedByX, k)[1]
                value = benefit(dataSortedByX, left, right)
                if value > bestValue:
                    bestValue = value
                k += 1

            bestValues = np.append(bestValues, bestValue)
            j += 1
        
        i += 1

    return bestValueAndAttributeIndex(bestValues)


# print(bestInfoGainIndAttr(trainData))

# print(bestInfoGainIndAttr(dummyFeatures))



# arrOne = np.empty([1,4])
arrTwo = np.array([[7,8,9,10], [1,2,3,4], [4,3,2,1], [6,4,5,3]])

# arrOne = np.append(arrOne, arrTwo, axis=0)
# arrOne = np.delete(arrOne, 0, axis=0)

print(np.split(arrTwo, [1,3], axis=0))

# def benefitOfSplit(data, leftSub, rightSub): # maybe split index rather than two addtl arrays?
#     base = len(data) 
#     pLeft = len(leftSub) / base
#     pRight = len(rightSub) / base
#     return entropy(data) - ( pLeft * entropy(leftSub) + pRight * entropy(rightSub) )

# print(entropy(dummyFeatures))








# just in case .. 

 # num = len(dataSortedByX)
# lowest = dataSortedByX[example][j]
# highest = dataSortedByX[num - 1][j]
# step = (highest - lowest) / num
# theta = step

# def getDataLeft(data, val)
#     arrLeft = np.array([])
#     for el in data:
#         if(el < val):
#             arrLeft.append(el)
#     return arrLeft

# def getDataRight(data, val)
#     arrLeft = np.array([])
#     for el in data:
#         if(el >= val):
#             arrLeft.append(el)
#     return arrLeft

