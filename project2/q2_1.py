# Project 2, Question 2_1 - Tarren Engberg (engbergt): sole group member.

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainFilePath = sys.argv[1]
testFilePath = sys.argv[2]

# load data
# knnTrainData = pd.read_csv(trainFilePath, header=None)
# knnTestData = pd.read_csv(testFilePath, header=None)

# separate into labels and features

dummyFeatures = [[1, 19.59, 25, 127.7, 1191],
                 [1, 24.25, 20.2, 166.2, 1761],
                 [-1, 14.8, 17.66, 95.88, 674.8],
                 [-1, 11.6, 12.84, 74.34, 412.6],
                 [-1, 11.08, 14.71, 70.21, 372.7]]

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

def benefitOfSplit(data, subDataLeft, subDataRight):
    return entropy(data) - ( entropy(subDataLeft) + entropy(subDataRight) )

def bestInfoGainIndex(data):

    for example in data:

        for attribute in example:
            



# def benefitOfSplit(data, leftSub, rightSub): # maybe split index rather than two addtl arrays?
#     base = len(data) 
#     pLeft = len(leftSub) / base
#     pRight = len(rightSub) / base
#     return entropy(data) - ( pLeft * entropy(leftSub) + pRight * entropy(rightSub) )

print(entropy(dummyFeatures))
