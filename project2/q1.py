# Project 2, Question 1 - Tarren Engberg (engbergt): sole group member.

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainFilePath = sys.argv[1]
testFilePath = sys.argv[2]
chosenK = int(sys.argv[3])

# load data
knnTrainData = pd.read_csv(trainFilePath, header=None)
knnTestData = pd.read_csv(testFilePath, header=None)

# separate into labels and features
knnTrainLabels = knnTrainData.iloc[:, :1].values
knnTrainFeatures = knnTrainData.iloc[:, 1:].values 

knnTestLabels = knnTestData.iloc[:, :1].values
knnTestFeatures = knnTestData.iloc[:, 1:].values

def distanceBetweenPoints(pointOneArray, pointTwoArray):
    index = 0
    total = 0
    for x in pointOneArray:
        sub = pointOneArray[index] - pointTwoArray[index]
        total = total + np.square(sub)
        index += 1
    return np.sqrt(total)

def findNearestIndex(point, points, ignoreIndices):
    bestIndex = 0
    bestDistance = math.inf
    while(bestIndex in ignoreIndices): # ensure the initialization isnt an ignored index.
        bestIndex += 1
    index = 0
    for p in points:
        if (not index in ignoreIndices):
            thisDistance = distanceBetweenPoints(p, point)
            # bestDistanceSoFar = distanceBetweenPoints(points[bestIndex], point)
            if (thisDistance < bestDistance):
                bestIndex = index
                bestDistance = thisDistance
        index += 1
    return bestIndex

def findKNearest(k, point, points):
    bestIndices = []
    for i in range(k):
        nearestPointIndex = findNearestIndex(point, points, bestIndices)
        bestIndices.extend([nearestPointIndex])
    return bestIndices

def getPredictedLabel(k, point, points, labels):
    totalVote = 0
    kBestIndicesArray = findKNearest(k, point, points)
    for index in kBestIndicesArray:
        thisVote = labels[index]
        totalVote += thisVote
    if totalVote < 0:
        return -1
    elif totalVote > 0:
        return 1
    else:
        print('Tie in vote occured! Pick an odd K value next time.')
        return 0

def floatToPercent(fl):
    return "{0:.3%}".format(fl)

def findError(k, trainFeatures, testFeatures, trainLabels, testLabels):
    correct = 0
    index = 0
    for feat in testFeatures:
        predictedLabel = getPredictedLabel(k, feat, trainFeatures, trainLabels)
        if (predictedLabel == testLabels[index]):
            correct += 1
        index += 1
    return floatToPercent(1 - (correct / len(testFeatures)))


print('training error: ', findError(chosenK, knnTrainFeatures, knnTrainFeatures, knnTrainLabels, knnTrainLabels))
# LEAVE THIS COMMENTED OUT. It takes too long to run.
# # print('leave-one-out cross-validation error: ', findError(len(knnTrainFeatures), knnTrainFeatures, knnTestFeatures, knnTrainLabels, knnTestLabels))
print('leave-one-out cross-validation error: omitted due to run time (discussed in report)')
print('testing error: ', findError(chosenK, knnTrainFeatures, knnTestFeatures, knnTrainLabels, knnTestLabels))
