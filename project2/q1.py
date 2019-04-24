import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd


# load data
knnDummyTrainData = pd.read_csv('data/knn_dummy_train.csv', header=None)
knnDummyTestData = pd.read_csv('data/knn_dummy_test.csv', header=None)
# knnTrainData = pd.read_csv('data/knn_train.csv', header=None)
# knnTestData = pd.read_csv('data/knn_test.csv', header=None)


# separate data into labels and features
tryA = np.array([1, 2, 3])
tryB = np.array([[2, 3, 4], 
                [1.12, 20, 3], 
                [9, 10, 11]])
tryL = np.array([1, 1, 1])

knnDummyTrainLabels = knnDummyTrainData.iloc[:, :1].values
knnDummyTrainFeatures = knnDummyTrainData.iloc[:, 1:].values

knnDummyTestLabels = knnDummyTestData.iloc[:, :1].values
knnDummyTestFeatures = knnDummyTestData.iloc[:, 1:].values

# knnTrainLabels = knnTrainData.iloc[:, :1].values
# knnTrainFeatures = knnTrainData.iloc[:, 1:].values 

# knnTestLabels = knnTestData.iloc[:, :1].values
# knnTestFeatures = knnTestData.iloc[:, 1:].values


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
    while(bestIndex in ignoreIndices): # ensure the initialization isnt an ignored index.
        bestIndex += 1
    index = 0
    for p in points:
        if (not index in ignoreIndices):
            thisDistance = distanceBetweenPoints(p, point)
            bestDistanceSoFar = distanceBetweenPoints(points[bestIndex], point)
            if (thisDistance < bestDistanceSoFar):
                bestIndex = index
        index += 1
    return bestIndex

def findKNearest(k, point, points):
    bestIndices = []
    for i in range(k):
        nearestPointIndex = findNearestIndex(point, points, bestIndices)
        bestIndices.extend([nearestPointIndex])
    return bestIndices

def vote(k, point, points, labels):
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
        print('Tie in vote occured!')
        return 0 # tie situation

# print(vote(5, knnDummyTestFeatures[0], knnDummyTrainFeatures, knnDummyTrainLabels))
print(knnDummyTestLabels[0][0])
