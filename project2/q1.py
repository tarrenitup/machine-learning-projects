import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd



# load data
knnDummyData = pd.read_csv('data/knn_dummy.csv', header=None)
# knnTrainData = pd.read_csv('data/knn_train.csv', header=None)
# knnTestData = pd.read_csv('data/knn_test.csv', header=None)



# separate data into labels and features
tryA = np.array([1, 2, 3])
tryB = np.array([[2, 3, 4], 
        [1.12, 20, 3], 
        [9, 10, 11]])

knnDummyLabels = knnDummyData.iloc[:, :1].values
knnDummyFeatures = knnDummyData.iloc[:, 1:].values 

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
    while(bestIndex in ignoreIndices):
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

print(findKNearest(2, tryA, tryB))
