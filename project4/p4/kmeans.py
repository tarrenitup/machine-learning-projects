import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### utilities ###
np.random.seed(250) # maybe make this based on time or something..

def numRows(df):
    return len(df)

def numCols(df):
    return len(df.columns)

def getRand(cap):
    return np.random.randint(0,cap)


### get data ###

# dfS1 = pd.read_csv('S1-data.txt', dtype='int', delimiter = ',', header=None)
dfS2 = pd.read_csv('S2-data.txt', dtype='int', delimiter = ',', header=None)
# dfF2 = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
# df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)


### get parameter(s) ###

k = 3 # default
if len(sys.argv) > 1:
    k = int(sys.argv[1]) # if user supplies parameter


### pick k random seeds ###

def getKSeeds(k, df):
    seeds = {}
    for i in range(k):
        randIdx = getRand(numRows(df))
        seeds[i] = df.iloc[randIdx]
    return seeds

# seedsS1 = getKSeeds(2, dfS1)
seedsS2 = getKSeeds(2, dfS2)
# seedsF2 = getKSeeds(k, dfF2)
# seeds = getKSeeds(k, df)


### assign all points to its nearest seed ###

def eucDist(vector): # distance from the origin 0
    sum = 0
    for i in range(vector.size):
        vs = vector[i] * vector[i]
        sum = sum + vs
    return math.sqrt(sum)

def getDistanceBetween(vectorA, vectorB): # optimization possibly: dist = numpy.linalg.norm(a-b)
    return math.fabs( eucDist(vectorA) - eucDist(vectorB) )

def findClosestCentroid(rowVector, centroids):
    smallestDist = float("inf")
    bestCentroidInd = 0
    for centroid in centroids:
        currDist = getDistanceBetween(centroids[centroid], rowVector)
        if currDist < smallestDist:
            smallestDist = currDist
            bestCentroidInd = centroid
    return bestCentroidInd

def assign(df, centroids):
    assignment = np.zeros(numRows(df)) # a rows long list of which each index is the k orientation.
    for rowIdx in range(numRows(df)):
        assignment[rowIdx] = findClosestCentroid(df.iloc[rowIdx], centroids)
    return assignment

# initAssignmentS1 = assign(dfS1, seedsS1)
initAssignmentS2 = assign(dfS2, seedsS2)
# initAssignmentF2 = assign(dfF2, seedsF2)
# initAssignment = assign(df, seeds)


#### compute k centroids ###

def getCentroidDim(df, assignment, colIdx, kval):
    sumC = 0
    count = 0
    for a in range(assignment.size):
        if assignment[a] == kval:
            count += 1
            currVal = df[colIdx][a]
            sumC += currVal
    centDimVal = float(sumC / count)
    return centDimVal

def getCentroid(df, assignment, kval):
    nc = numCols(df)
    centroid = np.zeros(nc)
    for colIdx in range(nc): # for each dimension in data
        centroid[colIdx] = getCentroidDim(df, assignment, colIdx, kval)
    return centroid

def getCentroids(df, k, assignment):
    centroids = {}
    for kval in range(k):
        centroids[kval] = getCentroid(df, assignment, kval)
    return centroids


# centroidsS1 = getCentroids(dfS1, 2, initAssignmentS1)
centroidsS2 = getCentroids(dfS2, 2, initAssignmentS2)
# centroidsF2 = getCentroids(dfF2, k, initAssignmentF2)
# centroids = getCentroids(df, k, initAssignment)

# print(centroidsF2)

### reassign all points to its nearest centroid ###

# newAssignmentS1 = assign(dfS1, centroidsS1)
newAssignmentS2 = assign(dfS2, centroidsS2)


print("data => ", dfS2)
print("seeds => ", seedsS2)
print("assignment => ", initAssignmentS2)
print("centroids => ", centroidsS2)
print("reassignment => ", newAssignmentS2)

# changeCheck should take the old and new assignments and return the amount of different indecies.
