import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### utilities ###
# np.random.seed(130) # reproducable
# 120 seed makes S2 go become reassigned 1 time.
# 122 seed caused the two random seeds to be the same and a divide by 0 error, fixed w/ while loop. Now goes through 1 time.

np.random.rand(4) # different each run


def numRows(df):
    return len(df)

def numCols(df):
    return len(df.columns)

def getRand(cap):
    return np.random.randint(0,cap)

def getSSE(df, centroid, assignment, kval):
    sums = 0
    count = 0
    for a in range(len(assignment)):
        if assignment[a] == kval:
            count += 1
            sums += getDistanceBetween(df.iloc[a], centroid)
    return float(sums) / float(count)

def getSSETotal(df, centroids, assignment, k):
    sums = 0
    for kval in range(k):
        sums += getSSE(df, centroids[kval], assignment, kval)
    return float(sums) / float(k)
    


### get data ###

dfS2 = pd.read_csv('S2-data.txt', dtype='int', delimiter = ',', header=None)
dfS3 = pd.read_csv('S3-data.txt', dtype='int', delimiter = ',', header=None)
dfS4 = pd.read_csv('S4-data.txt', dtype='int', delimiter = ',', header=None)
dfF2 = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)


### get parameter(s) ###

k = 2 # default
if len(sys.argv) > 1:
    k = int(sys.argv[1]) # if user supplies parameter


### pick k random seeds ###

def getKSeeds(k, df):
    seeds = {}
    chosen = np.array([])
    for i in range(k):
        randIdx = getRand(numRows(df))
        while(randIdx in chosen): # ensure that the same seed isn't chosen more than once.
             randIdx = getRand(numRows(df))
        seeds[i] = df.iloc[randIdx]
        chosen = np.append(chosen, randIdx)
    return seeds


### assign all points to its nearest seed ###

def eucDist(vector): # distance from the origin 0
    sums = 0
    for i in range(vector.size):
        vs = vector[i] * vector[i]
        sums = sums + vs
    return math.sqrt(sums)

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


#### compute k centroids ###

def getCentroidDim(df, assignment, colIdx, kval):
    sumC = 0
    count = 0
    for a in range(assignment.size):
        if assignment[a] == kval:
            count += 1
            currVal = df[colIdx][a]
            sumC += currVal
    centDimVal = float(sumC) / float(count)
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


### reassign loop ###

def mainLoop(df, k):
    seeds = getKSeeds(k, df)
    initAssign = assign(df, seeds)
    bestCentroids = {}
    bestAssignments = initAssign
    reassignment = 0
    while(True):
        centroids = getCentroids(df, k, initAssign)
        sse = getSSETotal(df, centroids, initAssign, k)
        print("On assign: #", reassignment, ", SSE: ", sse)
        newAssign = assign(df, centroids)
        if(np.array_equal(initAssign, newAssign)):
            bestCentroids = centroids
            bestAssignments = newAssign
            print("reassignment times: ", reassignment)
            break
        else:
            reassignment += 1
            initAssign = newAssign
    return {0: bestCentroids, 1: bestAssignments}

# bestsS2 = mainLoop(dfS2, 2, seedsS2, initAssignmentS2)
bests = mainLoop(df, k)

print("Best assign (k index orientation): #", bests[1])
