import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### utilities ###
np.random.seed(200) # maybe make this based on time or something..

def numRows(df):
    return len(df)

def numCols(df):
    return len(df.columns)

def getRand(cap):
    return np.random.randint(0,cap)

def dfRowToVector(df, rowIdx):
    vector = np.zeros(numCols(df))
    for i in range(vector.size):
        vector[i] = df.iloc[rowIdx][i]
    return vector

def getKSeeds(k, df):
    seeds = {}
    for i in range(k):
        randIdx = getRand(numRows(df))
        seeds[i] = dfRowToVector(df, randIdx)
    return seeds

def eucDist(vector): # distance from the origin 0
    sum = 0
    for i in range(vector.size):
        vs = vector[i] * vector[i]
        sum = sum + vs
    return math.sqrt(sum)

def getDistanceBetween(vectorA, vectorB):
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
        assignment[rowIdx] = findClosestCentroid(dfRowToVector(df, rowIdx), centroids)
    return assignment


### get data ###
dfSample = pd.read_csv('test-data.txt', dtype='int', delimiter = ',', header=None)
dfF2 = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None, nrows=2)
df = pd.read_csv('p4-data.txt', dtype='int', delimiter = ',', header=None)


### get parameter(s) ###
k = 3 # default
if len(sys.argv) > 1:
    k = int(sys.argv[1]) # if user supplies parameter


### pick k seeds ###
seedsSample = getKSeeds(2, dfSample)
seedsF2 = getKSeeds(k, dfF2)
seeds = getKSeeds(k, df)


### assign all points to its nearest seed ###
initAssignmentSample = assign(dfSample, seedsSample)
initAssignmentF2 = assign(dfF2, seedsF2)
initAssignment = assign(df, seeds)


#### compute k centroids ###


### reassign all points to its nearest centroid ###

# changeCheck should take the old and new assignments and return the amount of different indecies.

print(initAssignment)
