import numpy as np
# Project 1 - Tarren Engberg (engbergt), sole group member.

# 1) import Boston housing training data.

def fileTo2dFloatArr(path):
    return map(lambda line : map(lambda str: float(str), line.split()), open(path, 'r').read().splitlines())

training2dArr = fileTo2dFloatArr('data/housing_train.txt')

###
# load training data into corresponding X (features) & Y (desired outputs) matrices.
###

# Matrix (2D array) of features.

def removeLast(lst):
    newLst = []
    i = 0
    while i < len(lst) - 1:
        newLst.append(lst[i])
        i += 1
    return newLst

X = map(removeLast, training2dArr)

# Matirx of desired outputs.

Y = map(lambda line : line[len(line) - 1], training2dArr)


###
# add as first column of X the dummy variable of '1'.
###

###
# compute optimal weight vector w (may use package such as numpy).
###

###
# print the learned weight vector (w).
###

###
# 2) import Boston housing testing data.
###

###
# using w, compute the ASE of training and testing data.
###

###
# print out trainging ASE and testing ASE and which is largest.
###