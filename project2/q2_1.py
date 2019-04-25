# Project 2, Question 2_1 - Tarren Engberg (engbergt): sole group member.

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainFilePath = sys.argv[1]
testFilePath = sys.argv[2]

# load data
knnTrainData = pd.read_csv(trainFilePath, header=None)
knnTestData = pd.read_csv(testFilePath, header=None)


# separate into labels and features
trainLabels = knnTrainData.iloc[:, :1].values
trainFeatures = knnTrainData.iloc[:, 1:].values 

testLabels = knnTestData.iloc[:, :1].values
testFeatures = knnTestData.iloc[:, 1:].values

# Normalize?? 