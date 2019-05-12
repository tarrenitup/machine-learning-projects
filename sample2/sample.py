# loading libraries
import pandas as pd

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# loading training data
trainData = pd.read_csv('iris.data.txt', header=None, names=names)
trainData.head()


