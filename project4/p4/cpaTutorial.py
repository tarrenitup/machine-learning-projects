from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print("original matrix: ", A)
print("")
# calculate the mean of each column
M = mean(A.T, axis=1)
print("means: ", M)
print("")
# center columns by subtracting column means
C = A - M
print("center cols: ", C)
print("")
# calculate covariance matrix of centered matrix
V = cov(C.T)
print("covarience matrix: ", V)
print("")
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print("eigenvectors: ", vectors)
print("")
print("eigenvalues: ", values)
print("")


# project data
P = vectors.T.dot(C.T)
print("project data: ", P.T)
