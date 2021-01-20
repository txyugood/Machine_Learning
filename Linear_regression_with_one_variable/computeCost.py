import numpy as np
def computeCost(X, y, theta):
    m = y.shape[0]
    J = 0
    for i in range(m):
        h = np.matmul(X[i,:], theta)
        J = J + (h - y[i])**2
    J = J / 2 / m
    return J
