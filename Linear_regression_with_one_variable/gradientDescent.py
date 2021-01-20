import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    n = theta.shape[0]
    for iter in range(num_iters):
        k = np.zeros([m, 1])
        p = np.zeros([n, 1])
        for i in range(m):
            X_tmp = np.expand_dims(X[i,:],0)
            k[i] = np.matmul(X_tmp, theta) - y[i]
            p = p + k[i] * np.transpose(X_tmp,[1,0])
            pass
        p = p / m
        theta = theta - alpha * p
        pass
    return theta