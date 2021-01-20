import matplotlib.pyplot as plt
import csv
import numpy as np
from computeCost import computeCost
from gradientDescent import gradientDescent

csv_reader = csv.reader(open('ex1data1.txt', 'r'))
X = []
y = []
for line in csv_reader:
    X.append(float(line[0]))
    y.append(float(line[1]))
X = np.array(X)
y = np.array(y)
plt.scatter(X, y)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

theta = np.zeros([2,1])
X = np.hstack([np.ones([X.shape[0],1]), X[:,np.newaxis]])

J = computeCost(X, y, theta)

print('With theta = [0 ; 0]\nCost computed = %f' % (J,));
print('Expected cost value (approx) 32.07\n');


J = computeCost(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = %f' % (J,));
print('Expected cost value (approx) 54.24\n');

alpha = 0.01
iterations = 1500
theta = gradientDescent(X, y, theta, alpha, iterations)


print('Theta found by gradient descent:');
print(theta[0], theta[1]);
print('Expected theta values (approx)');
print(' -3.6303\n  1.1664');

plt.scatter(X[:,1], y)
plt.plot(X[:,1], np.matmul(X, theta),'r')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros([100, 100])
for i in range(100):
    for j in range(100):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        t = t[:,np.newaxis]
        J_vals[i, j] = computeCost(X, y, t)
        pass

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.transpose(J_vals, [1, 0])
surf = ax.plot_surface(X, Y, J_vals)
ax.set_xlabel("${\Theta_0}$")
ax.set_ylabel("${\Theta_1}$")
ax.set_zlabel('J')

plt.show()

plt.contour(X, Y, J_vals, np.logspace(-2, 3, 20))
plt.scatter(theta[0], theta[1])
plt.xlabel("${\Theta_0}$")
plt.ylabel("${\Theta_1}$")
plt.show()