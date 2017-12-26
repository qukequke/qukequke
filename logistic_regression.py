# coding=utf-8
# import pandas as pd
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ', data.shape)
    print(data[1:6, :])
    return data


def cost1(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))


def plotData(data, theta, label_x, label_y, label_pos, label_neg, axes=None):
    xx = np.arange(30, 90)
    yy = (theta[0] + theta[1] * xx) / (-theta[2])
    if axes == None:
        axes = plt.gca()

    regr = LinearRegression()
    regr.fit(data[:, 0].reshape(-1, 1), data[:, 1])
    axes.scatter(data[:, 0], data[:, 1], c=data[:, 2])
    # axes.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')
    axes.plot(xx, yy)
    plt.show()


# def sigmoid(z):
#     return 1.0/(1+np.exp(z))

def sigmoid(z):
    return(1.0 / (1 + np.exp(-z)))


def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))

    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))

    if np.isnan(J[0]):
        return (np.inf)
    return J[0]


def gradient_decent(theta, X, y):
    m = y.size
    iters = 500
    J = np.zeros(iters)
    alpha = 0.00001
    for i in range(iters):
        J[i] = costFunction(theta.reshape(-1, 1), X, y)
        h = sigmoid(X.dot(theta.reshape(-1, 1)))
        grad = (1.0 / m) * X.T.dot(h - y)
        theta -= alpha * grad
    return theta, J


def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))

    grad = (1.0 / m) * X.T.dot(h - y)

    # return grad.flatten()
    return grad.flatten()

data = loaddata('data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

# initial_theta = np.zeros((X.shape[1], 1))
# theta = gradient_decent(X, y, initial_theta)

# print(theta)
theta1 = np.array([-25.16131634,   0.2062316 ,   0.20147143])
initial_theta = np.zeros((X.shape[1], 1))

cost = costFunction(theta1, X, y)
print('Cost: \n', cost)
#
theta, j = gradient_decent(initial_theta, X, y)
print(theta)

plt.plot(j)
plt.show()

#res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})
# res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient_decent, options={'maxiter':400})
# print(res)
# print(sigmoid(np.array([1, 45, 85]).dot(res.x.T)))

