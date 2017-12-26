# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ', data.shape)
    print(data[1:6, :])
    return data


def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))

    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg / (2.0 * m)) * np.sum(np.square(theta[1:]))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)


def sigmoid(z):
    # print(1.0 / (1 + np.exp(-z)))
    return(1.0 / (1 + np.exp(-z)))


def gradientReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1, 1)))

    grad = (1.0 / m) * XX.T.dot(h - y) + 1.0 * (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())


def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))


fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

# 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
# Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
# Lambda = 1 : 这才是正确的打开方式
# Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界
data2 = loaddata('data2.txt', ',')

y = np.c_[data2[:,2]]
X = data2[:,0:2]

#plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')

poly = PolynomialFeatures(4)
XX = poly.fit_transform(data2[:,0:2])
print(XX.shape)
# 看看形状(特征映射后x有多少维了)

initial_theta = np.zeros(XX.shape[1])
print(costFunctionReg(initial_theta, 1, XX, y))

res2 = minimize(costFunctionReg, initial_theta, args=(1, XX, y), method='Newton-CG', jac=gradientReg)
#res2 = minimize(costFunctionReg, initial_theta, args=(100, XX, y), method='Newton-CG', jac=gradientReg, options={'maxiter': 3000})
print('---------------------------------')
print(res2)


for i, C in enumerate([0.0, 0.0, 0.0]):
    # 最优化 costFunctionReg
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), jac=gradientReg, options={'maxiter': 3000})

    # 准确率
    accuracy = 100.0 * sum(predict(res2.x, XX) == y.ravel()) / y.size

    # 对X,y的散列绘图
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])

    # 画出决策边界
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
# plt.show()