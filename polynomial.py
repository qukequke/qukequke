import numpy as np
import matplotlib.pyplot as plt
import random


def cost_function(X, y, theta = np.zeros((3, 1))):
    '''
    计算损失函数
    '''
    J = sum(np.multiply(X * theta - y, X * theta - y))
    return np.float(J)


def decent(X, Y, theta=np.zeros((3, 1))):
    '''梯度下降'''
    alpha = 0.008
    j_list = []
    for i in range(2500):
        h = X * theta
        error = h - Y
        theta = theta - (alpha) * X.transpose() * error
        j = cost_function(X, Y, theta)
        j_list.append(j)
    return j_list, theta


def plot_data(X, Y, theta = np.zeros((3, 1))):
    '''
    input：数据和theta
    没有输出就是画图
    '''
    x_min, x_max = np.min(X[:, 1]), np.max(X[:, 1])
    xx = np.arange(x_min, x_max, 0.01)
    yy = theta[0] + theta[1] * xx + theta[2] * xx ** 2
    plt.plot(xx, yy.reshape(np.size(xx), 1), c='r')
    plt.scatter(np.array(X[:, 1]), np.array(Y))
    plt.show()


def creat_data():
    '''
    此函数用于产生x,y
    Xinition为feature scaling 之前的特征，X，Y为之后的
    '''
    x = np.random.randint(0, 25, size=(100, 1))
    noise = [random.gauss(-5, 10) for i in range(100)]
    noise = np.mat(noise).reshape(100, 1)
    y = 0.1 * (x ** 2) + noise
    X_initial = np.c_[np.ones((100, 1)), np.mat(x), np.mat(x*x)]
    Y_initial = np.mat(y).reshape(100, 1)

    x1_min, x1_max = np.min(X_initial[:, 1]), np.max(X_initial[:, 1])
    x2_min, x2_max = np.min(X_initial[:, 2]), np.max(X_initial[:, 2])
    y_min, y_max = np.min(Y_initial[:, 0]), np.max(Y_initial[:, 0])
    mean = np.mean(X_initial, axis=0)
    mean = np.array(mean)
    mean_y = np.mean(Y_initial)
    X = np.mat(np.zeros(np.shape(X_initial)))
    Y = np.mat(np.zeros(np.shape(Y_initial)))
    X[:, 0] = 1
    X[:, 1] = (X_initial[:, 1] - mean[0][0]) / (x1_max - x1_min)
    X[:, 2] = (X_initial[:, 2] - mean[0][1]) / (x2_max - x2_min)
    Y[:, 0] = (Y_initial[:, 0] - mean_y) / (y_max - y_min)
    return X, Y


if __name__ == "__main__":
    X, Y = creat_data()
    theta1 = (X.transpose() * X).I * X.transpose() * Y
    print(theta1)
    j, theta = decent(X, Y)
    print(theta)
    plot_data(X, Y, theta1)
