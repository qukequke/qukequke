import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py


def forward_propagation(X, nh1, nh2, classes):
    """
    前向传播 加初始化参数
    """
    n, m = X.shape
    w1 = tf.get_variable('w1', [nh1, n], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [nh1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    w2 = tf.get_variable('w2', [nh2, nh1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [nh2, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    w3 = tf.get_variable('w3', [classes, nh2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [classes, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
    parameters = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3}

    Z1 = tf.add(tf.matmul(w1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(w2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(w3, A2), b3)
    return Z3, parameters


def compute_cost(Z3, Y):
    '''
    利用tf.nn 计算损失函数
    '''
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(Z3), labels=tf.transpose(Y)))


def creat_place_holder(nx, ny):
    '''
    创建x,y placeholder
    '''
    X = tf.placeholder(dtype=tf.float32, shape=[nx, None], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[ny, None], name='Y')
    return X, Y


def one_hot_change(x, classes):
    '''
    将y的0，1,2,3,4,5， 转变为独热码
    :param x:
    :param classes:
    :return:
    '''
    one_hot_matrix = tf.one_hot(x, classes, axis=0)
    with tf.Session() as sess:
        result = sess.run(one_hot_matrix)
    return result


def load_data():
    '''
    载入数据并处理  包括归一化 和 转变独热码
    '''
    train_dataset = h5py.File('datasets/train_signs.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    test_dataset = h5py.File('datasets/test_signs.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255

    train_set_y = one_hot_change(train_set_y_orig, len(classes))
    test_set_y = one_hot_change(test_set_y_orig, len(classes))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def mini_batch(X, Y, mini_batch_size, seed=0):
    '''
    将数据划分为minib
    '''
    n, m = X.shape
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    batch_num = m // mini_batch_size
    batches = []
    for i in range(batch_num):
        X_batch = shuffled_X[:, i*mini_batch_size:i*mini_batch_size+mini_batch_size]
        Y_batch = shuffled_Y[:, i*mini_batch_size:i*mini_batch_size+mini_batch_size]
        batch = (X_batch, Y_batch)
        batches.append(batch)
    if m % mini_batch_size != 0:
        X_batch = shuffled_X[:, batch_num*mini_batch_size:m]
        Y_batch = shuffled_Y[:, batch_num*mini_batch_size:m]
        batch = (X_batch, Y_batch)
        batches.append(batch)
    return batches


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_data()
    print(train_set_y.shape)
    costs=[] #画图用 存放损失函数
    batches = mini_batch(train_set_x, train_set_y, 64)

    n_x = train_set_x.shape[0]
    n_y = train_set_y.shape[0]
    X, Y = creat_place_holder(n_x, n_y)

    Z3, parameter = forward_propagation(X, 25, 12, 6)

    cost = compute_cost(Z3, Y)
    optimer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)

    init = tf.global_variables_initializer()
    num_minibatches = int(train_set_y.shape[1]/64)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1500):
            epoch_loss = 0
            for mini_X, mini_Y in batches:
                _, loss = sess.run([optimer, cost], feed_dict={X: mini_X, Y: mini_Y})
                epoch_loss += loss / num_minibatches
            if i % 20 == 0:
                print('epoch', i, ":", epoch_loss)
                costs.append(epoch_loss)

        parameter = sess.run(parameter)
        correct_pred = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accurcy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        print("train accury", sess.run(accurcy, feed_dict={X: train_set_x, Y: train_set_y}))
        print("test accury", sess.run(accurcy, feed_dict={X: test_set_x, Y: test_set_y}))
    plt.plot(costs)
    plt.show()
