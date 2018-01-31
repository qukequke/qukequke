import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py


def forward_propagation(X, classes):
    """
    前向传播 加初始化参数
    与普通nn不同在 构建网络 也就是前向传播不同
    """
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    w1 = tf.get_variable('w1', [4, 4, 3, 8], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w2 = tf.get_variable('w2', [2, 2, 8, 16], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {'w1': w1, 'w2': w2}

    Z1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, w2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=6, activation_fn=None)
    return Z3, parameters


def compute_cost(Z3, Y):
    '''
    利用tf.nn 计算损失函数
    与nn不同在与 数据的排列方式不同了，nn是样本再axis1，这个是在axis0
    '''
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))


def creat_place_holder(nH0, nW0, n_C0, n_y):
    '''
    创建x,y placeholder
    还是数据排列方式不同了
    '''
    X = tf.placeholder(dtype=tf.float32, shape=[None, nH0, nW0, n_C0], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name='Y')
    return X, Y


def one_hot_change(x, classes):
    '''
    将y的0，1,2,3,4,5， 转变为独热码
    数据排列方式不同，axis不同 0变成1
    '''
    one_hot_matrix = tf.one_hot(x, classes, axis=1)
    with tf.Session() as sess:
        result = sess.run(one_hot_matrix)
    return result



def load_data():
    '''
    载入数据并处理  包括归一化 和 转变独热码
    数据不同reshape了
    '''
    train_dataset = h5py.File('datasets/train_signs.h5', 'r')
    train_set_x = np.array(train_dataset['train_set_x'][:]) / 255
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    test_dataset = h5py.File('datasets/test_signs.h5', 'r')
    test_set_x = np.array(test_dataset['test_set_x'][:]) / 255
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y = one_hot_change(train_set_y_orig, len(classes))
    test_set_y = one_hot_change(test_set_y_orig, len(classes))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def mini_batch(X, Y, mini_batch_size, seed=0):
    '''
    将数据划分为minib
    也根据数据发生改变了
    '''
    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    batch_num = m // mini_batch_size
    batches = []
    for i in range(batch_num):
        X_batch = shuffled_X[i*mini_batch_size:i*mini_batch_size+mini_batch_size, :, :, :]
        Y_batch = shuffled_Y[i*mini_batch_size:i*mini_batch_size+mini_batch_size, :]
        batch = (X_batch, Y_batch)
        batches.append(batch)
    if m % mini_batch_size != 0:
        X_batch = shuffled_X[batch_num*mini_batch_size:m, :, :, :]
        Y_batch = shuffled_Y[batch_num*mini_batch_size:m, :]
        batch = (X_batch, Y_batch)
        batches.append(batch)
    return batches


if __name__ == '__main__':
    #最后结果训练正确率97%，测试85
    #测试计算图也发生了一点变化，axis defalt0变为1
    tf.reset_default_graph()
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_data()
    m, n_H0, n_W0, n_C0 = train_set_x.shape
    n_y = train_set_y.shape[1]
    costs=[] #画图用 存放损失函数
    batches = mini_batch(train_set_x, train_set_y, 64)

    X, Y = creat_place_holder(n_H0, n_W0, n_C0, n_y)

    Z3, parameter = forward_propagation(X, 6)

    cost = compute_cost(Z3, Y)
    optimer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)

    init = tf.global_variables_initializer()
    num_minibatches = int(train_set_y.shape[0]/64)
    with tf.Session() as sess:
        tf.summary.FileWriter('.', sess.graph)
        sess.run(init)
        for i in range(1000):
            epoch_loss = 0
            for mini_X, mini_Y in batches:
                _, loss = sess.run([optimer, cost], feed_dict={X: mini_X, Y: mini_Y})
                epoch_loss += loss / num_minibatches
            if i % 20 == 0:
                print('epoch', i, ":", epoch_loss)
                costs.append(epoch_loss)

        parameter = sess.run(parameter)
        correct_pred = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1)) #cnn样本是横着的，m*n所有要在轴1上比较
        accurcy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        print("train accury", sess.run(accurcy, feed_dict={X: train_set_x, Y: train_set_y}))
        print("test accury", sess.run(accurcy, feed_dict={X: test_set_x, Y: test_set_y}))
    plt.plot(costs)
    plt.show()
