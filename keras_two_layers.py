import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
import keras
import keras.backend as K
K.set_image_data_format('channels_last') #不用忘记设置这个
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Input, BatchNormalization, Flatten
from keras.models import Model
from keras.utils import plot_model



def model(input_shape):
    x_input = Input(input_shape)
    X = Conv2D(8, (4, 4), padding='SAME')(x_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((8, 8))(X) #strides = pool_size default
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(16, (2, 2), padding='SAME')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4))(X) #strides = pool_size default
    X = BatchNormalization(axis=3)(X)
    X = Flatten()(X)
    X = Dense(6, activation='softmax')(X)
    print(X)
    model = Model(inputs=x_input, outputs=X, name='my_model')
    return model


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

if __name__ == '__main__':
    # tf.reset_default_graph()
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_data()
    print(train_set_y.shape)
    print(train_set_x.shape[1:])
    my_model = model(train_set_x.shape[1:])
    print(my_model)
    my_model.compile(keras.optimizers.adam(lr=0.0001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # my_model.compile(keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5), loss='categorical_crossentropy', metrics=['accuracy'])
    my_model.fit(x=train_set_x, y=train_set_y.reshape(train_set_y.shape[1], train_set_y.shape[0]), batch_size=64, epochs=1)
    print(my_model.summary())
    plot_model(my_model, to_file='MyModel.png')
    # print(train_set_y.shape)
    # costs=[] #画图用 存放损失函数
    # batches = mini_batch(train_set_x, train_set_y, 64)
    #
    # n_x = train_set_x.shape[0]
    # n_y = train_set_y.shape[0]
    # X, Y = creat_place_holder(n_x, n_y)
    #
    # Z3, parameter = forward_propagation(X, 25, 12, 6)
    #
    # cost = compute_cost(Z3, Y)
    # optimer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)
    #
    # init = tf.global_variables_initializer()
    # num_minibatches = int(train_set_y.shape[1]/64)
    # with tf.Session() as sess:
    #     tf.summary.FileWriter('.', sess.graph)
    #     sess.run(init)
    #     for i in range(1500):
    #         epoch_loss = 0
    #         for mini_X, mini_Y in batches:
    #             _, loss = sess.run([optimer, cost], feed_dict={X: mini_X, Y: mini_Y})
    #             epoch_loss += loss / num_minibatches
    #         if i % 20 == 0:
    #             print('epoch', i, ":", epoch_loss)
    #             costs.append(epoch_loss)
    #
    #     parameter = sess.run(parameter)
    #     correct_pred = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    #     accurcy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
    #     print("train accury", sess.run(accurcy, feed_dict={X: train_set_x, Y: train_set_y}))
    #     print("test accury", sess.run(accurcy, feed_dict={X: test_set_x, Y: test_set_y}))
    # plt.plot(costs)
    # plt.show()
