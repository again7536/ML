import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def DataPreprocess(data, seq_len):
    X = []; Y = []
    for i in range(0, len(data)-seq_len):
        x_ = data[i:i+seq_len, :]
        y_ = data[i+seq_len, [-1]]
        X.append(x_); Y.append(y_)
    return np.array(X), np.array(Y)

data = np.loadtxt('sample.csv', dtype=np.float32, delimiter=',', usecols=(1,2,3,4,5))

data[:, 3], data[:, 4] = data[:, 4], data[:, 3].copy()

#Variables
seq_len = 7 ; data_dim = 5 ; hidden_dim = 7 ; output_dim = 1
epochs = 40 ; batch_size = 100

train_amount = (int) (len(data)*0.7)

train_data = data[:train_amount]
test_data = data[train_amount- seq_len:]
train_data = MinMaxScaler(train_data)
test_data = MinMaxScaler(test_data)

X_train, Y_train = DataPreprocess(train_data, seq_len)
X_test, Y_test = DataPreprocess(test_data, seq_len)

X = tf.placeholder(tf.float32, shape=[None, seq_len, data_dim])
Y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

#basic cell
cell = [tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True) for _ in range(2)]
#dropout
#dropout1 = tf.contrib.rnn.DropoutWrapper(cell[0], input_keep_prob=keep_prob, output_keep_prob=keep_prob)
#dropout2 = tf.contrib.rnn.DropoutWrapper(cell[1], input_keep_prob=keep_prob, output_keep_prob=keep_prob)
#multi cell
#cell = tf.contrib.rnn.MultiRNNCell([dropout1, dropout2], state_is_tuple=True)

outputs, _state = tf.nn.dynamic_rnn(dropout1, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)

loss = tf.reduce_mean(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_num = (int)(len(X_train) /batch_size)
    print("Start Training.\n")
    for epoch in range(epochs):
        loss_avg = 0
        for i in range(0, batch_num-1):
            x_batch = X_train[i*batch_size : (i+1)*batch_size]
            y_batch = Y_train[i*batch_size : (i+1)*batch_size]

            _, l = sess.run([train, loss], feed_dict={X:x_batch, Y:y_batch, keep_prob:0.5})
            loss_avg += l / batch_num
        print(f'Epoch: {epoch}\t Loss: {loss_avg:.6f}')
    
    #test
    pred = sess.run(Y_pred, feed_dict={X:X_test, keep_prob:1.0})
    plt.plot(pred)
    plt.plot(Y_test)
    plt.show()
    

