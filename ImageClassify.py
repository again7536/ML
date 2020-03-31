#############################
##### 75% CIFAR-10 Code #####
#############################

# import package
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

####################################### Preprocessing #######################################
#shape =[32, 32, 3]
def unpickle(file):
    import pickle 
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
        X = _dict[b'data']
        Y = _dict[b'labels']
        X = X.reshape(-1, 3, 32, 32)
        X = X.swapaxes(1, 3).swapaxes(1, 2)
        Y = np.array(Y)
        Y = np.eye(10)[Y]
    return X, Y

#training variables 
epochs = 160
batch_size = 512
nb_class = 10

X_data = [] ; Y_data = []
for i in range(1, 6):
    _x, _y = unpickle("CIFAR-10/data_batch_"+(str)(i))
    X_data.append(_x), Y_data.append(_y)
X_data = np.array(X_data) ; Y_data = np.array(Y_data)
X_test, Y_test = unpickle("CIFAR-10/test_batch")

#input standardization
mean = X_data.mean()
std = X_data.std()
X_data = (X_data - mean) / std
X_test = (X_test - mean) / std

#placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[None, nb_class])
keep_prob = tf.placeholder(tf.float32)

########################################## Model ###########################################

with tf.variable_scope('Layer1'):
    W1 = tf.Variable(tf.random_normal(shape=[3, 3, 3, 64], stddev=0.01), name='W1')
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME', name='Filter1')
    L1 = tf.nn.relu(L1, name='Lelu1')
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME', name='Pool1')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    #L1 = tf.layers.batch_normalization(L1)
    #shape reduces to [16, 16, 64]

with tf.variable_scope('Layer2'):
    W2 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), name='W2')
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME', name='Filter2')
    L2 = tf.nn.relu(L2, name='Lelu2')
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='Pool2')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    L2 = tf.layers.batch_normalization(L2)
    #shape reduces to [8, 8, 128]
    
with tf.variable_scope('Layer3'):
    W3 = tf.Variable(tf.random_normal(shape=[3, 3, 128, 256], stddev=0.01), name='W3')
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME', name='Filter3')
    L3 = tf.nn.relu(L3, name='Lelu3')
    L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='Pool3')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    #L3 = tf.layers.batch_normalization(L3)
    #shape reduces to [4, 4, 256]
    L3 = tf.reshape(L3, shape=[-1, 4*4*256])

with tf.variable_scope('Layer4'):
    W4 = tf.get_variable("W4", shape=[4*4*256, 512], initializer=tf.glorot_normal_initializer())
    b = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b, name='Lelu4')
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    L4 = tf.layers.batch_normalization(L4)

with tf.variable_scope('Layer5'):
    W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.glorot_normal_initializer())
    b2 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(L4, W5) + b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###################################### Training ##########################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(0, epochs):
        batch_num = (int)(len(Y_data[epoch % 5]) / batch_size)
        avg_cost=0
        for i in range(0, batch_num-1):
            batch_x = X_data[epoch % 5][i*batch_size:(i+1)*batch_size]
            batch_y = Y_data[epoch % 5][i*batch_size:(i+1)*batch_size]
            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.5})
            avg_cost += c / batch_num
        print(f'Epoch: {epoch}   Cost: {avg_cost:.6f}\t Accuracy: {a:.4f}')

    print('Accuracy:', sess.run(accuracy, feed_dict={X:X_test[0:batch_size], Y:Y_test[0:batch_size], keep_prob:1.0}))

####input example####
image = X_data[0][1]
plt.imshow(image)
plt.show()