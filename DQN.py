import numpy as np
from queue import deque
import tensorflow as tf

class DQN():
    def __init__(self, input_size, output_size, session, name="main"):
        self.input_size = input_size
        self.output_size = output_size
        self.network_name = name
        self.sess = session
        self.build()
    
    def build(self):
        with tf.variable_scope(self.network_name):
            self._X = tf.placeholder(tf.float32, shape=[None, self.input_size])
            self._keep_prob = tf.placeholder(tf.float32)

            W1 = tf.get_variable("Weight1", shape=[self.input_size, 15], initializer=tf.glorot_normal_initializer())
            b1 = tf.Variable(tf.random_normal(shape=[15], stddev=0.005), name="b1")
            H1 = tf.nn.relu(tf.matmul(self._X, W1)+b1)
            H1 = tf.nn.dropout(H1, keep_prob=keep_prob)

            W2 = tf.get_variable("Weight2", shape=[15, 5], initializer=tf.glorot_normal_initializer())
            b2 = tf.Variable(tf.random_normal(shape=[5], stddev=0.005), name="b2")
            H2 = tf.nn.relu(tf.matmul(H1, W2) + b2)
            H2 = tf.nn.dropout(H2, keep_prob=keep_prob)

            W3 = tf.get_variable("Weight3", shape=[5, self.output_size], initializer=tf.glorot_normal_initializer())
            b3 = tf.Variable(tf.random_normal(shape=[self.output_size], stddev=0.005), name="b2")
            self._Qpred = tf.matmul(H2, W3)+b3

        self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
        self._loss = tf.reduce_mean(tf.square(self._Qpred - self._Y))
        self._train = tf.train.AdamOptimizer(0.1).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.sess.run(self._Qpred, feed_dict={self._X:x})
        
    def update(self, x_stack, y_stack):
        return self.sess.run([self._loss, self._train], feed_dict={self._X:x_stack, self._Y:y_stack})
    
