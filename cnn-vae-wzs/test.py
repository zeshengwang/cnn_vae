import tensorflow as tf
import numpy as np
import sys
from shared_subgraphs import qy_graph, qz_graph, labeled_loss
from tensorbayes.layers import constant as Constant
from tensorbayes.layers import placeholder as Placeholder
from tensorbayes.layers import dense as Dense
from tensorbayes.layers import gaussian_sample as GaussianSample
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def px_graph(z, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        h1 = Dense(y, 64, 'h1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 64, 'h2', tf.nn.relu, reuse=reuse)
        zm = Dense(h2, 64, 'zm', reuse=reuse)
        zv = Dense(h2, 64, 'zv', tf.nn.softplus, reuse=reuse)
    # -- p(x)
    with tf.variable_scope('px'):
        # h1 = Dense(z, 512, 'layer1', tf.nn.relu, reuse=reuse)
        # h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        # h3 = Dense(h2, 512, 'layer3', tf.nn.relu, reuse=reuse)
        # px_logit = Dense(h3, 784, 'logit', reuse=reuse)
        h1 = Dense(z, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 28*14*14, 'layer2', tf.nn.relu, reuse=reuse)
        h2 = tf.reshape(h2,[-1,14,14,28])
        conv1 = tf.layers.conv2d_transpose(h2,28,[3,3],(1,1),padding="same",activation=tf.nn.relu,reuse=reuse)
        conv2 = tf.layers.conv2d_transpose(conv1,28,[3,3],(2,2),padding="same",activation=tf.nn.relu,reuse=reuse)
        conv3 = tf.layers.conv2d_transpose(conv2,1,[3,3],(1,1),padding="same",activation=tf.nn.relu,reuse=reuse)
        px_logit = tf.contrib.layers.flatten(conv3)
    return zm, zv, px_logit
x = tf.reshape(mnist.train.images[:3],[-1,28,28,1])
# h1 = tf.layers.conv2d(inputs=x,filters=16,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
# pool1 = tf.layers.max_pooling2d(inputs=h1, pool_size=[2, 2], strides=2)
# h2 = tf.layers.conv2d(inputs=pool1,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
# pool2 = tf.layers.max_pooling2d(inputs=h2, pool_size=[2, 2], strides=2)
# pool2_flat = tf.contrib.layers.flatten(pool2)
# dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
# dense = tf.layers.dense(inputs=dense, units=14*14*28, activation=tf.nn.relu)
# dense = tf.reshape(dense,[-1,14,14,28])
# conv1 = tf.layers.conv2d_transpose(dense,28,[3,3],(1,1),padding="same",activation=tf.nn.relu)
# conv2 = tf.layers.conv2d_transpose(conv1,28,[3,3],(2,2),padding="same",activation=tf.nn.relu)
# conv3 = tf.layers.conv2d_transpose(conv2,1,[3,3],(1,1),padding="same",activation=tf.nn.relu)
# logits = tf.contrib.layers.flatten(conv3)
xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
y_ = tf.fill(tf.stack([tf.shape(x)[0], 10]), 0.0)

z, zm, zv, zm_prior, zv_prior, px_logit = [[None] * 10 for i in xrange(6)]

# for i in range(10):
# 	y = tf.add(y_, Constant(np.eye(10)[i]))
# 	z[i], zm[i], zv[i] = qz_graph(xb, y)
# 	zm_prior[i], zv_prior[i], px_logit[i] = px_graph(z[i], y)
y = tf.add(y_, Constant(np.eye(10)[1]))
z[1], zm[1], zv[1] = qz_graph(xb, y)
y = tf.add(y_, Constant(np.eye(10)[2]))
z[2], zm[2], zv[2] = qz_graph(xb, y)
y = tf.add(y_, Constant(np.eye(10)[3]))
z[3], zm[3], zv[3] = qz_graph(xb, y)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(np.shape(sess.run(z[2])))




