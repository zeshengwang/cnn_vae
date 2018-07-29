import tensorflow as tf
#from tensorbayes.layers import Constant, Placeholder, Dense, GaussianSample
from tensorbayes.layers import constant as Constant
from tensorbayes.layers import placeholder as Placeholder
from tensorbayes.layers import dense as Dense
from tensorbayes.layers import gaussian_sample as GaussianSample
from tensorbayes.layers import conv2d as Conv2d
from tensorbayes.layers import max_pool as Max_pool
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
# from tensorbayes.tbutils import cross_entropy_with_logits
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as cross_entropy_with_logits
import numpy as np
import sys

# vae subgraphs
def qy_graph(x, k=10):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        x = tf.reshape(x,[-1,28,28,1])
        h1 = tf.layers.conv2d(inputs=x,filters=16,kernel_size=[3, 3],padding="same",activation=tf.nn.relu,reuse=reuse)
        pool1 = tf.layers.max_pooling2d(inputs=h1, pool_size=[2, 2], strides=1)
        h2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[3, 3],padding="same",activation=tf.nn.relu,reuse=reuse)
        pool2 = tf.layers.max_pooling2d(inputs=h2, pool_size=[2, 2], strides=1)
        pool2_flat = tf.contrib.layers.flatten(pool2)
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        qy_logit = Dense(dense, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        x = tf.reshape(x,[-1,28,28,1])
        h1 = tf.layers.conv2d(inputs=x,filters=16,kernel_size=[3, 3],padding="same",activation=tf.nn.relu,reuse=reuse)
        pool1 = tf.layers.max_pooling2d(inputs=h1, pool_size=[2, 2], strides=1)
        #h2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[3, 3],padding="same",activation=tf.nn.relu,reuse=reuse)
        h2 = Conv2d(pool1, 16, [3,3], [1,1], activation = tf.nn.relu, reuse = reuse, scope = 'convlayer2')
        pool2 = tf.layers.max_pooling2d(inputs=h2, pool_size=[2, 2], strides=1)
        pool2_flat = tf.contrib.layers.flatten(pool2)
        dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu,reuse=reuse)
        xy = tf.concat((dense, y), 1, name='xy/concat')
        h1 = Dense(xy, 256, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 128, 'layer2', tf.nn.relu, reuse=reuse)
        h3 = Dense(h2, 64, 'layer3', tf.nn.relu, reuse=reuse)
        zm = Dense(h3, 64, 'zm', reuse=reuse)
        zv = Dense(h3, 64, 'zv', tf.nn.softplus, reuse=reuse)
        z = GaussianSample(zm, zv, 'z')
    return z, zm, zv


def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)
