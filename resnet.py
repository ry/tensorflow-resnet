import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('input_size', 224, "input image size")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')


activation = tf.nn.relu


def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              bottleneck=True):
    is_training = tf.convert_to_tensor(is_training,
                                       dtype='bool',
                                       name='is_training')

    with tf.variable_scope('scale1'):
        x = _conv(x, 64, ksize=7, stride=2)
        x = _bn(x, is_training)
        x = activation(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        x = stack(x, num_blocks[0], 64, bottleneck, is_training, stride=1)

    with tf.variable_scope('scale3'):
        x = stack(x, num_blocks[1], 128, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale4'):
        x = stack(x, num_blocks[2], 256, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale5'):
        x = stack(x, num_blocks[3], 512, bottleneck, is_training, stride=2)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        logits = _fc(x, num_units_out=num_classes)

    return logits


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x,
                    is_training,
                    num_blocks=3, # 6n+2 total weight layers will be used.
                    num_classes=10):
    bottleneck = False
    is_training = tf.convert_to_tensor(is_training,
                                       dtype='bool',
                                       name='is_training')

    with tf.variable_scope('scale1'):
        x = _conv(x, 16, ksize=3, stride=1)
        x = _bn(x, is_training)
        x = activation(x)

        x = stack(x, num_blocks, 16, bottleneck, is_training, stride=1)

    with tf.variable_scope('scale2'):
        x = stack(x, num_blocks, 32, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale3'):
        x = stack(x, num_blocks, 64, bottleneck, is_training, stride=2)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        logits = _fc(x, num_units_out=num_classes)

    return logits


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(3, 3, rgb * 255.0)
    bgr = tf.concat(3, [blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.scalar_summary('loss', loss_)

    return loss_


def stack(x, num_blocks, filters_internal, bottleneck, is_training, stride):
    for n in range(num_blocks):
        s = stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x,
                      filters_internal,
                      bottleneck=bottleneck,
                      is_training=is_training,
                      stride=s)
    return x


def block(x, filters_internal, is_training, stride, bottleneck):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    if bottleneck:
        filters_out = 4 * filters_internal
    else:
        filters_out = filters_internal

    shortcut = x  # branch 1

    if bottleneck:
        with tf.variable_scope('a'):
            x = _conv(x, filters_internal, ksize=1, stride=stride)
            x = _bn(x, is_training)
            x = activation(x)

        with tf.variable_scope('b'):
            x = _conv(x, filters_internal, ksize=3, stride=1)
            x = _bn(x, is_training)
            x = activation(x)

        with tf.variable_scope('c'):
            x = _conv(x, filters_out, ksize=1, stride=1)
            x = _bn(x, is_training)
    else:
        with tf.variable_scope('A'):
            x = _conv(x, filters_internal, ksize=3, stride=stride)
            x = _bn(x, is_training)
            x = activation(x)

        with tf.variable_scope('B'):
            x = _conv(x, filters_out, ksize=3, stride=1)
            x = _bn(x, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or stride != 1:
            shortcut = _conv(shortcut, filters_out, ksize=1, stride=stride)
            shortcut = _bn(shortcut, is_training)

    return activation(x + shortcut)


def _bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def _fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def _conv(x, filters_out, ksize=3, stride=1):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
