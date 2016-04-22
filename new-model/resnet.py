# Goals:
# - Inputs could be used for images or audio
# - Experiment with ResNets as RNNs. Specifically the ability to share weights within a single
#   section of blocks where they all share the same spacial size.
# - Ability to experiment with stochastic layers
# - Be able to load the caffe pre-trained models
# - Be able to use distributed TF

import tensorflow as tf
import slim

FLAGS = tf.app.flags.FLAGS


def inferene(inputs,
             num_classes=1000,
             is_training=True,
             num_layers=[2, 2, 2, 2], # defaults to 18-layer network
             bottleneck=False, 
             restore_logits=True,
             scope=''):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
    }
    #with tf.op_scope([inputs], scope, 'ResNet'):

    with slim.arg_scope([slim.conv2d, slim.fc], weight_decay=0.00004),
         slim.arg_scope([slim.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        is_training=is_training,
                        batch_norm_params=batch_norm_params),
         slim.arg_scope([resnet_block], bottleneck=bottleneck):
        # pre-net
        x = slim.conv2d(inputs, 64, [7, 7], stride=2, scope="Conv1")

        # ResNet

        with tf.variable_scope('Conv2'):
            x = slim.max_pool(x, [3, 3], stride=2, padding='VALID')
            x = slim.repeat_op(num_layers[0], x, resnet_block, 64)

        with tf.variable_scope('Conv3'):
            x = slim.repeat_op(num_layers[1], x, resnet_block, 128)

        with tf.variable_scope('Conv4'):
            x = slim.repeat_op(num_layers[2], x, resnet_block, 256)

        with tf.variable_scope('Conv5'):
            x = slim.repeat_op(num_layers[3], x, resnet_block, 512)

        # post-net
        x = slim.avg_pool(x, [7,7])
        x = slim.fc(x, num_units_out=num_classes, bias=0.0, restore=restore_logits)

        return tf.nn.softmax(x) 


@scopes.add_arg_scope
def resnet_block(inputs,
                 num_filters,
                 kernel_size,
                 stride=1,
                 bottleneck=False,
                 scope=None,
                 reuse=False):
    input_shape = inputs.get_shape().as_list()
    input_depth = input_shape[-1]

    # Note: num_filters isn't how many filters are outputed. That is the case when bottleneck=False
    # but when bottleneck is True, num_filters*4 filters are outputed. num_filters is how many filters
    # the 3x3 convs output.
    if bottleneck:
        num_filters_out = 4 * num_filters
    else:
        num_filters_out = num_filters
  
    with tf.variable_op_scope([inputs], scope, 'Block', reuse=reuse):
        b1 = inputs
        b2 = inputs

        with tf.variable_scope('Branch1'):
            if input_depth != num_filters_out:
                b1 = slim.conv2d(b1,
                                num_filters=num_filters,
                                kernel_size=[1, 1],
                                strides=strides,
                                activation=None)
  
        with tf.variable_scope('Branch2'):
            if bottleneck:
                b2 = slim.conv2d(b2,
                                num_filters_out=num_filters,
                                kernel_size=[1, 1],
                                strides=strides)

                b2 = slim.conv2d(b2,
                                num_filters_out=num_filters,
                                kernel_size=[3, 3],
                                strides=1)

                b2 = slim.conv2d(b2,
                                num_filters_out=num_filters_out,
                                kernel_size=[1, 1],
                                strides=1,
                                activation=None)
                
            else:
                b2 = slim.conv2d(b2,
                                num_filters_out=num_filters,
                                kernel_size=[3, 3],
                                strides=strides)

                b2 = slim.conv2d(b2,
                                num_filters_out=num_filters_out,
                                kernel_size=[3, 3],
                                strides=1,
                                activation=None)
  
        return tf.nn.relu(b1 + b2)
