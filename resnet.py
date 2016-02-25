import tensorflow as tf
import numpy as np

def depth(x):
    return x.get_shape().as_list()[-1]

# Input to this model should be RGB images with [0,1] pixels
class Model():
    def __init__(self, param_provider):
        self.param_provider = param_provider

    def preprocess(self, rgb):
        rgb_scaled = rgb * 255.0

        red, green, blue = tf.split(3, 3, rgb_scaled) 

        # In the original caffe model they subtract per-pixel means
        # but that forces the model to have 224 x 224 inputs. Because
        # channel variance isn't huge, we do what VGG does and just center
        # the data based on channel means. 
        # https://github.com/KaimingHe/deep-residual-networks/issues/5#issuecomment-183578514

        #blue var 35.377156
        #green var 24.383125
        #red var 10.690761
        blue_mean = 103.062624
        green_mean = 115.902883
        red_mean = 123.151631

        bgr = tf.concat(3, [
            blue - blue_mean,
            green - green_mean,
            red - red_mean,
        ], name="centered_bgr")

        return bgr    
    
    

    def bn(self, bn_name, scale_name, x):
        d = depth(x)
        mean, var, gamma, beta = self.param_provider.bn_params(bn_name, scale_name, d)
        return tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, 1e-12, scale_after_normalization=True, name='bn')

    def conv(self, name, x, out_chans, shape, strides):
        in_chans = x.get_shape().as_list()[3]
        kernel = self.param_provider.conv_kernel(name, in_chans, out_chans, shape, strides)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def _bottleneck_block(self, name, x, num_units, out_chan1, out_chan2, down_stride, use_letters):
        for i in range(0, num_units):
            ds = (i == 0 and down_stride)
            if i == 0:
                unit_name = '%sa' % name
            elif use_letters:
                unit_name = '%s%c' % (name, ord('a') + i) 
            else:
                unit_name = '%sb%d' % (name, i) 
                
            x = self.bottleneck_unit(unit_name, x, out_chan1, out_chan2, ds)
        return x

    def bottleneck_block(self, name, x, num_units, out_chan1, out_chan2, down_stride):
        return self._bottleneck_block(name, x, num_units, out_chan1, out_chan2, down_stride, use_letters=False)

    def bottleneck_block_letters(self, name, x, num_units, out_chan1, out_chan2, down_stride):
        return self._bottleneck_block(name, x, num_units, out_chan1, out_chan2, down_stride, use_letters=True)

    # name should be of the form '2a' or '2b', because it will attempt
    # to expand name to match the layers in caffe
    def bottleneck_unit(self, name, x, out_chan1, out_chan2, down_stride=False):
        in_chans = x.get_shape().as_list()[3]

        if down_stride:
            first_stride = 2
        else:
            first_stride = 1

        with tf.variable_scope('res%s' % name):
            if in_chans == out_chan2:
                b1 = x
            else:
                with tf.variable_scope('branch1'):
                    b1 = self.conv('res%s_branch1' % name, x, out_chans=out_chan2, shape=1, strides=first_stride)
                    b1 = self.bn('bn%s_branch1' % name, 'scale%s_branch1' % name, b1)

            with tf.variable_scope('branch2a'):
                b2 = self.conv('res%s_branch2a' % name, x, out_chans=out_chan1, shape=1, strides=first_stride)
                b2 = self.bn('bn%s_branch2a' % name, 'scale%s_branch2a' % name, b2)
                b2 = tf.nn.relu(b2, name='relu')

            with tf.variable_scope('branch2b'):
                b2 = self.conv('res%s_branch2b' % name, b2, out_chans=out_chan1, shape=3, strides=1)
                b2 = self.bn('bn%s_branch2b' % name, 'scale%s_branch2b' % name, b2)
                b2 = tf.nn.relu(b2, name='relu')

            with tf.variable_scope('branch2c'):
                b2 = self.conv('res%s_branch2c' % name, b2, out_chans=out_chan2, shape=1, strides=1)
                b2 = self.bn('bn%s_branch2c' % name, 'scale%s_branch2c' % name, b2)

            x = b1 + b2
            return tf.nn.relu(x, name='relu')

    def fc(self, name, x, n):
        x_shape = x.get_shape().as_list()
        in_chans = x_shape[-1]
        assert x_shape[1] == 1
        assert x_shape[2] == 1
        x = tf.squeeze(x, [1, 2])

        w, b = self.param_provider.fc_params(name)

        x = tf.matmul(x, w)
        x = tf.nn.bias_add(x, b)
        return x

    def build(self, x, layers):
        with tf.variable_scope('preprocess'):
            x = self.preprocess(x)

        with tf.variable_scope('conv1'):
            x = self.conv('conv1', x, out_chans=64, shape=7, strides=2)
            x = self.bn('bn_conv1', 'scale_conv1', x)
            x = tf.nn.relu(x, name='relu')

        x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding='SAME', name='pool1')

        x = self.bottleneck_block_letters('2', x, 3, 64, 256, False)

        if layers == 50:
            x = self.bottleneck_block_letters('3', x, 4, 128, 512, True)
            x = self.bottleneck_block_letters('4', x, 6, 256, 1024, True)
        elif layers == 101:
            x = self.bottleneck_block('3', x, 4, 128, 512, True)
            x = self.bottleneck_block('4', x, 23, 256, 1024, True)
        elif layers == 152:
            x = self.bottleneck_block('3', x, 8, 128, 512, True)
            x = self.bottleneck_block('4', x, 36, 256, 1024, True)
        else:
            pr
            assert False, "bad layers val"

        x = self.bottleneck_block_letters('5', x, 3, 512, 2048, True)

        # avg pool. Use reduce_mean instead of tf.nn.avg_pool so we don't 
        # have to fix a kernel size and allow the network to handle variable
        # sized input.
        x = tf.reduce_mean(x, [1, 2], keep_dims=True, name='pool5')

        x = self.fc('fc1000', x, 1000)
        return tf.nn.softmax(x, name='prob') 

