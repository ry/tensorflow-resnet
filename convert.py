import os
os.environ["GLOG_minloglevel"] = "2"
import sys
import caffe
import numpy as np
import tensorflow as tf
import utils
import skimage
from skimage.io import imsave
from caffe.proto import caffe_pb2


def depth(x):
    return x.get_shape().as_list()[-1]

def preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    mean_bgr = load_mean_bgr()
    out = np.copy(img) * 255.0
    out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
    out -= mean_bgr
    return out
    
def assert_almost_equal(caffe_tensor, tf_tensor):
    t = tf_tensor[0]
    c = caffe_tensor[0].transpose((1,2,0))

    #for i in range(0, t.shape[-1]):
    #    print "tf", i,  t[:,i]
    #    print "caffe", i,  c[:,i]

    if t.shape != c.shape:
        print "t.shape", t.shape
        print "c.shape", c.shape
        sys.exit(1)

    d = np.linalg.norm(t - c)
    print "d", d
    assert d < 500

def same_tensor(a, b):
    return np.linalg.norm(a - b) < 0.1


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

class ResNet():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net


    def bn(self, bn_name, scale_name, x):
        d = depth(x)

        mean = self.caffe_net.params[bn_name][0].data
        var = self.caffe_net.params[bn_name][1].data

        assert mean.shape == (d,)
        assert var.shape == (d,)

        gamma = self.caffe_net.params[scale_name][0].data
        beta = self.caffe_net.params[scale_name][1].data

        assert gamma.shape == (d,)
        assert beta.shape == (d,)

        return tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, 1e-12, scale_after_normalization=True, name='bn')

    def conv(self, name, x, out_chans, shape, strides):
        in_chans = x.get_shape().as_list()[3]

        k = self.caffe_net.params[name][0].data
        #b = self.caffe_net.params[name][1].data

        # caffe uses [out_channels, in_channels, filter_height, filter_width] 
        #             0             1            2              3
        # tensorflow uses [filter_height, filter_width, in_channels, out_channels]
        #                  2               3            1            0

        if strides == 2 and shape == 1:
            # TensorFlow doesn't let you do this. BUG.
            # https://github.com/tensorflow/tensorflow/issues/889
            # Going to hack around this by creating a 2x2 kernel where the top left
            # is the 1x1 kernel from caffe. 
            # k is [out_chans, in_chans, 1, 1]
            # need [2, 2, in_chans, out_chans]
            z = np.zeros((2, 2, in_chans, out_chans))
            k = k[:, :, 0, 0].transpose((1,0))
            assert k.shape == (in_chans, out_chans)
            z[0,0,:,:] = k
            kernel = tf.constant(z,  dtype='float32', name="kernel")
            kernel_shape = kernel.get_shape().as_list()
            assert kernel_shape[0] == 2
            assert kernel_shape[1] == 2
        else:
            kernel = tf.constant(k.transpose((2, 3, 1, 0)),  dtype='float32', name="kernel")
            kernel_shape = kernel.get_shape().as_list()
            assert kernel_shape[0] == shape
            assert kernel_shape[1] == shape

        assert kernel_shape[2] == in_chans
        assert kernel_shape[3] == out_chans
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

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

        weights = self.caffe_net.params[name][0].data
        assert weights.shape == (n, in_chans)
        bias = self.caffe_net.params[name][1].data
        assert bias.shape == (n,)

        w = tf.constant(weights.transpose((1,0)), name="weights", dtype="float32")
        b = tf.constant(bias, name="bias", dtype="float32")

        x = tf.matmul(x, w)
        x = tf.nn.bias_add(x, b)
        return x


    def build50(self, x):
        with tf.variable_scope('conv1'):
            x = self.conv('conv1', x, out_chans=64, shape=7, strides=2)
            x = self.bn('bn_conv1', 'scale_conv1', x)
            x = tf.nn.relu(x, name='relu')

        x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding='SAME', name='pool1')

        x = self.bottleneck_unit('2a', x, 64, 256, False)
        x = self.bottleneck_unit('2b', x, 64, 256, False)
        x = self.bottleneck_unit('2c', x, 64, 256, False)

        x = self.bottleneck_unit('3a', x, 128, 512, True)
        x = self.bottleneck_unit('3b', x, 128, 512, False)
        x = self.bottleneck_unit('3c', x, 128, 512, False)
        x = self.bottleneck_unit('3d', x, 128, 512, False)

        x = self.bottleneck_unit('4a', x, 256, 1024, True)
        x = self.bottleneck_unit('4b', x, 256, 1024, False)
        x = self.bottleneck_unit('4c', x, 256, 1024, False)
        x = self.bottleneck_unit('4d', x, 256, 1024, False)
        x = self.bottleneck_unit('4e', x, 256, 1024, False)
        x = self.bottleneck_unit('4f', x, 256, 1024, False)

        x = self.bottleneck_unit('5a', x, 512, 2048, True)
        x = self.bottleneck_unit('5b', x, 512, 2048, False)
        x = self.bottleneck_unit('5c', x, 512, 2048, False)

        print x.get_shape()

        x = tf.nn.avg_pool(x, [1,7,7,1], [1,1,1,1], padding='VALID', name='pool5')

        x = self.fc('fc1000', x, 1000)
        return tf.nn.softmax(x, name='prob') 

def load_mean_bgr():
    """ bgr mean pixel value image, [0, 255]. [height, width, 3] """
    with open("ResNet_mean.binaryproto", mode='rb') as f:
        data = f.read()
    blob = caffe_pb2.BlobProto()
    blob.ParseFromString(data)

    mean_bgr = caffe.io.blobproto_to_array(blob)[0]
    assert mean_bgr.shape == (3, 224, 224)

    return mean_bgr.transpose((1,2,0))

def load_caffe(img_p):
    caffe.set_mode_cpu()
    net = caffe.Net("ResNet-50-deploy.prototxt", "ResNet-50-model.caffemodel", caffe.TEST)

    net.blobs['data'].data[0] = img_p.transpose((2,0,1))
    assert net.blobs['data'].data[0].shape == (3, 224, 224)
    net.forward()

    caffe_prob = net.blobs['prob'].data[0]

    utils.print_prob(caffe_prob)

    return net


def main(_):
    img = load_image("cat.jpg")
    img_p = preprocess(img)

    net = load_caffe(img_p)

    with tf.device('/cpu:0'):
        images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
        m = ResNet(net)
        m.build50(images)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    graph = tf.get_default_graph()
    #conv1_conv = graph.get_operation_by_name("conv1/conv").outputs[0]

    i = [
        graph.get_tensor_by_name("conv1/relu:0"),
        graph.get_tensor_by_name("pool1:0"),
        graph.get_tensor_by_name("res2a/relu:0"),
        graph.get_tensor_by_name("res2b/relu:0"),
        graph.get_tensor_by_name("res2c/relu:0"),
        graph.get_tensor_by_name("res3a/relu:0"),
        graph.get_tensor_by_name("res3b/relu:0"),
        graph.get_tensor_by_name("res5c/relu:0"),
        graph.get_tensor_by_name("pool5:0"),
        graph.get_tensor_by_name("prob:0"),
    ]

    o = sess.run(i, {
        images: img_p[np.newaxis,:]
    })

    assert_almost_equal(net.blobs['conv1'].data, o[0])
    assert_almost_equal(net.blobs['pool1'].data, o[1])
    assert_almost_equal(net.blobs['res2a'].data, o[2])
    assert_almost_equal(net.blobs['res2b'].data, o[3])
    assert_almost_equal(net.blobs['res2c'].data, o[4])
    assert_almost_equal(net.blobs['res3a'].data, o[5])
    assert_almost_equal(net.blobs['res3b'].data, o[6])
    assert_almost_equal(net.blobs['res5c'].data, o[7])
    assert_almost_equal(net.blobs['pool5'].data, o[8])
    assert same_tensor(net.blobs['prob'].data, o[9])

    utils.print_prob(o[9][0])



if __name__ == '__main__':
    tf.app.run()
