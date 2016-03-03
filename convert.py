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

import resnet


class CaffeParamProvider():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    # It's unfortunate that it needs all these parameters but due
    # to the bug mentioned below we have to special case the creation of
    # the kernel.
    def conv_kernel(self, name, in_chans, out_chans, shape, strides):
        k =  self.caffe_net.params[name][0].data

        # caffe      [out_channels, in_channels, filter_height, filter_width] 
        #             0             1            2              3
        # tensorflow [filter_height, filter_width, in_channels, out_channels]
        #             2              3             1            0

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
            kernel = tf.constant(k.transpose((2, 3, 1, 0)),  dtype='float32',
                name="kernel")
            kernel_shape = kernel.get_shape().as_list()
            assert kernel_shape[0] == shape
            assert kernel_shape[1] == shape

        assert kernel_shape[2] == in_chans
        assert kernel_shape[3] == out_chans

        return kernel

    def bn_params(self, bn_name, scale_name, depth):
        mean = self.caffe_net.params[bn_name][0].data
        assert mean.shape == (depth,)
        mean = tf.constant(mean, dtype='float32', name='mean')

        var = self.caffe_net.params[bn_name][1].data
        assert var.shape == (depth,)
        var = tf.constant(var, dtype='float32', name='var')

        gamma = self.caffe_net.params[scale_name][0].data
        assert gamma.shape == (depth,)
        gamma = tf.constant(gamma, dtype='float32', name='gamma')

        beta = self.caffe_net.params[scale_name][1].data
        assert beta.shape == (depth,)
        beta = tf.constant(beta, dtype='float32', name='beta')

        return mean, var, gamma, beta

    def fc_params(self, name):
        weights = self.caffe_net.params[name][0].data
        bias = self.caffe_net.params[name][1].data
        w = tf.constant(weights.transpose((1,0)), name="weights", dtype="float32")
        b =  tf.constant(bias, name="bias", dtype="float32")
        return w, b


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

def load_mean_bgr():
    """ bgr mean pixel value image, [0, 255]. [height, width, 3] """
    with open("ResNet_mean.binaryproto", mode='rb') as f:
        data = f.read()
    blob = caffe_pb2.BlobProto()
    blob.ParseFromString(data)

    mean_bgr = caffe.io.blobproto_to_array(blob)[0]
    assert mean_bgr.shape == (3, 224, 224)

    return mean_bgr.transpose((1,2,0))

def load_caffe(img_p, layers=50):
    caffe.set_mode_cpu()

    prototxt = "ResNet-%d-deploy.prototxt" % layers
    caffemodel = "ResNet-%d-model.caffemodel" % layers
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    net.blobs['data'].data[0] = img_p.transpose((2,0,1))
    assert net.blobs['data'].data[0].shape == (3, 224, 224)
    net.forward()

    caffe_prob = net.blobs['prob'].data[0]
    utils.print_prob(caffe_prob)

    return net


def save_graph(save_path):
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    print "graph_def byte size", graph_def.ByteSize()
    graph_def_s = graph_def.SerializeToString()

    with open(save_path, "wb") as f:
      f.write(graph_def_s)

    print "saved model to %s" % save_path

def convert(graph, img, img_p, layers):
    net = load_caffe(img_p, layers)
    param_provider = CaffeParamProvider(net)
    with tf.device('/cpu:0'):
        images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
        m = resnet.Model(param_provider)
        m.build(images, layers)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())


    i = [
        graph.get_tensor_by_name("conv1/relu:0"),
        graph.get_tensor_by_name("pool1:0"),
        graph.get_tensor_by_name("res2a/relu:0"),
        graph.get_tensor_by_name("res2b/relu:0"),
        graph.get_tensor_by_name("res2c/relu:0"),
        graph.get_tensor_by_name("res3a/relu:0"),
        graph.get_tensor_by_name("res5c/relu:0"),
        graph.get_tensor_by_name("pool5:0"),
        graph.get_tensor_by_name("prob:0"),
    ]

    o = sess.run(i, {
        images: img[np.newaxis,:]
    })

    assert_almost_equal(net.blobs['conv1'].data, o[0])
    assert_almost_equal(net.blobs['pool1'].data, o[1])
    assert_almost_equal(net.blobs['res2a'].data, o[2])
    assert_almost_equal(net.blobs['res2b'].data, o[3])
    assert_almost_equal(net.blobs['res2c'].data, o[4])
    assert_almost_equal(net.blobs['res3a'].data, o[5])
    assert_almost_equal(net.blobs['res5c'].data, o[6])
    assert_almost_equal(net.blobs['pool5'].data, o[7])

    utils.print_prob(o[8][0])

    prob_dist = np.linalg.norm(net.blobs['prob'].data - o[8])
    print 'prob_dist ', prob_dist
    assert prob_dist < 0.2 # XXX can this be tightened?

    save_graph("resnet-%d.tfmodel" % layers)

def main(_):
    img = load_image("cat.jpg")
    img_p = preprocess(img)

    for layers in [50, 101, 152]:
        g = tf.Graph()
        with g.as_default():
            print "CONVERT", layers
            convert(g, img, img_p, layers)


if __name__ == '__main__':
    tf.app.run()
