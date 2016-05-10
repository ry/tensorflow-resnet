import os
os.environ["GLOG_minloglevel"] = "2"
import sys
import re
import caffe
import numpy as np
import tensorflow as tf
import skimage.io
from caffe.proto import caffe_pb2
from synset import *

import resnet


class CaffeParamProvider():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    def conv_kernel(self, name):
        k = self.caffe_net.params[name][0].data
        # caffe      [out_channels, in_channels, filter_height, filter_width] 
        #             0             1            2              3
        # tensorflow [filter_height, filter_width, in_channels, out_channels]
        #             2              3             1            0
        return k.transpose((2, 3, 1, 0))
        return k

    def bn_gamma(self, name):
        return self.caffe_net.params[name][0].data

    def bn_beta(self, name):
        return self.caffe_net.params[name][1].data

    def bn_mean(self, name):
        return self.caffe_net.params[name][0].data

    def bn_variance(self, name):
        return self.caffe_net.params[name][1].data

    def fc_weights(self, name):
        w = self.caffe_net.params[name][0].data
        w = w.transpose((1, 0))
        return w

    def fc_biases(self, name):
        b = self.caffe_net.params[name][1].data
        return b


def preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    mean_bgr = load_mean_bgr()
    print 'mean blue', np.mean(mean_bgr[:, :, 0])
    print 'mean green', np.mean(mean_bgr[:, :, 1])
    print 'mean red', np.mean(mean_bgr[:, :, 2])
    out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out -= mean_bgr
    return out


def assert_almost_equal(caffe_tensor, tf_tensor):
    t = tf_tensor[0]
    c = caffe_tensor[0].transpose((1, 2, 0))

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
def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


def load_mean_bgr():
    """ bgr mean pixel value image, [0, 255]. [height, width, 3] """
    with open("data/ResNet_mean.binaryproto", mode='rb') as f:
        data = f.read()
    blob = caffe_pb2.BlobProto()
    blob.ParseFromString(data)

    mean_bgr = caffe.io.blobproto_to_array(blob)[0]
    assert mean_bgr.shape == (3, 224, 224)

    return mean_bgr.transpose((1, 2, 0))


def load_caffe(img_p, layers=50):
    caffe.set_mode_cpu()

    prototxt = "data/ResNet-%d-deploy.prototxt" % layers
    caffemodel = "data/ResNet-%d-model.caffemodel" % layers
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
    assert net.blobs['data'].data[0].shape == (3, 224, 224)
    net.forward()

    caffe_prob = net.blobs['prob'].data[0]
    print_prob(caffe_prob)

    return net


# returns the top1 string
def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print "Top1: ", top1
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1


def parse_tf_varnames(p, tf_varname, num_layers):
    if tf_varname == 'scale1/weights':
        return p.conv_kernel('conv1')

    elif tf_varname == 'scale1/gamma':
        return p.bn_gamma('scale_conv1')

    elif tf_varname == 'scale1/beta':
        return p.bn_beta('scale_conv1')

    elif tf_varname == 'scale1/moving_mean':
        return p.bn_mean('bn_conv1')

    elif tf_varname == 'scale1/moving_variance':
        return p.bn_variance('bn_conv1')

    elif tf_varname == 'fc/weights':
        return p.fc_weights('fc1000')

    elif tf_varname == 'fc/biases':
        return p.fc_biases('fc1000')

    # scale2/block1/shortcut/weights
    # scale3/block2/c/moving_mean
    # scale3/block6/c/moving_variance
    # scale4/block3/c/moving_mean
    # scale4/block8/a/beta
    re1 = 'scale(\d+)/block(\d+)/(shortcut|a|b|c|A|B)'
    m = re.search(re1, tf_varname)

    def letter(i):
        return chr(ord('a') + i - 1)

    scale_num = int(m.group(1))

    block_num = int(m.group(2))
    if scale_num == 2:
        # scale 2 always uses block letters
        block_str = letter(block_num)
    elif scale_num == 3 or scale_num == 4:
        # scale 3 uses block letters for l=50 and numbered blocks for l=101, l=151
        # scale 4 uses block letters for l=50 and numbered blocks for l=101, l=151
        if num_layers == 50:
            block_str = letter(block_num)
        else:
            if block_num == 1:
                block_str = 'a'
            else:
                block_str = 'b%d' % (block_num - 1)
    elif scale_num == 5:
        # scale 5 always block letters
        block_str = letter(block_num)
    else:
        raise ValueError("unexpected scale_num %d" % scale_num)

    branch = m.group(3)
    if branch == "shortcut":
        branch_num = 1
        conv_letter = ''
    else:
        branch_num = 2
        conv_letter = branch.lower()

    x = (scale_num, block_str, branch_num, conv_letter)
    #print x

    if 'weights' in tf_varname:
        return p.conv_kernel('res%d%s_branch%d%s' % x)

    if 'gamma' in tf_varname:
        return p.bn_gamma('scale%d%s_branch%d%s' % x)

    if 'beta' in tf_varname:
        return p.bn_beta('scale%d%s_branch%d%s' % x)

    if 'moving_mean' in tf_varname:
        return p.bn_mean('bn%d%s_branch%d%s' % x)

    if 'moving_variance' in tf_varname:
        return p.bn_variance('bn%d%s_branch%d%s' % x)

    raise ValueError('unhandled var ' + tf_varname)


def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers


def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers


def convert(graph, img, img_p, layers):
    caffe_model = load_caffe(img_p, layers)

    #for i, n in enumerate(caffe_model.params):
    #    print n

    param_provider = CaffeParamProvider(caffe_model)

    if layers == 50:
        num_blocks = [3, 4, 6, 3]
    elif layers == 101:
        num_blocks = [3, 4, 23, 3]
    elif layers == 152:
        num_blocks = [3, 8, 36, 3]

    with tf.device('/cpu:0'):
        images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
        logits = resnet.inference(images,
                                  is_training=False,
                                  num_blocks=num_blocks,
                                  preprocess=True,
                                  bottleneck=True)
        prob = tf.nn.softmax(logits, name='prob')

    # We write the metagraph first to avoid adding a bunch of
    # assign ops that are used to set variables from caffe.
    # The checkpoint is written to at the end.
    tf.train.export_meta_graph(filename=meta_fn(layers))

    vars_to_restore = tf.all_variables()
    saver = tf.train.Saver(vars_to_restore)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    assigns = []
    for var in vars_to_restore:
        #print var.op.name
        data = parse_tf_varnames(param_provider, var.op.name, layers)
        #print "caffe data shape", data.shape
        #print "tf shape", var.get_shape()
        assigns.append(var.assign(data))
    sess.run(assigns)

    #for op in tf.get_default_graph().get_operations():
    #    print op.name

    i = [
        graph.get_tensor_by_name("scale1/Relu:0"),
        graph.get_tensor_by_name("scale2/MaxPool:0"),
        graph.get_tensor_by_name("scale2/block1/Relu:0"),
        graph.get_tensor_by_name("scale2/block2/Relu:0"),
        graph.get_tensor_by_name("scale2/block3/Relu:0"),
        graph.get_tensor_by_name("scale3/block1/Relu:0"),
        graph.get_tensor_by_name("scale5/block3/Relu:0"),
        graph.get_tensor_by_name("avg_pool:0"),
        graph.get_tensor_by_name("prob:0"),
    ]

    o = sess.run(i, {images: img[np.newaxis, :]})

    assert_almost_equal(caffe_model.blobs['conv1'].data, o[0])
    assert_almost_equal(caffe_model.blobs['pool1'].data, o[1])
    assert_almost_equal(caffe_model.blobs['res2a'].data, o[2])
    assert_almost_equal(caffe_model.blobs['res2b'].data, o[3])
    assert_almost_equal(caffe_model.blobs['res2c'].data, o[4])
    assert_almost_equal(caffe_model.blobs['res3a'].data, o[5])
    assert_almost_equal(caffe_model.blobs['res5c'].data, o[6])
    #assert_almost_equal(np.squeeze(caffe_model.blobs['pool5'].data), o[7])

    print_prob(o[8][0])

    prob_dist = np.linalg.norm(caffe_model.blobs['prob'].data - o[8])
    print 'prob_dist ', prob_dist
    assert prob_dist < 0.2  # XXX can this be tightened?

    # We've already written the metagraph to avoid a bunch of assign ops.
    saver.save(sess, checkpoint_fn(layers), write_meta_graph=False)


def save_graph(save_path):
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    print "graph_def byte size", graph_def.ByteSize()
    graph_def_s = graph_def.SerializeToString()

    with open(save_path, "wb") as f:
        f.write(graph_def_s)

    print "saved model to %s" % save_path


def main(_):
    img = load_image("data/cat.jpg")
    print img
    img_p = preprocess(img)

    for layers in [50, 101, 152]:
        g = tf.Graph()
        with g.as_default():
            print "CONVERT", layers
            convert(g, img, img_p, layers)


if __name__ == '__main__':
    tf.app.run()
