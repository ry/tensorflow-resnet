import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import resnet
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np

from synset import *
from image_processing import distorted_inputs

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/imagenet-tf-records/',
                           'imagenet dir')


class DataSet:
    def __init__(self, data_dir):
        self.subset = 'train'

    def reader(self):
        return tf.TFRecordReader()

    def data_files(self):
        tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s/%s at %s' %
                  (self.name, self.subset, FLAGS.data_dir))
            sys.exit(-1)
        return data_files


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames


def load_data(data_dir):
    data = []
    i = 0

    print "listing files in", data_dir
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    print "took %f sec" % duration

    for img_fn in files:
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = re.search(r'(n\d+)', img_fn).group(1)
        fn = os.path.join(data_dir, img_fn)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

    return data


# Returns a numpy array of shape [size, size, 3]
def load_image(path, size):
    img = skimage.io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]

    img = skimage.transform.resize(crop_img, (size, size))

    # if it's a black and white photo, we need to change it to 3 channel
    # or raise an error if we're not allowing b&w (which we do during training)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    assert img.shape == (size, size, 3)
    assert (0 <= img).all() and (img <= 1.0).all()

    return img


def main(_):
    dataset = DataSet(FLAGS.data_dir)
    images, labels = distorted_inputs(dataset, FLAGS.batch_size)
    resnet.train(images, labels)


if __name__ == '__main__':
    tf.app.run()
