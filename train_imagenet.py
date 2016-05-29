import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
from resnet_train import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np

from synset import *
from image_processing import image_preprocessing

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')


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


def distorted_inputs():
    data = load_data(FLAGS.data_dir)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]

    filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    num_preprocess_threads = 4
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        image_buffer = tf.read_file(filename)

        bbox = []
        train = True
        image = image_preprocessing(image_buffer, bbox, train, thread_id)
        images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2 * num_preprocess_threads * FLAGS.batch_size)

    height = FLAGS.input_size
    width = FLAGS.input_size
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    return images, tf.reshape(label_index_batch, [FLAGS.batch_size])


def main(_):
    images, labels = distorted_inputs()

    logits = inference(images,
                       num_classes=1000,
                       is_training=True,
                       bottleneck=False,
                       num_blocks=[2, 2, 2, 2])
    train(logits, images, labels)


if __name__ == '__main__':
    tf.app.run()
