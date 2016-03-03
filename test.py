import tensorflow as tf
import numpy as np
import time
import utils
import re
import os
from synset import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('max', 100000, 'max images to test')

def load_data():
    dir_txt = "/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train.txt"
    data_dir = os.path.splitext(dir_txt)[0]
    data = []
    # data contain the label "n01440764/n01440764_10048.JPEG"
    # so we need to extract "n01440764"
    pattern = r'(n\d+)_'
    with open(dir_txt, mode='r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)

            label_name = re.search(pattern, fn).group(1)

            data.append({
                "filename": fn,
                "label_name": label_name,
                "label_index": synset_map[label_name]["index"],
                "desc": synset_map[label_name]["desc"],
            })
    return data

def main(_):
    data = load_data()
    np.random.shuffle(data)

    # size = 65 # minimum input size
    size = 224 # training size

    with open("resnet-152.tfmodel", mode='rb') as f:
      fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float", [None, size, size, 3])

    tf.import_graph_def(graph_def, input_map={ "images": images })
    print "graph loaded from disk"

    graph = tf.get_default_graph()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    prob_tensor = graph.get_tensor_by_name("import/prob:0")

    top1_wrong = 0
    top5_wrong = 0
    total = 0

    while total < FLAGS.max or total + FLAGS.batch_size > len(data):
        start_time = time.time()

        data_batch = data[total:FLAGS.batch_size + total]

        imgs = [ utils.load_image(d['filename'], size) for d in data_batch ] 
        #fns = [ d['filename'] for d in data_batch ] 
        #print fns
        batch = np.stack(imgs)
        assert batch.shape == (FLAGS.batch_size, size, size, 3)
        feed_dict = { images: batch }

        prob = sess.run(prob_tensor, feed_dict=feed_dict)

        for i in range(0, FLAGS.batch_size):
            pred = np.argsort(prob[i])[::-1]
            d = data_batch[i]
            in_top1 = (pred[0] == d['label_index'])
            in_top5 = d['label_index'] in pred[0:5]

            #print d['filename']
            #print "in_top1", in_top1
            #print "in_top5", in_top5

            if not in_top1: top1_wrong += 1
            if not in_top5: top5_wrong += 1
            total += 1

        top5_error = 100.0 * top5_wrong / total
        top1_error = 100.0 * top1_wrong / total

        duration = time.time() - start_time
        duration_per_img = duration / FLAGS.batch_size
        print('%d top5 = %.1f%% top1 = %.1f%% (%.1f s, %.1f s/img)' %
            (total, top5_error, top1_error, duration, duration_per_img))


if __name__ == '__main__':
    tf.app.run()
