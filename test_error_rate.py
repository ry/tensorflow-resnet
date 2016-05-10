import tensorflow as tf
import numpy as np
import time
import re
import os
from synset import *
import skimage.io
import skimage.transform

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_integer('max', 100000, 'max images to test')
flags.DEFINE_integer('crop_size', 224, 'crop size')
flags.DEFINE_integer('resize_size', 256,
                     'short edge of image is resized to this')
flags.DEFINE_boolean('debug', False, 'Print network output')

flags.DEFINE_string(
    'xml_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_bbox_val_v3_CLS_LOC/val',
    'dir with CLS-LOC xml files')
flags.DEFINE_string('data_dir',
                    '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_val',
                    'data dir with images')
flags.DEFINE_string('model', 'resnet-101.tfmodel', 'model file to test')


def load_image(path, crop_size=224, resize_size=256):
    img = skimage.io.imread(path)
    img = img / 255.0

    short_edge = min(img.shape[:2])
    img = skimage.transform.resize(img, (resize_size, resize_size))

    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    img = img[yy:yy + crop_size, xx:xx + crop_size]

    if len(img.shape) == 2:
        # if it's a black and white photo, we need to change it to 3 channel
        img = np.stack([img, img, img], axis=-1)

    assert img.shape == (crop_size, crop_size, 3)

    return img

#
#def load_image(path, size=224):
#    # load image
#    img = skimage.io.imread(path)
#    img = img / 255.0
#    assert (0 <= img).all() and (img <= 1.0).all()
#    #print "Original Image Shape: ", img.shape
#    # we crop image from center
#    short_edge = min(img.shape[:2])
#    yy = int((img.shape[0] - short_edge) / 2)
#    xx = int((img.shape[1] - short_edge) / 2)
#    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
#    # resize to 224, 224
#    img = skimage.transform.resize(crop_img, (size, size))
#
#    if len(img.shape) == 2:
#        # if it's a black and white photo, we need to change it to 3 channel
#        img = np.stack([img, img, img], axis=-1)
#
#    return img


def get_label(img_fn):
    base = os.path.splitext(img_fn)[0]

    xml_fn = os.path.join(FLAGS.xml_dir, base + ".xml")

    with open(xml_fn, mode='r') as f:
        xml_data = f.read()

    label_name = re.search(r'(n\d+)', xml_data).group(1)

    #print xml_fn
    #print label_name

    return label_name


def load_data():
    data = []
    i = 0
    for img_fn in os.listdir(FLAGS.data_dir):
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = get_label(img_fn)
        fn = os.path.join(FLAGS.data_dir, img_fn)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

        i += 1

    return data


def main(_):
    data = load_data()
    np.random.shuffle(data)

    with open(FLAGS.model, mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float", [None, FLAGS.crop_size, FLAGS.crop_size,
                                      3])

    tf.import_graph_def(graph_def, input_map={"images": images})
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

        imgs = [load_image(d['filename'], FLAGS.crop_size, FLAGS.resize_size)
                for d in data_batch]
        #fns = [ d['filename'] for d in data_batch ]
        #print fns
        batch = np.stack(imgs)
        assert batch.shape == (FLAGS.batch_size, FLAGS.crop_size,
                               FLAGS.crop_size, 3)
        feed_dict = {images: batch}

        prob = sess.run(prob_tensor, feed_dict=feed_dict)

        for i in range(0, FLAGS.batch_size):
            pred = np.argsort(prob[i])[::-1]
            d = data_batch[i]
            in_top1 = (pred[0] == d['label_index'])
            in_top5 = d['label_index'] in pred[0:5]

            if FLAGS.debug:
                print d['filename']
                print "truth", d['desc']
                print "prediction", synset[pred[0]]
                print "in_top1", in_top1
                print "in_top5", in_top5
                print ""

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
