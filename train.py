import skimage.io # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.training import moving_averages
import resnet
from dataset import DataSet
import time
import datetime
import numpy as np
import os

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch size")
tf.app.flags.DEFINE_integer('input_size', 224, "input image size")
tf.app.flags.DEFINE_boolean('continue', False, 'resume from latest saved state')


def train(dataset):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    labels = tf.placeholder("int32", [None, 1], name="labels")
    images = tf.placeholder("float",
                            [None, FLAGS.input_size, FLAGS.input_size, 3],
                            name="images")
    tf.image_summary('images', images)


    logits = resnet.inference(images,
                              num_classes=1000,
                              is_training=True,
                              preprocess=True,
                              bottleneck=False,
                              num_blocks=[2,2,2,2])

    loss = resnet.loss(logits, labels, batch_size=FLAGS.batch_size)
    tf.scalar_summary('loss', loss)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(resnet.MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(resnet.UPDATE_OPS_COLLECTION, ema.apply([loss]))
    loss_avg = ema.average(loss)
    tf.scalar_summary('loss_avg', loss_avg)

    tf.scalar_summary('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss)
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.__getattr__('continue'):
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "continue", latest
        saver.restore(sess, latest)

    while True:
        start_time = time.time()

        images_, labels_ = dataset.get_batch(FLAGS.batch_size, FLAGS.input_size)

        step = sess.run(global_step)
        i = [train_op, loss]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i, {
            images: images_,
            labels: labels_,
        })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

def main(_):
    dataset = DataSet(FLAGS.data_dir)
    train(dataset)

if __name__ == '__main__':
    tf.app.run()
