import tensorflow as tf
import resnet
from dataset import DataSet
import time

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 2, "batch size")


def train(dataset):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    num_categories = 1000

    images = tf.placeholder("float", [None, 224, 224, 3], name="images")
    labels = tf.placeholder("int32", [None, 1], name="labels")

    logits = resnet.inference(images, num_classes=1000, is_training=True,
                              num_blocks=[2, 2, 2, 2])

    loss = resnet.loss(logits, labels, batch_size=FLAGS.batch_size)
       

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

    sess = tf.Session()
    sess.run(init)
    
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    while True: 
        start_time = time.time()
        images_, labels_ = dataset.get_batch(FLAGS.batch_size)
        feed_dict = {
            images: images_,
            labels: labels_,
        }
        _, loss_value = sess.run([train_op, loss], feed_dict)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
  
        if step % 10 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                              examples_per_sec, duration))
  
        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
  
        # Save the model checkpoint periodically.
        if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

def main(_):
    dataset = DataSet(FLAGS.data_dir)
    train(dataset)

if __name__ == '__main__':
    tf.app.run()
