from __future__ import print_function

from datetime import datetime
import tensorflow as tf
import resnet


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          """learning rate.""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")


def loss(inputs, lables):
    logits = resnet.inference(inputs)
    resnet.loss(logits, labels, batch_size=FLAGS.batch_size)
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

def train(dataset):
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    loss = loss(inputs, labels)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9)
    apply_gradient_op = opt.minimize(loss, global_step=global_step)

    # Group all updates to into a single train op.
    batchnorm_updates = tf.get_collection(
        slim.ops.UPDATE_OPS_COLLECTION, scope)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_summary(summaries)

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    if FLAGS.pretrained_model_checkpoint_path:
        assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
        print('%s: Pre-trained model restored from %s' %
              (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

    summary_writer = tf.train.SummaryWriter(
        FLAGS.train_dir,
        graph_def=sess.graph.as_graph_def(add_shapes=True))

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
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



