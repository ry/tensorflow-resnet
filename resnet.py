import tensorflow as tf
import os
import re
import data
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('summary_dir', 'log', 'Where to put the summary logs')
flags.DEFINE_string('images', None, 'where to find training images')


def bn_conv_layer(data, filter_shape, wh_strides):
  out_channels = filter_shape[3]
  filt = tf.Variable(tf.truncated_normal(filter_shape))

  x = tf.nn.conv2d(data, filt, strides=[1, wh_strides, wh_strides, 1], padding="SAME")

  mean, variance = tf.nn.moments(x, [0,1,2]) 

  beta = tf.Variable(tf.zeros([out_channels]), name="beta")
  gamma = tf.Variable(tf.truncated_normal([out_channels]), name="gamma")

  bn = tf.nn.batch_norm_with_global_normalization(
      x, mean, variance, beta, gamma, 0.001,
      scale_after_normalization=True)

  return tf.nn.relu(bn)


def res_3x3_pair(bottom, out_depth, down_sample):
  bottom_depth = bottom.get_shape().as_list()[3]
  
  if down_sample:
    bottom = tf.nn.max_pool(bottom, [1,2,2,1], [1,2,2,1], padding='SAME')

  conv1 = bn_conv_layer(bottom, [3, 3, bottom_depth, out_depth], 1)
  conv2 = bn_conv_layer(conv1, [3, 3, out_depth, out_depth], 1)

  if bottom_depth != out_depth:
    assert down_sample
    # XXX 1x1 convolution here?
    bottom_pad = tf.pad(bottom, [[0,0], [0,0], [0,0], [0, out_depth - bottom_depth]])
  else:
    assert not down_sample
    bottom_pad = bottom

  res = conv2 + bottom_pad

  return res


def model18(rgb):
  with tf.variable_scope("conv1"):
    # [filter_height, filter_width, in_channels, out_channels]
    conv1 = bn_conv_layer(rgb, [7, 7, 3, 64], 2)
    assert conv1.get_shape().as_list()[1:] == [112, 112, 64]

  with tf.variable_scope("conv2"):
    conv2_1 = tf.nn.max_pool(conv1, [1,3,3,1], [1,2,2,1], padding='SAME')
    res2_1 = res_3x3_pair(conv2_1, 64, False)
    conv2 = res_3x3_pair(res2_1, 64, False)
    assert conv2.get_shape().as_list()[1:] == [56, 56, 64]

  with tf.variable_scope("conv3"):
    res3_1 = res_3x3_pair(conv2, 128, True)
    conv3 = res_3x3_pair(res3_1, 128, False)
    assert conv3.get_shape().as_list()[1:] == [28, 28, 128]

  with tf.variable_scope("conv4"):
    res4_1 = res_3x3_pair(conv3, 256, True)
    conv4 = res_3x3_pair(res4_1, 256, False)
    assert conv4.get_shape().as_list()[1:] == [14, 14, 256]

  with tf.variable_scope("conv5"):
    res5_1 = res_3x3_pair(conv4, 512, True)
    conv5 = res_3x3_pair(res5_1, 512, False)
    assert conv5.get_shape().as_list()[1:] == [7, 7, 512]

  avg_pool = tf.reduce_mean(conv5, [1,2])
  assert avg_pool.get_shape().as_list()[1:] == [512]

  fc_weight = tf.Variable(tf.truncated_normal([512, 1000]), name="fc_weight")
  fc_bias = tf.Variable(tf.zeros([1000]), name="fc_bias")
  logits = tf.nn.bias_add(tf.matmul(avg_pool, fc_weight), fc_bias)
  assert logits.get_shape().as_list()[1:] == [1000]

  return logits

def run_training():
  data_set = data.DataSet(FLAGS.images)

  step_var = tf.Variable(0)
  inc_step = step_var.assign_add(1) 

  rgb = tf.placeholder("float", [FLAGS.batch_size, 224, 224, 3])
  labels = tf.placeholder("float", [FLAGS.batch_size, 1000])

  tf.image_summary("summary", rgb)

  logits = model18(rgb)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
  tf.scalar_summary("loss", loss)

  opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  train_op = opt.minimize(loss)
  summary_op = tf.merge_all_summaries()

  saver = tf.train.Saver()
  summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir)
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())

  while True:
    start_time = time.time()

    batch, batch_labels = data_set.next_batch(FLAGS.batch_size) 
    feed_dict = { rgb: batch, labels: batch_labels }

    _, step, loss_value, summary_str = sess.run([train_op, inc_step,
        loss, summary_op], feed_dict=feed_dict)

    if step % 3 == 0:
      summary_writer.add_summary(summary_str, step)
      saver.save(sess, "checkpoint", global_step=step)

    duration = time.time() - start_time

    print('Step %d: loss = %.2f (%.1f sec)' % (step, loss_value, duration))
  # End train loop

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
