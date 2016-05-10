from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf

layers = 50

img = load_image("data/cat.jpg")

sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(layers))
new_saver.restore(sess, checkpoint_fn(layers))

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
images = graph.get_tensor_by_name("images:0")
for op in graph.get_operations():
    print op.name

#init = tf.initialize_all_variables()
#sess.run(init)
print "graph restored"

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}

prob = sess.run(prob_tensor, feed_dict=feed_dict)

print_prob(prob[0])
