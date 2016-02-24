import tensorflow as tf
import utils

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

cat = utils.load_image("cat.jpg", size)

with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  print "variables initialized"

  batch = cat.reshape((1, size, size, 3))

  feed_dict = { images: batch }

  prob_tensor = graph.get_tensor_by_name("import/prob:0")
  prob = sess.run(prob_tensor, feed_dict=feed_dict)

utils.print_prob(prob[0])


