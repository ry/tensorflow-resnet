import skimage
import skimage.io
import skimage.transform
import numpy as np
from synset import *

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
  # load image
  img = skimage.io.imread(path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  img = skimage.transform.resize(crop_img, (size, size))

  if len(img.shape) == 2:
    # if it's a black and white photo, we need to change it to 3 channel
    img = np.stack([img, img, img], axis=-1)

  return img

# returns the top1 string
def print_prob(prob):
  #print prob
  pred = np.argsort(prob)[::-1]

  # Get top1 label
  top1 = synset[pred[0]]
  print "Top1: ", top1
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print "Top5: ", top5
  return top1
