import os
import re
import numpy as np
import skimage.io
import skimage.transform
from os.path import isfile, join

IMAGE_SIZE = 224

def save_image(path, a):
  skimage.io.imsave(path, a)

# Returns a numpy array of shape [height, width, 3]
def load_image(path):
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
  # resize to IMAGE_SIZE, IMAGE_SIZE

  img = skimage.transform.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))

  # if it's a black and white photo, we need to change it to 3 channel
  # or raise an error if we're not allowing b&w (which we do during training)
  if len(img.shape) == 2:
    img = np.stack([img, img, img], axis=-1)

  assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)

  return img

label_map = {}
with open("synset.txt") as f:
  i = 0
  for line in f:
    code, _ = line.split(" ", 1)
    label_map[code] = i
    i += 1
# n01440764/n01440764_10066.JPEG
imagenet_filename_pattern = re.compile(".*(n\d+)_\d+.JPEG")

print "label_map", len(label_map)

class DataSet(object):
  def __init__(self, dir_txt):
    self.data_dir = os.path.splitext(dir_txt)[0]
    self._examples = []

    with open(dir_txt, 'r') as f:
      for line in f:
        if line[0] == '.': continue
        m = imagenet_filename_pattern.match(line)
        if not m: continue

        line = line.rstrip()
        fn = os.path.join(self.data_dir, line)

        label = m.groups()[0]
        label_index = label_map[label]

        self._examples.append((fn, label_index, label))

    assert len(self._examples) > 0

    self._epochs_completed = 0
    self._index_in_epoch = 0

    self._shuffle()

  def _shuffle(self):
    np.random.shuffle(self._examples)

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def epoch_complete(self):
    # Finished epoch
    self._epochs_completed += 1
    # Shuffle the data
    self._shuffle()
    # Start next epoch
    self._index_in_epoch = 0

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    assert batch_size <= len(self._examples)

    imgs = []
    labels = []
    while len(imgs) < batch_size:
      try:
        fn, label_index, _ = self._examples[self._index_in_epoch]
        img = load_image(fn)
        imgs.append(img)
        labels.append(label_index)
        self._index_in_epoch += 1 
        if self._index_in_epoch >= len(self._examples):
          self.epoch_complete()
      except ValueError:
        print "error loading", fn
        del self._examples[self._index_in_epoch]

    batch = np.stack(imgs)
    assert batch.shape == (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3)

    batch_labels = np.zeros([batch_size, 1000])
    for i in range(0, len(labels)):
      label_index = labels[i]
      batch_labels[i, label_index] = 1.0
    
    return batch, batch_labels

