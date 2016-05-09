from synset import *
import time
import os
import re
import numpy as np
import skimage.io
import skimage.transform

class DataSet:
    def __init__(self, data_dir):
        self.data = load_data(data_dir)
        np.random.shuffle(self.data)
        self.index = 0
        self.epochs_completed = 0

    def epoch_complete(self, batch_size):
        np.random.shuffle(self.data)
        self.index = 0
        self.epochs_completed += 1

    def get_batch(self, batch_size, input_size):
        imgs = []
        labels = []
        while len(imgs) < batch_size:
            try:
                fn = self.data[self.index]['filename']
                img = load_image(fn, input_size)
                imgs.append(img)
                labels.append(self.data[self.index]['label_index'])
                self.index += 1
                if self.index >= len(self.data):
                    self.epoch_complete()

            except ValueError:
                del self.data[self.index]

        batch_images = np.stack(imgs)
        assert batch_images.shape == (batch_size, input_size, input_size, 3)

        batch_labels = np.asarray(labels).reshape((batch_size, 1))
        assert batch_labels.shape == (batch_size, 1)

        return batch_images, batch_labels


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames

def load_data(data_dir):
    data = []
    i = 0

    print "listing files in", data_dir
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    print "took %f sec" % duration

    for img_fn in files:
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = re.search(r'(n\d+)', img_fn).group(1)
        fn = os.path.join(data_dir, img_fn)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

    return data


# Returns a numpy array of shape [size, size, 3]
def load_image(path, size):
    img = skimage.io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]

    img = skimage.transform.resize(crop_img, (size, size))

    # if it's a black and white photo, we need to change it to 3 channel
    # or raise an error if we're not allowing b&w (which we do during training)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    assert img.shape == (size, size, 3)
    assert (0 <= img).all() and (img <= 1.0).all()

    return img
