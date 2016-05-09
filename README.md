# ResNet Model in TensorFlow

This is the second version of my ResNet implementation.

## Goals

* Be able to use the pre-trained model's that [Kaiming He has provided for
  Caffe](https://github.com/KaimingHe/deep-residual-networks). The `convert.py`
  will convert the weights for use with TensorFlow.

* Implemented in the style of
  [Inception](https://github.com/tensorflow/models/tree/master/inception/inception)
  not using any classes and making heavy use of variable scope. It should be
  easily usable in other models.

* Expierment with changes to ResNet like [stochastic
  depth](https://arxiv.org/abs/1603.09382), [shared weights at each
  scale](https://arxiv.org/abs/1604.03640), and 1D convolutions for audio.

* ResNet is fully convolutional and the implementation should allow inputs to
  be any size.

## Pretrained Model

Instead of running `convert.py`, which depends on Caffe, you can download the converted model thru BitTorrent:

[tensorflow-resnet-pretrained-20160509.tar.gz.torrent](https://raw.githubusercontent.com/ry/tensorflow-resnet/master/data/tensorflow-resnet-pretrained-20160509.tar.gz.torrent)  464M


## Notes

* This code depends on [TensorFlow git commit
  cf7ce8](https://github.com/tensorflow/tensorflow/commit/cf7ce8a7879b6a7ba90441724ea3f8353917a515)
  or later because ResNet needs 1x1 convolutions with stride 2. TF 0.8 is not new
  enough.

* The `convert.py` script checks that activations are similiar to the caffe version
  but it's not exactly the same. This is probably due to differences between how
  TF and Caffe handle padding. Also preprocessing is done with color-channel means 
  instead of pixel-wise means.

* ResNet is full convolutional. You can resize the network input down to 65x65 images.
