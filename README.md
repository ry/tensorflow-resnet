# Pre-trained ResNet Models in TensorFlow

ResNet was the winner of ILSVRC 2015. It's currently (2/2016) the most accurate
image classification model.

The authors of ResNet have
[published](https://github.com/KaimingHe/deep-residual-networks) pre-trained
models for Caffe. This is a script to convert those exact models for use in
TensorFlow.

Running the conversion script, of course, depends on both Caffe and TensorFlow
both being installed.  You will also need to downlaod the Caffe pretrained
models from [here](https://github.com/KaimingHe/deep-residual-networks).
Running the converted model only depends on TensorFlow. Download it
with BitTorrent:

[resnet-50-20160223.tfmodel.torrent](https://github.com/ry/tensorflow-resnet/raw/master/resnet-50-20160223.tfmodel.torrent) (138M)

[resnet-101-20160223.tfmodel.torrent](https://github.com/ry/tensorflow-resnet/raw/master/resnet-101-20160223.tfmodel.torrent) (210M)

[resnet-152-20160223.tfmodel.torrent](https://github.com/ry/tensorflow-resnet/raw/master/resnet-152-20160223.tfmodel.torrent) (270M)

The forward.py script shows how to use it.

## Notes

* The convert.py script checks that activations are similiar to the caffe version
  but it's not exactly the same. This is because TensorFlow handles padding slightly
  differently and because I preprocess by subtracting channel level means instead of 
  pixel means.

* I wanted to experiement with resizing the network to handle input other than
  224 x 224 images. Therefore I do not use the pixel-wise means in preprocessing
  but instead channel means. You can resize input down to 65x65 images.

* I have not yet checked that the error rates are the same as MSRA's model. I
  will do this soon and update.
