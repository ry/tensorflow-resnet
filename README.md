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

