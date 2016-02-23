# MSRA's ResNet in TensorFlow

ResNet was the winner of ILSVRC 2015. It's currently (Feb 2016) the most
accurate image classification model.

The authors have published pretrained models for Caffe https://github.com/KaimingHe/deep-residual-networks
This is a script to convert those exact models for use in TensorFlow.

Running the conversion script, of course, depends on both Caffe and TensorFlow both being installed.
You will also need to downlaod the Caffe pretrained models from [here](https://github.com/KaimingHe/deep-residual-networks).
Running the converted model only depends on TensorFlow. You can download it with BitTorrent:

https://github.com/ry/tensorflow-resnet/raw/master/resnet-50-20160223.tfmodel.torrent (138M)

The forward.py script shows how to use it.

