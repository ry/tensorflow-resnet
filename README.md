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

[resnet-50-20160305.tfmodel.torrent](https://github.com/ry/tensorflow-resnet/raw/master/resnet-50-20160305.tfmodel.torrent) (138M)

[resnet-101-20160305.tfmodel.torrent](https://github.com/ry/tensorflow-resnet/raw/master/resnet-101-20160305.tfmodel.torrent) (210M)

[resnet-152-20160305.tfmodel.torrent](https://github.com/ry/tensorflow-resnet/raw/master/resnet-152-20160305.tfmodel.torrent) (270M)

The forward.py script shows how to use it.

## Notes

* The convert.py script checks that activations are similiar to the caffe version
  but it's not exactly the same. This is because TensorFlow handles padding slightly
  differentl.

* Resnet is full convolutional. You can resize the network input down to 65x65 images.

## Error Rates

Unfortunately, it runs less accurately than the published error rates for
Caffe - I suspect it's due to the padding algorithms being slightly different.
Maybe fine-tuning will help. Use `test_error_rate.py` to check:

  model|top-1|top-5
  :---:|:---:|:---:
  [VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http
  TF ResNet-50|27.4%|9.0%
  TF ResNet-101|26.1%|8.1%
  Caffe ResNet-50|24.7%|7.8%
  TF ResNet-152|25.2%|7.2%
  Caffe ResNet-101|23.6%|7.1%
  Caffe ResNet-152|23.0%|6.7%


