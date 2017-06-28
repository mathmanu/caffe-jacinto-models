# Caffe-jacinto
###### Caffe-jacinto - embedded deep learning framework

Caffe-jacinto-models provides example scripts for training sparse models using [tidsp/caffe-jacinto](https://github.com/tidsp/caffe-jacinto), which in-turn is derived from [BVLC/Caffe](https://github.com/BVLC/caffe). These scripts enable training of sparse CNN models - resulting in low complexity models that can be used in embedded platforms. 

For example, the semantic segmentation example (see below) shows how to train a model that is nearly 80% sparse (only 20% non-zero coefficients) and 8-bit quantized. This reduces the complexity of convolution layers by <b>5x</b>. An inference engine designed to efficiently take advantage of sparsity can run <b>significantly faster</b> by using such a model. 

Care has to be taken to strike the right balance between quality and speedup. We have obtained more than 4x overall speedup for CNN inference on embedded device by applying sparsity. Since 8-bit multiplier is sufficient (instead of floating point), the speedup can be even higher on some platforms.

### Installation
* After cloning this repository, switch to the branch caffe-0.16, if it is not checked out already.
-- *git checkout caffe-0.16*

* Please see the [installation instructions](INSTALL.md) for installing the dependencies and building the code. 

### Features

Note that Caffe-jacinto-models does not directly support any embedded/low-power device. But the models trained by it can be used for fast inference on such a device due to the sparsity and quantization.

### Examples
###### Semantic segmentation:
* Note that ImageNet training (see below) is recommended before doing this segmentation training to create the pre-trained weights. The segmentation training will read the ImageNet trained caffemodel for doing the fine tuning on segmentation. However it is possible to directly do segmentation training without ImageNet training, but the quality might be inferior.
* [Train sparse, quantized CNN for semantic segmentation](examples/tidsp/docs/Cityscapes_Segmentation_README.md) on the cityscapes dataset. Inference script is also provided to test out the final model.

###### Classification:
* [Training on ILSVRC ImageNet dataset](examples/tidsp/docs/Imagenet_Classification_README.md). The 1000 class ImageNet trained weights is useful for fine tuning other tasks.
* [Train sparse, quantized CNN on cifar10 dataset](examples/tidsp/docs/Cifar10_Classification_README.md) for classification. Note that this is just a toy example and no inference script is provided to test the final model.

