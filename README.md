# Caffe-jacinto
###### Caffe-jacinto - embedded deep learning framework

Caffe-jacinto-models provides example scripts for training sparse models using [tidsp/caffe-jacinto](https://github.com/tidsp/caffe-jacinto). These scripts enable training of sparse CNN models - resulting in low complexity models that can be used in embedded platforms. 

For example, the semantic segmentation example shows how to train a model that is nearly 80% sparse (only 20% non-zero coefficients) and 8-bit quantized. This reduces the complexity of convolution layers by nearly <b>5x</b>. An inference engine designed to efficiently take advantage of sparsity can run <b>significantly faster</b> by using such a model. 

Care has to be taken to strike the right balance between quality and speedup. We have obtained more than 4x overall speedup for CNN inference on embedded device by applying sparsity. Since 8-bit multiplier is sufficient (instead of floating point), the speedup can be even higher on some platforms.

### Prerequisite
Please see the documentation of [tidsp/caffe-jacinto](https://github.com/tidsp/caffe-jacinto). The build procedure is same as the building of any other Caffe fork. Make sure that the following is done, before attempting to use the scripts in this directory. 
1.  Clone caffe-jacinto. caffe-jacinto and caffe-jacinto-models should be at the same directory level. For example, if the path to this repository is /user/tomato/work/caffe-jacinto-models, then the path to caffe-jacinto should be /user/tomato/work/caffe-jacinto
2.  Checkout the correct branch
    *git checkout caffe-0.15*
3.  Build caffe-jacinto. Make sure to build the libraries, tools and pycaffe. Make sure that the pycaffe folder (for example:  /user/tomato/work/caffe-jacinto/python) is in your environment variable PYTHONPATH defined in .bashrc. Also make sure that PYTHONPATH starts with a : so that the import of local folders work.
Example:<br>
export PYTHONPATH=:/user/tomato/work/caffe-jacinto/python:$PYTHONPATH

### Installation
* After cloning this repository, switch to the branch caffe-0.15, if it is not checked out already.
-- *git checkout caffe-0.15*

### Features

Note that Caffe-jacinto-models does not directly support any embedded/low-power device. But the models trained by it can be used for fast inference on such a device due to the sparsity and quantization.

### Examples
The scripts for the following examples are provided in the folder caffe-jacinto-models/scripts. Change dierctory into the scripts folder first, before attempting to start training. For example:
cd /user/tomato/work/caffe-jacinto-models/scripts

###### Semantic segmentation:
* Note that ImageNet training (see below) is recommended before doing this segmentation training to create the pre-trained weights. The segmentation training will read the ImageNet trained caffemodel for doing the fine tuning on segmentation. However it is possible to directly do segmentation training without ImageNet training, but the quality might be inferior.
* [Train sparse, quantized CNN for semantic segmentation](docs/Cityscapes_Segmentation_README.md) on the cityscapes dataset. Inference script is also provided to test out the final model.

###### Classification:
* [Training on ILSVRC ImageNet dataset](docs/Imagenet_Classification_README.md). The 1000 class ImageNet trained weights is useful for fine tuning other tasks.
* [Train sparse, quantized CNN on cifar10 dataset](docs/Cifar10_Classification_README.md) for classification. Note that this is just a toy example and no inference script is provided to test the final model.

### Notes
* Quantization is supported in the code. However, it is not enabled by default in the scripts as an improvement is in the pipeline that will enable quantization automatically during test/inference.


