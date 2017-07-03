# Sparse, Quantized CNN training for segmentation

### Pre-requisites
It is assumed here, that all the pre-requisites required for running Caffe-jacinto are met. Open a bash terminal and change directory into the scripts folder, as explainded earlier.

### Dataset preparation
The details about how to obtain the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) can be seen from their website. Download and unzip gtFine and leftImg8bit as sub-directories into a suitable folder.

Change directory to the tidsp folder. All the remaining scripts are to be executed from this folder.
* cd $CAFFE_HOME/examples/tidsp

Before training, create list files needed to train on cityscapes dataset.
* Open the file ./tools/create_cityscapes_lists.sh (eg. vi ./tools/create_cityscapes_lists.sh) and change the DATASETPATH to the location where you have downloaded the dataset. Under this folder, the gtFine and leftImg8bit folders of Cityscapes should be present.
* Then execute ./tools/create_cityscapes_lists.sh. This script creates the image and label lists used for training. It also does label transformation. 
* We have chosen a smaller set of 5-classes for training. 32 classes of cityscapes are converted into 5-classes - so the trained model will learn to segment 5-classes (background, road, person, road signs, vehicle). 
* Note: The number of classes and class mappings can be easily changed in this script. The network model prototxt may also need to be changed if the number of classes chosen is more than the output channels in the model.
* Note: this 5-class training is different from the typical [19-class training done for cityscapes](https://github.com/mcordts/cityscapesScripts) and reported on the benchmark website. 


### Execution
* Open the file train_cityscapes_segmentation.sh  and look at the gpus variable. This should reflect the number of gpus that you have. For example, if you have two NVIDIA CUDA supported gpus, the gpus variable should be set to "0,1". If you have more GPUs, modify this field to reflect it so that the training will complete faster.

* Execute the training by running the training script: ./train_cityscapes_segmentation.sh. 

* The training takes around 22 hours, when using one NVIDIA GTX 1080 GPU.

* This script will perform all the stages required to generate a sparse CNN model. The quantized model will be placed in a folder inside scripts/training.

### Results

The validation accuracy is printed in the training log. Following is what we got for the 5-class (background, road, person, road signs, vehicle) training.


|Configuration                                    |Pixel Accuracy  |Mean IOU  |
|-------------------------------------------------|----------------|----------|
|Initial L2 regularized training                  |96.20           |83.23     |
|L1 regularized fine tuning                       |96.32           |<b>83.94  |
|Sparse fine tuned(nearly 80% zero coefficients)  |96.11           |82.85     |
|Sparse(80%), Quantized(8-bit dynamic fixed point)|95.91           |<b>82.15  |
|<b>Overall impact due to sparse+quant            |<b>-0.42        |<b>-1.79  |

* 80% sparsity (i.e. zero coefficients in convolution weights) implies that the complexity of inference can be potentially reduced by 5x - by using a suitable sparse convolution implementation.

* It is possible to change the value of sparsity applied - see the training script for more details.

### Inference using the trained model
* This section explains how the trained model can be used for inference on a PC using Caffe-jacinto.
* Open the file infer_cityscapes_segmentation.sh using a text editor and change the path of deploy model and weights (caffemodel) to the one that is generated in the recent training.
* Run the file infer_cityscapes_segmentation.sh
This will create the output images or video in the location corresponding to the output parameter mentioned in the script.
