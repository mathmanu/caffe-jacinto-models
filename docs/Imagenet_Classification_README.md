# ILSVRC ImageNet training

ImageNet training is a pre-requisite before doing several other training tasks. This will create pre-trained weights that can be fine tuned for a variety of other tasks.

Here a low complexity model called JacintoNet11 is used as an example to demonstrate ImageNet training.

### Dataset preparation

* Open a bash terminal and change directory into the scripts folder, as explainded earlier.

* The following website gives details of how to obtain the ImageNet dataset and organize the data: 
https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data

* The above webpage also explains how to create lmdb database. It can also be created by executing  ./tools/utils/create_imagenet_classification_lmdb.sh. Before executing, open this file and modify the DATA field to point to the location where ImageNet train and val folders are placed.

* After creating the lmdb database, make sure that ilsvrc12_train_lmdb and ilsvrc12_val_lmdb folders are present in ./data folder. (If they are not there, you can either move them or create soft links there)

### Training 
* Open the file train_imagenet_classification.sh  and look at the gpus variable. This should reflect the number of gpus that you have. For example, if you have two NVIDIA CUDA supported gpus, the gpus variable should be set to "0,1". If you have more GPUs, modify this field to reflect it so that the training will complete faster.

* Execute the script ./tools/utils/train_imagenet_classification.sh to do the ImageNet training. This will take several hours or days, depending on your GPU configuration. We use polynomial learning rate with 320,000 iterations and an effective batch size of 256, as in [1].

* The training takes around 32 hours when using one NVIDIA GTX 1080 GPU.

* This script will perform all the stages required to generate a sparse CNN model. The final model will be placed in a folder inside scripts/training.

### Results 

###### Validation accuracy and complexity 
Complexity is reported in GigaMACS for one image crop. Size of the crop is 224x224 for all networks except AlexNet. AlexNet uses 227x227.

As can be seen below, JacintoNet11 provides better accuracy compared to BVLC-AlexNet at lower complexity. It also has lower complexity compared to ResNet10.

|Configuration      |Top-1 accuracy   | Top-5 accuracy |Complexity for 1000 classes|
|-------------------|-----------------|----------------|---------------------------|
|<b>JacintoNet11    |<b>60.91         |<b>83.05        |<b>0.410                   |
|ResNet10 [1]       |63.9             |85.2            |0.910                      |
|BVLC AlexNet [2]   |57.1             |80.2            |0.687                      |


[1] "ImageNet pre-trained models with batch normalization", https://arxiv.org/pdf/1612.01452.pdf, https://github.com/cvjena/cnn-models <br>
[2] "BVLC/caffe/models/bvlc_alexnet" https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
