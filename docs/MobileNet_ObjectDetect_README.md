# MobileNet based CNN training and inference for Object detect

### Pre-requisites
* Please go throuhg the prequisites for object detection given here [VOC0712 Object Detection](VOC0712_ObjectDetect_README.md).
* As explained in the link above, please go through [Caffe SSD Object Detection Training](https://github.com/weiliu89/caffe/tree/ssd) in detail for understanding the Caffe SSD training procedure.

### Introduction
The training procedure is quite similar to any other Caffe SSD based object detection, so go through the documentation in the links above.

MobileNet based networks may not need sparsification as the complexity is already low. Complexity Vs accuracy trade off can be easily achieved by changing the width (number of channels). For example mobilenet-0.5 has roughly 1/4 th complexity of mobilenet-1.0. We suggest to use mobiledetnet-0.5 for low complexity Object Detection.

### Training Execution

* The main training script is located [/scripts/train_mobilenet_object_detection.sh](../scripts/train_mobilenet_object_detection.sh). 

* Two example configurations for two different image resolutions are provided in this script. Please create the LMDB files of the resolution of interest befoer executing the training script.

* Go through this script to understand the expected paths of LMDB files, number of GPUs configured, batch_size used etc. Change these parameters if necessary according to your setup.

* In the bash prompt change directory to the scripts folder and execute the above script.

### Results

The validation accuracy in the form of mean average precision (mAP) is printed in the run.log in the respective folder for each stage. 

|Model Name       | Image Resolution | Train Dataset | Val Dataset | mAP     | Notes     |
| ---             | ---              | ---           | ---         | ---     | ---       |
|mobiledetnet-0.5 | 512x256          | VOV0712       | VOC07       | 62.51   |           |
|mobiledetnet-0.5 | 512x512          | VOV0712       | VOC07       | TODO    |           |


