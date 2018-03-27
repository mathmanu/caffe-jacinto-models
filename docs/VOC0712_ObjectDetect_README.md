# Sparse CNN training and inference for Object detect

### Pre-requisites
It is assumed here, that all the pre-requisites required for running Caffe-jacinto are met. Open a bash terminal and change directory into the scripts folder, as explained earlier.

### Dataset preparation
We use the same LMDB format as used by original [Caffe-SSD implementation](https://github.com/weiliu89/caffe/blob/4817bf8b4200b35ada8ed0dc378dceaf38c539e4/README.md#citing-ssd). 


### Training Execution

* The main training script is located [../scripts/train_image_object_detection.sh](../scripts/train_image_object_detection.sh). 

* There are three example configurations provided in the script, one for PASCAL VOC0712 and other two for custom datasets.
* Appropriate dataset can be set at this [location](https://github.com/tidsp/caffe-jacinto-models/blob/79621dde7528bb33f4740fb9a760162b15ec2fd6/scripts/train_image_object_detection.sh#L12). 
* For custom dataset the following parameters need to be set to appropriate values,
* > train_data,   test_data,   name_size_file,   label_map_file,   num_test_image and 
  num_classes. 
* Also solver params need to be set based on the size of one epoch in the dataset.
* Look at gpus variable at this [location](https://github.com/tidsp/caffe-jacinto-models/blob/79621dde7528bb33f4740fb9a760162b15ec2fd6/scripts/train_image_object_detection.sh#L8). This should reflect the number of gpus that you have. For example, if you have two NVIDIA CUDA supported gpus, the gpus variable should be set to "0,1". If you have more GPUs, modify this field to reflect it so that the training will complete faster.

* Execute the training by running the training script, 
* > ./train_image_object_detection.sh. 

* There are three stages in this training.

	**Stage-1: Initial stage with L2 regularization training**
    
    Uses imagenet pre-trained model and trains it for object detect task for the dataset set earlier. For PASCAL VCOC0712, this stage runs for 120k iteration which approximately takes 20 hrs on 2 GTX 1080 GPUs. The trained model is stored at ./training/dataset/model_name/folder_name/initial/. The folder_name is specified at this [location](https://github.com/tidsp/caffe-jacinto-models/blob/79621dde7528bb33f4740fb9a760162b15ec2fd6/scripts/train_image_object_detection.sh#L13). Similarly dataset and model_name are specified in the file, [./train_image_object_detection.sh](https://github.com/tidsp/caffe-jacinto-models/blob/79621dde7528bb33f4740fb9a760162b15ec2fd6/scripts/train_image_object_detection.sh)
	
	**Stage-2: L1 regularization training**
    This stage fine tunes stage-1 trained model to make CNN n/w amenable for sparsification.The trained model is stored at ./training/dataset/model_name/folder_name/l1reg/.

	**Stage-3: Sparsification training** 
    This stage starts with trained model in stage-2 and induces sparsity gradually. The config parameters can be adjusted to achieve desired level of sparsity at this [location](https://github.com/tidsp/caffe-jacinto-models/blob/79621dde7528bb33f4740fb9a760162b15ec2fd6/scripts/train_image_object_detection.sh#L183-L184).The trained model is stored at ./training/dataset/model_name/folder_name/sparse/.
  

### Results

The validation accuracy in the form of mean average precision (mAP) is printed in the run.log in the respective folder for each stage. 

|Configuration-Dataset VOC0712                    |mAP        |
| :---                                            |  :---:    |
|Initial L2 regularized training                  |  68.66%   |
|L1 regularized fine tuning                       |  68.07%   |
|Sparse fine tuned(nearly 61% zero coefficients)  |  65.77%   |
|<b>Overall impact due to sparseness              |   2.89%   |


* 61.1% sparsity (i.e. zero coefficients in convolution weights) implies that the complexity of inference can be potentially reduced by 2.5x - by using a suitable sparse convolution implementation.

* It is possible to change the value of sparsity applied - see the training script for more details.

### Pre-trained Model
*The pre-trained models are made available for PASCAL VOC0712 and TI Internal automotive dataset.

* **PASCAL VOC0712**: [SSD512x512(L2)](../trained/object_detection/voc0712/JDetNet/ssd512x512_ds_PSP_dsFac_32_fc_0_hdDS8_1_kerMbox_3_1stHdSameOpCh_1/initial/voc0712_ssdJacintoNetV2_iter_106000.caffemodel), [SSD512x512(Sparsed)](../trained/object_detection/voc0712/JDetNet/ssd512x512_ds_PSP_dsFac_32_fc_0_hdDS8_1_kerMbox_3_1stHdSameOpCh_1/sparse/voc0712_ssdJacintoNetV2_iter_48000.caffemodel)
* **TI, Auto Dataset**: [SSD720x368](../trained/object_detection/ti-720x368/JDetNet/ssd720x368_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/initial/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_10000.caffemodel), [SSD720x368(Sparsed)](../trained/object_detection/ti-720x368/JDetNet/ssd720x368_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/sparse/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_38000.caffemodel), [SSD768x320](../trained/object_detection/ti-720x368/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/initial/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_40000.caffemodel), [SSD768x320(Sparsed)](../trained/object_detection/ti-720x368/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/sparse/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_46000.caffemodel)


### Inference using the trained model
The script to run trained model through video files can be executed by the following simple commands.
* >cd _$root/scripts/_
* >python ./infer_video_object.py
* Set _caffe_root_ path to folder pointing to _caffe_jacinto_ in _./infer_video_object.py_. 
* The path of the input video needs to be updated along with video names by updating _dataset_ at the location [].
* Output videos with detected objects are stored at the path provided by, params.OpPath at location [].
* Detected outputs are stored in the text files too.    
