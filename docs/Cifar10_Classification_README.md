# Sparse, Quantized CNN training for classification

In this section, cifar10 dataset is used as an example to explain the training procedure. This is  a toy example to do the flow flushing of the entire training procedure. No inference script is provided to test the resulting model.

Change directory to the scripts folder. All the remaining scripts are to be executed from this folder.
* cd scripts

### Dataset preparation

Fetch the cifar10 dataset by executing:
* ./tools/utils/get_cifar10.sh

Create LMDB folders for the cifar10 dataset by executing:
* ./tools/utils/create_cifar10.sh

### Execution

Execute the script:
* Open the file train_cifar10_classification.sh  and look at the gpus variable. This should reflect the number of gpus that you have. For example, if you have two NVIDIA CUDA supported gpus, the gpus variable should be set to "0,1". If you have more GPUs, modify this field to reflect it so that the training will complete faster.

* Finally execute the script.  
train_cifar10_classification.sh

This script will perform all the stages required to generate a sparse CNN model. The final model will be placed in a folder inside scripts/training.


