#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
model_name=jacintonet11v2
dataset=cifar10
folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#------------------------------------------------
caffe="../../caffe-jacinto/build/tools/caffe.bin"

#-------------------------------------------------------
gpus="0,1,2"
max_iter=64000
base_lr=0.1
type=SGD
batch_size=64
stride_list="[1,1,2,1,2]"
#-------------------------------------------------------
solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000}"

#-------------------------------------------------------
#initial training from scratch
stage="initial"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':None,\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"
config_name_prev=$config_name

#-------------------------------------------------------
#incremental sparsification and finetuning
stage="sparse"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

max_iter=64000
stepvalue1=32000
stepvalue2=56000
type=SGD
base_lr=0.01

sparse_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'sparse_mode':1,'display_sparsity':1000,\
'sparsity_target':0.8,'sparsity_start_iter':0,'sparsity_start_factor':0.0,\
'sparsity_step_iter':1000,'sparsity_step_factor':0.02}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name


#-------------------------------------------------------
#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50,\
'caffe':'$caffe test'}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param
config_name_prev=$config_name


