#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
model_name=jacintonet11v2
dataset=imagenet
folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#------------------------------------------------
caffe="../../caffe-jacinto/build/tools/caffe.bin"

#-------------------------------------------------------
gpus="0,1"
max_iter=100 #320000
base_lr=0 #0.1
#threshold_step_factor=1e-6
type=SGD
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter}"

#-------------------------------------------------------
weights='/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.06.new_script/caffe-0.15/jacintonet11_imagenet_2017.06.12_lmdb_caffe-0.15-2gpu(60.89%)/stage0/jacintonet11_iter_320000.caffemodel'

stage="initial"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights'}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"
config_name_prev=$config_name


#-------------------------------------------------------
#incremental sparsification and finetuning
stage="sparse"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

max_iter=160000
#stepvalue1=80000
#stepvalue2=120000
type=SGD
base_lr=0.01  #use a lower lr for fine tuning
sparse_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'sparse_mode':1,'display_sparsity':1000,\
'sparsity_target':0.8,'sparsity_start_iter':0,'sparsity_start_factor':0.0,\
'sparsity_step_iter':1000,'sparsity_step_factor':0.01}"
#'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights'}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name


#-------------------------------------------------------
#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'sparse_mode':1,'display_sparsity':1000}"
#'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':1000,'image_width':224,'image_height':224,'crop_size':224,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/ilsvrc12_train_lmdb','test_data':'./data/ilsvrc12_val_lmdb',\
'num_test_image':50000,'test_batch_size':50,\
'caffe':'$caffe test'}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param
config_name_prev=$config_name


