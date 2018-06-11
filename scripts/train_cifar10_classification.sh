#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#------------------------------------------------
gpus="0,1,2"          #IMPORTANT: change this to "0" if you have only one GPU

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
#l1 regularized training before sparsification
stage="l1reg"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

max_iter=64000
base_lr=0.01
type=SGD

l1reg_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'regularization_type':'L1','weight_decay':1e-5}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$l1reg_solver_param
config_name_prev=$config_name


#-------------------------------------------------------
#incremental sparsification and finetuning
stage="sparse"
#Using more than one GPU for this step gives strange results. Imbalanced accuracy between the GPUs.
gpus="0" #"0,1,2"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

max_iter=64000
base_lr=0.01
type=SGD

sparse_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000,\
'sparsity_target':0.8,'sparsity_start_iter':4000,'sparsity_start_factor':0.0,\
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
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50,\
'caffe_cmd':'test'}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param
#config_name_prev=$config_name


#-------------------------------------------------------
#test_quantize
stage="test_quantize"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'test_interval':1000,\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'stride_list':$stride_list,'pretrain_model':'$weights',\
'num_output':10,'image_width':32,'image_height':32,'crop_size':32,\
'accum_batch_size':$batch_size,'batch_size':$batch_size,\
'train_data':'./data/cifar10_train_lmdb','test_data':'./data/cifar10_test_lmdb',\
'num_test_image':10000,'test_batch_size':50,\
'caffe_cmd':'test'}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param

echo "quantize: true" > $config_name/deploy_new.prototxt
cat $config_name/deploy.prototxt >> $config_name/deploy_new.prototxt
mv --force $config_name/deploy_new.prototxt $config_name/deploy.prototxt

echo "quantize: true" > $config_name/test_new.prototxt
cat $config_name/test.prototxt >> $config_name/test_new.prototxt
mv --force $config_name/test_new.prototxt $config_name/test.prototxt

#config_name_prev=$config_name


#-------------------------------------------------------
#run
list_dirs=`command ls -d1 "$folder_name"/*/ | command cut -f3 -d/`
for f in $list_dirs; do "$folder_name"/$f/run.sh; done

