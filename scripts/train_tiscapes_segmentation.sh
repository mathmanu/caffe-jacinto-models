#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
model_name=jsegnet21v2
dataset=cityscapes5
folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


#------------------------------------------------
gpus="0,1,2"
max_iter=32000
stepvalue=24000
base_lr=1e-4
use_image_list=0 #known issue - use_image_list=0 && shuffle=1 => hang.
shuffle=0        #Note shuffle is used only in training

solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue]}"

#------------------------------------------------
#Download the pretrained weights
weights_dst="training/imagenet_jacintonet11v2_iter_320000.caffemodel"
if [ -f $weights_dst ]; then
  echo "Using pretrained model $weights_dst"
else
  weights_src="https://github.com/tidsp/caffe-jacinto-models/blob/caffe-0.15/trained/image_classification/imagenet_jacintonet11v2/initial/imagenet_jacintonet11v2_iter_320000.caffemodel?raw=true"
  wget $weights_src -O $weights_dst
fi

#-------------------------------------------------------
#Initial training
stage="initial"
weights=$weights_dst
config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'image_width':1024,'image_height':512}" 

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$solver_param
config_name_prev=$config_name

#-------------------------------------------------------
#l1 regularized training before sparsification
stage="l1reg"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

base_lr=1e-5  #use a lower lr for fine tuning
l1reg_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue],\
'regularization_type':'L1','weight_decay':1e-5}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'image_width':1024,'image_height':512}" 

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$l1reg_solver_param
config_name_prev=$config_name

#-------------------------------------------------------
#incremental sparsification and finetuning
stage="sparse"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

base_lr=1e-5  #use a lower lr for fine tuning
sparse_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue],\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000,\
'sparsity_target':0.8,'sparsity_start_iter':0,'sparsity_start_factor':0.8,\
'sparsity_step_iter':1000,'sparsity_step_factor':0.01}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'image_width':1024,'image_height':512}" 

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name

#-------------------------------------------------------
#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue],\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'image_width':1024,'image_height':512,\
'num_test_image':500,'test_batch_size':10,\
'caffe_cmd':'test'}" 

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$test_solver_param
#config_name_prev=$config_name

#-------------------------------------------------------
#test_quantize
stage="test_quantize"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'Adam','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue],\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'image_width':1024,'image_height':512,\
'num_test_image':500,'test_batch_size':10,\
'caffe_cmd':'test'}" 

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$test_solver_param

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



