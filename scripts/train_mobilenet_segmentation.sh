#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#------------------------------------------------
#IMPORTANT: change this to "0" if you have only one GPU
gpus="0,1"        #"0,1,2"

#-------------------------------------------------------
model_name=mobilesegnetv2-1.0    #jsegnet21v2 #mobilesegnetv2-1.0 #mobilesegnetv2-0.5
dataset=cityscapes5
folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#------------------------------------------------
#Download the pretrained weights
weights_dst="/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2018.06/net_surgery_from_shicai_mobilenetv2/imagenet_mobilenet_v2_shicaistrides_fusedbn/test_quantize/MobileNetV2_new_NV_fused_bn.caffemodel"

#-------------------------------------------------------
#Initial training
stage="initial"
weights=$weights_dst

type="SGD"       #"SGD"   #Adam    #"Adam"
max_iter=60000   #120000  #64000   #32000
stepvalue1=30000 #60000   #32000   #16000
stepvalue2=45000 #90000   #48000   #24000
base_lr=1e-2     #1e-2    #1e-4    #1e-3

use_image_list=0
shuffle=1

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2]}"
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':19,\
'image_width':1024,'image_height':512,'crop_size':256}"

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$solver_param
config_name_prev=$config_name

#-------------------------------------------------------
#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':19,\
'image_width':1024,'image_height':512,\
'num_test_image':500,'test_batch_size':10,\
'caffe_cmd':'test','display_sparsity':1}" 

python ./models/image_segmentation.py --config_param="$config_param" --solver_param=$test_solver_param
#config_name_prev=$config_name

#-------------------------------------------------------
#test_quantize
stage="test_quantize"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':19,\
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



