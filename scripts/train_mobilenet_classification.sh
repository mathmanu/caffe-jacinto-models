#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
#IMPORTANT: change gpus depending on the number of GPUs available.
#IMPORTANT: reduce the batch size, if the script crashes due to GPU memory shortage
gpus="0"           #"0,1,2,3"  #"0,1"   #"0"
batch_size=64      #256        #128     #64

#-------------------------------------------------------
model_name=mobilenet-0.5  #mobilenet-1.0 #mobilenet-0.5  #mobilenetv2-1.0 #mobilenetv2t3-1.0
dataset=imagenet
folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#------------------------------------------------
caffe="../../caffe-jacinto/build/tools/caffe.bin"

max_iter=320000        #320000    #1000000
base_lr=0.1            #0.1        #0.045
type=SGD
weight_decay=1e-4  #4e-5
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter,'weight_decay':$weight_decay,'ignore_shape_mismatch':1}"

train_transform_param="{'mirror':1,'mean_value':[103.94,116.78,123.68],'crop_size':224,'scale':0.017}"
test_transform_param="{'mirror':0,'mean_value':[103.94,116.78,123.68],'crop_size':224,'scale':0.017}"

#------------------------------------------------
#Download the pretrained weights and place in this location, if you want to finetune
weights_src="training/"$model_name".caffemodel"

#-------------------------------------------------------
stage="initial"
weights=$weights_src
config_name=$folder_name/$stage;mkdir $config_name
#config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
#'pretrain_model':'$weights','mean_value':0,'train_transform_param':$train_transform_param,'test_transform_param':$test_transform_param}" 

config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'mean_value':0,'train_transform_param':$train_transform_param,'test_transform_param':$test_transform_param}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"

echo "python cmd completed"
config_name_prev=$config_name



#-------------------------------------------------------
#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'regularization_type':'L2','weight_decay':$weight_decay,\
'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'pretrain_model':'$weights','mean_value':0,'train_transform_param':$train_transform_param,'test_transform_param':$test_transform_param,\
'num_output':1000,'image_width':224,'image_height':224,'crop_size':224,\
'caffe_cmd':'test'}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param
#config_name_prev=$config_name


#-------------------------------------------------------
#test_quantize
stage="test_quantize"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'regularization_type':'L2','weight_decay':$weight_decay,\
'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'pretrain_model':'$weights','mean_value':0,'train_transform_param':$train_transform_param,'test_transform_param':$test_transform_param,\
'num_output':1000,'image_width':224,'image_height':224,'crop_size':224,\
'caffe_cmd':'test'}"


python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param

echo "quantize: true" > $config_name/deploy_new.prototxt
cat $config_name/deploy.prototxt >> $config_name/deploy_new.prototxt
mv --force $config_name/deploy_new.prototxt $config_name/deploy.prototxt

echo "quantize: true" > $config_name/test_new.prototxt
cat $config_name/test.prototxt >> $config_name/test_new.prototxt
mv --force $config_name/test_new.prototxt $config_name/test.prototxt
#config_name_prev=$config_name


##-------------------------------------------------------
#run
list_dirs=`command ls -d1 "$folder_name"/*/ | command cut -f3 -d/`
for f in $list_dirs; do "$folder_name"/$f/run.sh; done



