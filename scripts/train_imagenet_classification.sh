#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#-------------------------------------------------------
#IMPORTANT: change gpus depending on the number of GPUs available.
#IMPORTANT: reduce the batch size, if the script crashes due to GPU memory shortage
gpus="0"               #"0,1,2,3"  #"0,1"   #"0"
batch_size=256         #256        #128     #64

#-------------------------------------------------------
model_name=jacintonet11v2
dataset=imagenet
folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"



#-------------------------------------------------------
max_iter=320000
base_lr=0.1
type=SGD
solver_param="{'type':'SGD','base_lr':$base_lr,'max_iter':$max_iter}"

#-------------------------------------------------------
stage="initial"
config_name=$folder_name/$stage;mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'pretrain_model':None}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param="$solver_param"
config_name_prev=$config_name


#-------------------------------------------------------
#incremental sparsification and finetuning
stage="sparse"
#Using more than one GPU for this step gives strange results. Imbalanced accuracy between the GPUs.
gpus="0" #"0,1,2"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

max_iter=160000 #320000
type=SGD
base_lr=0.01  #use a lower lr for fine tuning
sparsity_start_iter=20000
sparse_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000,\
'sparsity_target':0.8,'sparsity_start_iter':$sparsity_start_iter,'sparsity_start_factor':0.0,\
'sparsity_step_iter':1000,'sparsity_step_factor':0.01}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'pretrain_model':'$weights'}" 
python ./models/image_classification.py --config_param="$config_param" --solver_param=$sparse_solver_param
config_name_prev=$config_name


#-------------------------------------------------------
#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'pretrain_model':'$weights',\
'num_output':1000,'image_width':224,'image_height':224,'crop_size':224,\
'caffe_cmd':'test','display_sparsity':1}" 

python ./models/image_classification.py --config_param="$config_param" --solver_param=$test_solver_param
#config_name_prev=$config_name


#-------------------------------------------------------
#test_quantize
stage="test_quantize"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus','batch_size':$batch_size,\
'pretrain_model':'$weights',\
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


#-------------------------------------------------------
#run
list_dirs=`command ls -d1 "$folder_name"/*/ | command cut -f3 -d/`
for f in $list_dirs; do "$folder_name"/$f/run.sh; done



