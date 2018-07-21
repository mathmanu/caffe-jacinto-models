#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y%m%d_%H-%M'`
#-------------------------------------------------------

#------------------------------------------------
#IMPORTANT: change gpus depending on the number of GPUs available.
#IMPORTANT: reduce the batch size, if the script crashes due to GPU memory shortage
gpus="0"            #"0,1,2,3" #"0,1" #"0"
batch_size=16       #32        #16    #"8"

#-------------------------------------------------------
model_name=mobiledetnet-0.5      #mobiledetnet-0.5  #mobiledetnet-1.0 #ssdJacintoNetV2  
dataset=voc0712-512x512          #voc0712-512x256, voc0712-512x512
#------------------------------------------------

#Download the pretrained weights
weights_dst="../trained/image_classification/imagenet_mobilenet-0.5/initial/imagenet_mobilenet-0.5_iter_320000.caffemodel"

#------------------------------------------------
#make sure the train_data and test_data lmdb's (below) point to the correct size dataset

#------------------------------------------------
#ssd-size:'512x512', '300x300','256x256', '512x256', '768x320'
ssd_size='512x256'

#0:[1,2,1/2] for each reg head, 1:like orig SSD
aspect_ratios_type=1

#max donwsampling factor: 16,32
ds_fac=32

#down sampling type: 'DFLT', 'PSP'
ds_type='PSP'

#regression head at downsamling 8 layer: 0,1 
reg_head_at_ds8=1

#use concat layers for regression heads
concat_reg_head=0

#kernel size for mbox_loc, mbox_conf conv
ker_mbox_loc_conf=1

#if unintialized value: Average of W,H will be used as min dim
min_dim=-1

#1:model tuned for small objects, 0:model tuned for moderate size object like PASCAL VOC
small_objs=0

#"step", "multistep", "poly"
lr_policy="multistep"

#needed for coco training for gray scale images
force_color=0
stepvalue3=300000 

#num op ch for creating mbox heads
num_intermediate=512

rhead_name_non_linear=0

#1: difficult GT will be used for eval, 0: difficult GT will not be used for evaluation
evaluate_difficult_gt=0

#:Experimntal option. Does not have  any effect now.
ignore_difficult_gt=0

#set to True for VOC0712
use_difficult_gt=1

#"poly","multistep"
lr_policy="multistep"

#set it to 4.0 for poly
power=1.0

#0.0005 (orignal SSD), 0.0001
#wd of dw layers may be lowered even furter inside the script using decay_mult
weight_decay_L2=1e-4

#0:linear,1:log,2:like original SSD (min/max ratio will be recomputed)
log_space_steps=0

#does not have effect when log_space_steps=2
#adjust these suitably to get the best accuracy
min_ratio=5
max_ratio=95

#1:FC layer like originalk SSD, 0: no FC layer
fully_conv_at_end=0

#1: connect 3 head in base n/w. Experimental. set it to 0
base_nw_3_head=0

#1:first head num of op channel same as other layers, 0: first hd double the number of op channel
first_hd_same_op_ch=1

#To chop of last few heads. It will make max/min size computed based on original number of heads
chop_num_heads=0

#known issue - use_image_list=0 && shuffle=1 => hang.
use_image_list=0 

#Note shuffle is used only in training
shuffle=0        

#-------------------------------------------------------
if [ $dataset = "voc0712-512x256" ]
then

  train_data="../../caffe-jacinto/examples/VOC0712/VOC0712_trainval_lmdb"
  test_data="../../caffe-jacinto/examples/VOC0712/VOC0712_test_lmdb"

  name_size_file="../../caffe-jacinto/data/VOC0712/test_name_size.txt"
  label_map_file="../../caffe-jacinto/data/VOC0712/labelmap_voc.prototxt"
 
  num_test_image=4952
  num_classes=21

  #IMPOARTANT: the detection accuracy seems to be quite sensitive to this parameter
  #try -1 or width or height and see which one gives the best result.
  #set to -1 to use auto mode - average of min and max dimensions
  min_dim=512 #-1

  resize_width=512  
  resize_height=256 
  crop_width=512    
  crop_height=256

  type="SGD"         #"SGD"   #Adam    #"Adam"
  max_iter=120000    #120000  #64000   #32000
  stepvalue1=60000   #60000   #32000   #16000
  stepvalue2=90000   #90000   #48000   #24000
  base_lr=1e-2       #1e-2    #1e-4    #1e-3
  
  #use batch norm ofr mbox layer1:enable,0:disable
  use_batchnorm_mbox=1

elif [ $dataset = "voc0712-512x512" ]
then
  train_data="../../caffe-jacinto/examples/VOC0712/VOC0712_trainval_lmdb"
  test_data="../../caffe-jacinto/examples/VOC0712/VOC0712_test_lmdb"

  name_size_file="../../caffe-jacinto/data/VOC0712/test_name_size.txt"
  label_map_file="../../caffe-jacinto/data/VOC0712/labelmap_voc.prototxt"
 
  num_test_image=4952
  num_classes=21

  #IMPOARTANT: the detection accuracy seems to be quite sensitive to this parameter
  #try -1 or width or height and see which one gives the best result.
  #set to -1 to use auto mode - average of min and max dimensions
  min_dim=-1

  resize_width=512
  resize_height=512 
  crop_width=512 
  crop_height=512

  type="SGD"         #"SGD"   #Adam    #"Adam"
  max_iter=120000    #120000  #64000   #32000
  stepvalue1=60000   #60000   #32000   #16000
  stepvalue2=90000   #90000   #48000   #24000
  base_lr=1e-2       #1e-2    #1e-4    #1e-3
  
  #use batch norm ofr mbox layer1:enable,0:disable
  use_batchnorm_mbox=1
  
else
  echo "Invalid dataset name"
  exit
fi

model_name_to_print=$model_name 
if [ $model_name = 'ssdJacintoNetV2' ]
then
  model_name_to_print="JDetNet" 
fi  

folder_name=training/"$dataset"/"$model_name_to_print"/"$dataset"_"$model_name"_"$DATE_TIME";mkdir training/"$dataset";mkdir training/"$dataset"/"$model_name_to_print";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#-------------------------------------------------------
#Initial training
stage="initial"
weights=$weights_dst

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'$lr_policy','power':$power,'stepvalue':[$stepvalue1,$stepvalue2,$stepvalue3],'weight_decay':$weight_decay_L2}"
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,'ignore_difficult_gt':$ignore_difficult_gt,'evaluate_difficult_gt':$evaluate_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,\
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,\
'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type',\
'rhead_name_non_linear':$rhead_name_non_linear,'force_color':$force_color,'num_intermediate':$num_intermediate,\
'use_batchnorm_mbox':$use_batchnorm_mbox,'chop_num_heads':$chop_num_heads}"

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$solver_param
config_name_prev=$config_name

#-------------------------------------------------------

#test
stage="test"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'$lr_policy','power':$power,'stepvalue':[$stepvalue1,$stepvalue2,$stepvalue3],\
'regularization_type':'L2','weight_decay':$weight_decay_L2}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,'ignore_difficult_gt':$ignore_difficult_gt,'evaluate_difficult_gt':$evaluate_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'test_batch_size':10,'caffe_cmd':'test_detection','display_sparsity':1,\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type','rhead_name_non_linear':$rhead_name_non_linear,'force_color':$force_color,'num_intermediate':$num_intermediate,'use_batchnorm_mbox':$use_batchnorm_mbox,'chop_num_heads':$chop_num_heads}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$test_solver_param
#config_name_prev=$config_name

#-------------------------------------------------------
#test_quantize
stage="test_quantize"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

test_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'$lr_policy','power':$power,'stepvalue':[$stepvalue1,$stepvalue2,$stepvalue3],\
'regularization_type':'L2','weight_decay':$weight_decay_L2}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,'ignore_difficult_gt':$ignore_difficult_gt,'evaluate_difficult_gt':$evaluate_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'test_batch_size':10,'caffe_cmd':'test_detection',\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,\
'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type','rhead_name_non_linear':$rhead_name_non_linear,\
'force_color':$force_color,'num_intermediate':$num_intermediate,'use_batchnorm_mbox':$use_batchnorm_mbox,'chop_num_heads':$chop_num_heads}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$test_solver_param

echo "quantize: true" > $config_name/deploy_new.prototxt
cat $config_name/deploy.prototxt >> $config_name/deploy_new.prototxt
mv --force $config_name/deploy_new.prototxt $config_name/deploy.prototxt

echo "quantize: true" > $config_name/test_new.prototxt
cat $config_name/test.prototxt >> $config_name/test_new.prototxt
mv --force $config_name/test_new.prototxt $config_name/test.prototxt

#-------------------------------------------------------
#run
list_dirs=`command ls -d1 "$folder_name"/*/ | command cut -f5 -d/`
for f in $list_dirs; do "$folder_name"/$f/run.sh; done
