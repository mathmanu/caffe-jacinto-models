#!/bin/bash

#-------------------------------------------------------
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
#-------------------------------------------------------

#------------------------------------------------
gpus="0,1" #"0,1,2"

#-------------------------------------------------------
model_name=ssdJacintoNetV2       #jdetnet21v2, jdetnet21v2-s8, jdetnet21v2-fpn,mobilenet-x.x, ssdJacintoNetV2       
dataset=ti201712-1024x256        #cityscapes-ssd768x384,cityscapes-ssd512x512, cityscapes-ssd512x256, voc0712od-ssd512x512, ti201712-ssd720x368,cityscapes-ssd720x368-ticat 

#------------------------------------------------
#Download the pretrained weights
#weights_dst="training/imagenet_jacintonet11v2_iter_320000.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/voc0712od-ssd512x512_jdetnet21v2_2017-09-19_16-17-34_pyr-max-pool_1x1head_(72.82%_n500)_69.05%/initial/voc0712od-ssd512x512_jdetnet21v2_iter_120000.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/cityscapes-ssd768x384_jdetnet21v2_2017-09-21_21-24-36_(32.40%)/initial/cityscapes-ssd768x384_jdetnet21v2_iter_60000.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto/models/ssd/voc0712od-ssd512x512_jdetnet21v2_2017-09-19_16-17-34_pyr-max-pool_1x1head_(72.82%)/initial/voc0712od-ssd512x512_jdetnet21v2_iter_120000.caffemodel"
#weights_dst="/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.08/caffe-0.16/imagenet_mobilenet-1.0_2017-08-17_14-27-05_(71.5%)_(finetune)/initial/imagenet_mobilenet-1.0_iter_32000.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/voc0712od-ssd512x512_jdetnet21v2-s8_2017-12-18_22-23-00_63.74/initial/voc0712od-ssd512x512_jdetnet21v2-s8_iter_120000.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/voc0712od-ssd512x512_ssdJacintoNetV2_2018-01-08_12-08-00/initial/voc0712od-ssd512x512_ssdJacintoNetV2_iter_96000_intermediate_67.8.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/cityscapes-ssd720x368-ticat_ssdJacintoNetV2_2018-01-09_16-24-25/initial/cityscapes-ssd720x368-ticat_ssdJacintoNetV2_iter_6000_intermediate_31_0.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-09_19-30-42/l1reg/ti201712-ssd720x368_ssdJacintoNetV2_iter_40000_52.1.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_PSP_2018-01-10_19-22-21_partial_51.6/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_8000_51.6.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_PSP_dsFac32_2018-01-10_22-44-54_51.7/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_28000_51_70.caffemodel"
#weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ssdJacintoNetV2_2018-01-11_10-13-46/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_26000_51.45.caffemodel"
weights_dst="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti201712-ssd720x368_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_ssdJacintoNetV2_2018-01-16_13-10-14/initial/ti201712-ssd720x368_ssdJacintoNetV2_iter_60000.caffemodel"
if [ -f $weights_dst ]; then
  echo "Using pretrained model $weights_dst"
else
  weights_src="https://github.com/tidsp/caffe-jacinto-models/blob/caffe-0.15/trained/image_classification/imagenet_jacintonet11v2/initial/imagenet_jacintonet11v2_iter_320000.caffemodel?raw=true"
  wget $weights_src -O $weights_dst
fi

#------------------------------------------------
use_image_list=0 #known issue - use_image_list=0 && shuffle=1 => hang.
shuffle=0        #Note shuffle is used only in training
batch_size=16    #32    #16
resize_width=512
resize_height=512
crop_width=512
crop_height=512
#unintialized value: Average of W,H will be used as min dim
min_dim=-1
small_objs=0
#ssd-size:'512x512', '300x300','256x256'
ssd_size='512x512'
#0:[1,2,1/2] for each reg head, 1:like orig SSD
aspect_ratios_type=0
min_ratio=10
max_ratio=90
#16,32
ds_fac=16

#'DFLT', 'PSP'
ds_type='PSP'

fully_conv_at_end=1
reg_head_at_ds8=1
concat_reg_head=0
#kernel size for mbox_loc, mbox_conf conv
ker_mbox_loc_conf=3

#-------------------------------------------------------
if [ $dataset = "voc0712od-ssd512x512" ]
then
  type="SGD"        #"SGD"    #Adam    #"Adam"  #"Adam"
  max_iter=120000   #120000   #120000  #60000   #30000
  stepvalue1=60000  #60000    #60000   #30000   #15000
  stepvalue2=90000  #90000    #90000   #45000   #25000
  base_lr=1e-3      #1e-2     #1e-2    #1e-4    #1e-3

  #train_data="/data/hdd/data/datasets/object-detect/other/pascal-voc/combined/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb"
  #test_data="/data/hdd/data/datasets/object-detect/other/pascal-voc/combined/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb"
  train_data="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/VOC0712/VOC0712_trainval_lmdb"
  test_data="/user/a0875091/files/work/github/weiliu89/caffe-ssd/examples/VOC0712/VOC0712_test_lmdb"

  name_size_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/VOC0712/test_name_size.txt"
  label_map_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/VOC0712/labelmap_voc.prototxt"
  
  num_test_image=4952
  num_classes=21
  param_aspect_ratios="2,3"
  min_ratio=7 #10
  max_ratio=90
  log_space_steps=0
  use_difficult_gt=1
elif [ $dataset = "cityscapes-ssd512x512" ]
then
  type="SGD"        #"SGD"   #Adam    #"Adam"
  max_iter=60000    #120000  #64000   #32000
  stepvalue1=30000  #60000   #32000   #16000
  stepvalue2=45000  #90000   #48000   #24000
  base_lr=1e-2      #1e-2    #1e-4    #1e-3
  
  train_data="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x512/CITY_512x512_train_lmdb"
  test_data="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x512/CITY_512x512_test_lmdb"
  name_size_file="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x512/test_name_size.txt"
  label_map_file="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x512/labelmap.prototxt"
  
  num_test_image=498
  num_classes=9
  min_ratio=10
  max_ratio=90
  log_space_steps=0
  use_difficult_gt=1
elif [ $dataset = "cityscapes-ssd768x384" ]
then
  type="SGD"        #"SGD"   #Adam    #"Adam"
  max_iter=60000    #120000  #64000   #32000
  stepvalue1=30000  #60000   #32000   #16000
  stepvalue2=45000  #90000   #48000   #24000
  base_lr=1e-2      #1e-2    #1e-4    #1e-3
  
  train_data="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_768x384/CITY_768x384_train_lmdb"
  test_data="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_768x384/CITY_768x384_test_lmdb"
  name_size_file="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_768x384/test_name_size.txt"
  label_map_file="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_768x384/labelmap.prototxt"
  
  num_test_image=498
  num_classes=8 #9
  min_ratio=10
  max_ratio=90
  log_space_steps=0
  use_difficult_gt=1
  
  resize_width=768
  resize_height=384
  crop_width=768
  crop_height=384
elif [ $dataset = "cityscapes-ssd720x368-ticat" ]
then
  type="SGD"        #"SGD"   #Adam    #"Adam"
  max_iter=36000    #120000  #64000   #32000
  stepvalue1=24000  #60000   #32000   #16000
  stepvalue2=30000  #90000   #48000   #24000
  base_lr=1e-3      #1e-2    #1e-4    #1e-3
  
  train_data="/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/CITY_720x368_TI_CATEGORY/lmdb/CITY_720x368_TI_CATEGORY_train_lmdb"
  test_data="/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/CITY_720x368_TI_CATEGORY/lmdb/CITY_720x368_TI_CATEGORY_test_lmdb"
  name_size_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/test_name_size.txt"
  label_map_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/CITY_720x368_TI_CATEGORY/labelmap.prototxt"
  
  num_test_image=498
  num_classes=4
	min_dim=368
	ssd_size='512x512'
	#like orig SSD
  aspect_ratios_type=1
  small_objs=1 
	#0:log,1:linear,2:like original SSD (min/max ratio will be recomputed)
  log_space_steps=2
  use_difficult_gt=0
  
  resize_width=720
  resize_height=368
  crop_width=720
  crop_height=368

elif [ $dataset = "cityscapes-ssd512x256" ]
then
  type="SGD"        #"SGD"   #Adam    #"Adam"
  max_iter=60000    #120000  #64000   #32000
  stepvalue1=30000  #60000   #32000   #16000
  stepvalue2=45000  #90000   #48000   #24000
  base_lr=1e-2      #1e-2    #1e-4    #1e-3
  
  train_data="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x256/CITY_512x256_train_lmdb"
  test_data="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x256/CITY_512x256_test_lmdb"
  name_size_file="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x256/test_name_size.txt"
  label_map_file="/data/hdd/datasets/object-detect/other/cityscapes/TI_Derivatives/CITY_object_detect/lmdb/lmdb_512x256/labelmap.prototxt"
  
  num_test_image=498
  num_classes=8 #9
  min_ratio=10
  max_ratio=90
  log_space_steps=0
  use_difficult_gt=1
  
  resize_width=512
  resize_height=256
  crop_width=512
  crop_height=256
elif [ $dataset = "ti201712-ssd720x368" ]
then
  type="SGD"        #"SGD"   #Adam    #"Adam"
  max_iter=60000    #120000  #64000   #32000
  stepvalue1=30000  #60000   #32000   #16000
  stepvalue2=45000  #90000   #48000   #24000
  base_lr=1e-3      #1e-2    #1e-4    #1e-3
  
  train_data="/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/TI_201712_720x368_V1/lmdb/TI_201712_720x368_V1_train_lmdb"
  test_data="/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/TI_201712_720x368_V1/lmdb/TI_201712_720x368_V1_test_lmdb"
  name_size_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/test_name_size.txt"
  label_map_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_720x368_V1/labelmap.prototxt"
  
  num_test_image=3609
  num_classes=4 #9

	min_dim=368
	ssd_size='512x512'
	#1:like orig SSD
  aspect_ratios_type=1
  small_objs=1 
	#0:log,1:linear,2:like original SSD (min/max ratio will be recomputed)
  log_space_steps=2
  #'DFLT', 'PSP'
  ds_type='PSP'
  ds_fac=32
  fully_conv_at_end=0
  reg_head_at_ds8=1
  concat_reg_head=0
  base_nw_3_head=0
  ker_mbox_loc_conf=1
  first_hd_same_op_ch=1
  
  resize_width=720
  resize_height=368
  crop_width=720
  crop_height=368
  #ignore lables are marked as diff in TI dataset 
  use_difficult_gt=0

elif [ $dataset = "ti201712-1024x256" ]
then
  type="SGD"        #"SGD"   #Adam    #"Adam"
  max_iter=60000    #120000  #64000   #32000
  stepvalue1=30000  #60000   #32000   #16000
  stepvalue2=45000  #90000   #48000   #24000
  base_lr=1e-3      #1e-2    #1e-4    #1e-3
  
  train_data="/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/TI_201712_1024x256/lmdb/TI_201712_1024x256_train_lmdb"
  test_data="/user/a0875091/files/data/datasets/object-detect/ti/detection/xml/TI_201712_1024x256/lmdb/TI_201712_1024x256_test_lmdb"
  name_size_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_1024x256/test_name_size.txt"
  label_map_file="/user/a0875091/files/work/github/weiliu89/caffe-ssd/data/TI_201712_1024x256/labelmap.prototxt"
  
  num_test_image=3609
  num_classes=4 #9

	min_dim=256
	ssd_size='512x512'
	#1:like orig SSD
  aspect_ratios_type=1
  small_objs=1 
	#0:log,1:linear,2:like original SSD (min/max ratio will be recomputed)
  log_space_steps=2
  #'DFLT', 'PSP'
  ds_type='PSP'
  ds_fac=32
  fully_conv_at_end=0
  reg_head_at_ds8=1
  concat_reg_head=0
  base_nw_3_head=0
  ker_mbox_loc_conf=1
  first_hd_same_op_ch=1
  
  resize_width=1024
  resize_height=256
  crop_width=1024
  crop_height=256
  #ignore lables are marked as diff in TI dataset 
  use_difficult_gt=0

else
  echo "Invalid dataset name"
  exit
fi

folder_name=training/"$dataset"_"$model_name"_"$DATE_TIME"_"ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1";mkdir $folder_name

#------------------------------------------------
LOG=$folder_name/train-log_"$DATE_TIME".txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#-------------------------------------------------------
#Initial training
stage="initial"
weights=$weights_dst
#weights="/data/mmcodec_video2_tier3/users/manu/experiments/object/detection/2017/2017.09/caffe-0.16/voc0712od-ssd512x512_jdetnet21v2_2017-09-19_16-17-34_pyr-max-pool_1x1head_(72.82%)/initial/voc0712od-ssd512x512_jdetnet21v2_iter_120000.caffemodel"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2]}"
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type'}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$solver_param
config_name_prev=$config_name


#-------------------------------------------------------
#l1 regularized training before sparsification
stage="l1reg"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel

max_iter=60000
stepvalue1=30000
stepvalue2=45000
#base_lr=1e-2

l1reg_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'regularization_type':'L1','weight_decay':1e-5}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type'}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$l1reg_solver_param
config_name_prev=$config_name

#-------------------------------------------------------
#incremental sparsification and finetuning
stage="sparse"
weights=$config_name_prev/"$dataset"_"$model_name"_iter_$max_iter.caffemodel
sparse_solver_param="{'type':'$type','base_lr':$base_lr,'max_iter':$max_iter,'lr_policy':'multistep','stepvalue':[$stepvalue1,$stepvalue2],\
'regularization_type':'L1','weight_decay':1e-5,\
'sparse_mode':1,'display_sparsity':1000,\
'sparsity_target':0.8,'sparsity_start_iter':0,'sparsity_start_factor':0.8,\
'sparsity_step_iter':1000,'sparsity_step_factor':0.01}"

config_name="$folder_name"/$stage; echo $config_name; mkdir $config_name
config_param="{'config_name':'$config_name','model_name':'$model_name','dataset':'$dataset','gpus':'$gpus',\
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type'}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$sparse_solver_param
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
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'test_batch_size':10,'caffe_cmd':'test_detection','display_sparsity':1,\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type'}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$test_solver_param
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
'train_data':'$train_data','test_data':'$test_data','name_size_file':'$name_size_file','label_map_file':'$label_map_file',\
'num_test_image':$num_test_image,'num_classes':$num_classes,'min_ratio':$min_ratio,'max_ratio':$max_ratio,
'log_space_steps':$log_space_steps,'use_difficult_gt':$use_difficult_gt,\
'pretrain_model':'$weights','use_image_list':$use_image_list,'shuffle':$shuffle,'num_output':8,\
'resize_width':$resize_width,'resize_height':$resize_height,'crop_width':$crop_width,'crop_height':$crop_height,'batch_size':$batch_size,\
'test_batch_size':10,'caffe_cmd':'test_detection',\
'aspect_ratios_type':$aspect_ratios_type,'ssd_size':'$ssd_size','small_objs':$small_objs,'min_dim':$min_dim,'concat_reg_head':$concat_reg_head,
'fully_conv_at_end':$fully_conv_at_end,'first_hd_same_op_ch':$first_hd_same_op_ch,'ker_mbox_loc_conf':$ker_mbox_loc_conf,'base_nw_3_head':$base_nw_3_head,'reg_head_at_ds8':$reg_head_at_ds8,'ds_fac':$ds_fac,'ds_type':'$ds_type'}" 

python ./models/image_object_detection.py --config_param="$config_param" --solver_param=$test_solver_param

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



