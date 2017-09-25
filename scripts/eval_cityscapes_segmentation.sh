#-------------------------------------------------------
LOG="training/eval-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
caffe=../../build/tools/caffe.bin
#-------------------------------------------------------

#GLOG_minloglevel=3 
#--v=5

gpu="1,0" #'0'


val_crop=0 #"1024 512"
val_resize="1024 512"
val_input="./data/val-image-list.txt"
val_label="./data/val-label-list.txt"
val_classes=5 #34
val_weights=0
num_images=500 #100000

#for 19 or 20 classes training of cityscapes, first convert to original labelIds and then apply the pallete
#label_dict_20_to_34="{0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20, 8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27, 15:28, 16:31, 17:32, 18:33, 19:0}"

#class grouping and ignored - 1:19 extra class that we added is also mapped to ignored (1:255)
#class_dict="{0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 255:255}"

#7 categories
#class_dict="{0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:1, 8:1, 9:255, 10:255, 11:2, 12:2, 13:2, 14:255, 15:255, 16:255, 17:3, 18:255, 19:3, 20:3, 21:4, 22:4, 23:5, 24:6, 25:6, 26:7, 27:7, 28:7, 29:255, 30:255, 31:7, 32:7, 33:7, 255:255}"


#------------------------------------------------
#initial model
val_model="../trained/image_segmentation/cityscapes5_jsegnet21v2/initial/deploy.prototxt"
val_weights="../trained/image_segmentation/cityscapes5_jsegnet21v2/initial/cityscapes5_jsegnet21v2_iter_120000.caffemodel"
python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back 
#--label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
echo 'initial eval.'

#------------------------------------------------
#l1reg model
val_model="../trained/image_segmentation/cityscapes5_jsegnet21v2/l1reg/deploy.prototxt"
val_weights="../trained/image_segmentation/cityscapes5_jsegnet21v2/l1reg/cityscapes5_jsegnet21v2_iter_60000.caffemodel"
python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back 
#--label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
echo 'l1reg eval.'

#------------------------------------------------
#sparse model
val_model="../trained/image_segmentation/cityscapes5_jsegnet21v2/sparse/deploy.prototxt"
val_weights="../trained/image_segmentation/cityscapes5_jsegnet21v2/sparse/cityscapes5_jsegnet21v2_iter_60000.caffemodel"
python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back 
#--label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
echo 'sparse eval.'

#------------------------------------------------
#quantized model
val_model="../trained/image_segmentation/cityscapes5_jsegnet21v2/test_quantize/deploy.prototxt"
val_weights="../trained/image_segmentation/cityscapes5_jsegnet21v2/sparse/cityscapes5_jsegnet21v2_iter_60000.caffemodel"
python ./tools/utils/infer_segmentation.py --crop $val_crop --resize $val_resize --model $val_model --weights $val_weights --input $val_input --label $val_label --num_classes=$val_classes --num_images=$num_images --resize_back 
#--label_dict="$label_dict_20_to_34" --class_dict="$class_dict"
echo 'sparse eval.'




