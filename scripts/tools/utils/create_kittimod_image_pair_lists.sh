#!/bin/bash

#-------------------------------------------------------
rm -rf data/*-folder
rm -rf data/*-list.txt
#-------------------------------------------------------

#-------------------------------------------------------
LOG="training/train-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#-------------------------------------------------------
#export PYTHONPATH=../../../../python:$PYTHONPATH

#DM: This is for image pair
export DATASETPATH=/user/a0132471/Files/datasets/KITTI_MOD_rgb_pair_caffe

#DM: This is for {dof,rgb} pair
#export DATASETPATH=/user/a0132471/Files/datasets/KITTI_MOD_rgb_dof_caffe
#-------------------------------------------------------

#-------------------------------------------------------
#Use original 34 classes
#export LABEL_MAP_DICT="[]"

#Default cityscapes mapping - 33 to 19 class
#see definition of the list 'labels' in https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
#export LABEL_MAP_DICT="{0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 255:255}"

#ignored classes mapped to a 20th class (id 19)
#export LABEL_MAP_DICT="{0:255, 1:19, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 255:255}"

#Custom mapping - 33 to 5 class
export LABEL_MAP_DICT="{0:0, 255:1}"
#-------------------------------------------------------


#-------------------------------------------------------
#The label images are converted copied to local folder. This is because paletted pngs need the index to be stored and image. 
#OpenCV (in caffe) will otherwise use the color values, instead of the index.

echo "creating train lists"

./tools/utils/create_image_folder.py --width 1248 --height 384 --image_dir=$DATASETPATH/train/image_frame1 --search_string="*.png" --output_dir=data/train-image-folder1
find ./data/train-image-folder1 -name *.png | sort > data/train-image-list1.txt

./tools/utils/create_image_folder.py --width 1248 --height 384 --image_dir=$DATASETPATH/train/image_frame2 --search_string="*.png" --output_dir=data/train-image-folder2
find ./data/train-image-folder2 -name *.png | sort > data/train-image-list2.txt


./tools/utils/create_image_folder.py --label --width 1248 --height 384 --image_dir=$DATASETPATH/train/mask --search_string="*.png" --output_dir=./data/train-label-folder --label_dict="$LABEL_MAP_DICT"
find ./data/train-label-folder -name *.png | sort > data/train-label-list.txt
#-------------------------------------------------------


#-------------------------------------------------------
echo "creating val lists"
./tools/utils/create_image_folder.py --width 1248 --height 384 --image_dir=$DATASETPATH/test/image_frame1 --search_string="*.png" --output_dir=data/test-image-folder1
find ./data/test-image-folder1 -name *.png | sort > data/test-image-list1.txt

./tools/utils/create_image_folder.py --width 1248 --height 384 --image_dir=$DATASETPATH/test/image_frame2 --search_string="*.png" --output_dir=data/test-image-folder2
find ./data/test-image-folder2 -name *.png | sort > data/test-image-list2.txt


./tools/utils/create_image_folder.py --label --width 1248 --height 384 --image_dir=$DATASETPATH/test/mask --search_string="*.png" --output_dir=./data/test-label-folder --label_dict="$LABEL_MAP_DICT"
find ./data/test-label-folder -name *.png | sort > data/test-label-list.txt
#-------------------------------------------------------


