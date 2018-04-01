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
export MODIFY_LABEL=1
export DATASETPATH=/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/data/mapillary
#-------------------------------------------------------

#-------------------------------------------------------
#for mapillary need to swap 0 and 255
#total categories(51) : 0 to 49 + 255  
#export LABEL_MAP_DICT="{0:255,8:1,10:2,13:3,16:4,22:5,26:6,32:7,33:8,40:9,46:10,47:11,58:12,70:13,76:14,78:15,79:16,83:17,84:18,90:19,94:20,96:21,103:22,105:23,108:24,110:25,113:26,115:27,118:28,119:29,120:30,126:31,128:32,133:33,147:34,150:35,153:36,164:37,170:38,171:39,173:40,174:41,178:42,192:43,193:44,195:45,196:46,210:47,220:48,241:49,255:0}"

#total categories(5+1) : 0:background, 1:person, 2:vehicle, 3:marking/snow, 4:road, 255:ignore 
export LABEL_MAP_DICT="{0:255,8:2,10:2,13:2,16:2,22:2,26:0,32:0,33:0,40:0,46:0,47:2,58:2,70:0,76:1,78:0,79:0,83:0,84:1,90:4,94:0,96:0,103:0,105:0,108:0,110:0,113:0,115:0,118:0,119:0,120:0,126:0,128:0,133:0,147:0,150:0,153:0,164:0,170:0,171:0,173:0,174:0,178:0,192:0,193:0,195:0,196:0,210:0,220:0,241:0,255:3}"
RESIZE_WIDTH=1024
RESIZE_HEIGHT=768

#-------------------------------------------------------
#The label images are converted copied to local folder. This is because paletted pngs need the index to be stored and image. 
#OpenCV (in caffe) will otherwise use the color values, instead of the index.

echo "creating train lists"
> data/train-image-list.txt
> data/train-label-list.txt

WILDCARD_TR_IMG="*.jpg"
find $DATASETPATH/leftImg8bit/train/ -name $WILDCARD_TR_IMG | sort > data/train-image-list.txt

if [ $MODIFY_LABEL -eq 1 ]
then #of if
LABEL_FOLDER=data/train-label-folder
WILDCARD_TR_LBL="*.png"
#mapillary
./tools/utils/create_image_folder.py --label --image_dir=$DATASETPATH/gtFine/train --width=$RESIZE_WIDTH --height=$RESIZE_HEIGHT --search_string="$WILDCARD_TR_LBL" --output_dir=$LABEL_FOLDER --label_dict="$LABEL_MAP_DICT"

else #of if
LABEL_FOLDER=$DATASETPATH/gtFine/train
WILDCARD_TR_LBL="*.png"
fi #of if

find $LABEL_FOLDER -name $WILDCARD_TR_LBL | sort > data/train-label-list.txt
#-------------------------------------------------------


#-------------------------------------------------------
echo "creating val lists"
> data/val-image-list.txt
> data/val-label-list.txt
WILDCARD_VAL_IMG="*.jpg"
find $DATASETPATH/leftImg8bit/val/ -name $WILDCARD_VAL_IMG | sort > data/val-image-list.txt
if [ $MODIFY_LABEL -eq 1 ]
then #of if
LABEL_FOLDER=data/val-label-folder
WILDCARD_VAL_LBL="*.png"
#mapillary
./tools/utils/create_image_folder.py --label --image_dir=$DATASETPATH/gtFine/val --width=$RESIZE_WIDTH --height=$RESIZE_HEIGHT --search_string="$WILDCARD_VAL_LBL" --output_dir=$LABEL_FOLDER --label_dict="$LABEL_MAP_DICT"
else #of if
LABEL_FOLDER=$DATASETPATH/gtFine/val
WILDCARD_VAL_LBL="*.png"
fi #of if

find $LABEL_FOLDER -name $WILDCARD_VAL_LBL | sort > data/val-label-list.txt
#-------------------------------------------------------
