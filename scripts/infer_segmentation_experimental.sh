#-------------------------------------------------------
LOG="training/infer-log-`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
#-------------------------------------------------------

#------------------------------------------------------
#palette used to translate id's to colors - for 5 classes
#palette5="[[0,0,0],[128,64,128],[220,20,60],[250,170,30],[0,0,142],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]"

#for 19 or 20 classes training of cityscapes, first convert to original labelIds and then apply the pallete
#label_dict_20_to_34="{0:7, 1:8, 2:11, 3:12, 4:13, 5:17, 6:19, 7:20, 8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27, 15:28, 16:31, 17:32, 18:33, 19:0}"

#34 class pallette - for visualization
#palette34="[(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(111, 74,  0),( 81,  0, 81),(128, 64,128),(244, 35,232),(250,170,160),(230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),(150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),(220,220,  0),(107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),(  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32),(  0,  0,142)]"

#palette256="[(128,18,23),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,70),(0,0,0),(0,0,90),(0,0,0),(0,0,0),(0,0,110),(0,0,0),(0,0,0),(0,0,142),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,192),(0,0,0),(0,0,0),(0,0,0),(0,0,230),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(32,32,32),(33,33,33),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(40,40,40),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(119,11,32),(0,60,100),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,80,100),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(70,70,70),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(255,0,0),(0,0,0),(250,0,30),(165,42,42),(0,0,0),(0,0,0),(0,0,0),(128,64,64),(220,20,60),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(128,64,128),(0,0,0),(0,0,0),(0,0,0),(222,40,40),(0,0,0),(96,96,96),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,170,30),(0,0,0),(128,64,255),(0,0,0),(0,0,0),(102,102,156),(0,0,0),(110,110,110),(0,0,0),(0,0,0),(0,192,0),(0,0,0),(150,100,100),(0,0,0),(0,0,0),(70,130,180),(107,142,35),(244,35,232),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(150,120,90),(0,0,0),(128,128,128),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(100,170,30),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(140,140,200),(0,0,0),(0,0,0),(200,128,128),(0,0,0),(0,0,0),(153,153,153),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(190,153,153),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(170,170,170),(180,165,180),(0,0,0),(230,150,140),(210,170,100),(0,0,0),(0,0,0),(0,0,0),(250,170,30),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(192,192,192),(250,170,160),(0,0,0),(220,220,0),(196,196,196),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(152,251,152),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(220,220,220),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(255,255,128),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]"

#palette51="[(255,255,255),(0,0,70),(0,0,90),(0,0,110),(0,0,142),(0,0,192),(0,0,230),(32,32,32),(33,33,33),(40,40,40),(119,11,32),(0,60,100),(0,80,100),(70,70,70),(255,0,0),(250,0,30),(165,42,42),(128,64,64),(220,20,60),(128,64,128),(222,40,40),(96,96,96),(0,170,30),(128,64,255),(102,102,156),(110,110,110),(0,192,0),(150,100,100),(70,130,180),(107,142,35),(244,35,232),(64,170,64),(150,120,90),(128,128,128),(100,170,30),(140,140,200),(200,128,128),(153,153,153),(190,153,153),(170,170,170),(180,165,180),(230,150,140),(210,170,100),(250,170,30),(192,192,192),(250,170,160),(220,220,0),(196,196,196),(152,251,152),(220,220,220),(255,255,128)]"

#Mapillary Category -5
#bg,person,vehicle,marking,road
palette5="[(0,0,0), (220,20,60), (0,0,142), (119,11,32),(128,64,128)]"

num_images=2000 #10 #1000
crop=0 #"1024 512"
resize="1024 768"

#------------------------------------------------------
#model="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/tipsdscapes_jsegnet21v2_2018-04-14_01-33-05/initial/deploy.prototxt"
#weights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/tipsdscapes_jsegnet21v2_2018-04-14_01-33-05/initial/tipsdscapes_jsegnet21v2_iter_120000.caffemodel"

#model="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/mapillary_jsegnet21v2_2018-04-17_13-08-59/initial/deploy.prototxt"
#weights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/mapillary_jsegnet21v2_2018-04-17_13-08-59/initial/mapillary_jsegnet21v2_iter_120000.caffemodel"

#Mapillary - 5 category
model="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/mapillary_jsegnet21v2_cat5/initial/deploy.prototxt"
weights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/mapillary_jsegnet21v2_cat5/initial/mapillary_jsegnet21v2_iter_120000.caffemodel"

#Infer
#input="input/sample"
#output="output/sample"

#TI_PSD Test
#input="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/data/tipsdscapes/val-image-list.txt"
#output="/data/mmcodec_video2_tier3/users/soyeb/semantic/test_psd/20180426_mapilary_tipsdscapes_2_blend1"

#Mapillary Val Set
input="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/data/val-image-list.txt"
output="/data/mmcodec_video2_tier3/users/soyeb/semantic/test/20180426_mapilary5Cat_val_blend2_temp"
val_lable_list="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/data/val-label-list.txt"

#Generate output images for chroma blended visualization
#--resize_back

#eval
./tools/utils/infer_segmentation.py --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --num_images $num_images --label="$val_lable_list" --num_classes=5 --resize_op_to_ip_size

#./tools/utils/infer_segmentation.py --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --num_images $num_images --num_classes=5 --palette="$palette5" --blend 1 --resize_op_to_ip_size

#--palette="$palette34" --label_dict="$label_dict_20_to_34"


#Generate output images for running the IOU measurement (using the measure_...  script)
#./tools/utils/infer_segmentation.py --crop $crop --resize $resize --model $model --weights $weights --input $input --output $output --num_images $num_images --resize_back --label_dict="$label_dict_20_to_34"
