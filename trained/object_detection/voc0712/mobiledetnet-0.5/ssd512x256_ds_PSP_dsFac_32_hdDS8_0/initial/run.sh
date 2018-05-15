/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin train \
--solver="training/voc0712/mobiledetnet-0.5/20180510_14-48_ds_PSP_dsFac_32_hdDS8_1/initial/solver.prototxt" \
--weights="../trained/image_classification/imagenet_mobilenet-0.5/initial/imagenet_mobilenet-0.5_iter_320000.caffemodel" \
--gpu "0" 2>&1 | tee training/voc0712/mobiledetnet-0.5/20180510_14-48_ds_PSP_dsFac_32_hdDS8_1/initial/run.log
