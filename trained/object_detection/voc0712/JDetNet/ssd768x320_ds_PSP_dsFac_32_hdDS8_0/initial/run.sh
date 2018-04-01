/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/voc0712/JDetNet/20180512_18-25_ds_PSP_dsFac_32_hdDS8_1/initial/solver.prototxt" \
--weights="training/imagenet_jacintonet11v2_iter_320000.caffemodel" \
--gpu "0,1" 2>&1 | tee training/voc0712/JDetNet/20180512_18-25_ds_PSP_dsFac_32_hdDS8_1/initial/run.log
