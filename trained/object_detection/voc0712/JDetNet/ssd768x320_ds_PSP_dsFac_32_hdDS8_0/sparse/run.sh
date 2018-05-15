/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/voc0712/JDetNet/20180512_18-25_ds_PSP_dsFac_32_hdDS8_1/sparse/solver.prototxt" \
--weights="training/voc0712/JDetNet/20180512_18-25_ds_PSP_dsFac_32_hdDS8_1/l1reg/voc0712_ssdJacintoNetV2_iter_60000.caffemodel" \
--gpu "0" 2>&1 | tee training/voc0712/JDetNet/20180512_18-25_ds_PSP_dsFac_32_hdDS8_1/sparse/run.log
