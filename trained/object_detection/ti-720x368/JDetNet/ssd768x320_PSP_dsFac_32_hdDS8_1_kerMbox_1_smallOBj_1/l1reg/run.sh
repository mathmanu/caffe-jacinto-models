/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/ti-vgg-720x368-v2/JDetNet/20180323_22-48_ds_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/l1reg/solver.prototxt" \
--weights="training/ti-vgg-720x368-v2/JDetNet/20180323_22-48_ds_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/initial/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_40000_51.41.caffemodel" \
--gpu "0,1" 2>&1 | tee training/ti-vgg-720x368-v2/JDetNet/20180323_22-48_ds_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/l1reg/run.log
