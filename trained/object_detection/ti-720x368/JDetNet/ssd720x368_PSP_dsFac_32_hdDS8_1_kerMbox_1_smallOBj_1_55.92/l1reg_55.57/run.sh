/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/l1reg/solver.prototxt" \
--weights="training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/initial/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_10000_55.92.caffemodel" \
--gpu "0,1" 2>&1 | tee training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/l1reg/run.log
