/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/initial/solver.prototxt" \
--snapshot="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/initial/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_18000.solverstate" \
--gpu "0,1" 2>&1 | tee -a training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/initial/run.log
