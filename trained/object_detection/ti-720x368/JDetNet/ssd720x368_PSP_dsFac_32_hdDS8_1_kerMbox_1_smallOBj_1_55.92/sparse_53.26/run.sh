#export CUDA_VISIBLE_DEVICES=1
/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/sparse/solver.prototxt" \
--weights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/l1reg_55.57/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_38000_55.57.caffemodel" \
--gpu "0" 2>&1 | tee training/ti-vgg-720x368-v2/JDetNet/ssd768x320_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1_51.41/sparse/run.log
