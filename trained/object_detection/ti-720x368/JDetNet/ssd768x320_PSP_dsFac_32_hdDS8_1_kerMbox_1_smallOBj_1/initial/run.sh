/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/ti-vgg-720x368-v2/JDetNet/20180323_22-48_ds_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/initial/solver.prototxt" \
--weights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/ti-vgg-720x368-v2/JDetNet/20180211_01-20_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1/sparse_fac0.5_0.8_53.26/ti-vgg-720x368-v2_ssdJacintoNetV2_iter_38000_53.26.caffemodel" \
--gpu "0,1" 2>&1 | tee training/ti-vgg-720x368-v2/JDetNet/20180323_22-48_ds_PSP_dsFac_32_hdDS8_1_kerMbox_1_smallOBj_1/initial/run.log
