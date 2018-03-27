/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/voc0712/JDetNet/20180317_09-32_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_3_1stHdSameOpCh_1_bnMbox_1/initial/solver.prototxt" \
--weights="/user/a0875091/files/work/bitbucket_TI/caffe-jacinto-models/scripts/training/voc0712/JDetNet/20180316_12-11_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_1_1stHdSameOpCh_1_bnMbox_1/initial/voc0712_ssdJacintoNetV2_iter_94000_65.10.caffemodel" \
--gpu "0,1" 2>&1 | tee training/voc0712/JDetNet/20180317_09-32_ds_PSP_dsFac_32_fc_0_hdDS8_1_cnctHD_0_baseNW3hd_0_kerMbox_3_1stHdSameOpCh_1_bnMbox_1/initial/run.log
