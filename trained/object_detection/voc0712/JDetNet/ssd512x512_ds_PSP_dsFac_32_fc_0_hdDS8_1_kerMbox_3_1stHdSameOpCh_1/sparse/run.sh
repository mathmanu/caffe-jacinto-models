/user/a0875091/files/work/bitbucket_TI/caffe-jacinto//build/tools/caffe.bin train \
--solver="training/voc0712/JDetNet/ssd512x512_ds_PSP_dsFac_32_fc_0_hdDS8_1_baseNW3hd_0_kerMbox_3_1stHdSameOpCh_1_68.66/sparse_0.25_0.8/solver.prototxt" \
--weights="training/voc0712/JDetNet/ssd512x512_ds_PSP_dsFac_32_fc_0_hdDS8_1_baseNW3hd_0_kerMbox_3_1stHdSameOpCh_1_68.66/l1reg_68.07/voc0712_ssdJacintoNetV2_iter_60000_68.07.caffemodel" \
--gpu "0" 2>&1 | tee training/voc0712/JDetNet/ssd512x512_ds_PSP_dsFac_32_fc_0_hdDS8_1_baseNW3hd_0_kerMbox_3_1stHdSameOpCh_1_68.66/sparse_0.25_0.8/run.log
