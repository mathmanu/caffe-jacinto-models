/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin test_detection \
--model="training/voc0712/mobiledetnet-0.5/20180510_14-48_ds_PSP_dsFac_32_hdDS8_1/test_quantize/test.prototxt" \
--iterations="496" \
--weights="training/voc0712/mobiledetnet-0.5/20180510_14-48_ds_PSP_dsFac_32_hdDS8_1/initial/voc0712_mobiledetnet-0.5_iter_120000.caffemodel" \
--gpu "0" 2>&1 | tee training/voc0712/mobiledetnet-0.5/20180510_14-48_ds_PSP_dsFac_32_hdDS8_1/test_quantize/run.log
