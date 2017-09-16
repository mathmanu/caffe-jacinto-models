cd /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin
/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin test \
--model="training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/test_quantize/test.prototxt" \
--iterations="50" \
--weights="training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/sparse/cityscapes5_jsegnet21v2_iter_60000.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/test_quantize/run.log
