cd /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin
/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/l1reg/solver.prototxt" \
--weights="training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/initial/cityscapes5_jsegnet21v2_iter_120000.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/l1reg/run.log
