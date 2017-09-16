cd /user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin
/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/initial/solver.prototxt" \
--weights="training/imagenet_jacintonet11v2_iter_320000.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/cityscapes5_jsegnet21v2_2017-09-16_10-06-43/initial/run.log
