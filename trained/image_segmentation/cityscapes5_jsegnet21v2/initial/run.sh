cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cityscapes5_jsegnet21v2_2017-08-15_19-04-07/initial/solver.prototxt" \
--weights="training/imagenet_jacintonet11v2_iter_320000.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/cityscapes5_jsegnet21v2_2017-08-15_19-04-07/initial/run.log
