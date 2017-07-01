cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cityscapes20_jsegnet21v2_2017-06-30_19-26-17/initial/solver.prototxt" \
--weights="training/imagenet_jacintonet11_v2_bn_iter_160000.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/cityscapes20_jsegnet21v2_2017-06-30_19-26-17/initial/train.log
