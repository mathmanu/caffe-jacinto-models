cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cityscapes5_jsegnet21v2_2017-07-02_23-02-42/initial/solver.prototxt" \
--weights="training/imagenet_jacintonet11_v2_bn_iter_160000.caffemodel" \
--gpu "0,1" 2>&1 | tee training/cityscapes5_jsegnet21v2_2017-07-02_23-02-42/initial/train.log
