cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cifar10_jacintonet11v2_2017-06-30_01-13-02/initial/solver.prototxt" \
--gpu "0,1,2" 2>&1 | tee training/cifar10_jacintonet11v2_2017-06-30_01-13-02/initial/train.log
