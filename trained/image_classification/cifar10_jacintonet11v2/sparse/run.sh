cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/cifar10_jacintonet11v2_2017-08-01_13-11-28/sparse/solver.prototxt" \
--weights="training/cifar10_jacintonet11v2_2017-08-01_13-11-28/l1reg/cifar10_jacintonet11v2_iter_64000.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/cifar10_jacintonet11v2_2017-08-01_13-11-28/sparse/run.log
