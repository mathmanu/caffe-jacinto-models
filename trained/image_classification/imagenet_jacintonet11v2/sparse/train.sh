cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/imagenet_jacintonet11v2_2017-07-05_21-31-44/sparse/solver.prototxt" \
--weights="training/imagenet_jacintonet11v2_2017-07-05_21-31-44/initial/imagenet_jacintonet11v2_iter_100.caffemodel" \
--gpu "0,1,2" 2>&1 | tee training/imagenet_jacintonet11v2_2017-07-05_21-31-44/sparse/train.log
