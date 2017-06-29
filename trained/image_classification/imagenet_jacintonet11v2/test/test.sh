cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin test \
--model="training/imagenet_jacintonet11v2_2017-06-28_19-45-45/test/test.prototxt" \
--iterations="1000" \
--weights="training/imagenet_jacintonet11v2_2017-06-28_19-45-45/sparse/imagenet_jacintonet11v2_iter_160000.caffemodel" \
--gpu "0,1" 2>&1 | tee training/imagenet_jacintonet11v2_2017-06-28_19-45-45/test/test.log
