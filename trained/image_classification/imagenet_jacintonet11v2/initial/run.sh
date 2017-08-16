cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/imagenet_jacintonet11v2_2017-08-14_19-50-34/initial/solver.prototxt" \
--gpu "0,1,2" 2>&1 | tee training/imagenet_jacintonet11v2_2017-08-14_19-50-34/initial/run.log
