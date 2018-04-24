cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/imagenet_mobilenet-0.5_2017-08-24_18-29-42/initial/solver.prototxt" \
--gpu "0,1,2" 2>&1 | tee training/imagenet_mobilenet-0.5_2017-08-24_18-29-42/initial/run.log
