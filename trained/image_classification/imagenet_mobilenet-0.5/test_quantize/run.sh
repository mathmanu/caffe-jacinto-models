/user/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto/build/tools/caffe.bin train \
--solver="training/imagenet_mobilenet-0.5_2018-06-07_22-55-32/test_quantize/solver.prototxt" \
--weights="training/imagenet_mobilenet-0.5_2018-06-07_22-55-32/initial/imagenet_mobilenet-0.5_iter_320000.caffemodel" \
--gpu "0" 2>&1 | tee training/imagenet_mobilenet-0.5_2018-06-07_22-55-32/test_quantize/run.log
