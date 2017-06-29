cd /home/a0393608/files/work/code/vision/ti/bitbucket/algoref/caffe-jacinto-models/scripts
../../caffe-jacinto/build/tools/caffe.bin train \
--solver="training/imagenet_jacintonet11v2_2017-06-28_19-45-45/initial/solver.prototxt" \
--weights="/data/mmcodec_video2_tier3/users/manu/experiments/object/classification/2017.06.new_script/caffe-0.15/jacintonet11_imagenet_2017.06.12_lmdb_caffe-0.15-2gpu(60.89%)/stage0/jacintonet11_iter_320000.caffemodel" \
--gpu "0,1" 2>&1 | tee training/imagenet_jacintonet11v2_2017-06-28_19-45-45/initial/train.log
